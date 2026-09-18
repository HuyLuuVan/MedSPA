#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
from typing import Dict, Any, List, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
from tqdm import tqdm  

from radgraph import RadGraph

CONDITIONS = [
    "enlarged_cardiomediastinum",
    "cardiomegaly",
    "lung_opacity",
    "lung_lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural_effusion",
    "pleural_other",
    "fracture",
    "support_devices",
    "no_finding",
]

LABEL_MEANING = {
    0: "blank",
    1: "positive",
    2: "negative",
    3: "uncertain",
}


RADGRAPH_MODEL = RadGraph(model_type="radgraph")


def extract_findings_from_text(text: str, radgraph_model: RadGraph = RADGRAPH_MODEL) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""

    annotations = radgraph_model([text])
    if not annotations:
        return ""

    ann = list(annotations.values())[0]
    entities = ann.get("entities", {}) or {}
    if not entities:
        return ""

    def _build_modifiers_map(entities: Dict[str, dict]) -> Dict[str, List[str]]:
 
        modifiers_for: Dict[str, List[str]] = {}
        for eid, ent in entities.items():
            for rel in ent.get("relations", []):
                if len(rel) != 2:
                    continue
                rel_type, target_id = rel
                if rel_type == "modify" and target_id in entities:
                    modifiers_for.setdefault(target_id, []).append(eid)
        return modifiers_for

    def _is_pure_modifier(ent: dict) -> bool:
        
        has_modify = False
        has_strong_rel = False  # located_at / suggestive_of

        for rel_type, tgt in ent.get("relations", []):
            if rel_type == "modify":
                has_modify = True
            if rel_type in ("located_at", "suggestive_of"):
                has_strong_rel = True

        # Pure modifier = has "modify" but no strong relation
        return has_modify and not has_strong_rel

    def _build_phrase(
        eid: str,
        entities: Dict[str, dict],
        modifiers_for: Dict[str, List[str]],
    ) -> str:
        ent_ids = [eid] + modifiers_for.get(eid, [])
        items = []
        for i in ent_ids:
            ent = entities[i]
            tokens = (ent.get("tokens") or "").strip()
            if not tokens:
                continue

            label = ent.get("label", "")
            if label.startswith("ANAT") and any(ch.isdigit() for ch in tokens):
                continue

            start = ent.get("start_ix", 0)
            items.append((start, tokens))

        if not items:
            return ""

        items.sort(key=lambda x: x[0])

        phrase_parts = [tok for _, tok in items]
        phrase = " ".join(phrase_parts).strip()
        phrase = phrase.replace(" . ", ". ")
        phrase = phrase.replace(" ,", ",")
        return phrase

    def _radgraph_to_facts_single(ann_local: Dict) -> List[str]:
        
        entities_local = ann_local.get("entities", {}) or {}
        if not entities_local:
            return []

        modifiers_for_local = _build_modifiers_map(entities_local)
        facts_local: List[str] = []

        for eid, ent in entities_local.items():
            label = ent.get("label", "")
            if not label.startswith("OBS"):  # only Observation entities
                continue

            # Skip OBS that act only as pure modifiers
            if _is_pure_modifier(ent):
                continue

            obs_phrase = _build_phrase(eid, entities_local, modifiers_for_local)
            if not obs_phrase:
                continue

            # Prefix based on certainty level
            prefix = ""
            if label.endswith("DA"):
                prefix = "No "
            elif label.endswith("U"):
                prefix = "Possible "

            anat_ids = [
                tgt_id
                for (rel_type, tgt_id) in ent.get("relations", [])
                if rel_type == "located_at" and tgt_id in entities_local
            ]
            anat_phrases = []
            for a_id in anat_ids:
                anat_phrase = _build_phrase(a_id, entities_local, modifiers_for_local)
                if anat_phrase:
                    anat_phrases.append(anat_phrase)

            sugg_ids = [
                tgt_id
                for (rel_type, tgt_id) in ent.get("relations", [])
                if rel_type == "suggestive_of" and tgt_id in entities_local
            ]
            sugg_phrases = []
            for s_id in sugg_ids:
                s_phrase = _build_phrase(s_id, entities_local, modifiers_for_local)
                if s_phrase:
                    sugg_phrases.append(s_phrase)

            core = f"{prefix}{obs_phrase}".strip()
            if anat_phrases:
                core = f"{core} at {', '.join(anat_phrases)}"
            if sugg_phrases:
                core = f"{core}, suggestive of {', '.join(sugg_phrases)}"

            if not core:
                continue

            if len(core.split()) < 2:
                continue

            if not core.endswith("."):
                core += "."
            core = core[0].upper() + core[1:]

            if core not in facts_local:
                facts_local.append(core)

        return facts_local

    facts = _radgraph_to_facts_single(ann)
    if not facts:
        return ""

    return " ".join(facts).strip()


class CheXbert(nn.Module):
    def __init__(self, checkpoint_path: str, device: str, p: float = 0.1):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            model_max_length=512,
        )
        config = BertConfig.from_pretrained("bert-base-uncased")

        with torch.no_grad():
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)
            hidden_size = self.bert.pooler.dense.in_features

            self.linear_heads = nn.ModuleList(
                [nn.Linear(hidden_size, 4, bias=True) for _ in range(13)]
            )
            self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

            state = torch.load(checkpoint_path, map_location=device)[
                "model_state_dict"
            ]
            new_state = OrderedDict()
            for k, v in state.items():
                if "bert" in k:
                    nk = k.replace("module.bert.", "bert.")
                elif "linear_heads" in k:
                    nk = k.replace("module.linear_heads.", "linear_heads.")
                else:
                    nk = k
                new_state[nk] = v

            incompatible = self.load_state_dict(new_state, strict=False)
            if getattr(incompatible, "unexpected_keys", []):
                print(
                    f"[chexbert] ignoring unexpected keys: "
                    f"{sorted(list(incompatible.unexpected_keys))}"
                )
            if getattr(incompatible, "missing_keys", []):
                print(
                    f"[chexbert] missing keys: "
                    f"{sorted(list(incompatible.missing_keys))}"
                )

        self.eval()

    @torch.no_grad()
    def _chunk_text_to_encodings(
        self, text: str, stride: int = 128
    ) -> Dict[str, torch.Tensor]:
        s = text if isinstance(text, str) else ""
        s = s.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")
        s = re.sub(r"\s+", " ", s)
        enc = self.tokenizer(
            s,
            truncation=True,
            max_length=512,
            stride=stride,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def _predict_chunk_batch(self, enc_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run BERT + linear heads over one batch of chunks and return per-chunk labels.
        """
        last_hidden = self.bert(**enc_batch)[0]
        cls = self.dropout(last_hidden[:, 0, :])
        logits = [head(cls) for head in self.linear_heads]
        labels = torch.stack([lg.argmax(dim=1) for lg in logits], dim=1)
        return labels

    @torch.no_grad()
    def _aggregate_labels_over_chunks(
        self,
        labels_chunks: torch.Tensor,
        agg: str = "vote",
        treat_uncertain_as_positive: bool = False,
    ) -> torch.Tensor:
        
        num_labels = labels_chunks.size(1)
        out = []
        for j in range(num_labels):
            col = labels_chunks[:, j]  # [num_chunks]
            if agg == "vote":
                val = torch.mode(col, dim=0).values.item()
                out.append(val)
            else:
                # "any_positive" mode (kept for completeness, not used here)
                if j < 13:
                    is_pos = col == 1
                    if treat_uncertain_as_positive:
                        is_pos = is_pos | (col == 3)
                    val = 1 if is_pos.any().item() else 0
                else:
                    # no_finding
                    val = 1 if (col == 1).any().item() else 0
                out.append(val)
        return torch.tensor(out, dtype=torch.long, device=labels_chunks.device)

    @torch.no_grad()
    def forward(
        self,
        texts: List[str],
        stride: int = 128,
        agg: str = "vote",
        treat_uncertain_as_positive: bool = False,
        mbatch_size: int = 16,
    ) -> torch.Tensor:
        
        results = []
        for s in texts:
            enc = self._chunk_text_to_encodings(s, stride=stride)
            n = enc["input_ids"].size(0)
            chunk_labels = []
            for i in range(0, n, mbatch_size):
                sl = slice(i, min(i + mbatch_size, n))
                batch = {
                    "input_ids": enc["input_ids"][sl],
                    "attention_mask": enc["attention_mask"][sl],
                }
                lbl = self._predict_chunk_batch(batch)
                chunk_labels.append(lbl)
            chunk_labels = torch.cat(chunk_labels, dim=0)
            merged = self._aggregate_labels_over_chunks(
                chunk_labels,
                agg=agg,
                treat_uncertain_as_positive=treat_uncertain_as_positive,
            )
            results.append(merged.unsqueeze(0))
        return torch.cat(results, dim=0)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, data: Any):
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def build_reasoning_text(sample: Dict[str, Any]) -> str:
    
    steps = sample.get("steps", [])
    if not isinstance(steps, list):
        return ""

    chunks: List[str] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        title = (st.get("title") or "").strip()
        content = (st.get("content") or "").strip()
        if title and content:
            chunks.append(f"{title}: {content}")
        elif title:
            chunks.append(title)
        elif content:
            chunks.append(content)

    return " ".join(chunks).strip()



def build_memory_label(steps: List[Dict[str, Any]]) -> Dict[str, str]:
    
    memory: Dict[str, str] = {}
    for st in steps or []:
        d = (st or {}).get("chexbert_step_labels", {}) or {}
        for k, v in d.items():
            if k not in CONDITIONS:
                continue
            if v is None:
                continue
            if v != "blank":
                memory[k] = v
    return memory



def sanitize_sample_paths(sample: Dict[str, Any]) -> Dict[str, Any]:
    
    if not isinstance(sample, dict):
        return sample

    sanitized = dict(sample)

    for key in ("image", "image_path"):
        value = sanitized.get(key)

        if isinstance(value, str) and value.strip():
            sanitized[key] = os.path.basename(value.strip())

        elif isinstance(value, list):
            sanitized[key] = [
                os.path.basename(v.strip()) if isinstance(v, str) and v.strip() else v
                for v in value
            ]

    return sanitized


def reorder_sample_keys(sample: Dict[str, Any]) -> Dict[str, Any]:
    
    if not isinstance(sample, dict):
        return sample

    preferred_order = [
        "image",
        "caption",
        "predicted_caption",                
        "trusted_finding",
        "predicted_finding",                
        "steps",                            
        "memory_label",                     
        "chexbert_caption_labels",
        "chexbert_predicted_caption_labels",  
        "reason_finding",                   
        "chexbert_finding_labels",
        "chexbert_predicted_finding_labels",  
        "chexbert_stepfinding_labels",
        "chexbert_reason_labels",
    ]

    new_sample = OrderedDict()

    for key in preferred_order:
        if key in sample:
            new_sample[key] = sample[key]

    for key, value in sample.items():
        if key not in new_sample:
            new_sample[key] = value

    return new_sample


def attach_chexbert_labels(
    input_path: str,
    out_json: str,
    chexbert_ckpt: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 16,
    save_every: int = 100,
    skip_step_finding: bool = False,
):

    if chexbert_ckpt is None:
        chexbert_ckpt = ".../chexbert.pth"

    if not os.path.exists(chexbert_ckpt):
        raise FileNotFoundError("CheXbert checkpoint was not found.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[info] CheXbert checkpoint     : configured")
    print(f"[info] Device                 : {device}")
    print(f"[info] skip_step_finding      : {skip_step_finding}")

    if os.path.exists(out_json):
        print("[info] Found existing output JSON. Resuming.")
        data = load_json(out_json)
    else:
        print("[info] Loading input JSON.")
        data = load_json(input_path)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON to be a list of samples.")

    print("[info] Loading CheXbert model ...")
    model = CheXbert(chexbert_ckpt, device=device).to(device)
    model.eval()

    caption_entries: List[Dict[str, Any]] = []
    caption_texts: List[str] = []

    finding_entries: List[Dict[str, Any]] = []
    finding_texts: List[str] = []

    reason_entries: List[Dict[str, Any]] = []
    reason_texts: List[str] = []

    predicted_caption_entries: List[Dict[str, Any]] = []
    predicted_caption_texts: List[str] = []

    predicted_finding_entries: List[Dict[str, Any]] = []
    predicted_finding_texts: List[str] = []

    stepfinding_entries: List[Dict[str, Any]] = []
    stepfinding_texts: List[str] = []

    perstep_entries: List[Dict[str, Any]] = []
    perstep_texts: List[str] = []

    for sample in data:
        if not isinstance(sample, dict):
            continue

        steps = sample.get("steps", [])
        step_finding_list: List[str] = []

        if not skip_step_finding:
            if isinstance(steps, list) and steps:
                for st in steps:
                    if not isinstance(st, dict):
                        continue

                    title = (st.get("title") or "").strip()
                    content = (st.get("content") or "").strip()

                    if title and content:
                        step_text = f"{title}: {content}"
                    elif title:
                        step_text = title
                    elif content:
                        step_text = content
                    else:
                        step_text = ""

                    # Only call RadGraph if step_finding does not already exist
                    existing_sf = (st.get("step_finding") or "").strip()
                    if not existing_sf and step_text:
                        step_finding = extract_findings_from_text(step_text)
                        st["step_finding"] = step_finding
                    else:
                        step_finding = existing_sf

                    if step_finding:
                        # Add to the concatenated sample-level step_finding
                        step_finding_list.append(step_finding)

                        # If this step has no labels yet, queue it for per-step CheXbert
                        if "chexbert_step_labels" not in st:
                            perstep_entries.append(st)
                            perstep_texts.append(step_finding)

            if step_finding_list:
                step_concat_text = " ".join(step_finding_list).strip()
            else:
                step_concat_text = ""
        else:
            step_concat_text = ""

        reason_finding_existing = (sample.get("reason_finding") or "").strip()
        if not reason_finding_existing:
            reason_text = build_reasoning_text(sample)
            if reason_text:
                reason_finding_new = extract_findings_from_text(reason_text)
                sample["reason_finding"] = reason_finding_new

        reason_finding_text = (sample.get("reason_finding") or "").strip()

        cap = (sample.get("caption") or "").strip()
        if cap and "chexbert_caption_labels" not in sample:
            caption_entries.append(sample)
            caption_texts.append(cap)

        finding = (sample.get("trusted_finding") or "").strip()
        if finding and "chexbert_finding_labels" not in sample:
            finding_entries.append(sample)
            finding_texts.append(finding)

        if reason_finding_text and "chexbert_reason_labels" not in sample:
            reason_entries.append(sample)
            reason_texts.append(reason_finding_text)

        if (not skip_step_finding) and step_concat_text and "chexbert_stepfinding_labels" not in sample:
            stepfinding_entries.append(sample)
            stepfinding_texts.append(step_concat_text)

        predicted_cap = (sample.get("predicted_caption") or "").strip()
        if predicted_cap:
            predicted_find = (sample.get("predicted_finding") or "").strip()
            if not predicted_find:
                predicted_find = extract_findings_from_text(predicted_cap)
                sample["predicted_finding"] = predicted_find

            if "chexbert_predicted_caption_labels" not in sample:
                predicted_caption_entries.append(sample)
                predicted_caption_texts.append(predicted_cap)

            if predicted_find and "chexbert_predicted_finding_labels" not in sample:
                predicted_finding_entries.append(sample)
                predicted_finding_texts.append(predicted_find)

    print(f"[info] caption texts to run CheXbert on      : {len(caption_texts)}")
    print(f"[info] finding texts to run CheXbert on      : {len(finding_texts)}")
    print(f"[info] reasoning findings to run CheXbert on : {len(reason_texts)}")
    print(f"[info] predicted captions count              : {len(predicted_caption_texts)}")
    print(f"[info] predicted findings count              : {len(predicted_finding_texts)}")
    print(f"[info] step_finding_concat (internal) count  : {len(stepfinding_texts)}")
    print(f"[info] per-step step_finding count           : {len(perstep_texts)}")

    # Helper: run model on a list of texts in batches and return list of label vectors
    def run_chexbert_in_batches(texts: List[str], desc: str) -> List[List[int]]:
        if not texts:
            return []
        labels_all: List[List[int]] = []
        n = len(texts)
        with torch.no_grad():
            for i in tqdm(range(0, n, batch_size), desc=desc):
                j = min(i + batch_size, n)
                batch = texts[i:j]
                lbl = model(
                    batch,
                    stride=128,
                    agg="vote",
                    treat_uncertain_as_positive=False,
                    mbatch_size=batch_size,
                )  # (B, 14)
                labels_all.extend(lbl.cpu().numpy().tolist())
        return labels_all

    # Run CheXbert for each text type with progress bars
    print("[chexbert] Running for captions ...")
    caption_labels = run_chexbert_in_batches(caption_texts, desc="Captions")

    print("[chexbert] Running for trusted findings ...")
    finding_labels = run_chexbert_in_batches(finding_texts, desc="Trusted findings")

    print("[chexbert] Running for reason_finding (global) ...")
    reason_labels = run_chexbert_in_batches(reason_texts, desc="Reason findings")

    print("[chexbert] Running for predicted_caption ...")
    predicted_caption_labels = run_chexbert_in_batches(
        predicted_caption_texts, desc="Predicted captions"
    )

    print("[chexbert] Running for predicted_finding ...")
    predicted_finding_labels = run_chexbert_in_batches(
        predicted_finding_texts, desc="Predicted findings"
    )

    if not skip_step_finding:
        print("[chexbert] Running for step_finding (concat, internal) ...")
        stepfinding_labels = run_chexbert_in_batches(
            stepfinding_texts, desc="Step findings concat"
        )
        print("[chexbert] Running for per-step step_finding ...")
        perstep_labels = run_chexbert_in_batches(
            perstep_texts, desc="Step findings (per-step)"
        )
    else:
        print("[info] Skipping step_finding CheXbert inference because --skip_step_finding is set.")
        stepfinding_labels = []
        perstep_labels = []

    labeled_count = 0  

    def save_reordered():
        reordered_data = [
            reorder_sample_keys(sanitize_sample_paths(s))
            if isinstance(s, dict)
            else s
            for s in data
        ]
        atomic_write_json(out_json, reordered_data)

    def attach_labels(
        entries: List[Dict[str, Any]],
        labels_list: List[List[int]],
        names_key: str,
    ):
        nonlocal labeled_count
        for sample, lab_vec in zip(entries, labels_list):

            if names_key == "chexbert_step_labels":
                label_names_dict: Dict[str, str] = {}
                for idx, cond in enumerate(CONDITIONS):
                    lab_id = int(lab_vec[idx])
                    lab_name = LABEL_MEANING.get(lab_id, f"unknown({lab_id})")
                    if lab_name == "blank":
                        continue
                    label_names_dict[cond] = lab_name

                if not label_names_dict:
                    continue

            else:
                label_names_dict: Dict[str, str] = {}
                for idx, cond in enumerate(CONDITIONS):
                    lab_id = int(lab_vec[idx])
                    label_names_dict[cond] = LABEL_MEANING.get(
                        lab_id, f"unknown({lab_id})"
                    )

            sample[names_key] = label_names_dict

            labeled_count += 1
            if save_every > 0 and labeled_count % save_every == 0:
                print(
                    f"[info] save_every={save_every}: saving intermediate JSON "
                    f"(newly_labeled_count_in_this_run={labeled_count})"
                )
                save_reordered()

    attach_labels(
        caption_entries,
        caption_labels,
        names_key="chexbert_caption_labels",
    )
    attach_labels(
        finding_entries,
        finding_labels,
        names_key="chexbert_finding_labels",
    )
    attach_labels(
        reason_entries,
        reason_labels,
        names_key="chexbert_reason_labels",
    )
    attach_labels(
        predicted_caption_entries,
        predicted_caption_labels,
        names_key="chexbert_predicted_caption_labels",
    )
    attach_labels(
        predicted_finding_entries,
        predicted_finding_labels,
        names_key="chexbert_predicted_finding_labels",
    )

    if not skip_step_finding:
        attach_labels(
            stepfinding_entries,
            stepfinding_labels,
            names_key="chexbert_stepfinding_labels",
        )
        attach_labels(
            perstep_entries,
            perstep_labels,
            names_key="chexbert_step_labels",
        )

    for sample in data:
        if not isinstance(sample, dict):
            continue
        if "memory_label" in sample:
            continue
        steps = sample.get("steps", []) or []
        if isinstance(steps, list):
            sample["memory_label"] = build_memory_label(steps)
        else:
            sample["memory_label"] = {}

    save_reordered()
    print("[done] Saved labeled JSON.")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="shard*.json",
        help="Path to qwen_shard JSON (list of samples).",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output JSON path with CheXbert labels attached (and used for resume).",
    )
    ap.add_argument(
        "--chexbert_ckpt",
        default="../chexbert.pth",
        help="Path to the CheXbert checkpoint file.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="(Optional) cuda/cpu. If None → auto (cuda if available, else cpu).",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for CheXbert inference.",
    )
    ap.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Save intermediate JSON every N newly labeled entries (0 = only final save).",
    )
    ap.add_argument(
        "--skip_step_finding",
        action="store_true",
        help="If set, skip RadGraph extraction per step and CheXbert stepfinding labeling.",
    )

    args = ap.parse_args()

    attach_chexbert_labels(
        input_path=args.input,
        out_json=args.output,
        chexbert_ckpt=args.chexbert_ckpt,
        device=args.device,
        batch_size=args.batch_size,
        save_every=args.save_every,
        skip_step_finding=args.skip_step_finding,
    )


if __name__ == "__main__":
    main()