#!/usr/bin/env python3

import os
import re
import json
import argparse
import time
import tempfile
from typing import Dict, List, Any, Tuple

import numpy as np
from radgraph import F1RadGraph

from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer


def safe_float_or_neg1(x) -> float:
    try:
        if x is None:
            return -1.0
        return float(x)
    except Exception:
        return -1.0


def _load_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def atomic_write(path: str, obj: Any):
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise


def _iter_input_samples(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and "samples" in payload and isinstance(payload["samples"], list):
        return payload["samples"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported input format. Expect a list OR a dict with key 'samples'.")


def _norm_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _public_image_id(value: str) -> str:
    value = _norm_text(value)
    if not value:
        return ""
    if os.path.isabs(value):
        return os.path.basename(value)
    return value


def _get_str_field(sample: Dict[str, Any], key: str) -> str:
    val = sample.get(key, None)
    return _norm_text(val)


def _get_image_id(sample: Dict[str, Any]) -> str:
    img = sample.get("image", None)
    if isinstance(img, str) and img.strip():
        return _public_image_id(img)
    img2 = sample.get("image_path", None)
    if isinstance(img2, str) and img2.strip():
        return _public_image_id(img2)
    return ""


def _index_by_image(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        img = _get_image_id(s)
        if img:
            out[img] = s
    return out


def _get_gt_for_sample(sample: Dict[str, Any]) -> str:
    return _get_str_field(sample, "caption")


def _compute_text_metrics_bulk(
    gts: Dict[str, List[str]],
    res: Dict[str, List[str]],
    lowercase: bool = False,
    skip_meteor: bool = False
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    if not gts or not res:
        global_scores: Dict[str, float] = {
            "BLEU_1": 0.0,
            "BLEU_2": 0.0,
            "BLEU_3": 0.0,
            "BLEU_4": 0.0,
            "ROUGE_L": 0.0,
        }
        if not skip_meteor:
            global_scores["METEOR"] = 0.0
        return global_scores, {}

    import re as _re
    try:
        from pycocoevalcap.bleu.bleu import Bleu
    except Exception as e:
        raise ImportError(f"Cannot import BLEU from pycocoevalcap: {type(e).__name__}")
    try:
        from pycocoevalcap.rouge.rouge import Rouge
    except Exception:
        from pycocoevalcap.rouge import Rouge

    MeteorClass = None
    if not skip_meteor:
        try:
            from pycocoevalcap.meteor.meteor import Meteor as _Meteor
            MeteorClass = _Meteor
        except Exception as e:
            print(f"[warn] METEOR disabled: {type(e).__name__}")
            skip_meteor = True

    def norm_for_coco(s: str) -> str:
        if lowercase:
            s = s.lower()
        s = (s + " ").replace(". ", " . ").replace(" - ", "-")
        return s.strip()

    def sanitize_for_meteor(s: str) -> str:
        s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        s = _re.sub(r"\s+", " ", s).strip()
        return s

    gts_n = {k: [norm_for_coco(x) for x in v] for k, v in gts.items()}
    res_n = {k: [norm_for_coco(v[0])] for k, v in res.items()}
    image_ids = list(res_n.keys())

    global_scores: Dict[str, float] = {}
    per_image: Dict[str, Dict[str, float]] = {k: {} for k in image_ids}

    def _fill_per_image(container, metric_key: str):
        if isinstance(container, dict):
            for img, val in container.items():
                if img in per_image:
                    per_image[img][metric_key] = float(val)
        else:
            try:
                seq = list(container)
                for i, img in enumerate(image_ids):
                    if i < len(seq):
                        per_image[img][metric_key] = float(seq[i])
            except Exception:
                pass

    bleu = Bleu(4)
    try:
        bleu_scores, bleu_img = bleu.compute_score(gts_n, res_n, verbose=0)
    except TypeError:
        bleu_scores, bleu_img = bleu.compute_score(gts_n, res_n)
    global_scores["BLEU_1"], global_scores["BLEU_2"], global_scores["BLEU_3"], global_scores["BLEU_4"] = [
        float(x) for x in bleu_scores
    ]
    for n_idx, key in enumerate(["bleu1", "bleu2", "bleu3", "bleu4"]):
        if n_idx < len(bleu_img):
            _fill_per_image(bleu_img[n_idx], key)

    rouge = Rouge()
    try:
        rouge_global, rouge_img = rouge.compute_score(gts_n, res_n, verbose=0)
    except TypeError:
        rouge_global, rouge_img = rouge.compute_score(gts_n, res_n)
    global_scores["ROUGE_L"] = float(rouge_global)
    _fill_per_image(rouge_img, "rougeL")

    if not skip_meteor and MeteorClass is not None:
        try:
            gts_m = {k: [sanitize_for_meteor(x) for x in v] for k, v in gts.items()}
            res_m = {k: [sanitize_for_meteor(v[0])] for k, v in res.items()}
            meteor = MeteorClass()
            meteor_global, meteor_img = meteor.compute_score(gts_m, res_m)
            global_scores["METEOR"] = float(meteor_global)
            _fill_per_image(meteor_img, "meteor")
        except Exception as e:
            print(f"[warn] METEOR disabled at runtime: {type(e).__name__}")

    return global_scores, per_image


CONDITIONS = [
    'enlarged_cardiomediastinum', 'cardiomegaly', 'lung_opacity', 'lung_lesion',
    'edema', 'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax',
    'pleural_effusion', 'pleural_other', 'fracture', 'support_devices', 'no_finding',
]


class CheXbert(nn.Module):
    def __init__(self, checkpoint_path, device, p=0.1):
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=512)
        config = BertConfig.from_pretrained('bert-base-uncased')
        with torch.no_grad():
            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)
            hidden_size = self.bert.pooler.dense.in_features
            self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
            self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))  # no_finding head

            state = torch.load(checkpoint_path, map_location=device)['model_state_dict']
            new_state = OrderedDict()
            for k, v in state.items():
                if 'bert' in k:
                    nk = k.replace('module.bert.', 'bert.')
                elif 'linear_heads' in k:
                    nk = k.replace('module.linear_heads.', 'linear_heads.')
                else:
                    nk = k
                new_state[nk] = v
            incompatible = self.load_state_dict(new_state, strict=False)
            if getattr(incompatible, "unexpected_keys", []):
                print(f"[chexbert] ignoring unexpected keys: {sorted(list(incompatible.unexpected_keys))}")
            if getattr(incompatible, "missing_keys", []):
                print(f"[chexbert] missing keys: {sorted(list(incompatible.missing_keys))}")
        self.eval()

    @torch.no_grad()
    def _predict_chunk_batch(self, enc_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        last_hidden = self.bert(**enc_batch)[0]
        cls = self.dropout(last_hidden[:, 0, :])
        logits = [head(cls) for head in self.linear_heads]
        labels = torch.stack([lg.argmax(dim=1) for lg in logits], dim=1)
        return labels

    @torch.no_grad()
    def _chunk_text_to_encodings(self, text: str, stride: int = 128) -> Dict[str, torch.Tensor]:
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
            return_tensors="pt"
        )
        if len(enc["input_ids"]) > 1:
            print(f"[info] Long text split into {len(enc['input_ids'])} chunks "
                  f"({len(enc['input_ids'][0])} tokens each)")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def _aggregate_labels_over_chunks(
        self,
        labels_chunks: torch.Tensor,
        agg: str = "vote",
        treat_uncertain_as_positive: bool = False
    ) -> torch.Tensor:
        """
        Aggregate predictions over chunks into a single label per condition.
        """
        num_labels = labels_chunks.size(1)
        out = []
        for j in range(num_labels):
            col = labels_chunks[:, j]  # shape: [num_chunks]
            if agg == "vote":
                # majority vote over discrete label {0,1,2,3}
                val = torch.mode(col, dim=0).values.item()
                out.append(val)
            else:
                # "any_positive" style: is there any chunk flagged as positive?
                if j < 13:
                    is_pos = (col == 1)
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
        mbatch_size: int = 16
    ) -> torch.Tensor:
        """
        Return shape: [N, num_labels], each entry is a discrete label.
        """
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
                treat_uncertain_as_positive=treat_uncertain_as_positive
            )
            results.append(merged.unsqueeze(0))
        return torch.cat(results, dim=0)


class CheXbertMetrics:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        mbatch_size: int = 16,
        include_no_finding: bool = False,
        treat_uncertain_as_positive: bool = False,
    ):
        self.device = device
        self.bs = mbatch_size
        self.include_no_finding = include_no_finding
        self.treat_uncertain_as_positive = treat_uncertain_as_positive
        self.model = CheXbert(checkpoint_path, device).to(device)
        if include_no_finding:
            self.keep_idx = list(range(len(CONDITIONS)))
        else:
            # drop "no_finding" if not requested
            self.keep_idx = [i for i, c in enumerate(CONDITIONS) if c != "no_finding"]

    def compute(
        self,
        gts_texts: List[str],
        res_texts: List[str],
        stride: int = 128,
        agg: str = "vote"
    ) -> Dict[str, float]:
        """
        Compute CheXbert-based CE metrics:

        - ce_precision / ce_recall / ce_f1      = MICRO (standard CE in prior papers)
          (micro over all (sample, condition) pairs)

        - ce_macro_precision / ce_macro_recall / ce_macro_f1
          = MACRO over examples (per-report).
        """
        assert len(gts_texts) == len(res_texts)
        N = len(gts_texts)
        if N == 0:
            return {
                "ce_precision": 0.0,
                "ce_recall": 0.0,
                "ce_f1": 0.0,
                "ce_macro_precision": 0.0,
                "ce_macro_recall": 0.0,
                "ce_macro_f1": 0.0,
                "ce_num_examples": 0.0,
            }

        with torch.no_grad():
            g = self.model(
                gts_texts,
                stride=stride,
                agg=agg,
                treat_uncertain_as_positive=self.treat_uncertain_as_positive,
                mbatch_size=self.bs
            ).cpu().numpy()
            r = self.model(
                res_texts,
                stride=stride,
                agg=agg,
                treat_uncertain_as_positive=self.treat_uncertain_as_positive,
                mbatch_size=self.bs
            ).cpu().numpy()

        # Keep / drop "no_finding" according to self.keep_idx
        g = g[:, self.keep_idx]
        r = r[:, self.keep_idx]

        if self.treat_uncertain_as_positive:
            g_pos = (g == 1) | (g == 3)
            r_pos = (r == 1) | (r == 3)
        else:
            g_pos = (g == 1)
            r_pos = (r == 1)

        tp = (r_pos & g_pos).astype(float)
        fp = (r_pos & ~g_pos).astype(float)
        fn = (~r_pos & g_pos).astype(float)

        tp_total = float(tp.sum())
        fp_total = float(fp.sum())
        fn_total = float(fn.sum())

        if tp_total + fp_total > 0.0:
            ce_precision_micro = tp_total / (tp_total + fp_total)
        else:
            ce_precision_micro = 0.0

        if tp_total + fn_total > 0.0:
            ce_recall_micro = tp_total / (tp_total + fn_total)
        else:
            ce_recall_micro = 0.0

        if (tp_total + fp_total + fn_total) > 0.0:
            ce_f1_micro = tp_total / (tp_total + 0.5 * (fp_total + fn_total))
        else:
            ce_f1_micro = 0.0

        tp_eg = tp.sum(1)   # [N]
        fp_eg = fp.sum(1)
        fn_eg = fn.sum(1)

        precision_eg = np.nan_to_num(tp_eg / (tp_eg + fp_eg))
        recall_eg    = np.nan_to_num(tp_eg / (tp_eg + fn_eg))
        f1_eg        = np.nan_to_num(tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg)))

        ce_macro_precision = float(precision_eg.mean())
        ce_macro_recall    = float(recall_eg.mean())
        ce_macro_f1        = float(f1_eg.mean())

        return {
            "ce_precision": float(ce_precision_micro),
            "ce_recall": float(ce_recall_micro),
            "ce_f1": float(ce_f1_micro),

            "ce_macro_precision": ce_macro_precision,
            "ce_macro_recall": ce_macro_recall,
            "ce_macro_f1": ce_macro_f1,

            "ce_num_examples": float(N),
        }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=False,
        help=(
            "Input JSON file (list OR dict with key 'samples') "
            "with fields: image/image_path, caption, predicted_caption"
        ),
    )
    ap.add_argument(
        "--shard_dir",
        type=str,
        default=None,
        help=(
            "Directory containing many JSON shard files, e.g.: "
            "test_caption_shard0_label.json, test_caption_shard1_label.json, ..."
        ),
    )

    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path.",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="(Deprecated) Alias for --output.",
    )

    ap.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Checkpoint frequency (in NEW evaluated samples). 0 to disable. Ignored in --metrics_only mode.",
    )

    ap.add_argument("--lowercase", action="store_true", help="Lowercase text before COCO normalization.")
    ap.add_argument("--skip_meteor", action="store_true", help="Disable METEOR metric.")

    ap.add_argument(
        "--chexbert_ckpt",
        default="dataset/chexbert_ckpt/chexbert.pth",
        help="Path to CheXbert checkpoint .pth",
    )
    ap.add_argument("--device", default="cuda", help="Device for CheXbert and RadGraph models.")
    ap.add_argument("--mbatch_size", default=16, type=int, help="Mini-batch size for CheXbert.")
    ap.add_argument("--include_no_finding", action="store_true", help="Include 'no_finding' label in CE metrics.")
    ap.add_argument(
        "--treat_uncertain_as_positive",
        action="store_true",
        help="Treat 'uncertain' labels (3) as positive in CE metrics.",
    )
    ap.add_argument(
        "--ce_chunk_stride",
        type=int,
        default=128,
        help="Stride overlap (tokens) between CheXbert 512-token chunks.",
    )
    ap.add_argument(
        "--ce_chunk_agg",
        type=str,
        default="vote",
        choices=["vote", "any_positive"],
        help="Aggregation strategy over CheXbert chunks.",
    )

    ap.add_argument(
        "--index",
        type=int,
        default=0,
        help="Start index in input samples AFTER parsing (0-based).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="How many samples from --index to consider. None = until the end.",
    )

    ap.add_argument(
        "--metrics_only",
        action="store_true",
        help=(
            "If set, compute ONLY global metrics (text, CE, RadGraph) "
            "and save JSON with {config, metrics} without per-sample outputs."
        ),
    )

    args = ap.parse_args()

    if args.output is None and args.out_json is not None:
        args.output = args.out_json

    if not args.output:
        if args.input:
            in_dir = os.path.dirname(os.path.abspath(args.input))
        elif args.shard_dir:
            in_dir = os.path.abspath(args.shard_dir)
        else:
            raise ValueError("Either --input or --shard_dir must be provided.")
        default_name = "eval_metrics_only.json" if args.metrics_only else "eval_all.json"
        args.output = os.path.join(in_dir, default_name)

    print("[info] Output path resolved.")

    t0 = time.time()

    if args.shard_dir is not None:
        shard_dir = os.path.abspath(args.shard_dir)
        if not os.path.isdir(shard_dir):
            raise ValueError("--shard_dir is not a directory.")

        shard_files = [
            os.path.join(shard_dir, fn)
            for fn in sorted(os.listdir(shard_dir))
            if fn.endswith(".json")
        ]
        if not shard_files:
            raise ValueError("No JSON shard files found.")

        print(f"[info] Found {len(shard_files)} shard files.")

        in_samples_all: List[Dict[str, Any]] = []
        for fp in shard_files:
            try:
                payload = _load_json_any(fp)
                samples = _iter_input_samples(payload)
            except Exception as e:
                raise ValueError(f"Failed to load a shard JSON: {type(e).__name__}")
            in_samples_all.extend(samples)

        total_loaded = len(in_samples_all)
        print(f"[info] Loaded {total_loaded} samples from shards")

        if not args.input:
            args.input = "shard_dir"

    else:
        if not args.input:
            raise ValueError("Either --input or --shard_dir must be provided.")
        in_payload = _load_json_any(args.input)
        in_samples_all = _iter_input_samples(in_payload)
        total_loaded = len(in_samples_all)
        print(f"[info] Loaded {total_loaded} samples from --input")

    start = max(0, min(args.index, total_loaded))
    end = total_loaded if args.limit is None else max(start, min(start + max(0, args.limit), total_loaded))
    in_candidates = in_samples_all[start:end]

    if args.metrics_only:
        print("[mode] metrics_only=True → only compute global metrics, do not save per-sample outputs.")

        gts: Dict[str, List[str]] = {}
        res: Dict[str, List[str]] = {}
        gt_texts: List[str] = []
        pred_texts: List[str] = []

        valid_pairs = 0
        for local_idx, s in enumerate(in_candidates):
            gt = _get_gt_for_sample(s)
            pred = _get_str_field(s, "predicted_caption")
            if not gt or not pred:
                continue

            img = _get_image_id(s)
            if not img:
                img = f"idx_{start + local_idx}"

            gts[img] = [gt]
            res[img] = [pred]
            gt_texts.append(gt)
            pred_texts.append(pred)
            valid_pairs += 1

        print(f"[metrics_only] valid (gt, pred) pairs: {valid_pairs}")

        global_text, _ = _compute_text_metrics_bulk(
            gts, res, lowercase=args.lowercase, skip_meteor=args.skip_meteor
        )
        metrics = {f"test_{k}": v for k, v in global_text.items()}

        metrics["ce_status"] = "skipped"
        metrics["ce_num_examples"] = float(len(gt_texts))

        if args.chexbert_ckpt and os.path.exists(args.chexbert_ckpt):
            try:
                if len(gt_texts) == 0:
                    print("[warn] CE skipped: no valid (gt, pred) pairs.")
                    metrics["ce_status"] = "no_pairs"
                else:
                    ce_calc = CheXbertMetrics(
                        checkpoint_path=args.chexbert_ckpt,
                        device=args.device,
                        mbatch_size=args.mbatch_size,
                        include_no_finding=args.include_no_finding,
                        treat_uncertain_as_positive=args.treat_uncertain_as_positive,
                    )
                    ce = ce_calc.compute(
                        gt_texts,
                        pred_texts,
                        stride=args.ce_chunk_stride,
                        agg=args.ce_chunk_agg,
                    )
                    metrics.update(ce)
                    metrics["ce_status"] = "ok"
            except Exception as e:
                print(f"[warn] CE disabled at runtime: {type(e).__name__}")
                metrics["ce_status"] = f"error:{type(e).__name__}"
        else:
            if args.chexbert_ckpt:
                print("[warn] CE skipped: checkpoint not found.")
                metrics["ce_status"] = "ckpt_not_found"

        metrics["test_rg_e"] = 0.0
        metrics["test_rg_er"] = 0.0
        metrics["test_rg_bar_er"] = 0.0

        if len(gt_texts) > 0:
            try:
                print("[radgraph] Initializing F1RadGraph (reward_level='all', model_type='radgraph')...")
                f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph")
                print("[radgraph] F1RadGraph ready.")

                mean_reward, reward_list, hyp_annotations, ref_annotations = f1radgraph(
                    hyps=pred_texts,
                    refs=gt_texts,
                )
                if mean_reward is not None:
                    try:
                        if len(mean_reward) < 3:
                            print(f"[radgraph] mean_reward has <3 values: {mean_reward}")
                            metrics["test_rg_e"] = safe_float_or_neg1(mean_reward[0])
                        else:
                            metrics["test_rg_e"] = safe_float_or_neg1(mean_reward[0])
                            metrics["test_rg_er"] = safe_float_or_neg1(mean_reward[1])
                            metrics["test_rg_bar_er"] = safe_float_or_neg1(mean_reward[2])
                    except TypeError:
                        # mean_reward is scalar → treat it as RG_E only
                        metrics["test_rg_e"] = safe_float_or_neg1(mean_reward)
                else:
                    print("[radgraph] mean_reward is None, keep RG metrics = 0.")
            except Exception as e:
                print(f"[warn] RadGraph disabled: {type(e).__name__}")

        payload = {
            "config": {
                "lowercase": args.lowercase,
                "skip_meteor": args.skip_meteor,
                "total_loaded": total_loaded,
                "index": start,
                "limit": args.limit,
                "selected": int(valid_pairs),
                "range": [start, end],
                "ce_chunk_stride": args.ce_chunk_stride,
                "ce_chunk_agg": args.ce_chunk_agg,
                "include_no_finding": args.include_no_finding,
                "treat_uncertain_as_positive": args.treat_uncertain_as_positive,
                "device": args.device,
                "total_time_min": round((time.time() - t0) / 60.0, 2),
                "checkpoint": False,
                "metrics_only": True,
                "gt_field": "caption",
            },
            "metrics": metrics,
        }

        atomic_write(args.output, payload)

        print("\n=== RESULTS (metrics_only, global over selected samples) ===")
        for k in sorted(metrics.keys()):
            print(f"{k:>24}: {metrics[k]}")
        print("\n[DONE] Metrics-only JSON saved.")
        return  # End here in fast mode


    prev_samples: List[Dict[str, Any]] = []
    if os.path.exists(args.output):
        try:
            prev_payload = _load_json_any(args.output)
            if isinstance(prev_payload, dict) and isinstance(prev_payload.get("samples"), list):
                prev_samples = prev_payload["samples"]
        except Exception as e:
            print(f"[resume] cannot load existing JSON: {type(e).__name__}")

    prev_by_image = _index_by_image(prev_samples)
    print(f"[resume] found {len(prev_by_image)} existing samples.")

    gts: Dict[str, List[str]] = {}
    res: Dict[str, List[str]] = {}
    gt_texts: List[str] = []
    pred_texts: List[str] = []

    for x in prev_samples:
        img = _get_image_id(x)
        if not img:
            continue
        gt = _get_gt_for_sample(x)
        pred = _get_str_field(x, "predicted_caption")
        if gt and pred:
            gts[img] = [gt]
            res[img] = [pred]
            gt_texts.append(gt)
            pred_texts.append(pred)

    aggregated: List[Dict[str, Any]] = list(prev_samples)
    to_process: List[Dict[str, Any]] = []
    copied = 0

    for s in in_candidates:
        img = _get_image_id(s)
        if not img:
            continue

        if img in prev_by_image:
            copied += 1
            continue

        gt = _get_gt_for_sample(s)
        pred = _get_str_field(s, "predicted_caption")

        if not gt or not pred:
            continue

        to_process.append(s)

    print(
        f"[select] total_loaded={total_loaded}, range=[{start}:{end}), "
        f"copied_from_prev={copied}, to_eval_new={len(to_process)}"
    )

    evaluated = 0
    for s in to_process:
        img = _get_image_id(s)
        if not img:
            continue

        gt = _get_gt_for_sample(s)
        pred = _get_str_field(s, "predicted_caption")

        if not gt or not pred:
            continue

        gts[img] = [gt]
        res[img] = [pred]
        gt_texts.append(gt)
        pred_texts.append(pred)

        new_entry = dict(s)
        new_entry["image"] = img
        if "image_path" in new_entry and isinstance(new_entry["image_path"], str):
            new_entry["image_path"] = _public_image_id(new_entry["image_path"])
        new_entry["gt_caption_eval"] = gt
        new_entry["predicted_caption"] = pred
        aggregated.append(new_entry)

        evaluated += 1

        if args.save_every and evaluated % args.save_every == 0:
            global_text, per_image_scores = _compute_text_metrics_bulk(
                gts, res, lowercase=args.lowercase, skip_meteor=args.skip_meteor
            )
            pi_map = per_image_scores
            for ent in aggregated:
                im = _get_image_id(ent)
                if not im:
                    continue
                if "eval" in ent:
                    continue
                gt_ = _get_gt_for_sample(ent)
                pred_ = _get_str_field(ent, "predicted_caption")
                pi = pi_map.get(im, {})
                eval_block = {
                    "bleu1": float(pi.get("bleu1", 0.0)),
                    "bleu2": float(pi.get("bleu2", 0.0)),
                    "bleu3": float(pi.get("bleu3", 0.0)),
                    "bleu4": float(pi.get("bleu4", 0.0)),
                    "rougeL": float(pi.get("rougeL", 0.0)),
                    "len_pred": int(len(pred_.split())) if pred_ else 0,
                    "len_gt": int(len(gt_.split())) if gt_ else 0,
                }
                if not args.skip_meteor:
                    eval_block["meteor"] = float(pi.get("meteor", 0.0))
                ent["eval"] = eval_block

            metrics_ckpt = {f"test_{k}": v for k, v in global_text.items()}
            metrics_ckpt["ce_status"] = "pending"
            metrics_ckpt["ce_num_examples"] = float(len(gt_texts))

            payload_ckpt = {
                "config": {
                    "lowercase": args.lowercase,
                    "skip_meteor": args.skip_meteor,
                    "total_loaded": total_loaded,
                    "index": start,
                    "limit": args.limit,
                    "selected": len(aggregated),
                    "newly_evaluated": len(to_process[:evaluated]),
                    "copied_from_prev": copied,
                    "range": [start, end],
                    "ce_chunk_stride": args.ce_chunk_stride,
                    "ce_chunk_agg": args.ce_chunk_agg,
                    "include_no_finding": args.include_no_finding,
                    "treat_uncertain_as_positive": args.treat_uncertain_as_positive,
                    "device": args.device,
                    "total_time_min": round((time.time() - t0) / 60.0, 2),
                    "checkpoint": True,
                    "metrics_only": False,
                    "gt_field": "caption",
                },
                "metrics": metrics_ckpt,
                "samples": aggregated,
            }

            atomic_write(args.output, payload_ckpt)
            print(f"✅ Saved checkpoint at {evaluated} new samples", flush=True)

    global_text, per_image_scores = _compute_text_metrics_bulk(
        gts, res, lowercase=args.lowercase, skip_meteor=args.skip_meteor
    )

    for ent in aggregated:
        im = _get_image_id(ent)
        if not im:
            continue
        if "eval" in ent:
            continue
        gt = _get_gt_for_sample(ent)
        pred = _get_str_field(ent, "predicted_caption")
        pi = per_image_scores.get(im, {})
        eval_block = {
            "bleu1": float(pi.get("bleu1", 0.0)),
            "bleu2": float(pi.get("bleu2", 0.0)),
            "bleu3": float(pi.get("bleu3", 0.0)),
            "bleu4": float(pi.get("bleu4", 0.0)),
            "rougeL": float(pi.get("rougeL", 0.0)),
            "len_pred": int(len(pred.split())) if pred else 0,
            "len_gt": int(len(gt.split())) if gt else 0,
        }
        if not args.skip_meteor:
            eval_block["meteor"] = float(pi.get("meteor", 0.0))
        ent["eval"] = eval_block

    metrics = {f"test_{k}": v for k, v in global_text.items()}
    metrics["ce_status"] = "skipped"
    metrics["ce_num_examples"] = float(len(gt_texts))

    try:
        print("[radgraph] Initializing F1RadGraph (reward_level='all', model_type='radgraph-xl')...")
        f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
        print("[radgraph] F1RadGraph ready.")
    except Exception as e:
        print(f"[warn] RadGraph disabled: {type(e).__name__}")
        f1radgraph = None

    rg_e_vals: List[float] = []
    rg_er_vals: List[float] = []
    rg_bar_er_vals: List[float] = []

    if f1radgraph is not None:
        for ent in aggregated:
            gt_rg = _get_gt_for_sample(ent)
            pred_rg = _get_str_field(ent, "predicted_caption")

            if not gt_rg or not pred_rg:
                ent["rg_e"] = -1.0
                ent["rg_er"] = -1.0
                ent["rg_bar_er"] = -1.0
                continue

            try:
                mean_reward, reward_list, hyp_annotations, ref_annotations = f1radgraph(
                    hyps=[pred_rg],
                    refs=[gt_rg],
                )

                rg_e = -1.0
                rg_er = -1.0
                rg_bar_er = -1.0

                if mean_reward is not None:
                    try:
                        if len(mean_reward) < 3:
                            print(
                                f"[radgraph] mean_reward has <3 values for image {_get_image_id(ent)}: "
                                f"{mean_reward}"
                            )
                            rg_e = safe_float_or_neg1(mean_reward[0])
                        else:
                            rg_e = safe_float_or_neg1(mean_reward[0])
                            rg_er = safe_float_or_neg1(mean_reward[1])
                            rg_bar_er = safe_float_or_neg1(mean_reward[2])
                    except TypeError:
                        rg_e = safe_float_or_neg1(mean_reward)

                ent["rg_e"] = rg_e
                ent["rg_er"] = rg_er
                ent["rg_bar_er"] = rg_bar_er

                if rg_e >= 0.0:
                    rg_e_vals.append(rg_e)
                if rg_er >= 0.0:
                    rg_er_vals.append(rg_er)
                if rg_bar_er >= 0.0:
                    rg_bar_er_vals.append(rg_bar_er)

            except Exception as e:
                print(f"[warn] RadGraph scoring failed: {type(e).__name__}")
                ent["rg_e"] = -1.0
                ent["rg_er"] = -1.0
                ent["rg_bar_er"] = -1.0

    metrics["test_rg_e"] = float(np.mean(rg_e_vals)) if rg_e_vals else 0.0
    metrics["test_rg_er"] = float(np.mean(rg_er_vals)) if rg_er_vals else 0.0
    metrics["test_rg_bar_er"] = float(np.mean(rg_bar_er_vals)) if rg_bar_er_vals else 0.0

    if args.chexbert_ckpt and os.path.exists(args.chexbert_ckpt):
        try:
            if len(gt_texts) == 0:
                print("[warn] CE skipped: no valid (gt, pred) pairs.")
                metrics["ce_status"] = "no_pairs"
            else:
                ce_calc = CheXbertMetrics(
                    checkpoint_path=args.chexbert_ckpt,
                    device=args.device,
                    mbatch_size=args.mbatch_size,
                    include_no_finding=args.include_no_finding,
                    treat_uncertain_as_positive=args.treat_uncertain_as_positive,
                )
                ce = ce_calc.compute(
                    gt_texts,
                    pred_texts,
                    stride=args.ce_chunk_stride,
                    agg=args.ce_chunk_agg,
                )
                metrics.update(ce)
                metrics["ce_status"] = "ok"
        except Exception as e:
            print(f"[warn] CE disabled at runtime: {type(e).__name__}")
            metrics["ce_status"] = f"error:{type(e).__name__}"
    else:
        if args.chexbert_ckpt:
            print("[warn] CE skipped: checkpoint not found.")
            metrics["ce_status"] = "ckpt_not_found"

    payload = {
        "config": {
            "lowercase": args.lowercase,
            "skip_meteor": args.skip_meteor,
            "total_loaded": total_loaded,
            "index": start,
            "limit": args.limit,
            "selected": len(aggregated),
            "newly_evaluated": len(to_process),
            "copied_from_prev": copied,
            "range": [start, end],
            "ce_chunk_stride": args.ce_chunk_stride,
            "ce_chunk_agg": args.ce_chunk_agg,
            "include_no_finding": args.include_no_finding,
            "treat_uncertain_as_positive": args.treat_uncertain_as_positive,
            "device": args.device,
            "total_time_min": round((time.time() - t0) / 60.0, 2),
            "checkpoint": False,
            "metrics_only": False,
            "gt_field": "caption",
        },
        "metrics": metrics,
        "samples": aggregated,
    }

    atomic_write(args.output, payload)

    print("\n=== RESULTS (global over previous + new) ===")
    for k in sorted(metrics.keys()):
        print(f"{k:>24}: {metrics[k]}")
    print("\n[DONE] Evaluation JSON saved.")


if __name__ == "__main__":
    main()