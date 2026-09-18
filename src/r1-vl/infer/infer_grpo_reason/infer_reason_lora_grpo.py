#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Inference for the R-Aligned MedSPA PR Agent.

The inference policy is composed as:
  base model + merged SFT LoRA + GRPO LoRA.

Each PR step follows the same structured format used during R-Align:
  <action>...</action>
  <reason>...</reason>
  <decision>continue|summary</decision>

Previous full triplets are fed back to the PR Agent until it emits
<decision>summary</decision> or the safety cap is reached.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from peft import PeftModel
from PIL import Image, ImageFile, ImageOps
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


_ACTION_RE = re.compile(
    r"<action>(.*?)</action>",
    flags=re.DOTALL | re.IGNORECASE,
)
_REASON_RE = re.compile(
    r"<reason>(.*?)</reason>",
    flags=re.DOTALL | re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"<decision>(continue|summary)</decision>",
    flags=re.DOTALL | re.IGNORECASE,
)
_STEP_RE = re.compile(
    r"^\s*<action>(?P<action>.*?)</action>\s*"
    r"<reason>(?P<reason>.*?)</reason>\s*"
    r"<decision>(?P<decision>continue|summary)</decision>\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)


STEP_FORMAT_INSTRUCTION = (
    "Each reasoning step must follow exactly this format:\n"
    "<action>...</action>\n"
    "<reason>...</reason>\n"
    "<decision>continue</decision>\n"
    "Use <decision>summary</decision> when sufficient evidence has been collected.\n"
)


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).strip()


def parse_reasoning_step(step: str) -> Optional[Dict[str, str]]:
    """Parse one structured PR step."""
    if not step:
        return None

    match = _STEP_RE.match(step.strip())
    if match:
        action = _clean_text(match.group("action"))
        reason = _clean_text(match.group("reason"))
        decision = _clean_text(match.group("decision")).lower()
        if action and reason and decision in {"continue", "summary"}:
            return {
                "action": action,
                "reason": reason,
                "decision": decision,
            }

    action_match = _ACTION_RE.search(step)
    reason_match = _REASON_RE.search(step)
    decision_match = _DECISION_RE.search(step)

    if not action_match or not reason_match or not decision_match:
        return None

    action = _clean_text(action_match.group(1))
    reason = _clean_text(reason_match.group(1))
    decision = _clean_text(decision_match.group(1)).lower()

    if not action or not reason or decision not in {"continue", "summary"}:
        return None

    return {
        "action": action,
        "reason": reason,
        "decision": decision,
    }


def extract_decision(step: str) -> str:
    if not step:
        return "continue"
    match = _DECISION_RE.search(step)
    if not match:
        return "continue"
    decision = _clean_text(match.group(1)).lower()
    return decision if decision in {"continue", "summary"} else "continue"


def canonical_step(parsed: Dict[str, str], step_idx: int) -> str:
    return (
        f"Step {step_idx}:\n"
        f"<action>{parsed['action']}</action>\n"
        f"<reason>{parsed['reason']}</reason>\n"
        f"<decision>{parsed['decision']}</decision>"
    )


def concat_segments(segments: Iterable[str], sep: str = "\n") -> str:
    return sep.join(segment for segment in segments if segment)


def build_reasoning_prompt(history_steps: List[str]) -> str:
    """Build the same iterative PR prompt structure used during R-Align."""
    header = (
        "You are a highly experienced radiologist.\n"
        + STEP_FORMAT_INSTRUCTION
        + "\n"
    )

    if not history_steps:
        return (
            header
            + "There are no previous reasoning steps yet.\n\n"
            + "Generate ONLY the FIRST reasoning step based on the image.\n"
        )

    history_block = (
        "Here are the previous reasoning steps:\n\n"
        + concat_segments(history_steps, sep="\n\n")
        + "\n\n"
    )

    return (
        header
        + history_block
        + "Generate ONLY the NEXT reasoning step based on the image "
        + "and the previous steps above.\n"
    )


def load_image_index(index_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    with open(index_path, "r", encoding="utf-8") as handle:
        for line in handle:
            path = line.strip()
            if not path:
                continue
            mapping[os.path.basename(path)] = path

    print(f"[INFO] Loaded {len(mapping)} image paths from index.")
    return mapping


def load_mimic_annotation(
    json_path: str,
    split: str,
) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        print(f"[INFO] Input JSON is a flat list with {len(data)} items.")
        return data

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be a list or a split dictionary.")

    if split not in data or not isinstance(data[split], list):
        available = [key for key, value in data.items() if isinstance(value, list)]
        raise ValueError(
            f"Split '{split}' not found. Available list-valued splits: {available}"
        )

    items = data[split]
    print(f"[INFO] Using split '{split}' with {len(items)} items.")
    return items


def resolve_image_path(
    item: Dict[str, Any],
    index_map: Dict[str, str],
) -> str:
    raw = item.get("image_path")

    if isinstance(raw, list):
        if not raw:
            raise ValueError("image_path is empty.")
        rel = str(raw[0])
    elif isinstance(raw, str) and raw.strip():
        rel = raw.strip()
    else:
        raise ValueError("image_path missing or invalid in item.")

    filename = os.path.basename(rel)
    abs_path = index_map.get(filename)

    if abs_path is None:
        raise ValueError(f"Image filename '{filename}' not found in index.")

    return abs_path


def rel_image_path_from_abs(abs_path: str) -> str:
    if not abs_path:
        return ""

    marker = "/files/"
    idx = abs_path.rfind(marker)

    if idx != -1:
        return abs_path[idx + len(marker):]

    return os.path.basename(abs_path)


def load_findings_map(findings_json: str) -> Dict[str, str]:
    with open(findings_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Expected findings JSON to be a list of objects.")

    mapping: Dict[str, str] = {}

    for item in data:
        if not isinstance(item, dict):
            continue

        image_path = str(item.get("image_path", "")).strip()
        finding = str(item.get("finding", "")).strip()

        if image_path:
            mapping[image_path] = finding

    print(f"[INFO] Loaded {len(mapping)} findings entries.")
    return mapping


def atomic_write_json(path: str, data: Any) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    os.replace(tmp_path, path)


def load_existing_results(
    output_json: str,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    if not output_json or not os.path.isfile(output_json):
        return [], set()

    try:
        with open(output_json, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, list):
            print("[WARN] Existing output is not a list. Resume data ignored.")
            return [], set()

        done = {
            str(record["image"])
            for record in data
            if isinstance(record, dict) and record.get("image")
        }

        print(
            f"[INFO] Resume enabled: loaded {len(data)} records, "
            f"done={len(done)} images."
        )
        return data, done

    except Exception as exc:
        print(f"[WARN] Resume load failed: {type(exc).__name__}")
        return [], set()


def load_image_square(
    path: str,
    target_size: int = 1024,
    pad_color: int = 0,
) -> Image.Image:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("Image file is missing.")

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions.")

        scale = float(target_size) / float(max(width, height))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        try:
            resample = Image.Resampling.LANCZOS
        except Exception:
            resample = Image.LANCZOS

        image = image.resize((new_width, new_height), resample=resample)

        canvas = Image.new(
            "RGB",
            (target_size, target_size),
            (pad_color, pad_color, pad_color),
        )

        left = (target_size - new_width) // 2
        top = (target_size - new_height) // 2
        canvas.paste(image, (left, top))
        canvas.load()
        return canvas


def _is_vision_lang_config(cfg) -> bool:
    model_type = (
        getattr(cfg, "model_type", "") or ""
    ).lower().replace("-", "_")

    if model_type in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}:
        return True

    return hasattr(cfg, "vision_config") or hasattr(cfg, "image_embed_dim")


def load_auto_for_policy(model_name_or_path: str, **kwargs):
    kwargs.setdefault("trust_remote_code", True)

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=kwargs["trust_remote_code"],
    )

    if _is_vision_lang_config(config):
        return AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            **kwargs,
        )

    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **kwargs,
    )


class HFReasoner:
    """Base + merged SFT adapter + active GRPO adapter."""

    def __init__(
        self,
        base_model: str,
        sft_adapter_path: str,
        grpo_adapter_path: str,
        attn_implementation: str = "sdpa",
        img_size: int = 1024,
        pad_color: int = 0,
    ):
        self.base_model = base_model.strip()
        self.sft_adapter_path = os.path.abspath(sft_adapter_path)
        self.grpo_adapter_path = os.path.abspath(grpo_adapter_path)
        self.img_size = int(img_size)
        self.pad_color = int(pad_color)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        print(f"[INFO] Base model    : {self.base_model}")
        print(f"[INFO] SFT adapter  : {self.sft_adapter_path}")
        print(f"[INFO] GRPO adapter : {self.grpo_adapter_path}")
        print(f"[INFO] device/dtype : {self.device} / {self.dtype}")
        print(f"[INFO] attn_impl    : {attn_implementation}")
        print(f"[INFO] img_size     : {self.img_size}")

        self.processor = AutoProcessor.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )

        if hasattr(self.processor, "image_processor"):
            try:
                self.processor.image_processor.min_pixels = 3136
            except Exception:
                pass

        model_init_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
            "attn_implementation": attn_implementation,
        }

        if self.device == "cuda":
            model_init_kwargs["device_map"] = "auto"

        base = load_auto_for_policy(
            self.base_model,
            **model_init_kwargs,
        )

        if not os.path.isdir(self.sft_adapter_path):
            raise FileNotFoundError("SFT adapter directory not found.")

        model = PeftModel.from_pretrained(
            base,
            self.sft_adapter_path,
            adapter_name="sft",
            is_trainable=False,
        )

        if not hasattr(model, "merge_and_unload"):
            raise RuntimeError("PEFT does not support merge_and_unload().")

        print("[INFO] Merging SFT adapter into base weights.")
        model = model.merge_and_unload()

        if not os.path.isdir(self.grpo_adapter_path):
            raise FileNotFoundError("GRPO adapter directory not found.")

        model = PeftModel.from_pretrained(
            model,
            self.grpo_adapter_path,
            adapter_name="grpo",
            is_trainable=False,
        )
        model.set_adapter("grpo")

        print("[INFO] Active adapter: grpo (SFT merged into base).")
        self.model = model.eval()

    @torch.no_grad()
    def _generate_once(
        self,
        image: Image.Image,
        user_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_beams: int,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        model_device = next(self.model.parameters()).device

        for key, value in list(inputs.items()):
            if not torch.is_tensor(value):
                continue

            if key == "pixel_values" and self.device == "cuda":
                inputs[key] = value.to(
                    model_device,
                    dtype=self.dtype,
                    non_blocking=True,
                )
            else:
                inputs[key] = value.to(
                    model_device,
                    non_blocking=True,
                )

        do_sample = num_beams == 1
        use_cuda = self.device == "cuda"

        with torch.autocast(
            device_type="cuda",
            dtype=self.dtype if use_cuda else torch.bfloat16,
            enabled=use_cuda,
        ):
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                num_beams=num_beams,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]

        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

    @torch.no_grad()
    def generate_one(
        self,
        image_path: str,
        user_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_beams: int,
    ) -> str:
        image = load_image_square(
            image_path,
            target_size=self.img_size,
            pad_color=self.pad_color,
        )

        try:
            return self._generate_once(
                image=image,
                user_text=user_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
            )
        except Exception as exc:
            if "Image features and image tokens do not match" not in str(exc):
                raise

            retry_image = load_image_square(
                image_path,
                target_size=self.img_size,
                pad_color=self.pad_color,
            )
            retry_image.load()

            return self._generate_once(
                image=retry_image,
                user_text=user_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
            )


def run_reasoning_loop(
    reasoner: HFReasoner,
    image_path: str,
    max_rounds: int = 10,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    num_beams: int = 1,
) -> List[str]:
    history_steps: List[str] = []
    all_model_steps: List[str] = []

    for step_idx in range(1, max_rounds + 1):
        user_text = build_reasoning_prompt(history_steps)

        print(
            f"\n[Round {step_idx}] Assistant "
            f"({os.path.basename(image_path)}): ",
            end="",
            flush=True,
        )

        response = reasoner.generate_one(
            image_path=image_path,
            user_text=user_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
        )

        print(response, flush=True)
        all_model_steps.append(response)

        parsed = parse_reasoning_step(response)

        if parsed is not None:
            history_steps.append(
                canonical_step(parsed, step_idx)
            )
            decision = parsed["decision"]
        else:
            print(
                "[WARN] Malformed PR step. Keeping raw response in history.",
                flush=True,
            )
            history_steps.append(
                f"Step {step_idx}:\n{response.strip()}"
            )
            decision = extract_decision(response)

        if decision == "summary":
            print("[INFO] PR Agent selected summary. Stopping.")
            break

    return all_model_steps


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--sft_adapter_path", required=True)
    parser.add_argument("--grpo_adapter_path", required=True)

    parser.add_argument(
        "--mimic_json",
        type=str,
        default="dataset/mimic/mimic_annotation_.json",
    )
    parser.add_argument(
        "--all_images_path",
        type=str,
        default="dataset/mimic/mimic_cxr_all_images.txt",
    )
    parser.add_argument(
        "--findings_json",
        type=str,
        default="dataset/mimic/finding/mimic_gt_findings.json",
    )
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--max_rounds", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--pad_color", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if os.path.isdir(args.output_json):
        raise ValueError("--output_json must be a file path.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    index_map = load_image_index(args.all_images_path)
    items = load_mimic_annotation(args.mimic_json, args.split)
    findings_map = load_findings_map(args.findings_json)

    print(f"[INFO] Loaded {len(items)} MIMIC items.")

    results: List[Dict[str, Any]] = []
    done_images: Set[str] = set()

    if args.resume:
        results, done_images = load_existing_results(args.output_json)

    reasoner = HFReasoner(
        base_model=args.base_model,
        sft_adapter_path=args.sft_adapter_path,
        grpo_adapter_path=args.grpo_adapter_path,
        attn_implementation=args.attn_implementation,
        img_size=args.img_size,
        pad_color=args.pad_color,
    )

    processed = 0
    total_items = len(items)

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for idx, item in enumerate(items, start=1):
        item_id = item.get("id", f"sample_{idx}")
        print(f"\n===== [{idx}/{total_items}] id={item_id} =====")

        try:
            abs_image_path = resolve_image_path(item, index_map)
            rel_image_path = rel_image_path_from_abs(abs_image_path)

            if args.resume and rel_image_path in done_images:
                print(f"[SKIP] Already processed: {rel_image_path}")
                continue

            caption = item.get("report", "") or item.get("caption", "")
            trusted_finding = findings_map.get(rel_image_path, "")

            raw_steps = run_reasoning_loop(
                reasoner=reasoner,
                image_path=abs_image_path,
                max_rounds=args.max_rounds,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )

            structured_steps: List[Dict[str, Any]] = []

            for step_idx, raw_step in enumerate(raw_steps, start=1):
                parsed = parse_reasoning_step(raw_step)

                if parsed is None:
                    structured_steps.append(
                        {
                            "count": step_idx,
                            "title": "",
                            "content": raw_step.strip(),
                            "decision": extract_decision(raw_step),
                        }
                    )
                    continue

                structured_steps.append(
                    {
                        "count": step_idx,
                        "title": parsed["action"],
                        "content": parsed["reason"],
                        "decision": parsed["decision"],
                    }
                )

            record = {
                "image": rel_image_path,
                "split": args.split,
                "caption": caption,
                "trusted_finding": trusted_finding,
                "steps": structured_steps,
            }

            results.append(record)
            processed += 1
            done_images.add(rel_image_path)

            if processed % args.save_every == 0:
                print(f"[INFO] Atomic save after {processed} new samples.")
                atomic_write_json(args.output_json, results)

        except Exception as exc:
            print(f"[ERROR] id={item_id}: {type(exc).__name__}")

    print("[INFO] Final atomic save.")
    atomic_write_json(args.output_json, results)
    print(f"[OK] Saved {len(results)} records.")


if __name__ == "__main__":
    main()
