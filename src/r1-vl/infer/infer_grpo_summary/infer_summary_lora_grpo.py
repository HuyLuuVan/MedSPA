#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Paper-aligned MedSPA SUM-agent inference with SFT + GRPO LoRA adapters."""

import argparse
import contextlib
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple

import torch
from peft import PeftModel
from PIL import Image, ImageFile, ImageOps
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FULL_STEP_RE = re.compile(
    r"<action>.*?</action>\s*"
    r"<reason>.*?</reason>\s*"
    r"<decision>.*?</decision>",
    re.DOTALL | re.IGNORECASE,
)

SUM_PROMPT = """You are a highly experienced radiologist.
Here are the clinical reasoning steps for this image:
{reasoning}

Based on the provided image and the above reasoning steps, write a clear, clinically appropriate radiology report."""


def clean_predicted_caption(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = _THINK_RE.sub("", text).strip()
    return cleaned if cleaned else text.strip()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(x) for x in value).strip()
    return str(value).strip()


def build_rel_index(all_images_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    basename_candidates: Dict[str, List[str]] = {}
    marker = "/files/"

    with open(all_images_path, "r", encoding="utf-8") as handle:
        for line in handle:
            full = line.strip()
            if not full:
                continue

            marker_idx = full.rfind(marker)
            if marker_idx != -1:
                rel = full[marker_idx + len(marker):]
            else:
                rel = os.path.basename(full)

            mapping[rel] = full
            base = os.path.basename(full)
            basename_candidates.setdefault(base, []).append(full)

    for base, paths in basename_candidates.items():
        if len(paths) == 1 and base not in mapping:
            mapping[base] = paths[0]

    print(f"[INFO] Loaded {len(mapping)} image index keys.")
    return mapping


def _normalize_item_image_field(item: Dict[str, Any]) -> str:
    raw = item.get("image", item.get("image_path", ""))
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    return raw.strip() if isinstance(raw, str) else ""


def resolve_abs_image_from_item(item: Dict[str, Any], rel_index: Dict[str, str]) -> str:
    rel = _normalize_item_image_field(item)
    if not rel:
        raise ValueError("Missing image field.")

    if os.path.isabs(rel) and os.path.isfile(rel):
        return rel

    abs_path = rel_index.get(rel)
    if abs_path is None:
        abs_path = rel_index.get(os.path.basename(rel))
    if abs_path is None:
        raise ValueError("Image path could not be resolved.")
    return abs_path


def _format_raw_reasoning_steps(raw_steps: Any) -> List[str]:
    if not isinstance(raw_steps, list):
        return []

    output: List[str] = []
    for idx, step in enumerate(raw_steps, start=1):
        text = _as_text(step)
        if not text:
            continue

        if re.match(r"^\s*Step\s+\d+\s*:", text, re.IGNORECASE):
            output.append(text)
        else:
            output.append(f"Step {idx}: {text}")
    return output


def _format_structured_steps(steps: Any) -> List[str]:
    if not isinstance(steps, list):
        return []

    valid_steps = [step for step in steps if isinstance(step, dict)]
    valid_steps = sorted(valid_steps, key=lambda step: step.get("count", 0))

    output: List[str] = []
    for idx, step in enumerate(valid_steps, start=1):
        action = _as_text(step.get("title", step.get("action", "")))
        reason = _as_text(step.get("content", step.get("reason", "")))
        decision = _as_text(step.get("decision", "continue")).lower() or "continue"

        # Preserve a raw structured step if a malformed PR output was stored in content.
        if not action and _FULL_STEP_RE.search(reason):
            output.append(f"Step {idx}:\n{reason}")
            continue

        output.append(
            f"Step {idx}:\n"
            f"<action>{action}</action>\n"
            f"<reason>{reason}</reason>\n"
            f"<decision>{decision}</decision>"
        )

    return output


def item_to_stepwise_history(item: Dict[str, Any]) -> List[str]:
    reasoning_steps = item.get("reasoning_steps")
    if isinstance(reasoning_steps, list) and reasoning_steps:
        return _format_raw_reasoning_steps(reasoning_steps)

    prefix = item.get("prefix")
    if isinstance(prefix, list) and prefix:
        return _format_raw_reasoning_steps(prefix)

    return _format_structured_steps(item.get("steps", []))


def build_summary_prompt(stepwise_history: List[str]) -> str:
    if not stepwise_history:
        raise ValueError("Missing PR reasoning chain for SUM inference.")
    return SUM_PROMPT.format(reasoning="\n".join(stepwise_history))


def load_image_square(path: str, target_size: int = 1024, pad_color: int = 0) -> Image.Image:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("Image file not found.")

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image size.")

        scale = float(target_size) / float(max(width, height))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
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


def atomic_write_json(path: str, data: Any) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_existing_results(output_json: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
    if not output_json or not os.path.isfile(output_json):
        return [], set()

    try:
        with open(output_json, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, list):
            print("[WARN] Existing output is not a list; resume state ignored.")
            return [], set()

        successful: List[Dict[str, Any]] = []
        done: Set[str] = set()

        for record in data:
            if not isinstance(record, dict):
                continue

            image_key = _normalize_item_image_field(record)
            prediction = record.get("predicted_caption", "")
            prediction = prediction.strip() if isinstance(prediction, str) else ""

            # Keep only completed records. Failed/empty records will be retried.
            if image_key and prediction:
                successful.append(record)
                done.add(image_key)

        print(
            f"[INFO] Resume enabled: kept {len(successful)} successful records; "
            f"retrying {len(data) - len(successful)} incomplete records."
        )
        return successful, done

    except Exception as exc:
        print(f"[WARN] Resume state could not be loaded: {type(exc).__name__}")
        return [], set()


def _is_vision_lang_config(config: Any) -> bool:
    model_type = (getattr(config, "model_type", "") or "").lower().replace("-", "_")
    if model_type in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}:
        return True
    return hasattr(config, "vision_config") or hasattr(config, "image_embed_dim")


def load_auto_for_policy(model_name_or_path: str, **kwargs):
    kwargs.setdefault("trust_remote_code", True)
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=kwargs["trust_remote_code"],
    )
    if _is_vision_lang_config(config):
        return AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


class HFGRPOSummarizer:
    def __init__(
        self,
        base_model: str,
        sft_adapter_path: str,
        grpo_adapter_path: str,
        attn_implementation: str = "sdpa",
        img_size: int = 1024,
        pad_color: int = 0,
        min_pixels: int = 3136,
    ):
        self.base_model = base_model.strip()
        self.sft_adapter_path = os.path.abspath(sft_adapter_path)
        self.grpo_adapter_path = os.path.abspath(grpo_adapter_path)
        self.img_size = int(img_size)
        self.pad_color = int(pad_color)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            self.dtype = torch.float32

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

        try:
            if hasattr(self.processor, "image_processor"):
                self.processor.image_processor.min_pixels = int(min_pixels)
        except Exception:
            pass

        model_init_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
            "attn_implementation": attn_implementation,
        }
        if self.device == "cuda":
            model_init_kwargs["device_map"] = "auto"

        base = load_auto_for_policy(self.base_model, **model_init_kwargs)

        if not os.path.isdir(self.sft_adapter_path):
            raise FileNotFoundError("SUMMARY SFT adapter directory not found.")

        model = PeftModel.from_pretrained(
            base,
            self.sft_adapter_path,
            adapter_name="sft",
            is_trainable=False,
        )

        if not hasattr(model, "merge_and_unload"):
            raise RuntimeError("PEFT merge_and_unload() is unavailable.")

        print("[INFO] Merging SUMMARY SFT adapter into base weights.")
        model = model.merge_and_unload()

        if not os.path.isdir(self.grpo_adapter_path):
            raise FileNotFoundError("SUMMARY GRPO adapter directory not found.")

        model = PeftModel.from_pretrained(
            model,
            self.grpo_adapter_path,
            adapter_name="grpo",
            is_trainable=False,
        )
        model.set_adapter("grpo")
        print("[INFO] Active adapter: grpo (SUMMARY SFT merged into base).")

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

        for key, value in list(inputs.items()):
            if not torch.is_tensor(value):
                continue
            if key == "pixel_values":
                inputs[key] = value.to(
                    self.model.device,
                    dtype=self.dtype,
                    non_blocking=True,
                )
            else:
                inputs[key] = value.to(self.model.device, non_blocking=True)

        do_sample = num_beams == 1
        if do_sample and temperature <= 0:
            do_sample = False

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device == "cuda"
            else contextlib.nullcontext()
        )

        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "num_beams": int(num_beams),
        }
        if do_sample:
            generation_kwargs["temperature"] = float(temperature)
            generation_kwargs["top_p"] = float(top_p)

        with autocast_context:
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        return (text or "").strip()

    @torch.no_grad()
    def generate_caption(
        self,
        image_path: str,
        user_text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        num_beams: int = 1,
    ) -> str:
        image = load_image_square(
            image_path,
            target_size=self.img_size,
            pad_color=self.pad_color,
        )

        try:
            return self._generate_once(
                image,
                user_text,
                max_new_tokens,
                temperature,
                top_p,
                num_beams,
            )
        except Exception as exc:
            if "Image features and image tokens do not match" not in str(exc):
                raise

            image = load_image_square(
                image_path,
                target_size=self.img_size,
                pad_color=self.pad_color,
            )
            image.load()
            return self._generate_once(
                image,
                user_text,
                max_new_tokens,
                temperature,
                top_p,
                num_beams,
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--sft_adapter_path", required=True)
    parser.add_argument("--grpo_adapter_path", required=True)

    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument(
        "--all_images_path",
        type=str,
        default="dataset/mimic/mimic_cxr_all_images.txt",
    )
    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--pad_color", type=int, default=0)
    parser.add_argument("--min_pixels", type=int, default=3136)

    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if os.path.isdir(args.output_json):
        raise ValueError("--output_json must be a file path.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    with open(args.input_json, "r", encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError("Input JSON must contain a list of samples.")

    print(f"[INFO] Loaded {len(items)} input samples.")
    rel_index = build_rel_index(args.all_images_path)

    results: List[Dict[str, Any]] = []
    done_images: Set[str] = set()
    if args.resume:
        results, done_images = load_existing_results(args.output_json)

    summarizer = HFGRPOSummarizer(
        base_model=args.base_model,
        sft_adapter_path=args.sft_adapter_path,
        grpo_adapter_path=args.grpo_adapter_path,
        attn_implementation=args.attn_implementation,
        img_size=args.img_size,
        pad_color=args.pad_color,
        min_pixels=args.min_pixels,
    )

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    processed_new = 0
    total = len(items)

    for idx, item in enumerate(items, start=1):
        item_id = item.get("id", f"sample_{idx}") if isinstance(item, dict) else f"sample_{idx}"
        print(f"\n===== [{idx}/{total}] id={item_id} =====")

        if not isinstance(item, dict):
            print("[ERROR] Invalid sample type.")
            continue

        rel_image = _normalize_item_image_field(item)
        if args.resume and rel_image and rel_image in done_images:
            print(f"[SKIP] Already processed: {rel_image}")
            continue

        try:
            abs_image_path = resolve_abs_image_from_item(item, rel_index)
            step_history = item_to_stepwise_history(item)
            prompt_text = build_summary_prompt(step_history)

            raw_prediction = summarizer.generate_caption(
                image_path=abs_image_path,
                user_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
            )
            predicted_caption = clean_predicted_caption(raw_prediction)

            if not predicted_caption:
                raise RuntimeError("Generated report is empty.")

            new_item = dict(item)
            new_item["predicted_caption"] = predicted_caption
            results.append(new_item)

            if rel_image:
                done_images.add(rel_image)

            processed_new += 1
            print("[OK] Generated report.")

            if processed_new % args.save_every == 0:
                print(f"[INFO] Atomic save after {processed_new} new samples.")
                atomic_write_json(args.output_json, results)

        except Exception as exc:
            print(f"[ERROR] id={item_id}: {type(exc).__name__}")
            failed_item = dict(item)
            failed_item["predicted_caption"] = ""
            results.append(failed_item)

    print("[INFO] Final atomic save.")
    atomic_write_json(args.output_json, results)
    print(f"[OK] Saved {len(results)} records.")


if __name__ == "__main__":
    main()