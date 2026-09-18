#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Inference for the MedSPA SUM agent after SFT."""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import torch

llamafactory_src = os.getenv("LLAMAFACTORY_SRC")
if llamafactory_src:
    sys.path.append(llamafactory_src)

from llamafactory.chat.chat_model import ChatModel


_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
SYSTEM_PROMPT = ""


def clean_predicted_caption(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = _THINK_RE.sub("", text).strip()
    return cleaned if cleaned else text.strip()


def build_rel_index(all_images_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    marker = "/files/"

    with open(all_images_path, "r", encoding="utf-8") as f:
        for line in f:
            full = line.strip()
            if not full:
                continue

            idx = full.rfind(marker)
            rel = full[idx + len(marker):] if idx != -1 else os.path.basename(full)
            mapping[rel] = full
            mapping.setdefault(os.path.basename(full), full)

    print(f"[INFO] Loaded {len(mapping)} image index entries.")
    return mapping


def normalize_item_image(item: Dict[str, Any]) -> str:
    raw = item.get("image", "")
    if isinstance(raw, list):
        if not raw:
            return ""
        raw = raw[0]
    if not isinstance(raw, str):
        return ""
    return raw.strip()


def resolve_abs_image_from_item(item: Dict[str, Any], rel_index: Dict[str, str]) -> str:
    rel = normalize_item_image(item)
    if not rel:
        raise ValueError("Item missing or empty 'image' field.")

    abs_path = rel_index.get(rel)
    if abs_path is None:
        abs_path = rel_index.get(os.path.basename(rel))
    if abs_path is None:
        raise ValueError(f"Cannot resolve image path for '{rel}'")
    return abs_path


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(str(x) for x in value).strip()
    if value is None:
        return ""
    return str(value).strip()


def steps_to_stepwise_history(steps: Any) -> List[str]:
    if not isinstance(steps, list):
        return []

    valid_steps = [s for s in steps if isinstance(s, dict)]
    valid_steps.sort(key=lambda s: s.get("count", 0))

    history: List[str] = []
    for idx, step in enumerate(valid_steps, start=1):
        title = _to_text(step.get("title"))
        content = _to_text(step.get("content"))

        if title and content:
            body = f"{title}: {content}"
        elif content:
            body = content
        else:
            body = title

        if body:
            history.append(f"Step {idx}: {body}")

    return history


def build_summary_prompt(stepwise_history: List[str]) -> str:
    header = "<image>\nYou are a highly experienced radiologist.\n"

    if stepwise_history:
        reasoning_block = (
            "\nHere are the previous step-by-step clinical reasoning steps for this image:\n\n"
            + "\n".join(stepwise_history)
            + "\n\n"
        )
        instruction = (
            "Based on the above step-by-step reasoning and the image, "
            "write a clear, clinically appropriate radiology report.\n"
        )
    else:
        reasoning_block = "\n"
        instruction = (
            "Based on the provided image, write a clear, clinically appropriate radiology report.\n"
        )

    return header + reasoning_block + instruction


def build_chat_model(
    base_model: str,
    lora_path: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    num_beams: int,
) -> ChatModel:
    base_model = base_model.strip()
    lora_path = os.path.abspath(lora_path)

    if not os.path.isdir(lora_path):
        raise FileNotFoundError(f"LoRA adapter directory not found: {lora_path}")

    print(f"[INFO] Base model      : {base_model}")
    print(f"[INFO] LoRA adapter dir: {lora_path}")

    infer_args = {
        "model_name_or_path": base_model,
        "adapter_name_or_path": lora_path,
        "finetuning_type": "lora",
        "infer_backend": "huggingface",
        "template": "qwen3_vl",
        "trust_remote_code": True,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
    }

    return ChatModel(args=infer_args)


def stream_once(
    chat_model: ChatModel,
    messages: List[dict],
    system: str,
    images: List[str],
    **gen_args,
) -> str:
    response = ""
    for token in chat_model.stream_chat(
        messages=messages,
        system=system,
        images=images,
        **gen_args,
    ):
        print(token, end="", flush=True)
        response += token
    print()
    return response


def generate_caption_for_item(
    chat_model: ChatModel,
    abs_image_path: str,
    stepwise_history: List[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    user_prompt = build_summary_prompt(stepwise_history)
    messages = [{"role": "user", "content": user_prompt}]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    response = stream_once(
        chat_model=chat_model,
        messages=messages,
        system=SYSTEM_PROMPT,
        images=[abs_image_path],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = time.perf_counter() - start_time
    print(f"[LATENCY] SUM model call: {latency:.3f} s")

    return response.strip()


def atomic_write_json(path: str, data: Any) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_existing_results(output_json: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
    if not output_json or not os.path.isfile(output_json):
        return [], set()

    try:
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load existing output: {type(e).__name__}")
        return [], set()

    if not isinstance(data, list):
        print("[WARN] Existing output is not a JSON list. Resume is ignored.")
        return [], set()

    successful: List[Dict[str, Any]] = []
    done: Set[str] = set()

    for item in data:
        if not isinstance(item, dict):
            continue

        image = normalize_item_image(item)
        caption = item.get("predicted_caption", "")
        if image and isinstance(caption, str) and caption.strip():
            successful.append(item)
            done.add(image)

    print(f"[INFO] Resume loaded {len(successful)} completed items.")
    return successful, done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--all_images_path", type=str, default="dataset/mimic/mimic_cxr_all_images.txt")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(args.output_json):
        raise ValueError(f"--output_json must be a file path: {args.output_json}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    with open(args.input_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("Input JSON must contain a list of items.")

    print(f"[INFO] Loaded {len(items)} items from {args.input_json}")

    rel_index = build_rel_index(args.all_images_path)

    results: List[Dict[str, Any]] = []
    done_images: Set[str] = set()
    if args.resume:
        results, done_images = load_existing_results(args.output_json)

    chat_model = build_chat_model(
        base_model=args.base_model,
        lora_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    processed_new = 0
    total = len(items)

    for idx, item in enumerate(items, start=1):
        item_id = item.get("id", f"sample_{idx}") if isinstance(item, dict) else f"sample_{idx}"
        print(f"\n===== [{idx}/{total}] id={item_id} =====")

        if not isinstance(item, dict):
            print(f"[ERROR] id={item_id}: invalid item type")
            continue

        image = normalize_item_image(item)
        if args.resume and image and image in done_images:
            print(f"[SKIP] Already processed: {image}")
            continue

        try:
            abs_image_path = resolve_abs_image_from_item(item, rel_index)
            step_history = steps_to_stepwise_history(item.get("steps", []))

            raw_caption = generate_caption_for_item(
                chat_model=chat_model,
                abs_image_path=abs_image_path,
                stepwise_history=step_history,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )

            predicted_caption = clean_predicted_caption(raw_caption)

            new_item = dict(item)
            new_item["predicted_caption"] = predicted_caption
            results.append(new_item)

            if image and predicted_caption:
                done_images.add(image)

            processed_new += 1
            print("[OK] Generated caption.")

            if processed_new % args.save_every == 0:
                print(f"[INFO] Atomic save after {processed_new} new items -> {args.output_json}")
                atomic_write_json(args.output_json, results)

        except Exception as e:
            print(f"[ERROR] id={item_id}: {type(e).__name__}")

    print("[INFO] Final atomic save...")
    atomic_write_json(args.output_json, results)
    print(f"[OK] Saved {len(results)} items -> {args.output_json}")


if __name__ == "__main__":
    main()
