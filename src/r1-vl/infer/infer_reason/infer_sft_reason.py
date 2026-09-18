#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step-by-step PR-agent inference with Qwen3-VL and an SFT LoRA adapter."""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Set, Tuple

import torch

llamafactory_src = os.getenv("LLAMAFACTORY_SRC")
if llamafactory_src:
    sys.path.append(llamafactory_src)

from llamafactory.chat.chat_model import ChatModel


_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)
_DECISION_RE = re.compile(r"<decision>(.*?)</decision>", re.DOTALL | re.IGNORECASE)
SYSTEM_PROMPT = ""


def extract_thought_text(step: str) -> str:
    if not step:
        return ""
    match = _THOUGHT_RE.search(step)
    text = match.group(1).strip() if match else step.strip()
    return _DECISION_RE.sub("", text).strip()


def extract_decision(step: str) -> str:
    if not step:
        return "continue"
    match = _DECISION_RE.search(step)
    if not match:
        return "continue"
    decision = match.group(1).strip().lower()
    return decision if decision else "continue"


def concat_segments(segments: Iterable[str], sep: str = "\n") -> str:
    return sep.join(s for s in segments if s)


def build_reasoning_prompt(history_plain_steps: List[str]) -> str:
    header = "<image>\nYou are a highly experienced radiologist.\n"

    if not history_plain_steps:
        history_block = "There are no previous reasoning steps yet.\n\n"
        instruction = "Generate ONLY the FIRST reasoning step based on the image.\n"
    else:
        history_block = (
            "Here are the previous reasoning steps:\n\n"
            + concat_segments(history_plain_steps)
            + "\n"
        )
        instruction = (
            "Generate ONLY the NEXT reasoning step based on the image "
            "and the previous steps above.\n"
        )

    return header + history_block + instruction


def load_image_index(index_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            path = line.strip()
            if path:
                mapping[os.path.basename(path)] = path
    print(f"[INFO] Loaded {len(mapping)} image paths from index")
    return mapping


def load_annotation(json_path: str, split: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        print(f"[INFO] Loaded list with {len(data)} items")
        return data

    if not isinstance(data, dict):
        raise ValueError("Annotation JSON must be a list or a split dictionary.")

    if split in data and isinstance(data[split], list):
        print(f"[INFO] Using split '{split}' with {len(data[split])} items")
        return data[split]

    list_fields = [(key, value) for key, value in data.items() if isinstance(value, list)]
    if len(list_fields) == 1:
        key, value = list_fields[0]
        print(f"[INFO] Using only available split '{key}' with {len(value)} items")
        return value

    raise ValueError(f"Split '{split}' was not found in the annotation JSON.")


def resolve_image_path(item: Dict[str, Any], index_map: Dict[str, str]) -> str:
    raw = item.get("image_path")
    if isinstance(raw, list):
        if not raw:
            raise ValueError("image_path is empty.")
        rel_path = str(raw[0]).strip()
    elif isinstance(raw, str):
        rel_path = raw.strip()
    else:
        raise ValueError("image_path is missing or invalid.")

    filename = os.path.basename(rel_path)
    abs_path = index_map.get(filename)
    if abs_path is None:
        raise ValueError(f"Image '{filename}' was not found in the image index.")
    return abs_path


def rel_image_path_from_abs(abs_path: str) -> str:
    if not abs_path:
        return ""
    marker = "/files/"
    idx = abs_path.rfind(marker)
    return abs_path[idx + len(marker):] if idx != -1 else os.path.basename(abs_path)


def load_findings_map(findings_json: str) -> Dict[str, str]:
    with open(findings_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Findings JSON must be a list of objects.")

    mapping: Dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        image_path = str(item.get("image_path", "")).strip()
        finding = str(item.get("finding", "")).strip()
        if image_path:
            mapping[image_path] = finding

    print(f"[INFO] Loaded {len(mapping)} findings entries")
    return mapping


def parse_title_and_content(thought: str) -> Tuple[str, str]:
    text = (thought or "").strip()
    if not text:
        return "", ""

    parts = text.split(":", 1)
    if len(parts) == 2:
        title = parts[0].strip()
        content = parts[1].strip()
        if 0 < len(title.split()) <= 16:
            return title, content

    return text, ""


def atomic_write_json(path: str, data: Any) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def normalize_image_field(record: Dict[str, Any]) -> str:
    image = record.get("image", "")
    if isinstance(image, list):
        image = image[0] if image else ""
    return str(image).strip() if image is not None else ""


def load_existing_results(output_json: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
    if not output_json or not os.path.isfile(output_json):
        return [], set()

    try:
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not load existing output: {type(exc).__name__}")
        return [], set()

    if not isinstance(data, list):
        print("[WARN] Existing output is not a list; resume state ignored")
        return [], set()

    results: List[Dict[str, Any]] = []
    done: Set[str] = set()

    for record in data:
        if not isinstance(record, dict):
            continue
        image = normalize_image_field(record)
        steps = record.get("steps")
        if image and isinstance(steps, list) and len(steps) > 0:
            results.append(record)
            done.add(image)

    print(f"[INFO] Resume enabled: {len(done)} completed images loaded")
    return results, done


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

    print(f"[INFO] Base model: {base_model}")
    print(f"[INFO] LoRA adapter: {lora_path}")
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


def run_reasoning_loop(
    chat_model: ChatModel,
    image_path: str,
    max_rounds: int = 10,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    num_beams: int = 1,
) -> List[str]:
    images = [image_path]
    history_plain_steps: List[str] = []
    all_model_steps: List[str] = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_start = time.perf_counter()

    for step_idx in range(1, max_rounds + 1):
        user_content = build_reasoning_prompt(history_plain_steps)
        messages = [{"role": "user", "content": user_content}]

        print(
            f"\n[Round {step_idx}] Assistant ({os.path.basename(image_path)}): ",
            end="",
            flush=True,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.perf_counter()

        response = stream_once(
            chat_model,
            messages=messages,
            system=SYSTEM_PROMPT,
            images=images,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_latency = time.perf_counter() - step_start
        print(f"[LATENCY] PR step {step_idx}: {step_latency:.3f} s")

        all_model_steps.append(response)

        thought = extract_thought_text(response)
        if thought:
            history_plain_steps.append(f"Step {step_idx}: {thought}")

        if extract_decision(response) == "summary":
            print("[INFO] Summary decision detected. Stop.")
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_latency = time.perf_counter() - total_start
    num_calls = len(all_model_steps)
    avg_latency = total_latency / num_calls if num_calls else 0.0

    print("\n========== PR INFERENCE LATENCY ==========")
    print(f"PR model calls : {num_calls}")
    print(f"PR total time  : {total_latency:.3f} s")
    print(f"Avg. time/call : {avg_latency:.3f} s" if num_calls else "Avg. time/call : N/A")
    print("==========================================\n")

    return all_model_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--mimic_json", default="dataset/mimic/mimic_annotation_.json")
    parser.add_argument("--all_images_path", default="dataset/mimic/mimic_cxr_all_images.txt")
    parser.add_argument("--findings_json", default="dataset/mimic/finding/mimic_gt_findings.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_rounds", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained_only", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(args.output_json):
        raise ValueError("--output_json must be a file path, not a directory.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    index_map = load_image_index(args.all_images_path)
    items = load_annotation(args.mimic_json, args.split)
    findings_map = load_findings_map(args.findings_json)

    results: List[Dict[str, Any]] = []
    done_images: Set[str] = set()
    if args.resume:
        results, done_images = load_existing_results(args.output_json)

    chat_model = None
    if not args.pretrained_only:
        chat_model = build_chat_model(
            base_model=args.base_model,
            lora_path=args.model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )

    processed_new = 0
    total_items = len(items)

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

            record: Dict[str, Any] = {
                "image": rel_image_path,
                "split": args.split,
                "caption": caption,
                "trusted_finding": trusted_finding,
            }

            if not args.pretrained_only:
                raw_steps = run_reasoning_loop(
                    chat_model=chat_model,
                    image_path=abs_image_path,
                    max_rounds=args.max_rounds,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )

                structured_steps: List[Dict[str, Any]] = []
                for step_idx, raw_step in enumerate(raw_steps, start=1):
                    thought = extract_thought_text(raw_step)
                    title, content = parse_title_and_content(thought)
                    structured_steps.append(
                        {
                            "count": step_idx,
                            "title": title,
                            "content": content,
                            "decision": extract_decision(raw_step),
                        }
                    )
                record["steps"] = structured_steps

            results.append(record)
            if rel_image_path:
                done_images.add(rel_image_path)
            processed_new += 1

            if processed_new % args.save_every == 0:
                print(f"[INFO] Atomic save after {processed_new} new samples")
                atomic_write_json(args.output_json, results)

        except Exception as exc:
            print(f"[ERROR] id={item_id}: {type(exc).__name__}: {exc}")

    print("[INFO] Final atomic save")
    atomic_write_json(args.output_json, results)
    print(f"[OK] Saved {len(results)} records to {args.output_json}")


if __name__ == "__main__":
    main()