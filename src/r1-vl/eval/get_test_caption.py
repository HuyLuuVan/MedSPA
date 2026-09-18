#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

llamafactory_src = os.getenv("LLAMAFACTORY_SRC")
if llamafactory_src:
    sys.path.append(llamafactory_src)

from llamafactory.chat.chat_model import ChatModel


SYSTEM_PROMPT_REASON = (
    "You are a highly experienced radiologist. "
    "You must think step by step. "
    "For each response, output exactly one reasoning step in the format:\n"
    "<thought>...</thought><decision>continue|summary</decision>"
)

SYSTEM_PROMPT_SUMMARY = (
    "You are a highly experienced radiologist. "
    "Use the provided step-by-step reasoning chain and the image "
    "to write a concise, clinically faithful final report."
)

THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.I | re.S)
DECISION_RE = re.compile(r"<decision>(.*?)</decision>", re.I | re.S)
SAVE_EVERY = 100


def load_image_index(index_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            path = line.strip()
            if path:
                mapping[os.path.basename(path)] = path

    print(f"[INFO] Loaded {len(mapping)} image entries.")
    return mapping


def load_mimic_annotation(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("MIMIC annotation JSON must be an object containing dataset splits.")

    if isinstance(data.get("test"), list):
        return data["test"]

    for value in data.values():
        if isinstance(value, list):
            return value

    raise ValueError("Input JSON contains no usable dataset split.")


def load_medtrinity_captions(json_path: str) -> Dict[str, str]:
    if not json_path or not os.path.exists(json_path):
        print("[WARN] MedTrinity caption file not found; captions will be empty.")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("[WARN] MedTrinity caption file is not a list; captions will be empty.")
        return {}

    mapping: Dict[str, str] = {}

    for item in data:
        if not isinstance(item, dict):
            continue

        image_name = str(item.get("image", "")).strip()
        caption = str(item.get("caption", "")).strip()

        if image_name:
            mapping[image_name] = caption

    print(f"[INFO] Loaded {len(mapping)} MedTrinity captions.")
    return mapping


def resolve_image_path(item: Dict[str, Any], index_map: Dict[str, str]) -> str:
    image_field = item.get("image_path")

    if not isinstance(image_field, list) or not image_field:
        raise ValueError("image_path is missing or invalid.")

    image_name = os.path.basename(str(image_field[0]))
    image_path = index_map.get(image_name)

    if not image_path:
        raise ValueError("Image was not found in the image index.")

    return image_path


def get_public_image_path(item: Dict[str, Any], runtime_path: str) -> str:
    image_field = item.get("image_path")

    if isinstance(image_field, list) and image_field:
        return str(image_field[0])

    return os.path.basename(runtime_path)


def extract_thought_text(text: str) -> str:
    if not text:
        return ""

    match = THOUGHT_RE.search(text)
    value = match.group(1).strip() if match else text.strip()
    value = DECISION_RE.sub("", value)
    return value.strip()


def detect_summary(text: str) -> bool:
    if not text:
        return False

    match = DECISION_RE.search(text)
    if not match:
        return False

    return match.group(1).strip().lower() == "summary"


def join_chain(outputs: List[str]) -> str:
    steps: List[str] = []

    for index, output in enumerate(outputs, start=1):
        thought = extract_thought_text(output)
        if thought:
            steps.append(f"Step {index}: {thought}")

    return "\n".join(steps).strip()


def build_reasoning_prompt(history_steps: List[str]) -> str:
    header = "<image>\nYou are a highly experienced radiologist.\n"

    if not history_steps:
        return (
            header
            + "There are no previous reasoning steps yet.\n\n"
            + "Generate ONLY the FIRST reasoning step based on the image.\n"
        )

    return (
        header
        + "Here are the previous reasoning steps:\n\n"
        + "\n".join(history_steps)
        + "\nGenerate ONLY the NEXT reasoning step based on the image "
        + "and the previous steps above.\n"
    )


def build_summary_prompt_from_chain(chain_text: str) -> str:
    header = "<image>\nYou are a highly experienced radiologist.\n"

    if not chain_text.strip():
        return (
            header
            + "\nBased on the provided image, write a clear, clinically appropriate "
            + "radiology report.\n"
        )

    return (
        header
        + "\nHere are the previous step-by-step clinical reasoning steps for this image:\n\n"
        + chain_text
        + "\n\nBased on the above step-by-step reasoning and the image, "
        + "write a clear, clinically appropriate radiology report.\n"
    )


def build_chat_model(
    model_path: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    num_beams: int,
) -> ChatModel:
    model_path = os.path.abspath(model_path)

    if not os.path.isdir(model_path):
        raise FileNotFoundError("Local model path does not exist.")

    print("[INFO] Loading local model.")

    return ChatModel(
        args={
            "model_name_or_path": model_path,
            "infer_backend": "huggingface",
            "template": "qwen2_vl",
            "trust_remote_code": True,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
        }
    )


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


def atomic_write_json(path: str, data: Any) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tmp_path = path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, path)


def load_existing_output(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            print(f"[INFO] Resume: loaded {len(data)} existing records.")
            return data

        print("[WARN] Existing output is not a list; ignoring it for resume.")
        return []

    except Exception as error:
        print(f"[WARN] Failed to load existing output: {type(error).__name__}")
        return []


def run_reasoning(
    chat_model: ChatModel,
    image_path: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    outputs: List[str] = []

    for round_index in range(args.max_rounds):
        history_steps: List[str] = []

        for step_index, output in enumerate(outputs, start=1):
            thought = extract_thought_text(output)
            if thought:
                history_steps.append(f"Step {step_index}: {thought}")

        prompt = build_reasoning_prompt(history_steps)

        print(f"\n[Reasoning Round {round_index + 1}] {os.path.basename(image_path)}")

        response = stream_once(
            chat_model=chat_model,
            messages=[{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT_REASON,
            images=[image_path],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
        )

        outputs.append(response)

        if detect_summary(response):
            print("[INFO] Summary decision detected.")
            break

    return {"outputs": outputs}


def run_summary(
    chat_model: ChatModel,
    image_path: str,
    chain_text: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    prompt = build_summary_prompt_from_chain(chain_text)

    print(f"\n[Summary Generation] {os.path.basename(image_path)}")

    response = stream_once(
        chat_model=chat_model,
        messages=[{"role": "user", "content": prompt}],
        system=SYSTEM_PROMPT_SUMMARY,
        images=[image_path],
        max_new_tokens=args.final_max_new_tokens,
        temperature=args.final_temperature,
        top_p=args.final_top_p,
        num_beams=args.final_num_beams,
    )

    return {"caption": response.strip()}


def process_one(
    item: Dict[str, Any],
    args: argparse.Namespace,
    index_map: Dict[str, str],
    medtrinity_map: Dict[str, str],
    reason_chat_model: ChatModel,
    summary_chat_model: ChatModel,
) -> Dict[str, Any]:
    image_path = resolve_image_path(item, index_map)
    image_name = os.path.basename(image_path)

    mimic_caption = item.get("report", "") or item.get("caption", "")
    medtrinity_caption = medtrinity_map.get(image_name, "")

    reasoning = run_reasoning(reason_chat_model, image_path, args)
    chain_text = join_chain(reasoning["outputs"])

    summary = run_summary(summary_chat_model, image_path, chain_text, args)

    return {
        "id": item.get("id", ""),
        "image": get_public_image_path(item, image_path),
        "reasoning_output": reasoning["outputs"],
        "predicted_caption": summary["caption"],
        "mimic_caption": mimic_caption,
        "medtrinity_caption": medtrinity_caption,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--reason_model_path", required=True)
    parser.add_argument("--summary_model_path", required=True)

    parser.add_argument(
        "--mimic_json",
        default="dataset/mimic/mimic_annotation_.json",
    )
    parser.add_argument(
        "--all_images_path",
        default="dataset/mimic/mimic_cxr_all_images.txt",
    )
    parser.add_argument(
        "--medtrinity_json",
        default="dataset/medtrinity/medtrinity_llama4_cleaned.json",
    )
    parser.add_argument("--output_json", required=True)

    parser.add_argument("--max_rounds", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--final_temperature", type=float, default=0.2)
    parser.add_argument("--final_top_p", type=float, default=0.95)
    parser.add_argument("--final_max_new_tokens", type=int, default=256)
    parser.add_argument("--final_num_beams", type=int, default=1)

    args = parser.parse_args()

    index_map = load_image_index(args.all_images_path)
    items = load_mimic_annotation(args.mimic_json)
    medtrinity_map = load_medtrinity_captions(args.medtrinity_json)

    print(f"[INFO] Loaded {len(items)} MIMIC annotation items.")

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = load_existing_output(args.output_json)
    processed_ids = {
        record.get("id")
        for record in results
        if isinstance(record, dict) and "id" in record
    }

    print(f"[INFO] Already processed {len(processed_ids)} items.")

    reason_chat_model = build_chat_model(
        args.reason_model_path,
        args.temperature,
        args.top_p,
        args.max_new_tokens,
        args.num_beams,
    )

    summary_chat_model = build_chat_model(
        args.summary_model_path,
        args.final_temperature,
        args.final_top_p,
        args.final_max_new_tokens,
        args.final_num_beams,
    )

    newly_processed = 0
    total = len(items)

    for index, item in enumerate(items, start=1):
        item_id = item.get("id", "")

        if item_id in processed_ids:
            print(f"[{index}/{total}] SKIP id={item_id}")
            continue

        print(f"[{index}/{total}] Processing id={item_id}")

        try:
            record = process_one(
                item=item,
                args=args,
                index_map=index_map,
                medtrinity_map=medtrinity_map,
                reason_chat_model=reason_chat_model,
                summary_chat_model=summary_chat_model,
            )

            results.append(record)
            processed_ids.add(item_id)
            newly_processed += 1

            if newly_processed % SAVE_EVERY == 0:
                atomic_write_json(args.output_json, results)
                print(f"[INFO] Saved after {newly_processed} new samples.")

        except Exception as error:
            print(f"[ERROR] id={item_id}: {type(error).__name__}")

    atomic_write_json(args.output_json, results)
    print(f"[OK] Wrote {len(results)} total samples.")


if __name__ == "__main__":
    main()
