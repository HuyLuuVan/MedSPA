#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import List, Dict, Any

from radgraph import F1RadGraph


# ========================
# CONFIG (default paths)
# ========================
INPUT_JSON = "src/r1-vl/data/qwen3vl_reasoning_raw.json"
OUTPUT_JSON = "src/r1-vl/data/qwen3vl_reasoning_radgraph_score.json"
ALL_IMAGES_PATH = "dataset/mimic_cxr_all_images.txt"

MODEL_TYPE = "radgraph"
REWARD_LEVEL = "all"     
SAVE_EVERY = 1000         
# ========================


def load_json(path: str):
    """Load a JSON array or JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        else:
            data = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
            return data


def atomic_save_json(path: str, data: Any):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def safe_float_or_neg1(x) -> float:
    try:
        if x is None:
            return -1.0
        return float(x)
    except Exception:
        return -1.0


# [NEW] Load mapping from image filename → absolute path from annotation txt
def load_image_map(txt_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not txt_path:
        return mapping
    if not os.path.exists(txt_path):
        print("[WARN] Image-path annotation file was not found.")
        return mapping

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fname = os.path.basename(line)
                if not fname:
                    continue
                # last one wins if duplicated filenames
                mapping[fname] = line
    except Exception as e:
        print(f"[WARN] Failed to load image map: {type(e).__name__}")
    return mapping


def update_image_paths(data: List[Dict[str, Any]], image_map: Dict[str, str]) -> int:
    if not image_map:
        return 0

    missing_count = 0

    for sample in data:
        img = sample.get("image")
        if not img:
            missing_count += 1
            continue

        fname = os.path.basename(img)
        abs_path = image_map.get(fname)

        if abs_path:
            sample["image"] = abs_path
        else:
            missing_count += 1

    return missing_count


def load_existing_output(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        # fallback: unexpected structure
        return []
    except Exception as e:
        print(f"[WARN] Failed to load existing output: {type(e).__name__}")
        return []


def main():
    print("📥 Loading input data...")
    data = load_json(INPUT_JSON)
    print(f"  → Loaded {len(data)} samples")

    image_map: Dict[str, str] = {}
    if ALL_IMAGES_PATH:
        print("📄 Loading image path annotation map...")
        image_map = load_image_map(ALL_IMAGES_PATH)
        print(f"  → Loaded {len(image_map)} image paths")

        missing_input = update_image_paths(data, image_map)
        print(f"⚠️ Missing absolute paths in input data: {missing_input} samples")

    print("📂 Checking existing output...")
    existing_output = load_existing_output(OUTPUT_JSON)
    print(f"  → Found {len(existing_output)} already-scored samples in output (if any)")

    if image_map:
        missing_existing = update_image_paths(existing_output, image_map)
        print(f"⚠️ Missing absolute paths in existing output: {missing_existing} samples")

    # Build resume lookup using resolved paths, while keeping saved output private.
    existing_by_key: Dict[tuple, Dict[str, Any]] = {}
    for item in existing_output:
        img = item.get("image")
        cap = (item.get("caption") or "").strip()
        key = (img, cap)

        sanitized_item = dict(item)
        if img:
            sanitized_item["image"] = os.path.basename(img)

        existing_by_key[key] = sanitized_item

    print(f"⚙️ Initializing F1RadGraph...")
    f1radgraph = F1RadGraph(reward_level=REWARD_LEVEL, model_type=MODEL_TYPE)
    print("  → F1RadGraph ready")

    output_data: List[Dict[str, Any]] = []

    processed_count = 0  # count for progress & autosave (includes reused + newly scored)

    for idx, sample in enumerate(data):
        caption = (sample.get("caption") or "").strip()
        img = sample.get("image")
        key = (img, caption)

        if key in existing_by_key:
            output_data.append(existing_by_key[key])
            processed_count += 1

            if processed_count % 50 == 0:
                print(f"  → [RESUME] Reused {processed_count}/{len(data)} samples")

            if processed_count % SAVE_EVERY == 0:
                print(f"💾 Auto-saving at sample {processed_count} ...")
                atomic_save_json(OUTPUT_JSON, output_data)
                print("  → Auto-save done.")
            continue

        steps = sample.get("steps", []) or []
        trusted_finding = sample.get("trusted_finding")

        # If caption or steps are missing, skip scoring but keep structure
        if not caption or not steps:
            compact_sample = {
                "image": os.path.basename(sample.get("image") or ""),
                "split": sample.get("split"),
                "caption": caption,
                "trusted_finding": trusted_finding,
                "caption_annotation": None,
                # keep original steps (no scores)
                "steps": steps,
            }
            output_data.append(compact_sample)
            processed_count += 1

        else:
            compact_steps: List[Dict[str, Any]] = []
            caption_annotation = None  # will be taken from hyp_annotations[0] on first successful call

            for i, step in enumerate(steps):
                title = (step.get("title") or "").strip()
                content = (step.get("content") or "").strip()

                if title and content:
                    ref_text = f"{title}. {content}"
                elif content:
                    ref_text = content
                else:
                    ref_text = title

                # Default values in case of errors
                rg_e = -1.0
                rg_er = -1.0
                rg_bar_er = -1.0
                step_annotation = None

                if ref_text:
                    try:
                        # Call F1RadGraph per step:
                        #   hyp = final caption/report
                        #   ref = this reasoning step text
                        mean_reward, reward_list, hyp_annotations, ref_annotations = f1radgraph(
                            hyps=[caption],
                            refs=[ref_text],
                        )

                        # Capture caption annotation once (from first successful call)
                        if caption_annotation is None and hyp_annotations:
                            caption_annotation = hyp_annotations[0]

                        # Capture annotation for this reasoning step (ref side)
                        if ref_annotations:
                            step_annotation = ref_annotations[0]

                        # mean_reward = (RG_E, RG_ER, RG_BAR_ER)
                        if mean_reward is None:
                            print(f"[WARN] Sample {idx}, step {i}: mean_reward is None")
                        else:
                            try:
                                # mean_reward is iterable (list/tuple/tensor-like)
                                if len(mean_reward) < 3:
                                    print(
                                        f"[WARN] Sample {idx}, step {i}: "
                                        f"mean_reward has <3 values: {mean_reward}"
                                    )
                                    # still try to use the first element; others remain -1
                                    rg_e = safe_float_or_neg1(mean_reward[0])
                                    rg_er = -1.0
                                    rg_bar_er = -1.0
                                else:
                                    rg_e = safe_float_or_neg1(mean_reward[0])
                                    rg_er = safe_float_or_neg1(mean_reward[1])
                                    rg_bar_er = safe_float_or_neg1(mean_reward[2])
                            except TypeError:
                                # mean_reward is scalar → use it as rg_e only
                                rg_e = safe_float_or_neg1(mean_reward)
                                rg_er = -1.0
                                rg_bar_er = -1.0

                    except Exception as e:
                        print(
                            f"[WARN] RadGraph scoring failed for sample {idx}, "
                            f"step {i}: {type(e).__name__}"
                        )
                        # keep default -1.0 scores and None annotation

                # Store full step information + scores
                step_entry = {
                    "count": step.get("count", i + 1),
                    "title": step.get("title", ""),
                    "content": step.get("content", ""),
                    "decision": step.get("decision", ""),
                    # Text actually used for RadGraph scoring (for debugging/inspection)
                    "ref_text": ref_text,
                    "rg_e": rg_e,
                    "rg_er": rg_er,
                    "rg_bar_er": rg_bar_er,
                    "annotation": step_annotation,
                }
                compact_steps.append(step_entry)

            compact_sample = {
                "image": os.path.basename(sample.get("image") or ""),
                "split": sample.get("split"),
                "caption": caption,
                "trusted_finding": trusted_finding,
                "caption_annotation": caption_annotation,
                "steps": compact_steps,
            }
            output_data.append(compact_sample)
            processed_count += 1

        # Progress logging (applies to both reused + newly scored)
        if processed_count % 50 == 0:
            print(f"  → Processed {processed_count}/{len(data)} samples")

        # Periodic auto-save
        if processed_count % SAVE_EVERY == 0:
            print(f"💾 Auto-saving at sample {processed_count} ...")
            atomic_save_json(OUTPUT_JSON, output_data)
            print("  → Auto-save done.")

    # Final save
    print("💾 Saving final output...")
    atomic_save_json(OUTPUT_JSON, output_data)
    print("✅ Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute RadGraph-based stepwise reasoning scores for reasoning data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_JSON,
        help="Path to input JSON/JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_JSON,
        help="Path to output JSON file.",
    )
    # [NEW] CLI param for annotation txt file
    parser.add_argument(
        "--all_images_path",
        type=str,
        default=ALL_IMAGES_PATH,
        help="Path to a txt file listing image paths.",
    )
    args = parser.parse_args()

    INPUT_JSON = args.input
    OUTPUT_JSON = args.output
    ALL_IMAGES_PATH = args.all_images_path

    main()