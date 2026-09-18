#!/usr/bin/env python3

import os
import json
import argparse
from math import ceil
from typing import List, Dict, Any, Union


def load_split(path: str, split_name: str = "test") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data: Union[Dict[str, Any], List[Dict[str, Any]]] = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected top-level JSON to be a dict with key '{split_name}', "
            f"but got {type(data)}."
        )

    if split_name not in data:
        raise ValueError(
            f"Key '{split_name}' not found in JSON. "
            f"Available keys: {list(data.keys())}"
        )

    split_data = data[split_name]

    if not isinstance(split_data, list):
        raise ValueError(
            f"Expected data['{split_name}'] to be a list, "
            f"but got {type(split_data)}."
        )

    return split_data


def load_bad_samples(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_shard(
    out_path: str,
    split_name: str,
    items: List[Dict[str, Any]],
) -> None:
    shard_obj = {split_name: items}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(shard_obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        default="dataset/mimic/mimic_annotation_.json",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write shard JSON files",
    )
    parser.add_argument(
        "--bad_samples_path",
        default="src/r1-vl/data/mimic/bad_sample.txt",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=16,
        help="Number of shards to create",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        help="Name of the split to shard",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    items = load_split(
        args.input_json,
        split_name=args.split_name,
    )
    bad_samples = load_bad_samples(args.bad_samples_path)

    filtered_items: List[Dict[str, Any]] = []
    skipped = 0

    for item in items:
        image = item.get("image")

        if isinstance(image, str):
            if image in bad_samples:
                skipped += 1
                continue

        elif isinstance(image, list):
            if any(
                isinstance(x, str) and x in bad_samples
                for x in image
            ):
                skipped += 1
                continue

        filtered_items.append(item)

    items = filtered_items
    n = len(items)

    print(f"[INFO] Skipped bad samples: {skipped}")

    if n == 0:
        print("[WARN] Split is empty, nothing to shard.")
        return

    per_shard = ceil(n / args.num_shards)

    print(f"[INFO] Total '{args.split_name}' samples: {n}")
    print(
        f"[INFO] Creating {args.num_shards} shards, "
        f"~{per_shard} samples per shard"
    )

    total_written = 0

    for i in range(args.num_shards):
        start = i * per_shard
        end = min((i + 1) * per_shard, n)
        shard_items = items[start:end]

        if not shard_items:
            print(f"[INFO] Shard {i} is empty, skipping.")
            continue

        out_path = os.path.join(
            args.output_dir,
            f"annotation__{args.split_name}_shard{i}.json",
        )

        save_shard(
            out_path,
            args.split_name,
            shard_items,
        )

        print(
            f"[OK] Shard {i}: "
            f"{len(shard_items)} samples -> {out_path}"
        )

        total_written += len(shard_items)

    print(
        f"[INFO] Done. Total written samples across shards: "
        f"{total_written}"
    )


if __name__ == "__main__":
    main()