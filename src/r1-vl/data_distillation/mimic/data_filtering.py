#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(
    path: str,
    data: Any,
) -> None:
    output_dir = os.path.dirname(path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    tmp_path = path + ".tmp"

    with open(
        tmp_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        tmp_path,
        path,
    )


def is_valid_sample(
    sample: Dict[str, Any],
) -> bool:
    steps = sample.get("steps")

    if not isinstance(steps, list) or not steps:
        return False

    num_steps = len(steps)

    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            return False

        count = step.get("count")
        title = step.get("title")
        content = step.get("content")
        decision = step.get("decision")

        if not isinstance(count, int) or count != index + 1:
            return False

        if not isinstance(title, str):
            return False

        if not isinstance(content, str):
            return False

        if decision not in {
            "continue",
            "summary",
        }:
            return False

        expected_decision = (
            "summary"
            if index == num_steps - 1
            else "continue"
        )

        if decision != expected_decision:
            return False

    return True


def normalize_sample(
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = dict(sample)

    image = normalized.get("image")

    if isinstance(image, str):
        normalized["image"] = os.path.basename(
            image
        )

    return normalized


def process_shard(
    shard_path: str,
    backup: bool = True,
) -> Dict[str, int]:
    if not os.path.isfile(shard_path):
        print(
            f"[WARN] Missing shard: "
            f"{os.path.basename(shard_path)}"
        )

        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
        }

    try:
        data = load_json(
            shard_path
        )
    except Exception as exc:
        print(
            f"[ERROR] Failed to load "
            f"{os.path.basename(shard_path)}: "
            f"{type(exc).__name__}"
        )

        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
        }

    if not isinstance(data, list):
        print(
            f"[ERROR] Invalid shard format: "
            f"{os.path.basename(shard_path)}"
        )

        return {
            "total": 0,
            "valid": 0,
            "invalid": 0,
        }

    total = len(data)

    if backup and total > 0:
        backup_path = shard_path + ".bak"

        if not os.path.exists(backup_path):
            atomic_write_json(
                backup_path,
                data,
            )

            print(
                f"[INFO] Backup created for "
                f"{os.path.basename(shard_path)}"
            )
        else:
            print(
                f"[INFO] Backup already exists for "
                f"{os.path.basename(shard_path)}"
            )

    valid_samples: List[
        Dict[str, Any]
    ] = []

    invalid = 0

    for sample in data:
        if (
            isinstance(sample, dict)
            and is_valid_sample(sample)
        ):
            valid_samples.append(
                normalize_sample(sample)
            )
        else:
            invalid += 1

    valid = len(
        valid_samples
    )

    atomic_write_json(
        shard_path,
        valid_samples,
    )

    print(
        f"[SHARD] {os.path.basename(shard_path)} "
        f"total={total} "
        f"valid={valid} "
        f"invalid={invalid}"
    )

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
    }


def parse_shard_ids(
    value: str,
) -> List[int]:
    shard_ids = []

    for raw_id in value.split(","):
        raw_id = raw_id.strip()

        if not raw_id:
            continue

        if not raw_id.isdigit():
            print(
                f"[WARN] Invalid shard id: "
                f"{raw_id}"
            )
            continue

        shard_ids.append(
            int(raw_id)
        )

    return sorted(
        set(shard_ids)
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shard_dir",
        default=(
            "src/r1-vl/data/mimic/shard"
        ),
    )

    parser.add_argument(
        "--shard_ids",
        default="0,1,2,3,4,5,6,7",
    )

    parser.add_argument(
        "--no_backup",
        action="store_true",
    )

    parser.add_argument(
        "--merged_output",
        default=(
            "src/r1-vl/data/mimic/"
            "qwen3vl_reasoning_raw.json"
        ),
    )

    args = parser.parse_args()

    shard_ids = parse_shard_ids(
        args.shard_ids
    )

    if not shard_ids:
        raise ValueError(
            "No valid shard IDs were provided."
        )

    total_all = 0
    valid_all = 0
    invalid_all = 0

    for shard_id in shard_ids:
        shard_path = os.path.join(
            args.shard_dir,
            f"qwen_shard{shard_id}.json",
        )

        stats = process_shard(
            shard_path,
            backup=not args.no_backup,
        )

        total_all += stats[
            "total"
        ]

        valid_all += stats[
            "valid"
        ]

        invalid_all += stats[
            "invalid"
        ]

    print(
        f"[SUMMARY] total={total_all} "
        f"valid={valid_all} "
        f"invalid={invalid_all}"
    )

    merged: List[
        Dict[str, Any]
    ] = []

    for shard_id in shard_ids:
        shard_path = os.path.join(
            args.shard_dir,
            f"qwen_shard{shard_id}.json",
        )

        if not os.path.isfile(
            shard_path
        ):
            print(
                f"[WARN] Missing shard during merge: "
                f"qwen_shard{shard_id}.json"
            )
            continue

        try:
            data = load_json(
                shard_path
            )
        except Exception as exc:
            print(
                f"[ERROR] Failed to load "
                f"qwen_shard{shard_id}.json: "
                f"{type(exc).__name__}"
            )
            continue

        if not isinstance(
            data,
            list,
        ):
            print(
                f"[ERROR] Invalid shard format: "
                f"qwen_shard{shard_id}.json"
            )
            continue

        merged.extend(
            data
        )

    atomic_write_json(
        args.merged_output,
        merged,
    )

    print(
        f"[DONE] Merged {len(merged)} samples."
    )


if __name__ == "__main__":
    main()
