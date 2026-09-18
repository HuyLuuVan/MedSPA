#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from glob import glob
from typing import Any, Dict

import numpy as np


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


def label_to_binary(
    label: str,
    treat_uncertain_as_positive: bool,
) -> int:
    label = (label or "blank").lower()

    if label == "positive":
        return 1

    if label == "uncertain":
        return 1 if treat_uncertain_as_positive else 0

    return 0


def compute_ce_for_files(
    shard_dir: str,
    pattern: str,
    treat_uncertain_as_positive: bool,
    skip_blank_gt: bool,
    pred_key: str,
    gt_key: str,
) -> Dict[str, Any]:

    files = sorted(
        glob(
            os.path.join(
                shard_dir,
                pattern,
            )
        )
    )

    if not files:
        raise FileNotFoundError(
            "No matching JSON files were found."
        )

    tp = 0.0
    fp = 0.0
    fn = 0.0
    total_pairs = 0

    conditions = [
        condition
        for condition in CONDITIONS
        if condition != "no_finding"
    ]

    per_cond_tp = {
        condition: 0.0
        for condition in conditions
    }

    per_cond_fp = {
        condition: 0.0
        for condition in conditions
    }

    per_cond_fn = {
        condition: 0.0
        for condition in conditions
    }

    for path in files:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(
                f"[WARN] Skipping non-list file: "
                f"{os.path.basename(path)}"
            )
            continue

        for item in data:

            if not isinstance(item, dict):
                continue

            if gt_key == "trusted_finding":
                gt_labels = (
                    item.get(
                        "chexbert_finding_labels"
                    )
                    or {}
                )

            elif gt_key == "caption":
                gt_labels = (
                    item.get(
                        "chexbert_caption_labels"
                    )
                    or {}
                )

            else:
                gt_labels = {}

            if not isinstance(
                gt_labels,
                dict,
            ):
                gt_labels = {}

            if pred_key == "predicted_caption":
                pred_labels = (
                    item.get(
                        "chexbert_predicted_caption_labels"
                    )
                    or {}
                )

            elif pred_key == "predicted_finding":
                pred_labels = (
                    item.get(
                        "chexbert_predicted_finding_labels"
                    )
                    or {}
                )

            else:
                pred_labels = {}

            if not isinstance(
                pred_labels,
                dict,
            ):
                pred_labels = {}

            for condition in conditions:

                gt_label = (
                    gt_labels.get(
                        condition,
                        "blank",
                    )
                    or "blank"
                ).lower()

                if (
                    skip_blank_gt
                    and gt_label == "blank"
                ):
                    continue

                pred_label = (
                    pred_labels.get(
                        condition,
                        "blank",
                    )
                    or "blank"
                ).lower()

                gt_pos = label_to_binary(
                    gt_label,
                    treat_uncertain_as_positive,
                )

                pred_pos = label_to_binary(
                    pred_label,
                    treat_uncertain_as_positive,
                )

                if (
                    gt_pos == 1
                    and pred_pos == 1
                ):
                    tp += 1
                    per_cond_tp[condition] += 1

                elif (
                    gt_pos == 0
                    and pred_pos == 1
                ):
                    fp += 1
                    per_cond_fp[condition] += 1

                elif (
                    gt_pos == 1
                    and pred_pos == 0
                ):
                    fn += 1
                    per_cond_fn[condition] += 1

                total_pairs += 1

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        tp / (
            tp
            + 0.5 * (fp + fn)
        )
        if (tp + fp + fn) > 0
        else 0.0
    )

    per_cond_precision: Dict[str, float] = {}
    per_cond_recall: Dict[str, float] = {}
    per_cond_f1: Dict[str, float] = {}

    for condition in conditions:

        c_tp = per_cond_tp[condition]
        c_fp = per_cond_fp[condition]
        c_fn = per_cond_fn[condition]

        c_precision = (
            c_tp / (c_tp + c_fp)
            if (c_tp + c_fp) > 0
            else 0.0
        )

        c_recall = (
            c_tp / (c_tp + c_fn)
            if (c_tp + c_fn) > 0
            else 0.0
        )

        c_f1 = (
            c_tp / (
                c_tp
                + 0.5 * (c_fp + c_fn)
            )
            if (c_tp + c_fp + c_fn) > 0
            else 0.0
        )

        per_cond_precision[
            condition
        ] = c_precision

        per_cond_recall[
            condition
        ] = c_recall

        per_cond_f1[
            condition
        ] = c_f1

    macro_f1 = (
        float(
            np.mean(
                list(
                    per_cond_f1.values()
                )
            )
        )
        if per_cond_f1
        else 0.0
    )

    return {
        "num_pairs": float(
            total_pairs
        ),
        "micro_precision": float(
            precision
        ),
        "micro_recall": float(
            recall
        ),
        "micro_f1": float(
            f1
        ),
        "macro_f1": float(
            macro_f1
        ),
        "per_condition_precision": (
            per_cond_precision
        ),
        "per_condition_recall": (
            per_cond_recall
        ),
        "per_condition_f1": (
            per_cond_f1
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shard_dir",
        required=True,
    )

    parser.add_argument(
        "--pattern",
        default=(
            "test_caption_shard*_label.json"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    gt_keys = [
        "trusted_finding",
        "caption",
    ]

    pred_keys = [
        "predicted_caption",
        "predicted_finding",
    ]

    skip_blank_options = [
        True,
        False,
    ]

    treat_uncertain_options = [
        False,
        True,
    ]

    results: Dict[str, Any] = {}

    print(
        "=== Running CE for all option combinations ==="
    )

    for gt_key in gt_keys:

        for pred_key in pred_keys:

            for skip_blank_gt in (
                skip_blank_options
            ):

                for treat_uncertain_as_positive in (
                    treat_uncertain_options
                ):

                    key_name = (
                        f"gt={gt_key}"
                        f"__pred={pred_key}"
                        f"__skip_blank_gt="
                        f"{skip_blank_gt}"
                        f"__treat_uncertain_pos="
                        f"{treat_uncertain_as_positive}"
                    )

                    print(
                        "--------------------------------------------------"
                    )

                    print(
                        f"=== {key_name} ==="
                    )

                    stats = compute_ce_for_files(
                        shard_dir=args.shard_dir,
                        pattern=args.pattern,
                        treat_uncertain_as_positive=(
                            treat_uncertain_as_positive
                        ),
                        skip_blank_gt=(
                            skip_blank_gt
                        ),
                        pred_key=pred_key,
                        gt_key=gt_key,
                    )

                    results[
                        key_name
                    ] = stats

                    print(
                        "Num pairs: "
                        f"{stats['num_pairs']:.0f}"
                    )

                    print(
                        "Micro Precision: "
                        f"{stats['micro_precision']:.4f}"
                    )

                    print(
                        "Micro Recall:    "
                        f"{stats['micro_recall']:.4f}"
                    )

                    print(
                        "Micro F1:        "
                        f"{stats['micro_f1']:.4f}"
                    )

                    print(
                        "Macro F1:        "
                        f"{stats['macro_f1']:.4f}"
                    )

    output = {
        "config": {
            "pattern": args.pattern,
            "gt_keys": gt_keys,
            "pred_keys": pred_keys,
            "skip_blank_gt_options": (
                skip_blank_options
            ),
            "treat_uncertain_as_positive_options": (
                treat_uncertain_options
            ),
        },
        "results": results,
    }

    output_dir = os.path.dirname(
        args.output
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[DONE] CE results saved."
    )


if __name__ == "__main__":
    main()
