#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert reasoning data to Summary SFT format with automatic resolution
of full image paths using mimic_cxr_all_images.txt.
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List



CAPTION_CANDIDATE_KEYS = [
    "caption",
    "report",
    "answer",
]


def load_json_maybe_jsonl(
    path: str,
) -> List[Dict[str, Any]]:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        text = f.read().strip()

    if not text:
        return []

    # Detect JSONL.
    if (
        "\n" in text
        and text.lstrip().startswith("{")
        and text.count("\n{") >= 1
    ):
        items = []

        for line in text.splitlines():

            line = line.strip()

            if line:
                items.append(
                    json.loads(line)
                )

        return items

    data = json.loads(text)

    return (
        data
        if isinstance(data, list)
        else [data]
    )


def load_all_json_from_shard_input(
    shard_input: str,
) -> List[Dict[str, Any]]:
    """
    Accept either:
      - a directory containing JSON shards
      - a single JSON/JSONL file
    """

    if os.path.isdir(shard_input):

        paths = sorted(
            glob.glob(
                os.path.join(
                    shard_input,
                    "*.json",
                )
            )
        )

    elif os.path.isfile(shard_input):

        paths = [shard_input]

    else:

        raise FileNotFoundError(
            "Shard input was not found."
        )


    all_data: List[Dict[str, Any]] = []

    for path in paths:

        part = load_json_maybe_jsonl(
            path
        )

        all_data.extend(part)

        print(
            f"[INFO] Loaded {len(part)} samples "
            f"from shard {os.path.basename(path)}"
        )

    print(
        f"[INFO] Total examples loaded: "
        f"{len(all_data)}"
    )

    return all_data


def load_all_images_map(
    txt_path: str,
) -> Dict[str, str]:
    """
    Return a mapping:

        image basename -> full runtime image path
    """

    mapping: Dict[str, str] = {}

    if not os.path.exists(txt_path):

        print(
            "[WARN] Image index was not found."
        )

        return mapping


    with open(
        txt_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            full_path = line.strip()

            if not full_path:
                continue

            basename = os.path.basename(
                full_path
            )

            mapping[basename] = full_path


    print(
        f"[INFO] Loaded {len(mapping)} image paths."
    )

    return mapping


def steps_to_stepwise_history(
    steps: Any,
) -> List[str]:

    if not isinstance(
        steps,
        list,
    ):
        return []


    steps_sorted = sorted(
        steps,
        key=lambda s: s.get(
            "count",
            0,
        ),
    )


    output: List[str] = []


    for idx, step in enumerate(
        steps_sorted,
        start=1,
    ):

        title = (
            step.get("title")
            or ""
        ).strip()

        content = (
            step.get("content")
            or ""
        ).strip()


        if title and content:

            body = (
                f"{title}: {content}"
            )

        elif content:

            body = content

        else:

            body = title


        if body:

            output.append(
                f"Step {idx}: {body}"
            )


    return output


def build_summary_prompt(
    stepwise_history: List[str],
) -> str:

    header = "<image>\nYou are a highly experienced radiologist.\n"

    if stepwise_history:

        reasoning_block = (
            "\nHere are the previous step-by-step clinical reasoning steps for this image:\n\n"
            + "\n".join(stepwise_history)
            + "\n\n"
        )

        instruction = (
            "Based on the above step-by-step reasoning and the image, write a clear, clinically appropriate radiology report.\n"
        )

    else:

        reasoning_block = "\n"

        instruction = (
            "Based on the provided image, write a clear, clinically appropriate radiology report.\n"
        )

    return (
        header
        + reasoning_block
        + instruction
    )


def get_caption_from_item(
    item: Dict[str, Any],
) -> str:

    if isinstance(
        item.get("caption"),
        str,
    ):
        return item["caption"]


    for key in CAPTION_CANDIDATE_KEYS:

        if (
            key in item
            and isinstance(
                item[key],
                str,
            )
        ):
            return item[key]


    return ""


def resolve_abs_image(
    image_field: Any,
    all_map: Dict[str, str],
) -> str | None:
    """
    Resolve an image filename to its full runtime path
    using basename lookup.
    """

    if image_field is None:
        return None


    if isinstance(
        image_field,
        list,
    ):

        if not image_field:
            return None

        image_field = image_field[0]


    if not isinstance(
        image_field,
        str,
    ):
        return None


    basename = os.path.basename(
        image_field
    )

    abs_path = all_map.get(
        basename
    )


    if abs_path is None:

        print(
            f"[WARN] Cannot resolve image "
            f"'{basename}' -> skip"
        )


    return abs_path


def convert_item(
    item: Dict[str, Any],
    all_map: Dict[str, str],
    skip_missing_caption: bool = False,
):

    abs_image = resolve_abs_image(
        item.get("image"),
        all_map,
    )


    if abs_image is None:
        return None


    step_history = (
        steps_to_stepwise_history(
            item.get(
                "steps",
                [],
            )
        )
    )


    if not step_history:

        print(
            "[WARN] No steps found -> skip"
        )

        return None


    user_prompt = build_summary_prompt(
        step_history
    )


    caption = get_caption_from_item(
        item
    )


    if not caption:

        if skip_missing_caption:

            print(
                "[WARN] Missing caption -> skip"
            )

            return None

        print(
            "[WARN] Missing caption -> keep empty"
        )


    return {
        "images": [
            abs_image
        ],
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            },
            {
                "role": "assistant",
                "content": caption,
            },
        ],
    }


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--shard_input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--skip-missing-caption",
        action="store_true",
    )

    parser.add_argument(
        "--all-images",
        default="dataset/mimic_cxr_all_images.txt",
    )


    args = parser.parse_args()


    data = load_all_json_from_shard_input(
        args.shard_input
    )


    all_map = load_all_images_map(
        args.all_images
    )


    output_items: List[
        Dict[str, Any]
    ] = []


    for item in data:

        if not isinstance(
            item,
            dict,
        ):
            continue


        converted = convert_item(
            item,
            all_map,
            skip_missing_caption=(
                args.skip_missing_caption
            ),
        )


        if converted:

            output_items.append(
                converted
            )


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
            output_items,
            f,
            ensure_ascii=False,
            indent=2,
        )


    print(
        f"[OK] Wrote "
        f"{len(output_items)} samples."
    )


if __name__ == "__main__":
    main()