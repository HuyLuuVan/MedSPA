#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
from typing import Any, Dict, List


def load_json_maybe_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    if (
        "\n" in text
        and text.lstrip().startswith("{")
        and text.count("\n{") >= 1
    ):
        return [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]

    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def load_all_json_from_shard_input(
    shard_input: str,
) -> List[Dict[str, Any]]:

    if os.path.isdir(shard_input):
        paths = sorted(
            glob.glob(
                os.path.join(shard_input, "*.json")
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
        part = load_json_maybe_jsonl(path)
        all_data.extend(part)

        print(
            f"[INFO] Loaded {len(part)} samples "
            f"from {os.path.basename(path)}"
        )

    print(
        f"[INFO] Total examples loaded: {len(all_data)}"
    )

    return all_data


def steps_to_stepwise_history(
    steps: Any,
) -> List[str]:

    if not isinstance(steps, list):
        return []

    steps_sorted = sorted(
        steps,
        key=lambda s: s.get("count", 0),
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
            body = f"{title}: {content}"
        elif content:
            body = content
        else:
            body = title

        if body:
            output.append(
                f"Step {idx}: {body}"
            )

    return output


def build_qa_prompt(
    stepwise_history: List[str],
    question: str,
) -> str:

    header = (
        "<image>\n"
        "You are a highly experienced doctor.\n"
    )

    if stepwise_history:
        reasoning_block = (
            "\nHere are the previous step-by-step clinical reasoning steps for this image:\n\n"
            + "\n".join(stepwise_history)
            + "\n\n"
        )
    else:
        reasoning_block = "\n"

    instruction = (
        "Answer the following clinical question based on the image and the reasoning steps above.\n"
        "Be concise and clinically accurate.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return (
        header
        + reasoning_block
        + instruction
    )


def convert_item_to_qa_samples(
    item: Dict[str, Any],
) -> List[Dict[str, Any]]:

    image_path = item.get("image")

    if (
        not isinstance(image_path, str)
        or not image_path.strip()
    ):
        return []

    step_history = steps_to_stepwise_history(
        item.get("steps", [])
    )

    if not step_history:
        return []

    qa_list = item.get("qa")

    if not isinstance(qa_list, list):
        return []

    output: List[Dict[str, Any]] = []

    image_id = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    for idx, qa in enumerate(
        qa_list,
        start=1,
    ):
        if not isinstance(qa, dict):
            continue

        question = (
            qa.get("question")
            or ""
        ).strip()

        answer = (
            qa.get("answer")
            or ""
        ).strip()

        if not question or not answer:
            continue

        user_prompt = build_qa_prompt(
            step_history,
            question,
        )

        output.append(
            {
                "images": [image_path],
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ],
                "id": f"{image_id}_q{idx}",
            }
        )

    return output


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

    args = parser.parse_args()

    data = load_all_json_from_shard_input(
        args.shard_input
    )

    output_items: List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        output_items.extend(
            convert_item_to_qa_samples(item)
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
        f"[OK] Wrote {len(output_items)} samples."
    )


if __name__ == "__main__":
    main()