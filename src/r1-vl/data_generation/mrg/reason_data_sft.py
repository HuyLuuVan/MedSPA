#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, Iterable, List



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
        part = load_json_maybe_jsonl(path)

        all_data.extend(part)

        print(
            f"[INFO] Loaded {len(part)} samples "
            f"from shard {os.path.basename(path)}"
        )

    print(
        f"[INFO] Total examples loaded: {len(all_data)}"
    )

    return all_data


def to_array_image(x: Any) -> List[str]:
    if x is None:
        return []

    return (
        x
        if isinstance(x, list)
        else [str(x)]
    )


def concat_segments(
    xs: Iterable[str],
    sep: str = "\n",
) -> str:

    return sep.join(
        x for x in xs if x
    )


_THOUGHT_RE = re.compile(
    r"<thought>(.*?)</thought>",
    re.DOTALL | re.IGNORECASE,
)

_DECISION_RE = re.compile(
    r"<decision>(.*?)</decision>",
    re.DOTALL | re.IGNORECASE,
)


def extract_thought_text(step: str) -> str:

    if not step:
        return ""

    match = _THOUGHT_RE.search(step)

    text = (
        match.group(1).strip()
        if match
        else step
    )

    return _DECISION_RE.sub(
        "",
        text,
    ).strip()



def load_image_map_from_txt(
    txt_path: str,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    if not os.path.exists(txt_path):
        print(
            "[WARN] Image list was not found."
        )

        return mapping

    with open(
        txt_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            path = line.strip()

            if not path:
                continue

            filename = os.path.basename(path)

            mapping.setdefault(
                filename,
                path,
            )

    print(
        f"[INFO] Loaded {len(mapping)} image paths"
    )

    return mapping


def resolve_image_field(
    img_field: Any,
    image_map: Dict[str, str],
) -> List[str]:

    if img_field is None:
        return []

    images = (
        img_field
        if isinstance(img_field, list)
        else [img_field]
    )

    resolved: List[str] = []

    for image in images:

        filename = os.path.basename(
            str(image)
        )

        if filename not in image_map:
            return []

        resolved.append(
            image_map[filename]
        )

    return resolved


def image_id_prefix_from_example(
    example: Dict[str, Any],
) -> str:

    image = (
        example.get("image")
        or example.get("images")
    )

    if isinstance(image, list) and image:

        filename = os.path.basename(
            str(image[0])
        )

    elif isinstance(image, str):

        filename = os.path.basename(
            image
        )

    else:
        return ""

    return os.path.splitext(
        filename
    )[0]

def build_reasoning_prompt(
    history: List[str],
) -> str:

    header = (
        "<image>\n"
        "You are a highly experienced radiologist.\n"
    )

    if not history:

        return (
            header
            + "There are no previous reasoning steps yet.\n\n"
            + "Generate ONLY the FIRST reasoning step based on the image.\n"
        )

    return (
        header
        + "Here are the previous reasoning steps:\n\n"
        + concat_segments(history)
        + "\nGenerate ONLY the NEXT reasoning step based on the image "
        + "and the previous steps above.\n"
    )


def expand_example(
    example: Dict[str, Any],
    image_map: Dict[str, str],
) -> List[Dict[str, Any]]:

    image = (
        example.get("image")
        or example.get("images")
    )

    images = resolve_image_field(
        image,
        image_map,
    )

    if not images:
        return []

    image_prefix = image_id_prefix_from_example(
        example
    )

    steps_full: List[str] = []

    if isinstance(
        example.get("steps"),
        list,
    ):

        for step in sorted(
            example["steps"],
            key=lambda x: x.get("count", 0),
        ):

            title = (
                step.get("title")
                or ""
            ).strip()

            content = (
                step.get("content")
                or ""
            ).strip()

            decision = (
                step.get("decision")
                or "continue"
            ).strip()

            body = (
                f"{title}: {content}"
                if title and content
                else (content or title)
            )

            steps_full.append(
                f"<thought>{body}</thought>"
                f"<decision>{decision}</decision>"
            )


    elif isinstance(
        example.get("conversations"),
        list,
    ):

        for turn in example["conversations"]:

            if turn.get("from") == "gpt":

                steps_full.append(
                    turn.get(
                        "value",
                        "",
                    )
                )


    if not steps_full:
        return []


    output: List[Dict[str, Any]] = []

    history: List[str] = []

    for i, step in enumerate(
        steps_full,
        start=1,
    ):

        output.append(
            {
                "id": f"{image_prefix}_v{i}",

                "images": images,

                "messages": [
                    {
                        "role": "user",
                        "content": build_reasoning_prompt(
                            history
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": step,
                    },
                ],
            }
        )

        plain = extract_thought_text(
            step
        )

        if plain:

            history.append(
                f"Step {i}: {plain}"
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

    parser.add_argument(
        "--all_images_path",
        default="../mimic_cxr_all_images.txt",
    )

    args = parser.parse_args()


    data = load_all_json_from_shard_input(
        args.shard_input
    )

    image_map = load_image_map_from_txt(
        args.all_images_path
    )


    all_output: List[Dict[str, Any]] = []

    for example in data:

        all_output.extend(
            expand_example(
                example,
                image_map,
            )
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
            all_output,
            f,
            ensure_ascii=False,
            indent=2,
        )


    print(
        f"[DONE] Wrote {len(all_output)} samples."
    )


if __name__ == "__main__":
    main()