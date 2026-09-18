#!/usr/bin/env python3

import json
import os
import argparse
from typing import Dict, List

from tqdm import tqdm
from radgraph import RadGraph


RADGRAPH_MODEL = RadGraph(model_type="radgraph")


def _build_modifiers_map(
    entities: Dict[str, dict]
) -> Dict[str, List[str]]:
    modifiers_for: Dict[str, List[str]] = {}

    for eid, ent in entities.items():
        for rel in ent.get("relations", []):
            if len(rel) != 2:
                continue

            rel_type, target_id = rel

            if rel_type == "modify" and target_id in entities:
                modifiers_for.setdefault(target_id, []).append(eid)

    return modifiers_for


def _is_pure_modifier(ent: dict) -> bool:
    has_modify = False
    has_strong_rel = False

    for rel_type, _ in ent.get("relations", []):
        if rel_type == "modify":
            has_modify = True

        if rel_type in ("located_at", "suggestive_of"):
            has_strong_rel = True

    return has_modify and not has_strong_rel


def _build_phrase(
    eid: str,
    entities: Dict[str, dict],
    modifiers_for: Dict[str, List[str]],
) -> str:
    ent_ids = [eid] + modifiers_for.get(eid, [])
    items = []

    for entity_id in ent_ids:
        ent = entities[entity_id]

        tokens = (ent.get("tokens") or "").strip()
        if not tokens:
            continue

        label = ent.get("label", "")

        if label.startswith("ANAT") and any(
            ch.isdigit() for ch in tokens
        ):
            continue

        start = ent.get("start_ix", 0)
        items.append((start, tokens))

    if not items:
        return ""

    items.sort(key=lambda x: x[0])

    phrase = " ".join(
        token for _, token in items
    ).strip()

    phrase = phrase.replace(" . ", ". ").replace(" ,", ",")

    return phrase


def radgraph_to_facts_single(ann: Dict) -> List[str]:
    entities = ann.get("entities", {}) or {}

    if not entities:
        return []

    modifiers_for = _build_modifiers_map(entities)
    facts: List[str] = []

    for eid, ent in entities.items():
        label = ent.get("label", "")

        if not label.startswith("OBS"):
            continue

        if _is_pure_modifier(ent):
            continue

        obs_phrase = _build_phrase(
            eid,
            entities,
            modifiers_for,
        )

        if not obs_phrase:
            continue

        prefix = ""

        if label.endswith("DA"):
            prefix = "No "
        elif label.endswith("U"):
            prefix = "Possible "

        anat_ids = [
            target_id
            for rel_type, target_id in ent.get("relations", [])
            if rel_type == "located_at"
            and target_id in entities
        ]

        anat_phrases = []

        for anatomy_id in anat_ids:
            phrase = _build_phrase(
                anatomy_id,
                entities,
                modifiers_for,
            )

            if phrase:
                anat_phrases.append(phrase)

        sugg_ids = [
            target_id
            for rel_type, target_id in ent.get("relations", [])
            if rel_type == "suggestive_of"
            and target_id in entities
        ]

        sugg_phrases = []

        for suggestion_id in sugg_ids:
            phrase = _build_phrase(
                suggestion_id,
                entities,
                modifiers_for,
            )

            if phrase:
                sugg_phrases.append(phrase)

        core = (prefix + obs_phrase).strip()

        if anat_phrases:
            core = f"{core} at {', '.join(anat_phrases)}"

        if sugg_phrases:
            core = (
                f"{core}, suggestive of "
                f"{', '.join(sugg_phrases)}"
            )

        if len(core.split()) < 2:
            continue

        if not core.endswith("."):
            core += "."

        core = core[0].upper() + core[1:]

        if core not in facts:
            facts.append(core)

    return facts


def caption_to_findings(
    caption: str,
    radgraph_model: RadGraph = RADGRAPH_MODEL,
) -> List[str]:
    annotations = radgraph_model([caption])

    if not annotations:
        return []

    ann = list(annotations.values())[0]

    return radgraph_to_facts_single(ann)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def atomic_write_json(path: str, data):
    output_dir = os.path.dirname(path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tmp_path = path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(tmp_path, path)


def process_file(
    input_path: str,
    output_path: str,
    save_every: int = 100,
):
    raw = load_json(input_path)

    samples = []

    if isinstance(raw, dict):
        for split_name, items in raw.items():
            if not isinstance(items, list):
                continue

            for item in items:
                if (
                    "split" not in item
                    or not item["split"]
                ):
                    item["split"] = split_name

                samples.append(item)

    elif isinstance(raw, list):
        samples = raw

    else:
        raise ValueError(
            "Unsupported JSON structure: expected "
            "a dict or list at the top level."
        )

    if os.path.isfile(output_path):
        try:
            already = load_json(output_path)

            done = {
                item["image_path"]
                for item in already
                if "image_path" in item
            }

            out_list = already

            print(
                f"Resuming: loaded {len(out_list)} "
                "items from output."
            )

        except Exception:
            out_list = []
            done = set()

            print(
                "Failed to load existing output. "
                "Starting from scratch."
            )

    else:
        out_list = []
        done = set()

    counter = 0

    for item in tqdm(
        samples,
        desc="Extracting findings",
        ncols=100,
    ):
        if not isinstance(item, dict):
            continue

        img_list = item.get("image_path") or []
        image_path = img_list[0] if img_list else ""

        if not image_path:
            continue

        if image_path in done:
            continue

        report = (item.get("report") or "").strip()
        split = item.get("split", "")

        if not report:
            findings_str = ""
        else:
            findings = caption_to_findings(report)

            findings_str = " ".join(
                finding.strip()
                for finding in findings
                if finding.strip()
            )

        out_list.append(
            {
                "image_path": image_path,
                "report": report,
                "finding": findings_str,
                "split": split,
            }
        )

        done.add(image_path)
        counter += 1

        if counter % save_every == 0:
            atomic_write_json(
                output_path,
                out_list,
            )

            print(f"Saved {counter} new items...")

    atomic_write_json(
        output_path,
        out_list,
    )

    print(f"Finished. Total saved: {len(out_list)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="",
    )
    parser.add_argument(
        "--output",
        default="",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    process_file(
        args.input,
        args.output,
        args.save_every,
    )


if __name__ == "__main__":
    main()