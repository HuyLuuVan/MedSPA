#!/usr/bin/env python3

from __future__ import annotations
import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import tempfile
import time
from typing import List, Literal, Dict, Any, Set, Optional
from pydantic import BaseModel, field_validator
from tqdm import tqdm
from PIL import Image, ImageOps
from openai import OpenAI

DEFAULT_IMAGES_INDEX        = "../mimic_cxr_all_images.txt"
DEFAULT_MIMIC_FINDINGS      = "../dataset/mimic/finding/mimic_gt_findings.json"
DEFAULT_MIMIC_ANNOT_JSON    = "../dataset/mimic/mimic_annotation_.json"
DEFAULT_OUTPUT_JSON         = "../src/r1-vl/data/v4/qwen3vl_reasoning_mimic.json"

_JSON_GRAB = re.compile(r"\{.*?\}", re.DOTALL)


class Step(BaseModel):
    count: int
    title: str
    content: str
    decision: Literal["continue", "summary"]


class ReasoningChain(BaseModel):
    steps: List[Step]

    @field_validator("steps")
    @classmethod
    def _validate_counts(cls, steps: List[Step]):
        for i, s in enumerate(steps, start=1):
            if s.count != i:
                raise ValueError("step.count must be 1..n without gaps")
        return steps


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, data: Any) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_all_images_map(txt_path: str) -> Dict[str, str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    mapping: Dict[str, str] = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if os.path.splitext(p)[1].lower() in exts:
                mapping.setdefault(os.path.basename(p), p)
    return mapping


def build_mimic_finding_map(mimic_findings_path: str) -> Dict[str, str]:
    data = load_json(mimic_findings_path)
    m: Dict[str, str] = {}
    for it in data:
        img_path = (it.get("image_path") or "").strip()
        finding = (it.get("finding") or "").strip()
        if not img_path or not finding:
            continue
        base = os.path.basename(img_path)
        m[base] = finding
    return m


def load_only_keys(txt_path: str) -> Set[str]:
    keys: Set[str] = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace("\\", "/")
            if s.startswith("files/"):
                s = s[len("files/"):]
            keys.add(s)
    return keys


def _extract_json_object(text: str) -> str:
    if not text:
        return ""
    m = _JSON_GRAB.search(text)
    return m.group(0) if m else ""


def _encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _cache_dir(img_size: int, resize_mode: str) -> str:
    d = os.path.join(tempfile.gettempdir(), f"qwen_resize_cache_{resize_mode}_{img_size}")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_key(path: str, img_size: int, resize_mode: str) -> str:
    h = hashlib.sha1(f"{path}|{img_size}|{resize_mode}".encode("utf-8")).hexdigest()
    return os.path.join(_cache_dir(img_size, resize_mode), f"{h}.jpg")


def resize_xray(src_path: str, img_size: int = 1024, resize_mode: str = "fit") -> str:
    mode = resize_mode.lower()
    if mode not in {"fit", "pad"}:
        raise ValueError("resize_mode must be 'fit' or 'pad'")
    dst_path = _cache_key(src_path, img_size, mode)
    if os.path.isfile(dst_path):
        return dst_path

    with Image.open(src_path) as im:
        if im.mode != "L":
            im = im.convert("L")
        if mode == "fit":
            im_sq = ImageOps.fit(im, (img_size, img_size), method=Image.LANCZOS, centering=(0.5, 0.5))
        else:
            im_sq = ImageOps.pad(im, (img_size, img_size), color=0, method=Image.LANCZOS)

        im_sq.save(dst_path, format="JPEG", quality=92, subsampling=0, optimize=True)

    return dst_path


INTRO = """
You are a highly experienced radiologist that generates a clear, step-wise
interpretation of a medical image.

Your goal is to produce a concise, clinically grounded reasoning chain that:
- is consistent with the underlying radiographic truth,
- may internally use the provided findings and report ONLY as hidden guidance,
- and remains strictly image-grounded and medically consistent in its wording.
""".strip()

TRUSTED_FINDINGS_GUIDANCE = """
TRUSTED FINDINGS
- You will receive a list of TRUSTED FINDINGS (GROUND-TRUTH SEMANTICS) derived from an expert radiology report.
- These represent the most reliable judgment about what is present, absent, or uncertain.
- You MUST treat these findings as the primary source of truth about radiographic content.
- Do NOT invent new radiographic abnormalities that are not in the trusted findings
  or clearly visible in the image.
""".strip()

CERTAINTY_GUIDANCE = """
CERTAINTY AND EVIDENCE
- Trusted findings may implicitly encode certainty through words such as:
  - Present / definite: "is", "are", "clearly", "markedly", "definite", "enlarged", etc.
  - Absent: "no", "without", "absent", "not seen", "no evidence of".
  - Uncertain: "possible", "possibly", "may represent", "could be",
    "suggestive of", "suspicious for", "cannot exclude".

You MUST:
- Preserve the certainty level of each trusted finding:
  - Do NOT upgrade uncertain findings to definite statements.
  - Do NOT downgrade definite findings to "possible".
  - Do NOT turn clearly absent findings into possible or present ones.
- For every clinically important POSITIVE (present) or NEGATIVE (absent) statement,
  explicitly mention at least one visual cue or concise medical explanation that
  justifies that conclusion (e.g. "sharp costophrenic angles without blunting -> no pleural effusion").
- For every UNCERTAIN statement, explicitly mention:
  - partial or ambiguous visual evidence, and/or
  - a short medical explanation of why the image alone does not allow a definite conclusion.

Never speculate beyond what can be reasonably supported by the combination
of the image and the trusted findings.
""".strip()

CAPTION_GUIDANCE = """
- You will also receive the original radiology REPORT AS KNOWLEDGE SOURCE
- Treat the report solely as internal clinical context and hypothesis cues that assist you in interpreting what you observe.
- You may reuse its medical reasoning and detailed descriptions only when they are fully
  consistent with both the trusted findings and the image, and you must preserve the same
  level of diagnostic certainty.

You MUST NOT:
- override or contradict any trusted finding.
- upgrade an uncertain idea in the report to a definite statement if the
  trusted findings are uncertain or silent on that point.
- introduce new radiographic abnormalities that are neither in the trusted
  findings nor clearly visible in the image.
""".strip()

MEDICAL_TEMPLATE = """
CLINICAL REASONING TEMPLATE

1) First step: GLOBAL CONTEXT (MANDATORY)
   - The FIRST step title MUST be a short V-ing phrase such as:
       "Describing initial radiographic context" or
       "Reviewing overall radiographic setting".
   - In this first step, explicitly mention:
     * the image type and projection(s) (e.g. PA, AP, lateral),
     * presence or absence of obvious support devices or lines/tubes if visible,
     * If the trusted findings mention any clips, devices, lines or tubes,
     you MUST explicitly describe them and you MUST NOT claim that there are
     "no foreign bodies", "no devices", or "no lines/tubes".
   - Do NOT jump directly into a single abnormality; this step sets the scene.

2) Then explore MAJOR REGIONS in a natural, radiologist-like order
   (you do NOT have to follow the textual order of the trusted findings, as long as you remain consistent):
   - When you reach a region (e.g. bones, heart, mediastinum, left lung, right lung, pleura):
     * First describe the overall status and important NEGATIVE findings (what is normal / absent).
     * Then describe POSITIVE or UNCERTAIN abnormalities suggested by the trusted findings or image.
     * Normal areas are described first while abnormalities are described later.

3) Step structure and focus:
   - Each step should focus on ONE coherent idea or one anatomical region where a trusted finding applies.
   - Avoid mechanically scanning every possible region if it is not relevant to the trusted findings.
   - Make sure later steps do NOT contradict earlier steps.
   - Every step title MUST be a short V-ing phrase summarizing the purpose of that step
     (e.g., "Assessing global lung aeration", "Evaluating pleural spaces", "Reviewing osseous structures").

4) Final step: INTEGRATED IMPRESSION (MANDATORY)
   - The FINAL step title MUST start with a V-ing phrase such as:
       "Summarizing overall radiographic impression".
   - The final step should:
     * synthesize the main positive, negative, and uncertain findings from previous steps,
     * read like a concise, integrated radiologic impression (2–4 sentences),
       not a repetition of one local region only,
     * still be fully image-grounded and consistent with the trusted findings.
   - The final step MUST use "decision": "summary".
""".strip()

OUTPUT_FORMAT = """
OUTPUT FORMAT (STRICT)
Return ONLY ONE JSON object with EXACTLY this structure (no extra keys, no markdown, no backticks):
{
  "steps": [
    {
      "title": a short V-ing phrase summarizing the purpose of this step,
      "content": objective, image-grounded reasoning for this step,
      "decision": "continue" | "summary"
    },
    ...
  ]
}

- Use 3 to {max_steps} steps in total.
- The FIRST step must be a global context step as described above.
- All intermediate steps (if any) should normally use "decision": "continue".
- The FINAL step MUST:
  * have a title that starts with "Summarizing" (or a similar V-ing phrase that clearly indicates synthesis),
  * act as a global integrated impression of the whole case,
  * use "decision": "summary" once the main radiographic truth has been adequately supported and synthesized.
""".strip()


def build_system_rules(max_steps: int) -> str:
    return "\n\n".join([
        INTRO,
        TRUSTED_FINDINGS_GUIDANCE,
        CAPTION_GUIDANCE,
        CERTAINTY_GUIDANCE,
        MEDICAL_TEMPLATE,
        OUTPUT_FORMAT.replace("{max_steps}", str(max_steps)),
    ])


CHAIN_USER_PROMPT = """
You are given a medical image, a set of trusted radiographic findings,
and the original radiology report.

Trusted radiographic findings (ground-truth from expert radiologists):
{trusted_findings}

Radiology report:
{report}

Your task is generating a reasoning chain with multiple steps that:
- supports and organizes the underlying radiographic truth,
- enriches explanations and visual detail when compatible with that truth,
- remains fully image-grounded in its wording.

Order and structure:
1) First step: GLOBAL CONTEXT (MANDATORY)
   - The FIRST step title MUST be a short V-ing phrase such as:
       "Describing initial radiographic context" or
       "Reviewing overall radiographic setting".
   - In this first step, explicitly mention:
     * the image type and projection(s) (e.g. PA, AP, lateral),
     * presence or absence of obvious support devices or lines/tubes if visible,
     * If the trusted findings mention any clips, devices, lines or tubes,
     you MUST explicitly describe them and you MUST NOT claim that there are
     "no foreign bodies", "no devices", or "no lines/tubes".
   - Do NOT jump directly into a single abnormality; this step sets the scene.

2) Then explore MAJOR REGIONS in a natural, radiologist-like order
   (you do NOT have to follow the textual order of the trusted findings, as long as you remain consistent):
   - When you reach a region (e.g. bones, heart, mediastinum, left lung, right lung, pleura):
     * First describe the overall status and important NEGATIVE findings (what is normal / absent).
     * Then describe POSITIVE or UNCERTAIN abnormalities suggested by the trusted findings or image.
     * Normal areas are described first while abnormalities are described later.

3) Step structure and focus:
   - Each step should focus on ONE coherent idea or one anatomical region where a trusted finding applies.
   - Avoid mechanically scanning every possible region if it is not relevant to the trusted findings.
   - Make sure later steps do NOT contradict earlier steps.
   - Every step title MUST be a short V-ing phrase summarizing the purpose of that step
     (e.g., "Assessing global lung aeration", "Evaluating pleural spaces", "Reviewing osseous structures").

4) Final step: INTEGRATED IMPRESSION (MANDATORY)
   - The FINAL step title MUST start with a V-ing phrase such as:
       "Summarizing overall radiographic impression".
   - The final step should:
     * synthesize the main positive, negative, and uncertain findings from previous steps,
     * read like a concise, integrated radiologic impression (2–4 sentences),
       not a repetition of one local region only,
     * still be fully image-grounded and consistent with the trusted findings.
   - The final step MUST use "decision": "summary".

Constraints:
- Preserve and respect the certainty of the trusted findings (present / absent / uncertain).
- Do NOT upgrade uncertain ideas to definite statements.
- Do NOT downgrade clear absence to possible/present.
- Never mention the words "caption", "reference caption", "report", "report text",
  "input text", "trusted findings", "ground truth", "description", or "instructions".
  Your output must read as if it is based solely on the image.
- Do NOT narrate your own plan (no "as planned", "I am now inspecting", "I will now", "next I will").
- Use between 3 and {max_steps} steps.

Return ONLY ONE JSON object with a single key "steps", which is a list of step objects:
each with "title", "content", and "decision".
""".strip()


def _format_trusted_findings_block(finding_str: str) -> str:
    s = (finding_str or "").strip()
    if not s:
        return "(none provided)"
    parts = [p.strip() for p in s.split(".") if p.strip()]
    if not parts:
        return s
    lines = ["- " + p for p in parts]
    return "\n".join(lines)


def _normalize_summary_steps(steps: List[Step]) -> List[Step]:
    if not steps:
        return steps

    summary_indices = [i for i, st in enumerate(steps) if st.decision == "summary"]

    if len(summary_indices) > 1:
        for i in summary_indices[:-1]:
            steps[i].decision = "continue"

    for i, st in enumerate(steps, start=1):
        st.count = i

    return steps


def _extract_text_from_chat_response(resp) -> str:
    try:
        msg = resp.choices[0].message
    except Exception:
        return ""
    content = msg.content
    if isinstance(content, str):
        return content.strip()

    try:
        texts = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text")
                if t:
                    texts.append(t)
            else:
                t = getattr(part, "text", None)
                if t:
                    texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return str(content)


def generate_reasoning_for_sample(
    client: OpenAI,
    model_name: str,
    image_path: str,
    report_text: str,
    trusted_findings: str,
    max_steps: int = 10,
    img_size: int = 1024,
    resize_mode: str = "fit",
    max_tokens: int = 1024,
) -> ReasoningChain:
    pre_path = resize_xray(image_path, img_size=img_size, resize_mode=resize_mode)
    b64_img = _encode_image_base64(pre_path)
    mime = mimetypes.guess_type(pre_path)[0] or "image/jpeg"

    rules = build_system_rules(max_steps)
    tf_block = _format_trusted_findings_block(trusted_findings)
    user_text = CHAIN_USER_PROMPT.format(
        trusted_findings=tf_block,
        report=report_text,
        max_steps=max_steps,
    )

    messages = [
        {"role": "system", "content": rules},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}},
            ],
        },
    ]

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
            )
            raw = _extract_text_from_chat_response(resp)

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                js = _extract_json_object(raw)
                if not js:
                    raise RuntimeError("Empty/unsafe response text (no JSON)")
                data = json.loads(js)

            steps_raw = data.get("steps", [])
            if not isinstance(steps_raw, list) or not steps_raw:
                raise RuntimeError("No 'steps' list in response JSON")

            steps: List[Step] = []
            for i, st in enumerate(steps_raw, start=1):
                if not isinstance(st, dict):
                    raise RuntimeError(f"Step {i} is not a JSON object")
                st = dict(st)
                st["count"] = i
                steps.append(Step(**st))

            steps = _normalize_summary_steps(steps)

            if steps[-1].decision != "summary":
                print(
                    f"[WARN] Last step decision is "
                    f"'{steps[-1].decision}', not 'summary'."
                )

            return ReasoningChain(steps=steps)

        except Exception as e:
            if attempt == 0:
                print(f"[WARN] Qwen3-VL call failed: {type(e).__name__}; retrying once...")
                time.sleep(3)
            else:
                raise


def main():
    qwen_base  = os.getenv("QWEN_API_BASE", "http://localhost:8000/v1")
    qwen_key   = os.getenv("QWEN_API_KEY", "EMPTY")
    qwen_model = os.getenv("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

    print("[INFO] Qwen client configured.")

    client = OpenAI(base_url=qwen_base, api_key=qwen_key)

    ap = argparse.ArgumentParser()
    ap.add_argument("--images_index", default=DEFAULT_IMAGES_INDEX)
    ap.add_argument("--mimic_findings_json", default=DEFAULT_MIMIC_FINDINGS)
    ap.add_argument("--mimic_annotation_json", default=DEFAULT_MIMIC_ANNOT_JSON)
    ap.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)

    ap.add_argument(
        "--only_list",
        default="",
        help=(
            "Path to a txt file containing image_key entries (one per line), "
            "e.g. 'p11/p116.../xxx.jpg'. If set, ONLY these images will be processed."
        ),
    )

    ap.add_argument(
        "--split",
        default="train",
        help="Which split in mimic_annotation_.json to use: train/val/test (default: train)",
    )
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of samples in this shard to process (0 = no limit).",
    )
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--img_size", type=int, default=1024)
    ap.add_argument("--resize_mode", choices=["fit", "pad"], default="fit")
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help=(
            "Max completion tokens per sample. Must satisfy "
            "prompt_tokens + max_tokens <= model context length."
        ),
    )

    ap.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards (for multi-server distillation)",
    )
    ap.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Current shard id (0-based)",
    )

    ap.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Reserved for future intra-shard parallelism (currently unused).",
    )

    args = ap.parse_args()

    only_keys: Optional[Set[str]] = None
    if args.only_list:
        if not os.path.isfile(args.only_list):
            raise FileNotFoundError("--only_list file not found.")
        only_keys = load_only_keys(args.only_list)
        print(f"[INFO] Loaded {len(only_keys)} keys from --only_list.")

    out_list: List[dict] = []
    done_images = set()
    if os.path.isfile(args.output_json):
        try:
            out_list = load_json(args.output_json)
            for e in out_list:
                img_key = e.get("image")
                if img_key:
                    done_images.add(img_key)
            print(f"[INFO] Loaded {len(out_list)} existing samples.")
        except Exception:
            out_list, done_images = [], set()

    img_map      = load_all_images_map(args.images_index)
    finding_map  = build_mimic_finding_map(args.mimic_findings_json)
    annot_all    = load_json(args.mimic_annotation_json)

    if args.split not in annot_all:
        raise ValueError(
            f"Split '{args.split}' was not found. "
            f"Available keys: {list(annot_all.keys())}"
        )

    annot_split = annot_all[args.split]
    print(f"[INFO] Split '{args.split}': {len(annot_split)} entries.")

    selected: List[dict] = []
    skipped_no_image = 0
    skipped_no_finding = 0
    skipped_no_report = 0
    skipped_not_in_only_list = 0

    for it in annot_split:
        image_paths = it.get("image_path") or []
        if not image_paths:
            continue

        rel_path = image_paths[0]
        base = os.path.basename(rel_path)

        if base not in img_map:
            skipped_no_image += 1
            continue
        if base not in finding_map:
            skipped_no_finding += 1
            continue

        report = (it.get("report") or "").strip()
        if not report:
            skipped_no_report += 1
            continue

        full_img_path = img_map[base]
        image_key = full_img_path.split("files/")[-1].replace("\\", "/")

        if only_keys is not None and image_key not in only_keys:
            skipped_not_in_only_list += 1
            continue

        selected.append(
            {
                "image": full_img_path,
                "image_key": image_key,
                "split": args.split,
                "caption": report,
                "finding": finding_map[base]
            }
        )

    print(f"[INFO] Selected candidates: {len(selected)}")
    print(f"[INFO] Skipped without local image: {skipped_no_image}")
    print(f"[INFO] Skipped without finding: {skipped_no_finding}")
    print(f"[INFO] Skipped without report: {skipped_no_report}")
    if only_keys is not None:
        print(f"[INFO] Skipped outside only_list: {skipped_not_in_only_list}")

    if args.num_shards > 1:
        shard_sel = []
        for i, s in enumerate(selected):
            if i % args.num_shards == args.shard_id:
                shard_sel.append(s)
        print(f"[INFO] Shard {args.shard_id}/{args.num_shards}: {len(shard_sel)} samples.")
        selected = shard_sel
    else:
        print("[INFO] Single-shard mode.")

    to_process = [s for s in selected if s["image_key"] not in done_images]
    if args.limit and args.limit > 0:
        slice_end = args.index + args.limit
        to_process = to_process[args.index:slice_end]
    else:
        to_process = to_process[args.index:]

    print(f"[INFO] Will process {len(to_process)} samples.")

    processed_new = 0

    for s in tqdm(to_process, desc="Generating reasoning", ncols=100):
        try:
            chain = generate_reasoning_for_sample(
                client=client,
                model_name=qwen_model,
                image_path=s["image"],
                report_text=s["caption"],
                trusted_findings=s["finding"],
                max_steps=args.max_steps,
                img_size=args.img_size,
                resize_mode=args.resize_mode,
                max_tokens=args.max_tokens,
            )

            if chain.steps and chain.steps[-1].decision == "summary":
                out_list.append({
                    "image": s["image_key"],
                    "split": s["split"],
                    "caption": s["caption"],
                    "trusted_finding": s["finding"],
                    "steps": [st.model_dump() for st in chain.steps],
                })
                processed_new += 1
                if args.save_every > 0 and processed_new % args.save_every == 0:
                    atomic_write_json(args.output_json, out_list)
                    print(f"[INFO] Saved {processed_new} new entries.")
            else:
                print(f"[WARN] Skipped without summary: {s['image_key']}")

        except Exception as e:
            print(f"[ERROR] {s['image_key']}: {type(e).__name__}")

    atomic_write_json(args.output_json, out_list)
    print(f"[DONE] {len(out_list)} total entries saved.")


if __name__ == "__main__":
    main()
