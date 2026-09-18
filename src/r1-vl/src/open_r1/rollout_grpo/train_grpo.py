#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch

import json
import urllib.request
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

from datasets import load_dataset
from PIL import Image, ImageFile
from transformers import AutoProcessor

from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser

# NOTE:
# - GRPOTrainerRollout: Phase A reasoning rollout (multi-step)
# - GRPOTrainerSummary: Phase B summary/report rollout (single-step caption)
from trainer_grpo import GRPOTrainerRollout, GRPOTrainerSummary

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =============================================================================
# Debug helpers (env-controlled, rank0-only by default)
#   TRAIN_DEBUG=1                 -> basic prints
#   TRAIN_DEBUG_VERBOSE=1         -> includes snippets
#   TRAIN_DEBUG_RANK0_ONLY=1      -> only print on rank0 (default)
# =============================================================================

def _env_flag(name: str, default: str = "0") -> bool:
    v = str(os.getenv(name, default)).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}

def _get_rank() -> int:
    for k in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        if k in os.environ:
            try:
                return int(os.environ[k])
            except Exception:
                pass
    return 0

def _short(s: str, n: int = 220) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + " ..."

_TDEBUG = _env_flag("TRAIN_DEBUG", "0")
_TVERBOSE = _env_flag("TRAIN_DEBUG_VERBOSE", "0")
_TRANK0 = _env_flag("TRAIN_DEBUG_RANK0_ONLY", "1")

def _tprint(msg: str):
    if not _TDEBUG:
        return
    if _TRANK0 and _get_rank() != 0:
        return
    print(f"[TrainGRPO-DEBUG] {msg}", flush=True)

# =============================================================================
# Dataset helpers
# =============================================================================

def to_str_path(p):
    if isinstance(p, (list, tuple)):
        return p[0] if len(p) > 0 else ""
    return p or ""

def probe_image_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size

def _gt_dict_from_example(example: dict) -> dict:
    """
    Prefer clean gt dict; robust to multiple naming conventions.
    """
    if isinstance(example.get("gt_labels"), dict) and example["gt_labels"]:
        return example["gt_labels"]

    gt = (
        example.get("chestxpert_caption_labels")
        or example.get("chestxpert_caption_label")
        or example.get("chexbert_caption_labels")
        or example.get("chexbert_caption_label")
        or example.get("chexbert_finding_labels")
        or {}
    )
    return gt if isinstance(gt, dict) else {}

def _format_step_history(steps: Any) -> List[str]:
    """
    Convert `steps` (list of dicts or strings) to:
      ["Step 1: title: content", "Step 2: title: content", ...]
    """
    out: List[str] = []
    if not isinstance(steps, list) or not steps:
        return out

    for i, s in enumerate(steps, start=1):
        if isinstance(s, str):
            txt = s.strip()
            if txt:
                out.append(f"Step {i}: {txt}")
            continue

        if isinstance(s, dict):
            title = str(s.get("title", "") or "").strip()
            content = str(s.get("content", "") or "").strip()
            # fallback: some formats may store text as "text"
            if not content:
                content = str(s.get("text", "") or "").strip()
            if title and content:
                out.append(f"Step {i}: {title}: {content}")
            elif content:
                out.append(f"Step {i}: {content}")
            elif title:
                out.append(f"Step {i}: {title}")
            continue

        # fallback
        txt = str(s or "").strip()
        if txt:
            out.append(f"Step {i}: {txt}")

    return out

def _build_summary_prompt(stepwise_history: List[str]) -> str:
    """
    Aligned with your builder:
      - Always includes <image> + radiologist role.
      - If there is history: list Step i lines, then ask to write report.
      - If no history: ask to write report based on image alone.
    """
    header = "<image>\nYou are a highly experienced radiologist.\n"

    if stepwise_history:
        reasoning_block = (
            "\nHere are the previous step-by-step clinical reasoning steps for this image:\n\n"
            + "\n".join(stepwise_history)
            + "\n\n"
        )
        instruction = (
            "Based on the above step-by-step reasoning and the image, "
            "write a clear, clinically appropriate radiology report.\n"
        )
    else:
        reasoning_block = "\n"
        instruction = (
            "Based on the provided image, write a clear, clinically appropriate radiology report.\n"
        )

    return header + reasoning_block + instruction

def _build_reasoning_firststep_prompt() -> str:
    return (
        "<image>\n"
        "You are a highly experienced radiologist.\n"
        "There are no previous reasoning steps yet.\n\n"
        "Generate ONLY the FIRST reasoning step based on the image.\n"
    )

def make_prompt_record_reasoning(example: dict) -> dict:
    """
    Phase A dataset record:
      expects: image (ABS PATH) or image_path (ABS PATH), gt labels.
    """
    path = to_str_path(example.get("image_path", example.get("image", "")))
    gt = _gt_dict_from_example(example)

    user_text = _build_reasoning_firststep_prompt()

    rec = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ],
        "image_path": path,
        "gt_labels": gt,
        "sample_id": example.get("image", example.get("sample_id", None)),
    }

    if _TDEBUG and _TVERBOSE:
        _tprint(f"make_prompt_record_reasoning: image_path={path} gt_keys={len(gt.keys())} prompt_snippet={_short(user_text, 200)}")
    elif _TDEBUG:
        _tprint(f"make_prompt_record_reasoning: image_path={path} gt_keys={len(gt.keys())}")

    return rec

def make_prompt_record_summary(example: dict) -> dict:
    """
    Phase B dataset record (summary):
      expects:
        - image (ABS PATH) or image_path (ABS PATH)
        - steps: list[dict] (infer from GRPO-reasoning)
        - gt labels: chexbert_caption_labels (or same resolver)
    """
    path = to_str_path(example.get("image_path", example.get("image", "")))
    gt = _gt_dict_from_example(example)

    step_lines = _format_step_history(example.get("steps", []))
    user_text = _build_summary_prompt(step_lines)

    rec = {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ],
        "image_path": path,
        "gt_labels": gt,
        "sample_id": example.get("image", example.get("sample_id", None)),
    }

    if _TDEBUG and _TVERBOSE:
        _tprint(
            f"make_prompt_record_summary: image_path={path} steps={len(step_lines)} gt_keys={len(gt.keys())} "
            f"prompt_snippet={_short(user_text, 200)}"
        )
    elif _TDEBUG:
        _tprint(f"make_prompt_record_summary: image_path={path} steps={len(step_lines)} gt_keys={len(gt.keys())}")

    return rec

# =============================================================================
# Reward client (HTTP)
# =============================================================================

def _reward_call_server(
    trajectory_steps: List[List[str]],
    gt_labels: List[dict],
    sample_ids: Optional[List[Any]] = None,
    reward_mode: Optional[str] = None,
) -> dict:
    host = os.getenv("REWARD_HOST", "127.0.0.1")
    port = int(os.getenv("REWARD_PORT", "18080"))
    url = f"http://{host}:{port}/reward"

    payload_obj = {"trajectory_steps": trajectory_steps, "gt_labels": gt_labels, "sample_ids": sample_ids}
    if reward_mode:
        payload_obj["mode"] = reward_mode
    payload = json.dumps(payload_obj).encode("utf-8")

    if _TDEBUG:
        _tprint(
            f"_reward_call_server: url={url} mode={reward_mode} n={len(trajectory_steps)} "
            f"steps0={(len(trajectory_steps[0]) if trajectory_steps and isinstance(trajectory_steps[0], list) else 'NA')} "
            f"gt0_keys={(len(gt_labels[0].keys()) if gt_labels and isinstance(gt_labels[0], dict) else 'NA')} "
            f"bytes={len(payload)}"
        )
        if _TVERBOSE and trajectory_steps:
            first_step = trajectory_steps[0][0] if trajectory_steps[0] else ""
            _tprint(f"_reward_call_server: first_step_snippet={_short(first_step, 220)}")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    t1 = time.time()

    if _TDEBUG:
        stats = data.get("stats", {}) if isinstance(data, dict) else {}
        _tprint(
            f"_reward_call_server: done dt={t1 - t0:.3f}s mode={data.get('mode', None)} "
            f"reward_mean={stats.get('reward_mean', None)} n={stats.get('n', None)}"
        )

    return data

def chexbert_reward_remote(
    prompts,
    completions,
    gt_labels: List[dict],
    trajectory_steps=None,
    _reward_stats: Optional[dict] = None,
    sample_id: Optional[List[Any]] = None,
    sample_ids: Optional[List[Any]] = None,
    **kwargs,
):
    """
    One reward func for BOTH Phase A & Phase B.
    Decide mode from env REWARD_MODE (or overridden by --reward_mode in this script).
      - Mode "A" -> rollout reasoning reward (expects trainer sends trajectory_steps)
      - Mode "B" -> summary/caption reward (trainer likely sends completions; we wrap as [[caption]])
    """
    if trajectory_steps is None:
        trajectory_steps = [[c] if isinstance(c, str) else [str(c)] for c in completions]
        if _TDEBUG:
            _tprint(f"chexbert_reward_remote: trajectory_steps missing -> built from completions, n={len(trajectory_steps)}")

    sid = sample_ids if isinstance(sample_ids, list) else (sample_id if isinstance(sample_id, list) else None)
    reward_mode = os.getenv("REWARD_MODE", "A").strip() or "A"

    if _TDEBUG:
        _tprint(
            f"chexbert_reward_remote: mode={reward_mode} n={len(trajectory_steps)} "
            f"gt_labels_n={len(gt_labels) if isinstance(gt_labels, list) else 'NA'} "
            f"sample_ids_n={(len(sid) if isinstance(sid, list) else 'NA')}"
        )

    out = _reward_call_server(
        trajectory_steps=trajectory_steps,
        gt_labels=gt_labels,
        sample_ids=sid,
        reward_mode=reward_mode,
    )
    rewards = out.get("rewards", [])
    stats = out.get("stats", {})

    print("\n" + "=" * 80, flush=True)
    print("[TRAIN] Reward server returned:", flush=True)
    print(f"  mode={out.get('mode', None)} rewards_n={len(rewards)}", flush=True)
    print(f"  rewards (first 16)={rewards[:16]}", flush=True)
    if isinstance(stats, dict):
        print(f"  stats={stats}", flush=True)
    print("=" * 80 + "\n", flush=True)

    if _reward_stats is not None and isinstance(stats, dict):
        for k, v in stats.items():
            _reward_stats[k] = v

    return rewards

# =============================================================================
# PEFT helpers
# =============================================================================

def _parse_adapters_csv(s: Optional[str]) -> Optional[list[str]]:
    if not s:
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return parts or None

def _read_sft_adapter_config(adapter_dir: str) -> dict:
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_grpo_lora_config_from_sft(adapter_cfg: dict):
    from peft import LoraConfig
    target_modules = adapter_cfg.get("target_modules", None)
    if not target_modules:
        raise ValueError("SFT adapter_config.json missing `target_modules`")
    r = int(adapter_cfg.get("r", 16))
    lora_alpha = int(adapter_cfg.get("lora_alpha", 32))
    lora_dropout = float(adapter_cfg.get("lora_dropout", 0.0))
    bias = adapter_cfg.get("bias", "none")
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

def _safe_set_adapter(model, adapters: list[str]):
    if not adapters:
        return
    if hasattr(model, "set_adapter"):
        try:
            model.set_adapter(adapters)
            return
        except Exception:
            model.set_adapter(adapters[0])
            return
    raise ValueError("Model does not support set_adapter().")

def _safe_load_adapter(model, adapter_dir: str, adapter_name: str, is_trainable: bool = False):
    if hasattr(model, "load_adapter"):
        try:
            model.load_adapter(adapter_dir, adapter_name=adapter_name, is_trainable=is_trainable)
            return None
        except TypeError:
            model.load_adapter(adapter_dir, adapter_name=adapter_name)
            return None
    raise RuntimeError("Model does not support load_adapter(). Upgrade PEFT/Transformers.")

# =============================================================================
# args
# =============================================================================

@dataclass
class GRPOScriptArguments(ScriptArguments):
    dataset_name: str = field(default=None)
    min_pixels: Optional[int] = field(default=3_136)

    target_size: Optional[int] = field(default=1_024)
    pad_color: Optional[int] = field(default=0)

    # Phase A only
    max_rollout_steps: int = field(default=6)

    sft_adapter_path: Optional[str] = field(default=None)
    sft_adapter_name: str = field(default="sft")
    grpo_adapter_name: str = field(default="grpo")

    policy_adapters: Optional[str] = field(default=None)
    ref_adapters: Optional[str] = field(default=None)

    # Which training: "reasoning" (Phase A) or "summary" (Phase B)
    task: str = field(default="reasoning")

    # Optional: override REWARD_MODE for this script ("A" or "B")
    reward_mode: Optional[str] = field(default=None)

def main(script_args, training_args, model_args):
    os.environ.setdefault("GRPO_IMG_SIZE", str(int(script_args.target_size or 1024)))
    os.environ.setdefault("GRPO_PAD_COLOR", str(int(script_args.pad_color or 0)))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

    # task -> reward_mode default
    task = (script_args.task or "reasoning").strip().lower()
    if script_args.reward_mode:
        os.environ["REWARD_MODE"] = str(script_args.reward_mode).strip()
    else:
        os.environ["REWARD_MODE"] = ("B" if task == "summary" else "A")

    if _TDEBUG:
        _tprint(
            f"main: rank={_get_rank()} task={task} dataset={script_args.dataset_name} "
            f"min_pixels={script_args.min_pixels} target_size={script_args.target_size} pad_color={script_args.pad_color} "
            f"max_rollout_steps={script_args.max_rollout_steps}"
        )
        _tprint(
            f"main: model_name_or_path={model_args.model_name_or_path} attn_impl={model_args.attn_implementation} "
            f"output_dir={training_args.output_dir}"
        )
        _tprint(
            f"main: adapters: sft_adapter_path={script_args.sft_adapter_path} "
            f"policy_adapters_arg={script_args.policy_adapters} ref_adapters_arg={script_args.ref_adapters} "
            f"REWARD_MODE={os.getenv('REWARD_MODE', 'A')}"
        )

    t_load0 = time.time()
    raw = load_dataset("json", data_files={"train": script_args.dataset_name})
    if task == "summary":
        ds = raw["train"].map(make_prompt_record_summary, batched=False)
    else:
        ds = raw["train"].map(make_prompt_record_reasoning, batched=False)
    t_load1 = time.time()

    if _TDEBUG:
        _tprint(f"dataset: loaded+mapped in {t_load1 - t_load0:.3f}s n={len(ds)} columns={ds.column_names}")

    # fast probe: assumes ABS paths already
    def fast_probe(ex):
        p = ex.get("image_path") or ""
        try:
            if (not p) or (not os.path.isfile(p)):
                ex["__keep__"] = False
                return ex
            with Image.open(p) as im:
                im.verify()
            w, h = probe_image_size(p)
            area = int(w) * int(h)
            ex["__keep__"] = (area >= (script_args.min_pixels or 0))
        except Exception:
            ex["__keep__"] = False
        return ex

    t_probe0 = time.time()
    ds = ds.map(fast_probe, num_proc=1)
    ds = ds.filter(lambda ex: bool(ex["__keep__"]))
    if "__keep__" in ds.column_names:
        ds = ds.remove_columns(["__keep__"])
    t_probe1 = time.time()

    if _TDEBUG:
        _tprint(f"dataset: fast_probe+filter in {t_probe1 - t_probe0:.3f}s kept_n={len(ds)}")

    t_proc0 = time.time()
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    t_proc1 = time.time()

    if _TDEBUG:
        tok = getattr(processor, "tokenizer", None)
        _tprint(
            f"processor: loaded in {t_proc1 - t_proc0:.3f}s "
            f"has_tokenizer={tok is not None} pad_id={(getattr(tok, 'pad_token_id', None) if tok else None)}"
        )

    if not script_args.sft_adapter_path:
        raise ValueError("Requires --sft_adapter_path (LoRA SFT)")

    t_cfg0 = time.time()
    sft_cfg = _read_sft_adapter_config(script_args.sft_adapter_path)
    grpo_peft_config = _build_grpo_lora_config_from_sft(sft_cfg)
    t_cfg1 = time.time()

    if _TDEBUG:
        _tprint(
            f"adapter_config: read+build lora in {t_cfg1 - t_cfg0:.3f}s "
            f"target_modules={len(getattr(grpo_peft_config, 'target_modules', []) or [])} "
            f"r={getattr(grpo_peft_config, 'r', None)} alpha={getattr(grpo_peft_config, 'lora_alpha', None)}"
        )

    if _TDEBUG:
        _tprint("trainer: initializing ...")

    t_tr0 = time.time()

    if task == "summary":
        # ✅ USE GRPOTrainerSummary (your final class) for Phase B
        trainer = GRPOTrainerSummary(
            model=model_args.model_name_or_path,
            reward_funcs=[chexbert_reward_remote],
            args=training_args,
            train_dataset=ds,
            eval_dataset=None,
            peft_config=grpo_peft_config,
            attn_implementation=model_args.attn_implementation,
            max_pixels=None,
            min_pixels=script_args.min_pixels,
            processing_class=processor,
        )
    else:
        trainer = GRPOTrainerRollout(
            model=model_args.model_name_or_path,
            reward_funcs=[chexbert_reward_remote],
            args=training_args,
            train_dataset=ds,
            eval_dataset=None,
            peft_config=grpo_peft_config,
            attn_implementation=model_args.attn_implementation,
            max_pixels=None,
            min_pixels=script_args.min_pixels,
            processing_class=processor,
            max_rollout_steps=int(script_args.max_rollout_steps),
        )

    t_tr1 = time.time()

    if _TDEBUG:
        _tprint(f"trainer: init done in {t_tr1 - t_tr0:.3f}s")

    # Find GRPO trainable adapter internal name
    grpo_internal_name = "default"
    try:
        if hasattr(trainer.model, "active_adapter"):
            aa = getattr(trainer.model, "active_adapter")
            if isinstance(aa, str) and aa:
                grpo_internal_name = aa
        if hasattr(trainer.model, "active_adapters"):
            aa2 = getattr(trainer.model, "active_adapters")
            if isinstance(aa2, (list, tuple)) and len(aa2) > 0 and isinstance(aa2[0], str):
                grpo_internal_name = aa2[0]
    except Exception:
        pass

    if _TDEBUG:
        _tprint(f"adapters: detected grpo_internal_name={grpo_internal_name}")

    # Load SFT adapter (frozen)
    t_la0 = time.time()
    _safe_load_adapter(
        trainer.model,
        adapter_dir=script_args.sft_adapter_path,
        adapter_name=script_args.sft_adapter_name,
        is_trainable=False,
    )
    t_la1 = time.time()

    if _TDEBUG:
        _tprint(f"adapters: loaded SFT adapter in {t_la1 - t_la0:.3f}s name={script_args.sft_adapter_name}")

    # Configure policy/ref adapters for KL
    policy_adapters = _parse_adapters_csv(script_args.policy_adapters) or [
        script_args.sft_adapter_name,
        grpo_internal_name,
    ]
    ref_adapters = _parse_adapters_csv(script_args.ref_adapters) or [script_args.sft_adapter_name]

    t_sa0 = time.time()
    _safe_set_adapter(trainer.model, policy_adapters)
    t_sa1 = time.time()

    # Let trainer use these in compute_loss
    trainer._ref_adapters = ref_adapters
    trainer._policy_adapters = policy_adapters

    print("[GRPO] adapters configured:", flush=True)
    print("  - task:", task, flush=True)
    print("  - REWARD_MODE:", os.getenv("REWARD_MODE", "A"), flush=True)
    print("  - SFT adapter:", script_args.sft_adapter_name, "from", script_args.sft_adapter_path, flush=True)
    print("  - GRPO trainable adapter (internal):", grpo_internal_name, flush=True)
    print("  - policy_adapters:", policy_adapters, flush=True)
    print("  - ref_adapters:", ref_adapters, flush=True)

    if _TDEBUG:
        _tprint(f"adapters: set policy adapters in {t_sa1 - t_sa0:.3f}s policy={policy_adapters} ref={ref_adapters}")

    if _TDEBUG:
        _tprint("trainer.train(): start")
    t_train0 = time.time()
    trainer.train()
    t_train1 = time.time()
    if _TDEBUG:
        _tprint(f"trainer.train(): done dt={t_train1 - t_train0:.3f}s")

    if _TDEBUG:
        _tprint(f"trainer.save_model: output_dir={training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    if _TDEBUG:
        _tprint("trainer.save_model: done")

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    if _TDEBUG:
        _tprint(
            f"__main__: rank={_get_rank()} TRAIN_DEBUG={_TDEBUG} TRAIN_DEBUG_VERBOSE={_TVERBOSE} "
            f"RANK0_ONLY={_TRANK0}"
        )

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
