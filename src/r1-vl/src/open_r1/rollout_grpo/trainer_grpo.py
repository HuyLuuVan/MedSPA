# src/r1-vl/src/open_r1/rollout_grpo/trainer_grpo.py
# rollout_grpo/trainer_grpo.py
# Copyright 2025 The HuggingFace Team.
# Licensed under Apache License, Version 2.0

import os
import re
import textwrap
import copy
import contextlib
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version

from tqdm.auto import tqdm  # <<< ADDED (progress bar)

from transformers import (
    AriaForConditionalGeneration,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

try:
    from transformers.models.qwen2_vl import Qwen2VLConfig  # type: ignore
except Exception:
    Qwen2VLConfig = None

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# Ensure correct CUDA device per local rank
if torch.cuda.is_available():
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(lr)

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

# =============================================================================
# Debug helpers
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


def _short(s: str, n: int = 180) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + " ..."


def _image_key_from_input(ex: dict) -> str:
    try:
        if isinstance(ex.get("image_path", None), str) and ex.get("image_path"):
            return ex["image_path"]
        if isinstance(ex.get("image", None), str) and ex.get("image"):
            return ex["image"]
        if ex.get("image", None) is not None:
            return "<PIL.Image>"
    except Exception:
        pass
    return "<unknown_image>"


def _is_vision_lang_config(cfg) -> bool:
    if Qwen2VLConfig is not None and isinstance(cfg, Qwen2VLConfig):
        return True
    mt = (getattr(cfg, "model_type", "") or "").lower().replace("-", "_")
    if mt in {
        "qwen2_vl",
        "mllama",
        "phi4_multimodal",
        "git",
        "fuyu",
        "llava",
        "llava_next",
        "llava_onevision",
        "idefics2",
        "emu3",
    }:
        return True
    if hasattr(cfg, "vision_config") or hasattr(cfg, "image_embed_dim"):
        return True
    return False


def load_auto_for_policy(model_name_or_path: str, **kwargs):
    kwargs.setdefault("trust_remote_code", True)
    use_cache = kwargs.pop("use_cache", None)

    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=kwargs["trust_remote_code"])
    if _is_vision_lang_config(cfg):
        model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)

    if use_cache is not None:
        try:
            model.config.use_cache = bool(use_cache)
        except Exception:
            pass
    return model


# =========================
#   PEFT adapter helpers
# =========================


def _parse_adapters_env(env_key: str) -> Optional[list[str]]:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None


def _get_active_adapters(model) -> Optional[list[str]]:
    if hasattr(model, "active_adapters"):
        try:
            aa = getattr(model, "active_adapters")
            if isinstance(aa, (list, tuple)):
                return list(aa)
        except Exception:
            pass
    if hasattr(model, "active_adapter"):
        try:
            aa = getattr(model, "active_adapter")
            if isinstance(aa, str) and aa:
                return [aa]
        except Exception:
            pass
    return None


def _set_adapters(model, adapters: Optional[list[str]]):
    if adapters is None:
        return

    if hasattr(model, "set_adapter"):
        try:
            model.set_adapter(adapters)
            return
        except Exception:
            try:
                model.set_adapter(adapters[0])
                return
            except Exception:
                pass

    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "set_adapter"):
        try:
            base.set_adapter(adapters)
            return
        except Exception:
            try:
                base.set_adapter(adapters[0])
                return
            except Exception:
                pass

    raise ValueError(f"Model does not support set_adapter for adapters={adapters}")


@contextlib.contextmanager
def _temporary_adapters(model, adapters: Optional[list[str]]):
    prev = _get_active_adapters(model)
    if adapters is None:
        yield
        return

    _set_adapters(model, adapters)
    try:
        yield
    finally:
        if prev is not None:
            try:
                _set_adapters(model, prev)
            except Exception:
                pass


# =============================================================================
#  Base trainer (single-step GRPO)
# =============================================================================


class Qwen2VLGRPOTrainer(Trainer):
    """
    Base GRPO trainer (single-step completion GRPO).
    """

    @staticmethod
    def _to_str_path(p: Any) -> str:
        if isinstance(p, (list, tuple)):
            return p[0] if len(p) > 0 else ""
        return p or ""

    @staticmethod
    def _load_image_square(path: Any, target_size: int = 1024, pad_color: int = 0) -> Image.Image:
        path = Qwen2VLGRPOTrainer._to_str_path(path)
        if not path or (not os.path.isfile(path)):
            raise FileNotFoundError(f"Missing image file: {path}")

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            if max(w, h) <= 0:
                raise ValueError(f"Invalid image size for {path}: {(w,h)}")

            scale = float(target_size) / float(max(w, h))
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))

            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                resample = Image.LANCZOS

            im = im.resize((nw, nh), resample=resample)
            canvas = Image.new("RGB", (target_size, target_size), (pad_color, pad_color, pad_color))
            left = (target_size - nw) // 2
            top = (target_size - nh) // 2
            canvas.paste(im, (left, top))
            canvas.load()
            return canvas

    # -----------------------------
    # dtype helpers
    # -----------------------------
    def _policy_dtype(self) -> torch.dtype:
        try:
            return self.accelerator.unwrap_model(self.model).dtype
        except Exception:
            try:
                return self.model.dtype
            except Exception:
                return torch.bfloat16

    def _autocast_dtype(self) -> torch.dtype:
        # prefer model dtype; fall back bf16 on Ampere/Hopper
        dt = self._policy_dtype()
        if dt in (torch.float16, torch.bfloat16):
            return dt
        return torch.bfloat16

    def _preferred_torch_dtype(self) -> torch.dtype:
        """
        Dtype to load weights / run FlashAttention safely.
        """
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _move_and_cast_processor_outputs(self, batch: dict) -> dict:
        """
        Move processor outputs to device and ensure pixel_values is in fp16/bf16
        (FlashAttention requirement + reduce VRAM).
        """
        dev = self.accelerator.device
        dt = self._autocast_dtype()

        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                if k == "pixel_values":
                    batch[k] = v.to(dev, dtype=dt, non_blocking=True)
                else:
                    batch[k] = v.to(dev, non_blocking=True)
        return batch

    def _get_images_from_inputs(self, inputs: list[dict]) -> list[Image.Image]:
        target_size = getattr(self, "grpo_img_size", None)
        if target_size is None:
            target_size = int(os.getenv("GRPO_IMG_SIZE", "1024"))

        pad_color = int(os.getenv("GRPO_PAD_COLOR", "0"))

        images: list[Image.Image] = []
        for x in inputs:
            if "image_path" in x and x.get("image_path"):
                images.append(
                    self._load_image_square(x.get("image_path"), target_size=int(target_size), pad_color=pad_color)
                )
            elif "image" in x and x.get("image") is not None:
                images.append(x["image"])
            else:
                raise KeyError("Each sample must contain either `image_path` or `image`.")
        return images

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        self.grpo_img_size = int(getattr(args, "target_size", 0) or int(os.getenv("GRPO_IMG_SIZE", "1024")))

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation

        # ======================================================
        # FIX: FlashAttention2 only supports fp16/bf16.
        # If torch_dtype isn't explicitly provided, HF often loads fp32.
        # Force a safe dtype here.
        # ======================================================
        if "torch_dtype" not in model_init_kwargs or model_init_kwargs["torch_dtype"] is None:
            model_init_kwargs["torch_dtype"] = self._preferred_torch_dtype()

        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")

            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a torch.dtype string, "
                    f"but got {torch_dtype}."
                )

            model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            print(f"Loading model {model_id} with kwargs {model_init_kwargs}")

            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache", None)
                model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                model = load_auto_for_policy(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` but model is already instantiated. "
                    "Only allowed when `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # =========================
        # Reference model
        # =========================
        if is_deepspeed_zero3_enabled():
            if peft_config is None:
                if "Qwen2-VL" in model_id:
                    self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                elif "Qwen2.5-VL" in model_id:
                    self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                elif "Aria" in model_id:
                    self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                else:
                    self.ref_model = load_auto_for_policy(model_id, **model_init_kwargs)
            else:
                self.ref_model = None
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        self._ref_adapters = _parse_adapters_env("REF_ADAPTERS")
        self._policy_adapters = _parse_adapters_env("POLICY_ADAPTERS")

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                if processing_class.tokenizer.pad_token_id is None:
                    processing_class.tokenizer.pad_token = processing_class.tokenizer.eos_token
                pad_token_id = processing_class.tokenizer.pad_token_id
                eos_token_id = processing_class.tokenizer.eos_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                if processing_class.pad_token_id is None:
                    processing_class.pad_token = processing_class.eos_token
                pad_token_id = processing_class.pad_token_id
                eos_token_id = processing_class.eos_token_id

        # Ensure valid pad/eos for generation
        if hasattr(processing_class, "tokenizer"):
            tokenizer = processing_class.tokenizer
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            pad_token_id = tokenizer.pad_token_id
            eos_token_id = tokenizer.eos_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = eos_token_id
        else:
            pad_token_id = getattr(processing_class, "pad_token_id", None)
            eos_token_id = getattr(processing_class, "eos_token_id", None)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing classes
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("Number of reward_processing_classes must match reward_funcs.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        def data_collator(features):
            return features

        # Training args
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,
            top_p=1.0,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        self.beta = args.beta

        try:
            model.warnings_issued["estimate_tokens"] = True
        except Exception:
            pass

        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # ==========================================================
    #  FIX OOM: bypass Accelerate ConvertOutputsToFp32 wrapper
    # ==========================================================
    def _unwrap_for_forward(self, model):
        m = model
        try:
            m = self.accelerator.unwrap_model(m)
        except Exception:
            pass
        if hasattr(m, "module"):
            m = m.module
        for _ in range(16):
            cls = type(m).__name__
            if ("ConvertOutputsToFp32" in cls) and hasattr(m, "model"):
                m = m.model
                if hasattr(m, "module"):
                    m = m.module
                continue
            if hasattr(m, "model") and not isinstance(m, (PreTrainedModel, torch.nn.Module)):
                try:
                    m = m.model
                    continue
                except Exception:
                    pass
            if hasattr(m, "model") and isinstance(getattr(m, "model"), torch.nn.Module) and (
                "accelerate" in str(type(m)).lower()
            ):
                m = m.model
                if hasattr(m, "module"):
                    m = m.module
                continue
            if hasattr(m, "module"):
                m = m.module
                continue
            break
        return m

    def _forward_no_fp32_convert(self, model, **kwargs):
        m = self._unwrap_for_forward(model)
        kwargs.setdefault("return_dict", False)
        kwargs.setdefault("output_hidden_states", False)
        kwargs.setdefault("output_attentions", False)
        dt = self._autocast_dtype()
        with torch.autocast(device_type="cuda", dtype=dt):
            out = m(**kwargs)
        return out

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        outputs = self._forward_no_fp32_convert(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits

        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def _reduce_mean_scalar(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.accelerator.device)
        x = x.detach()
        if x.numel() != 1:
            x = x.mean()
        return self.accelerator.reduce(x, reduction="mean")

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(self, model_name: Optional[str] = None, dataset_name: Optional[str] = None, tags=None):
        if not self.is_world_process_zero():
            return
        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]
        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))


# =============================================================================
#  Rollout helpers (Reasoning + Summary)
# =============================================================================

_ROLLOUT_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", flags=re.DOTALL | re.IGNORECASE)
_ROLLOUT_DECISION_RE = re.compile(r"<decision>(.*?)</decision>", flags=re.DOTALL | re.IGNORECASE)


def _rollout_extract_decision(text: str) -> str:
    if not text:
        return "continue"
    m = _ROLLOUT_DECISION_RE.search(text)
    if not m:
        return "continue"
    d = (m.group(1) or "").strip().lower()
    return d if d else "continue"


def _rollout_extract_decision_raw(text: str) -> str:
    if not text:
        return ""
    m = _ROLLOUT_DECISION_RE.search(text)
    if not m:
        return ""
    return (m.group(1) or "").strip()


def _rollout_extract_thought_plain(text: str) -> str:
    if not text:
        return ""
    m = _ROLLOUT_THOUGHT_RE.search(text)
    if m:
        core = (m.group(1) or "").strip()
    else:
        core = text.strip()
    core = _ROLLOUT_DECISION_RE.sub("", core).strip()
    core = re.sub(r"\s+", " ", core).strip()
    return core


def _rollout_concat_segments(segments, sep="\n") -> str:
    return sep.join(s for s in segments if s is not None and s != "")


def _build_reasoning_prompt(history_plain_steps) -> str:
    header = "You are a highly experienced radiologist.\n"
    if not history_plain_steps:
        history_block = "There are no previous reasoning steps yet.\n\n"
        instruction = "Generate ONLY the FIRST reasoning step based on the image.\n"
    else:
        history_block = (
            "Here are the previous reasoning steps:\n\n" + _rollout_concat_segments(history_plain_steps, sep="\n") + "\n"
        )
        instruction = "Generate ONLY the NEXT reasoning step based on the image and the previous steps above.\n"
    return header + history_block + instruction


def _as_stepwise_history(x: Any) -> list[str]:
    """
    Normalize any of these to List[str]:
      - List[str]
      - Tuple[str]
      - str (splitlines)
      - None -> []
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out = []
        for it in x:
            if it is None:
                continue
            s = str(it).strip()
            if s:
                out.append(s)
        return out
    if isinstance(x, str):
        lines = [ln.strip() for ln in x.splitlines() if ln.strip()]
        return lines
    s = str(x).strip()
    return [s] if s else []


def build_summary_prompt(stepwise_history: list[str]) -> str:
    """
    Build the summary prompt, aligned with the SFT builder:

      - Always includes <image> + radiologist role (optionally).
      - If there is history: list Step i lines, then ask to write report.
      - If no history: ask to write report based on image alone.

    NOTE:
    In our multimodal chat-template path, the image is provided via {"type":"image"}.
    If you still want the literal "<image>" token in the text, set env:
      SUMMARY_PROMPT_INCLUDE_IMAGE_TOKEN=1
    """
    include_image_token = _env_flag("SUMMARY_PROMPT_INCLUDE_IMAGE_TOKEN", "0")

    header = "<image>\nYou are a highly experienced radiologist.\n" if include_image_token else "You are a highly experienced radiologist.\n"

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
        instruction = "Based on the provided image, write a clear, clinically appropriate radiology report.\n"

    return header + reasoning_block + instruction


# =============================================================================
#  GRPOTrainerRollout (multi-step reasoning rollout per sample; reward=trajectory-level)
# =============================================================================


class GRPOTrainerRollout(Qwen2VLGRPOTrainer):
    def __init__(self, *args, max_rollout_steps: int = 6, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_rollout_steps = int(max_rollout_steps)

    def _get_per_token_logps_completion_only(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        prompt_len: int,
    ):
        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
        )
        return per_token_logps[:, (prompt_len - 1) :]

    def _rollout_trajectory(self, model, image, gen_cfg_one):
        history_plain_steps = []
        step_texts = []
        step_prompt_texts = []

        for _ in range(self.max_rollout_steps):
            user_prompt = _build_reasoning_prompt(history_plain_steps)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]

            prompt_text = maybe_apply_chat_template({"prompt": messages}, self.processing_class)["prompt"]
            step_prompt_texts.append(prompt_text)

            prompt_inputs = self.processing_class(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            prompt_inputs = self._move_and_cast_processor_outputs(prompt_inputs)

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
                # FIX: ensure generate() runs under autocast (flash-attn requires bf16/fp16)
                dt = self._autocast_dtype()
                with torch.autocast(device_type="cuda", dtype=dt):
                    out_ids = unwrapped.generate(**prompt_inputs, generation_config=gen_cfg_one)

            prompt_len = prompt_inputs["input_ids"].size(1)
            completion_ids = out_ids[:, prompt_len:]
            step_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)[0]
            step_texts.append(step_text)

            plain = _rollout_extract_thought_plain(step_text)
            if plain:
                history_plain_steps.append(f"Step {len(history_plain_steps) + 1}: {plain}")

            if _rollout_extract_decision(step_text) == "summary":
                break

        return step_texts, step_prompt_texts

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        if self._policy_adapters is not None:
            try:
                _set_adapters(self.accelerator.unwrap_model(model), self._policy_adapters)
            except Exception:
                pass

        device = self.accelerator.device
        B = len(inputs)
        G = self.num_generations

        images = self._get_images_from_inputs(inputs)

        gen_cfg_one = copy.deepcopy(self.generation_config)
        gen_cfg_one.num_return_sequences = 1

        # ------------------------------------------------------
        # Rollout trajectories (WITH tqdm)
        # ------------------------------------------------------
        trajectories: list[list[str]] = []
        traj_prompts: list[list[str]] = []

        total_rollouts = B * G
        # show progress only on rank0 to avoid spam
        pbar = tqdm(
            total=total_rollouts,
            desc=f"rollout (B={B}, G={G}, steps<={self.max_rollout_steps})",
            disable=not self.is_world_process_zero(),
            mininterval=float(os.getenv("TQDM_MININTERVAL", "1.0")),
            smoothing=0.1,
        )

        for b in range(B):
            for g in range(G):
                steps_text, steps_prompt_text = self._rollout_trajectory(
                    model=model,
                    image=images[b],
                    gen_cfg_one=gen_cfg_one,
                )
                trajectories.append(steps_text)
                traj_prompts.append(steps_prompt_text)
                pbar.update(1)

        pbar.close()

        # ------------------------------------------------------
        # PRINT: prompt + completion + decision for each step
        # ------------------------------------------------------
        print("\n" + "=" * 90)
        print("ROLLOUT DEBUG: PROMPT + COMPLETION + <decision> (ALL)")
        print("=" * 90)

        for b in range(B):
            img_key = _image_key_from_input(inputs[b])
            print(f"\n[SAMPLE b={b}] image_key={img_key}")
            print("-" * 90)

            gt = inputs[b].get("gt_labels", None)
            if gt is None:
                gt = inputs[b].get("gt_label", None)
            if gt is not None:
                try:
                    import json

                    print("[GT_LABEL (from inputs)]")
                    print(json.dumps(gt, ensure_ascii=False, indent=2))
                except Exception:
                    print(f"[GT_LABEL (from inputs)] {gt}")
                print("-" * 90)

            for g in range(G):
                idx_show = b * G + g
                steps_text = trajectories[idx_show]
                steps_prompt_text = traj_prompts[idx_show]

                print(f"\n  [COMPLETION g={g}] steps={len(steps_text)}")
                print("  " + "-" * 86)

                for si, (p_txt, s_txt) in enumerate(zip(steps_prompt_text, steps_text), start=1):
                    dec_norm = _rollout_extract_decision(s_txt)
                    dec_raw = _rollout_extract_decision_raw(s_txt)
                    plain = _rollout_extract_thought_plain(s_txt)

                    print(f"\n    (step {si}) decision_norm='{dec_norm}' decision_raw='{dec_raw}'")
                    print("    " + "." * 86)

                    print("    [PROMPT]")
                    print(p_txt.strip())

                    print("\n    [COMPLETION RAW]")
                    print(s_txt.strip())

                    print("\n    [THOUGHT PLAIN]")
                    print(plain.strip())

                    print("\n    " + "." * 86)

        print("=" * 90 + "\n")

        # ------------------------------------------------------
        # Reward computation (trajectory-level)
        # ------------------------------------------------------
        completion_texts = ["\n".join(ts) for ts in trajectories]
        first_prompt_texts = [pp[0] if pp else "" for pp in traj_prompts]
        prompts_for_reward = first_prompt_texts
        completions_for_reward = completion_texts

        rewards_per_func = torch.zeros(B * G, len(self.reward_funcs), device=device)
        merged_stats = {}

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                texts = [p + "\n" + c for p, c in zip(prompts_for_reward, completions_for_reward)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                for k, v in list(reward_inputs.items()):
                    if torch.is_tensor(v):
                        reward_inputs[k] = v.to(device, non_blocking=True)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for ex in inputs:
                        reward_kwargs[key].extend([ex[key]] * G)

                reward_kwargs["trajectory_steps"] = trajectories
                reward_kwargs["trajectory_prompts"] = traj_prompts

                reward_stats = {}
                reward_kwargs["_reward_stats"] = reward_stats

                out_r = reward_func(
                    prompts=prompts_for_reward,
                    completions=completions_for_reward,
                    **reward_kwargs,
                )
                rewards_per_func[:, i] = torch.tensor(out_r, dtype=torch.float32, device=device)

                if reward_stats:
                    fn_name = getattr(reward_func, "__name__", f"reward_func_{i}")
                    for k, v in reward_stats.items():
                        merged_stats[f"{fn_name}/{k}"] = v

        rewards = rewards_per_func.sum(dim=1)

        # ------------------------------------------------------
        # PRINT: reward per completion (g=0..G-1) + per-func breakdown
        # ------------------------------------------------------
        print("\n" + "=" * 90)
        print("ROLLOUT DEBUG: REWARDS PER COMPLETION (g=0..G-1)")
        print("=" * 90)

        reward_names = []
        for rf in self.reward_funcs:
            if isinstance(rf, PreTrainedModel):
                reward_names.append(rf.config._name_or_path.split("/")[-1])
            else:
                reward_names.append(getattr(rf, "__name__", "reward_func"))

        rewards_cpu = rewards.detach().float().cpu().tolist()
        rpf_cpu = rewards_per_func.detach().float().cpu().tolist()

        for b in range(B):
            print(f"\n[SAMPLE b={b}] image_key={_image_key_from_input(inputs[b])}")
            print("-" * 90)
            for g in range(G):
                idx_show = b * G + g
                total_r = rewards_cpu[idx_show]
                per_func = rpf_cpu[idx_show]

                print(f"  g={g}  total_reward={total_r:.6f}")
                for name, val in zip(reward_names, per_func):
                    print(f"      - {name}: {float(val):.6f}")

        print("=" * 90 + "\n")

        # ------------------------------------------------------
        # Advantage normalization (A within each sample)
        # ------------------------------------------------------
        mean_grouped_b = rewards.view(B, G).mean(dim=1)
        std_grouped_b = rewards.view(B, G).std(dim=1)

        mean_grouped = mean_grouped_b.repeat_interleave(G, dim=0)
        std_grouped = std_grouped_b.repeat_interleave(G, dim=0)
        advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)

        print("\n" + "=" * 90)
        print("ROLLOUT DEBUG: ADVANTAGE (A) per sample")
        print("A = (reward_g - mean_reward_of_sample) / (std_reward_of_sample + 1e-4)")
        print("=" * 90)

        mean_cpu = mean_grouped_b.detach().float().cpu().tolist()
        std_cpu = std_grouped_b.detach().float().cpu().tolist()
        adv_cpu = advantages.detach().float().cpu().tolist()

        for b in range(B):
            img_key = _image_key_from_input(inputs[b])
            b_advs = adv_cpu[b * G : (b + 1) * G]
            g_best = int(max(range(G), key=lambda gg: b_advs[gg]))
            g_worst = int(min(range(G), key=lambda gg: b_advs[gg]))

            print(f"\n[SAMPLE b={b}] image_key={img_key}")
            print(f"  reward_mean={mean_cpu[b]:.6f}  reward_std={std_cpu[b]:.6f}")
            print(f"  winner_by_A: g={g_best}  A={b_advs[g_best]:+.6f}")
            print(f"  loser_by_A : g={g_worst} A={b_advs[g_worst]:+.6f}")
            print("  " + "-" * 86)
            for g in range(G):
                idx_show = b * G + g
                print(f"  g={g}  reward={rewards_cpu[idx_show]:.6f}  A={adv_cpu[idx_show]:+.6f}")

        print("=" * 90 + "\n")

        # ------------------------------------------------------
        # Compute per-step loss (sum over steps in trajectory)
        # ------------------------------------------------------
        total_loss = 0.0
        total_traj = 0
        total_step_count = 0

        idx = 0
        for b in range(B):
            for g in range(G):
                adv = advantages[idx]
                steps_text = trajectories[idx]
                steps_prompt_text = traj_prompts[idx]

                traj_loss = 0.0
                traj_steps = 0

                for (prompt_text, step_text) in zip(steps_prompt_text, steps_text):
                    prompt_inputs = self.processing_class(
                        text=[prompt_text],
                        images=[images[b]],
                        return_tensors="pt",
                        padding=True,
                        padding_side="left",
                        add_special_tokens=False,
                    )
                    prompt_inputs = super()._prepare_inputs(prompt_inputs)
                    prompt_inputs = self._move_and_cast_processor_outputs(prompt_inputs)

                    prompt_ids = prompt_inputs["input_ids"]
                    prompt_len = prompt_ids.size(1)

                    tok = self.processing_class.tokenizer if hasattr(self.processing_class, "tokenizer") else self.processing_class
                    comp_ids = tok(step_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
                        device, dtype=torch.long, non_blocking=True
                    )
                    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
                    attention_mask = torch.ones_like(input_ids, device=device, dtype=torch.long)

                    pixel_values = prompt_inputs["pixel_values"]
                    image_grid_thw = prompt_inputs["image_grid_thw"]

                    per_token_logps = self._get_per_token_logps_completion_only(
                        model, input_ids, attention_mask, pixel_values, image_grid_thw, prompt_len
                    )

                    with torch.inference_mode():
                        if self.ref_model is not None:
                            ref_per_token_logps = self._get_per_token_logps_completion_only(
                                self.ref_model, input_ids, attention_mask, pixel_values, image_grid_thw, prompt_len
                            )
                        else:
                            unwrapped = self.accelerator.unwrap_model(model)
                            if self._ref_adapters is not None:
                                with _temporary_adapters(unwrapped, self._ref_adapters):
                                    ref_per_token_logps = self._get_per_token_logps_completion_only(
                                        unwrapped, input_ids, attention_mask, pixel_values, image_grid_thw, prompt_len
                                    )
                            else:
                                if hasattr(unwrapped, "disable_adapter"):
                                    with unwrapped.disable_adapter():
                                        ref_per_token_logps = self._get_per_token_logps_completion_only(
                                            unwrapped, input_ids, attention_mask, pixel_values, image_grid_thw, prompt_len
                                        )
                                else:
                                    raise ValueError(
                                        "ref_model is None but model has no disable_adapter(). "
                                        "Set REF_ADAPTERS for PEFT adapter switching or enable ref_model."
                                    )

                    per_token_kl = (
                        torch.exp(ref_per_token_logps - per_token_logps)
                        - (ref_per_token_logps - per_token_logps)
                        - 1
                    )

                    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * adv
                    per_token_loss = -(per_token_loss - self.beta * per_token_kl)

                    traj_loss = traj_loss + per_token_loss.mean()
                    traj_steps += 1
                    total_step_count += 1

                if traj_steps > 0:
                    traj_loss = traj_loss / traj_steps
                total_loss = total_loss + traj_loss

                total_traj += 1
                idx += 1

        loss = total_loss / max(total_traj, 1)

        self._metrics["reward"].append(float(self._reduce_mean_scalar(rewards.mean()).item()))
        self._metrics["reward_std"].append(float(self._reduce_mean_scalar(std_grouped_b.mean()).item()))
        self._metrics["rollout_avg_steps"].append(float(total_step_count) / max(total_traj, 1))

        for k, v in merged_stats.items():
            try:
                self._metrics[f"reward_stats/{k}"].append(float(v))
            except Exception:
                pass

        return loss


# =============================================================================
#  GRPOTrainerSummary (single-step summary; reward=caption-level)
# =============================================================================

class GRPOTrainerSummary(Qwen2VLGRPOTrainer):
    """
    Phase B: Generate ONE radiology report (caption) conditioned on:
      - image
      - optional stepwise reasoning history (from dataset or external reasoner)

    Expected input fields per example (flexible):
      - image_path / image
      - gt_labels (for reward server)
      - stepwise_history OR reasoning_steps OR reasoning_history OR steps_history (optional)
        (List[str] preferred; str also ok)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_per_token_logps_completion_only(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        prompt_len: int,
    ):
        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
        )
        return per_token_logps[:, (prompt_len - 1) :]

    def _extract_history_for_example(self, ex: dict) -> list[str]:
        for k in (
            "stepwise_history",
            "reasoning_steps",
            "reasoning_history",
            "steps_history",
            "trajectory_steps_reasoning",
            "reason_steps",
        ):
            if k in ex and ex.get(k) is not None:
                return _as_stepwise_history(ex.get(k))
        return []

    def _build_summary_chat_prompt(self, stepwise_history: list[str]) -> str:
        user_text = build_summary_prompt(stepwise_history)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        return maybe_apply_chat_template({"prompt": messages}, self.processing_class)["prompt"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        if self._policy_adapters is not None:
            try:
                _set_adapters(self.accelerator.unwrap_model(model), self._policy_adapters)
            except Exception:
                pass

        device = self.accelerator.device
        B = len(inputs)
        G = self.num_generations

        images = self._get_images_from_inputs(inputs)

        gen_cfg_one = copy.deepcopy(self.generation_config)
        gen_cfg_one.num_return_sequences = 1

        # ------------------------------------------------------
        # Rollout (summary) trajectories (WITH tqdm)
        # Each "trajectory" is a single caption step: [caption]
        # ------------------------------------------------------
        trajectories: list[list[str]] = []
        traj_prompts: list[list[str]] = []

        total_rollouts = B * G
        pbar = tqdm(
            total=total_rollouts,
            desc=f"summary_rollout (B={B}, G={G})",
            disable=not self.is_world_process_zero(),
            mininterval=float(os.getenv("TQDM_MININTERVAL", "1.0")),
            smoothing=0.1,
        )

        for b in range(B):
            stepwise_history = self._extract_history_for_example(inputs[b])
            prompt_text = self._build_summary_chat_prompt(stepwise_history)

            for g in range(G):
                prompt_inputs = self.processing_class(
                    text=[prompt_text],
                    images=[images[b]],
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
                prompt_inputs = super()._prepare_inputs(prompt_inputs)
                prompt_inputs = self._move_and_cast_processor_outputs(prompt_inputs)

                with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
                    dt = self._autocast_dtype()
                    with torch.autocast(device_type="cuda", dtype=dt):
                        out_ids = unwrapped.generate(**prompt_inputs, generation_config=gen_cfg_one)

                prompt_len = prompt_inputs["input_ids"].size(1)
                completion_ids = out_ids[:, prompt_len:]
                caption = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)[0]

                trajectories.append([caption])
                traj_prompts.append([prompt_text])

                pbar.update(1)

        pbar.close()

        # ------------------------------------------------------
        # PRINT: prompt + caption
        # ------------------------------------------------------
        print("\n" + "=" * 90)
        print("SUMMARY DEBUG: PROMPT + CAPTION (ALL)")
        print("=" * 90)

        for b in range(B):
            img_key = _image_key_from_input(inputs[b])
            print(f"\n[SAMPLE b={b}] image_key={img_key}")
            print("-" * 90)

            hist = self._extract_history_for_example(inputs[b])
            if hist:
                print("[STEPWISE_HISTORY] (first 12 lines)")
                for ln in hist[:12]:
                    print("  " + str(ln))
                if len(hist) > 12:
                    print(f"  ... ({len(hist) - 12} more lines)")
                print("-" * 90)

            gt = inputs[b].get("gt_labels", None)
            if gt is None:
                gt = inputs[b].get("gt_label", None)
            if gt is not None:
                try:
                    import json
                    print("[GT_LABEL (from inputs)]")
                    print(json.dumps(gt, ensure_ascii=False, indent=2))
                except Exception:
                    print(f"[GT_LABEL (from inputs)] {gt}")
                print("-" * 90)

            for g in range(G):
                idx_show = b * G + g
                prompt_text = traj_prompts[idx_show][0]
                caption = trajectories[idx_show][0]

                print(f"\n  [COMPLETION g={g}]")
                print("  " + "-" * 86)
                print("\n    [PROMPT]")
                print(prompt_text.strip())
                print("\n    [CAPTION]")
                print(str(caption).strip())

        print("=" * 90 + "\n")

        # ------------------------------------------------------
        # Reward computation (caption-level)
        # ------------------------------------------------------
        completion_texts = [ts[0] if ts else "" for ts in trajectories]  # one caption per rollout
        prompt_texts = [pp[0] if pp else "" for pp in traj_prompts]

        prompts_for_reward = prompt_texts
        completions_for_reward = completion_texts

        rewards_per_func = torch.zeros(B * G, len(self.reward_funcs), device=device)
        merged_stats = {}

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                texts = [p + "\n" + c for p, c in zip(prompts_for_reward, completions_for_reward)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                for k, v in list(reward_inputs.items()):
                    if torch.is_tensor(v):
                        reward_inputs[k] = v.to(device, non_blocking=True)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for ex in inputs:
                        reward_kwargs[key].extend([ex[key]] * G)

                # For Phase B server, we typically don't need trajectory_steps;
                # but we still provide it for compatibility/debugging.
                reward_kwargs["trajectory_steps"] = trajectories
                reward_kwargs["trajectory_prompts"] = traj_prompts

                reward_stats = {}
                reward_kwargs["_reward_stats"] = reward_stats

                out_r = reward_func(
                    prompts=prompts_for_reward,
                    completions=completions_for_reward,
                    **reward_kwargs,
                )
                rewards_per_func[:, i] = torch.tensor(out_r, dtype=torch.float32, device=device)

                if reward_stats:
                    fn_name = getattr(reward_func, "__name__", f"reward_func_{i}")
                    for k, v in reward_stats.items():
                        merged_stats[f"{fn_name}/{k}"] = v

        rewards = rewards_per_func.sum(dim=1)

        # ------------------------------------------------------
        # PRINT: reward per completion + per-func breakdown
        # ------------------------------------------------------
        print("\n" + "=" * 90)
        print("SUMMARY DEBUG: REWARDS PER COMPLETION (g=0..G-1)")
        print("=" * 90)

        reward_names = []
        for rf in self.reward_funcs:
            if isinstance(rf, PreTrainedModel):
                reward_names.append(rf.config._name_or_path.split("/")[-1])
            else:
                reward_names.append(getattr(rf, "__name__", "reward_func"))

        rewards_cpu = rewards.detach().float().cpu().tolist()
        rpf_cpu = rewards_per_func.detach().float().cpu().tolist()

        for b in range(B):
            print(f"\n[SAMPLE b={b}] image_key={_image_key_from_input(inputs[b])}")
            print("-" * 90)
            for g in range(G):
                idx_show = b * G + g
                total_r = rewards_cpu[idx_show]
                per_func = rpf_cpu[idx_show]

                print(f"  g={g}  total_reward={total_r:.6f}")
                for name, val in zip(reward_names, per_func):
                    print(f"      - {name}: {float(val):.6f}")

        print("=" * 90 + "\n")

        # ------------------------------------------------------
        # Advantage normalization (A within each sample)
        # ------------------------------------------------------
        mean_grouped_b = rewards.view(B, G).mean(dim=1)
        std_grouped_b = rewards.view(B, G).std(dim=1)

        mean_grouped = mean_grouped_b.repeat_interleave(G, dim=0)
        std_grouped = std_grouped_b.repeat_interleave(G, dim=0)
        advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)

        print("\n" + "=" * 90)
        print("SUMMARY DEBUG: ADVANTAGE (A) per sample")
        print("A = (reward_g - mean_reward_of_sample) / (std_reward_of_sample + 1e-4)")
        print("=" * 90)

        mean_cpu = mean_grouped_b.detach().float().cpu().tolist()
        std_cpu = std_grouped_b.detach().float().cpu().tolist()
        adv_cpu = advantages.detach().float().cpu().tolist()

        for b in range(B):
            img_key = _image_key_from_input(inputs[b])
            b_advs = adv_cpu[b * G : (b + 1) * G]
            g_best = int(max(range(G), key=lambda gg: b_advs[gg]))
            g_worst = int(min(range(G), key=lambda gg: b_advs[gg]))

            print(f"\n[SAMPLE b={b}] image_key={img_key}")
            print(f"  reward_mean={mean_cpu[b]:.6f}  reward_std={std_cpu[b]:.6f}")
            print(f"  winner_by_A: g={g_best}  A={b_advs[g_best]:+.6f}")
            print(f"  loser_by_A : g={g_worst} A={b_advs[g_worst]:+.6f}")
            print("  " + "-" * 86)
            for g in range(G):
                idx_show = b * G + g
                print(f"  g={g}  reward={rewards_cpu[idx_show]:.6f}  A={adv_cpu[idx_show]:+.6f}")

        print("=" * 90 + "\n")

        # ------------------------------------------------------
        # Compute loss (single-step caption => 1 step per trajectory)
        # ------------------------------------------------------
        total_loss = 0.0
        total_traj = 0
        total_step_count = 0

        idx = 0
        for b in range(B):
            for g in range(G):
                adv = advantages[idx]
                # exactly one step
                prompt_text = traj_prompts[idx][0]
                caption_text = trajectories[idx][0]

                prompt_inputs = self.processing_class(
                    text=[prompt_text],
                    images=[images[b]],
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
                prompt_inputs = super()._prepare_inputs(prompt_inputs)
                prompt_inputs = self._move_and_cast_processor_outputs(prompt_inputs)

                prompt_ids = prompt_inputs["input_ids"]
                prompt_len = prompt_ids.size(1)

                tok = self.processing_class.tokenizer if hasattr(self.processing_class, "tokenizer") else self.processing_class
                comp_ids = tok(caption_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
                    device, dtype=torch.long, non_blocking=True
                )
                input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
                attention_mask = torch.ones_like(input_ids, device=device, dtype=torch.long)

                pixel_values = prompt_inputs["pixel_values"]
                image_grid_thw = prompt_inputs["image_grid_thw"]

                per_token_logps = self._get_per_token_logps_completion_only(
                    model, input_ids, attention_mask, pixel_values, image_grid_thw, prompt_len
                )

                with torch.inference_mode():
                    if self.ref_model is not None:
                        ref_per_token_logps = self._get_per_token_logps_completion_only(
                            self.ref_model, input_ids, attention_mask, pixel_values, image_grid_thw, prompt_len
                        )
                    else:
                        unwrapped = self.accelerator.unwrap_model(model)
                        if self._ref_adapters is not None:
                            with _temporary_adapters(unwrapped, self._ref_adapters):
                                ref_per_token_logps = self._get_per_token_logps_completion_only(
                                    unwrapped, input_ids, attention_mask, pixel_values, image_grid_thw, prompt_len
                                )
                        else:
                            if hasattr(unwrapped, "disable_adapter"):
                                with unwrapped.disable_adapter():
                                    ref_per_token_logps = self._get_per_token_logps_completion_only(
                                        unwrapped, input_ids, attention_mask, pixel_values, image_grid_thw, prompt_len
                                    )
                            else:
                                raise ValueError(
                                    "ref_model is None but model has no disable_adapter(). "
                                    "Set REF_ADAPTERS for PEFT adapter switching or enable ref_model."
                                )

                per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps)
                    - (ref_per_token_logps - per_token_logps)
                    - 1
                )

                per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * adv
                per_token_loss = -(per_token_loss - self.beta * per_token_kl)

                traj_loss = per_token_loss.mean()
                total_loss = total_loss + traj_loss

                total_traj += 1
                total_step_count += 1
                idx += 1

        loss = total_loss / max(total_traj, 1)

        self._metrics["reward"].append(float(self._reduce_mean_scalar(rewards.mean()).item()))
        self._metrics["reward_std"].append(float(self._reduce_mean_scalar(std_grouped_b.mean()).item()))
        self._metrics["rollout_avg_steps"].append(float(total_step_count) / max(total_traj, 1))

        for k, v in merged_stats.items():
            try:
                self._metrics[f"reward_stats/{k}"].append(float(v))
            except Exception:
                pass

        return loss
