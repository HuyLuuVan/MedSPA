# src/r1-vl/src/open_r1/trainer/grpo_trainer.py
# Copyright 2025 The HuggingFace Team.
# Licensed under Apache License, Version 2.0

import os
import textwrap
import re
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
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

import copy
import contextlib  # for adapter switch context manager

# PIL for runtime image loading (avoid putting PIL.Image inside HF datasets)
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- imports ---
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
)

try:
    # Qwen2-VL config class (may exist depending on transformers version)
    from transformers.models.qwen2_vl import Qwen2VLConfig  # type: ignore
except Exception:
    Qwen2VLConfig = None


def _is_vision_lang_config(cfg) -> bool:
    """
    Detect vision-language configs (use AutoModelForVision2Seq).
    Prefer isinstance(Qwen2VLConfig) if available; fallback to heuristics.
    """
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
    """
    Choose correct AutoModel class based on config:
      - Vision-Language -> AutoModelForVision2Seq
      - Else -> AutoModelForCausalLM
    """
    kwargs.setdefault("trust_remote_code", True)

    # don't pass use_cache into model __init__ via from_pretrained kwargs
    use_cache = kwargs.pop("use_cache", None)

    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=kwargs["trust_remote_code"])
    if _is_vision_lang_config(cfg):
        model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)

    # set on config AFTER load (safe)
    if use_cache is not None:
        try:
            model.config.use_cache = bool(use_cache)
        except Exception:
            pass

    return model


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


# =========================
#   PEFT adapter helpers
# =========================

def _parse_adapters_env(env_key: str) -> Optional[list[str]]:
    """
    Read env like:
      REF_ADAPTERS="sft"
      POLICY_ADAPTERS="sft,grpo"
    Return list[str] or None if not set/empty.
    """
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None


def _get_active_adapters(model) -> Optional[list[str]]:
    """
    Best-effort get current active adapters (PEFT).
    """
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
    """
    Best-effort set adapters on PEFT model.
    - Newer PEFT: model.set_adapter(list[str]) supported.
    - Older: model.set_adapter(str) only => fallback to first.
    """
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
    """
    Temporarily switch active adapters, then restore previous.
    If adapters is None: do nothing.
    """
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


class Qwen2VLGRPOTrainer(Trainer):
    """
    Base GRPO trainer (single-step completion GRPO).
    """

    # -------------------------
    # Image loading helpers
    # -------------------------
    @staticmethod
    def _to_str_path(p: Any) -> str:
        if isinstance(p, (list, tuple)):
            return p[0] if len(p) > 0 else ""
        return p or ""

    @staticmethod
    def _load_image_square(path: Any, target_size: int = 1024, pad_color: int = 0) -> Image.Image:
        """
        Load image from path and make it square (resize + pad).
        """
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

    def _get_images_from_inputs(self, inputs: list[dict]) -> list[Image.Image]:
        """
        Prefer using image_path (string) to avoid PIL.Image inside HF datasets.
        Fallback to x["image"] if user still provides it.
        """
        target_size = int(os.getenv("GRPO_IMG_SIZE", "1024"))
        pad_color = int(os.getenv("GRPO_PAD_COLOR", "0"))

        images: list[Image.Image] = []
        for x in inputs:
            if "image_path" in x and x.get("image_path"):
                images.append(self._load_image_square(x.get("image_path"), target_size=target_size, pad_color=pad_color))
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

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation

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
        # Reference model (CACH B)
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

        # Data collator (no collation needed)
        def data_collator(features):
            return features

        # Training args
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=float(getattr(args, "temperature", 1.0)),
            top_p=float(getattr(args, "top_p", 1.0)),
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True
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
        """
        Unwrap DDP then unwrap Accelerate ConvertOutputsToFp32 wrapper if present.
        We want to call the underlying model forward WITHOUT convert_to_fp32().
        """
        m = model
        # DDP -> .module
        if hasattr(m, "module"):
            m = m.module

        # Accelerate ConvertOutputsToFp32 wrapper has attribute .model_forward
        # Calling m(...) triggers convert_to_fp32; calling m.model_forward(...) bypasses it.
        return m

    def _forward_no_fp32_convert(self, model, **kwargs):
        """
        Forward pass in autocast, bypassing accelerate's convert_to_fp32 wrapper.
        Returns the HF outputs (with .logits).
        """
        m = self._unwrap_for_forward(model)

        # Use accelerator autocast if available (bf16/fp16) to keep logits in bf16 and save VRAM.
        with self.accelerator.autocast():
            if hasattr(m, "model_forward") and callable(getattr(m, "model_forward")):
                # This is very likely ConvertOutputsToFp32 wrapper
                return m.model_forward(**kwargs)
            return m(**kwargs)

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        # IMPORTANT: call forward without accelerate convert_to_fp32 (OOM on rank0)
        outputs = self._forward_no_fp32_convert(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        logits = outputs.logits

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

    # =========================
    #   OOM-safe metrics helpers
    # =========================
    def _reduce_mean_scalar(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.accelerator.device)
        x = x.detach()
        if x.numel() != 1:
            x = x.mean()
        return self.accelerator.reduce(x, reduction="mean")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        if self._policy_adapters is not None:
            try:
                _set_adapters(self.accelerator.unwrap_model(model), self._policy_adapters)
            except Exception:
                pass

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        images = self._get_images_from_inputs(inputs)

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        _dev = self.accelerator.device
        for _k, _v in list(prompt_inputs.items()):
            if torch.is_tensor(_v):
                prompt_inputs[_k] = _v.to(_dev, non_blocking=True)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
        )
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                )
            else:
                unwrapped = self.accelerator.unwrap_model(model)

                if self._ref_adapters is not None:
                    with _temporary_adapters(unwrapped, self._ref_adapters):
                        ref_per_token_logps = self._get_per_token_logps(
                            unwrapped, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                        )
                else:
                    if hasattr(unwrapped, "disable_adapter"):
                        with unwrapped.disable_adapter():
                            ref_per_token_logps = self._get_per_token_logps(
                                unwrapped, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                            )
                    else:
                        raise ValueError(
                            "ref_model is None but model has no disable_adapter(). "
                            "Set REF_ADAPTERS for PEFT adapter switching or enable ref_model."
                        )

        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # OOM-safe metrics
        completion_length = self._reduce_mean_scalar(completion_mask.sum(1).float().mean()).item()
        self._metrics["completion_length"].append(float(completion_length))

        reward_per_func_local = rewards_per_func.mean(0)
        reward_per_func_world = self.accelerator.reduce(reward_per_func_local.detach(), reduction="mean")
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(float(reward_per_func_world[i].item()))

        self._metrics["reward"].append(float(self._reduce_mean_scalar(rewards.mean()).item()))
        self._metrics["reward_std"].append(float(self._reduce_mean_scalar(std_grouped_rewards.mean()).item()))

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(float(self._reduce_mean_scalar(mean_kl).item()))

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
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
#  GRPOTrainerRollout (multi-step PR trajectory; paper-aligned R-Align)
# =============================================================================

_ROLLOUT_ACTION_RE = re.compile(r"<action>(.*?)</action>", flags=re.DOTALL | re.IGNORECASE)
_ROLLOUT_REASON_RE = re.compile(r"<reason>(.*?)</reason>", flags=re.DOTALL | re.IGNORECASE)
_ROLLOUT_DECISION_RE = re.compile(
    r"<decision>(continue|summary)</decision>",
    flags=re.DOTALL | re.IGNORECASE,
)
_ROLLOUT_STEP_RE = re.compile(
    r"^\s*<action>(?P<action>.*?)</action>\s*"
    r"<reason>(?P<reason>.*?)</reason>\s*"
    r"<decision>(?P<decision>continue|summary)</decision>\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)


def _rollout_normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).strip()


def _rollout_parse_step(text: str):
    if not text:
        return None
    m = _ROLLOUT_STEP_RE.match(text.strip())
    if not m:
        return None
    action = _rollout_normalize_text(m.group("action"))
    reason = _rollout_normalize_text(m.group("reason"))
    decision = (m.group("decision") or "").strip().lower()
    if not action or not reason or decision not in {"continue", "summary"}:
        return None
    return action, reason, decision


def _rollout_extract_decision(text: str) -> str:
    if not text:
        return "continue"
    m = _ROLLOUT_DECISION_RE.search(text)
    if not m:
        return "continue"
    decision = (m.group(1) or "").strip().lower()
    return decision if decision in {"continue", "summary"} else "continue"


def _rollout_history_step(step_index: int, step_text: str) -> str:
    parsed = _rollout_parse_step(step_text)
    if parsed is not None:
        action, reason, decision = parsed
        return (
            f"Step {step_index}:\n"
            f"<action>{action}</action>\n"
            f"<reason>{reason}</reason>\n"
            f"<decision>{decision}</decision>"
        )

    raw = (step_text or "").strip()
    return f"Step {step_index}:\n{raw}" if raw else f"Step {step_index}: [invalid step]"


def _rollout_concat_segments(segments, sep="\n\n") -> str:
    return sep.join(s for s in segments if s is not None and s != "")


def _build_reasoning_prompt(history_steps) -> str:
    format_block = (
        "Each step must contain exactly one action-reasoning-decision triplet in this format:\n"
        "<action>diagnostic focus</action>\n"
        "<reason>image-grounded reasoning for that focus</reason>\n"
        "<decision>continue</decision>\n"
        "Use <decision>summary</decision> only when the collected evidence is sufficient "
        "to terminate the reasoning chain.\n"
        "Return only the single next step. Do not add markdown or extra text.\n\n"
    )

    header = "You are a highly experienced radiologist.\n" + format_block

    if not history_steps:
        return (
            header
            + "There are no previous reasoning steps yet.\n\n"
            + "Generate ONLY the FIRST reasoning step based on the image.\n"
        )

    return (
        header
        + "Here are the previous reasoning steps:\n\n"
        + _rollout_concat_segments(history_steps)
        + "\n\nGenerate ONLY the NEXT reasoning step based on the image and the previous steps above.\n"
    )


class GRPOTrainerRollout(Qwen2VLGRPOTrainer):
    """
    Multi-step PR rollout trainer.

    For each input, the current PR policy samples G complete reasoning
    trajectories. Each trajectory is generated step by step using the same
    current policy and the accumulated action-reasoning-decision memory.
    A trajectory-level reward is then assigned uniformly to every generated
    token in that trajectory, with KL regularization against the reference
    policy.
    """

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
        history_steps = []
        step_texts = []
        step_prompt_texts = []

        for step_index in range(1, self.max_rollout_steps + 1):
            user_prompt = _build_reasoning_prompt(history_steps)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]

            prompt_text = maybe_apply_chat_template(
                {"prompt": messages},
                self.processing_class,
            )["prompt"]
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

            device = self.accelerator.device
            for key, value in list(prompt_inputs.items()):
                if torch.is_tensor(value):
                    prompt_inputs[key] = value.to(device, non_blocking=True)

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
                out_ids = unwrapped.generate(
                    **prompt_inputs,
                    generation_config=gen_cfg_one,
                )

            prompt_len = prompt_inputs["input_ids"].size(1)
            completion_ids = out_ids[:, prompt_len:]
            step_text = self.processing_class.batch_decode(
                completion_ids,
                skip_special_tokens=True,
            )[0].strip()

            step_texts.append(step_text)
            history_steps.append(_rollout_history_step(step_index, step_text))

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
        batch_size = len(inputs)
        group_size = self.num_generations

        if group_size < 2:
            raise ValueError("GRPOTrainerRollout requires num_generations >= 2.")

        images = self._get_images_from_inputs(inputs)

        gen_cfg_one = copy.deepcopy(self.generation_config)
        gen_cfg_one.num_return_sequences = 1

        trajectories = []
        traj_prompts = []
        for batch_index in range(batch_size):
            for _ in range(group_size):
                steps_text, steps_prompt_text = self._rollout_trajectory(
                    model=model,
                    image=images[batch_index],
                    gen_cfg_one=gen_cfg_one,
                )
                trajectories.append(steps_text)
                traj_prompts.append(steps_prompt_text)

        completion_texts = ["\n".join(steps) for steps in trajectories]
        first_prompt_texts = [prompts[0] if prompts else "" for prompts in traj_prompts]

        rewards_per_func = torch.zeros(
            batch_size * group_size,
            len(self.reward_funcs),
            device=device,
        )
        merged_stats = {}

        for reward_index, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                texts = [
                    prompt + "\n" + completion
                    for prompt, completion in zip(first_prompt_texts, completion_texts)
                ]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, reward_index] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {
                    key: []
                    for key in inputs[0].keys()
                    if key not in ["prompt", "completion"]
                }
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * group_size)

                reward_kwargs["trajectory_steps"] = trajectories
                reward_kwargs["trajectory_prompts"] = traj_prompts

                reward_stats = {}
                reward_kwargs["_reward_stats"] = reward_stats

                output_reward = reward_func(
                    prompts=first_prompt_texts,
                    completions=completion_texts,
                    **reward_kwargs,
                )

                if len(output_reward) != batch_size * group_size:
                    raise ValueError(
                        "Trajectory reward function must return one reward per sampled trajectory."
                    )

                rewards_per_func[:, reward_index] = torch.tensor(
                    output_reward,
                    dtype=torch.float32,
                    device=device,
                )

                if reward_stats:
                    fn_name = getattr(reward_func, "__name__", f"reward_func_{reward_index}")
                    for key, value in reward_stats.items():
                        merged_stats[f"{fn_name}/{key}"] = value

        rewards = rewards_per_func.sum(dim=1)
        rewards_grouped = rewards.view(batch_size, group_size)

        mean_grouped = rewards_grouped.mean(dim=1).repeat_interleave(group_size, dim=0)
        std_grouped = rewards_grouped.std(dim=1, unbiased=False).repeat_interleave(group_size, dim=0)

        # Paper-aligned group-relative advantage: A_i = R_i - mean_j R_j.
        advantages = rewards - mean_grouped

        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        total_trajectories = 0
        total_step_count = 0
        total_token_count = 0

        trajectory_index = 0
        for batch_index in range(batch_size):
            for _ in range(group_size):
                advantage = advantages[trajectory_index]
                steps_text = trajectories[trajectory_index]
                steps_prompt_text = traj_prompts[trajectory_index]

                trajectory_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
                trajectory_token_count = 0

                for prompt_text, step_text in zip(steps_prompt_text, steps_text):
                    prompt_inputs = self.processing_class(
                        text=[prompt_text],
                        images=[images[batch_index]],
                        return_tensors="pt",
                        padding=True,
                        padding_side="left",
                        add_special_tokens=False,
                    )
                    prompt_inputs = super()._prepare_inputs(prompt_inputs)

                    for key, value in list(prompt_inputs.items()):
                        if torch.is_tensor(value):
                            prompt_inputs[key] = value.to(device, non_blocking=True)

                    prompt_ids = prompt_inputs["input_ids"]
                    prompt_len = prompt_ids.size(1)

                    tokenizer = (
                        self.processing_class.tokenizer
                        if hasattr(self.processing_class, "tokenizer")
                        else self.processing_class
                    )
                    completion_ids = tokenizer(
                        step_text,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"].to(device)

                    if completion_ids.numel() == 0:
                        continue

                    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                    attention_mask = torch.ones_like(
                        input_ids,
                        device=device,
                        dtype=torch.long,
                    )

                    pixel_values = prompt_inputs["pixel_values"]
                    image_grid_thw = prompt_inputs["image_grid_thw"]

                    per_token_logps = self._get_per_token_logps_completion_only(
                        model,
                        input_ids,
                        attention_mask,
                        pixel_values,
                        image_grid_thw,
                        prompt_len,
                    )

                    with torch.inference_mode():
                        if self.ref_model is not None:
                            ref_per_token_logps = self._get_per_token_logps_completion_only(
                                self.ref_model,
                                input_ids,
                                attention_mask,
                                pixel_values,
                                image_grid_thw,
                                prompt_len,
                            )
                        else:
                            unwrapped = self.accelerator.unwrap_model(model)

                            if self._ref_adapters is not None:
                                with _temporary_adapters(unwrapped, self._ref_adapters):
                                    ref_per_token_logps = self._get_per_token_logps_completion_only(
                                        unwrapped,
                                        input_ids,
                                        attention_mask,
                                        pixel_values,
                                        image_grid_thw,
                                        prompt_len,
                                    )
                            elif hasattr(unwrapped, "disable_adapter"):
                                with unwrapped.disable_adapter():
                                    ref_per_token_logps = self._get_per_token_logps_completion_only(
                                        unwrapped,
                                        input_ids,
                                        attention_mask,
                                        pixel_values,
                                        image_grid_thw,
                                        prompt_len,
                                    )
                            else:
                                raise ValueError(
                                    "ref_model is None but no SFT reference adapter is configured."
                                )

                    per_token_kl = (
                        torch.exp(ref_per_token_logps - per_token_logps)
                        - (ref_per_token_logps - per_token_logps)
                        - 1
                    )

                    per_token_objective = (
                        torch.exp(per_token_logps - per_token_logps.detach()) * advantage
                    )
                    per_token_loss = -(per_token_objective - self.beta * per_token_kl)

                    trajectory_loss_sum = trajectory_loss_sum + per_token_loss.sum()
                    token_count = int(per_token_loss.numel())
                    trajectory_token_count += token_count
                    total_token_count += token_count
                    total_step_count += 1

                if trajectory_token_count > 0:
                    # Paper objective sums token contributions within each trajectory.
                    trajectory_loss = trajectory_loss_sum
                    total_loss = total_loss + trajectory_loss
                    total_trajectories += 1

                trajectory_index += 1

        if total_trajectories == 0:
            raise RuntimeError("No valid rollout tokens were available for GRPO loss computation.")

        loss = total_loss / total_trajectories

        self._metrics["reward"].append(
            float(self._reduce_mean_scalar(rewards.mean()).item())
        )
        self._metrics["reward_std"].append(
            float(self._reduce_mean_scalar(std_grouped.mean()).item())
        )
        self._metrics["rollout_avg_steps"].append(
            float(total_step_count) / max(total_trajectories, 1)
        )
        self._metrics["rollout_avg_tokens"].append(
            float(total_token_count) / max(total_trajectories, 1)
        )

        for key, value in merged_stats.items():
            try:
                self._metrics[f"reward_stats/{key}"].append(float(value))
            except Exception:
                pass

        return loss