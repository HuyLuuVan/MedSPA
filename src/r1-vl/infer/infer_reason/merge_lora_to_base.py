#!/usr/bin/env python3

import os
import argparse
import shutil
import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from peft import PeftModel


PROCESSOR_FILES = [
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "image_processor_config.json",
    "feature_extractor_config.json",
    "chat_template.jinja",
    "chat_template.json",
]


def _copy_if_missing(src_dir: str, dst_dir: str, filename: str) -> bool:
    src = os.path.join(src_dir, filename)
    dst = os.path.join(dst_dir, filename)

    if os.path.isfile(src) and not os.path.isfile(dst):
        shutil.copy2(src, dst)
        print(f"[merge][copy] {filename} <- {src_dir}")
        return True

    return False


def ensure_processor_files(base_model: str, adapter_path: str, output_dir: str):
    copied_any = False

    if os.path.isdir(base_model):
        for filename in PROCESSOR_FILES:
            copied_any |= _copy_if_missing(base_model, output_dir, filename)

    if os.path.isdir(adapter_path):
        for filename in PROCESSOR_FILES:
            copied_any |= _copy_if_missing(adapter_path, output_dir, filename)

    if not copied_any:
        print("[merge][copy] No additional processor files copied.")


def merge_and_save(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    dtype_str: str = "auto",
):
    os.makedirs(output_dir, exist_ok=True)

    if dtype_str == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        dtype = getattr(torch, dtype_str, torch.float32)

    device_map = "auto" if torch.cuda.is_available() else None

    print(f"[merge] Loading base model: {base_model} (dtype={dtype})")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    print("[merge] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
        trust_remote_code=True,
    )

    print(f"[merge] Loading PEFT adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(
        model,
        adapter_path,
        device_map=device_map,
    )

    print("[merge] Merging LoRA adapter into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"[merge] Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    ensure_processor_files(
        base_model=base_model,
        adapter_path=adapter_path,
        output_dir=output_dir,
    )

    print(f"[merge] Done. Merged model ready at: {output_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        required=True,
        help="Base Hugging Face model name or local path",
    )
    parser.add_argument(
        "--adapter_path",
        required=True,
        help="Path to the LoRA adapter",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for the merged model",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Torch dtype: bfloat16, float16, float32, or auto",
    )

    args = parser.parse_args()

    merge_and_save(
        args.base_model,
        args.adapter_path,
        args.output_dir,
        args.dtype,
    )


if __name__ == "__main__":
    main()