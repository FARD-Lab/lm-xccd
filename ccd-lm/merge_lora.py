#!/usr/bin/env python
# merge_lora.py
# Merge a base model with a LoRA adapter and save the result as a standalone model.

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora(
    base_model_id: str,
    lora_path: str,
    output_dir: str,
    dtype: torch.dtype = torch.float16,   # can be torch.bfloat16 or torch.float32
):
    """
    Load the base model and a LoRA adapter, fuse them, and save a single
    ‘normal’ model to `output_dir`.
    """
    # 1) Load the base model (on CPU to avoid GPU OOM during the merge)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="cpu",   # or "auto"
        torch_dtype=dtype,
    )

    # 2) Attach the LoRA adapter
    model = PeftModel.from_pretrained(base, lora_path)

    # 3) Merge LoRA weights into the base weights
    print("→ merging LoRA weights …")
    model = model.merge_and_unload()  # now `model` is a plain transformers model

    # 4) Save the merged model and tokenizer
    print(f"→ saving merged model to {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)  # .safetensors
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    tokenizer.save_pretrained(output_dir)

    print("✅ merge completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", required=True, help="Base model ID or local path")
    parser.add_argument("--lora", required=True, help="Path to LoRA checkpoint folder")
    parser.add_argument("--out",  required=True, help="Destination folder for merged model")
    args = parser.parse_args()

    merge_lora(args.base, args.lora, args.out)