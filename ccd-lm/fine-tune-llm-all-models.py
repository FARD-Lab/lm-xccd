#!/usr/bin/env python
"""
Unified LoRA-fine-tuning script that works for **Phi-3** *and* **Qwen
(Chat / Instruct / Coder)** models out of the box.

Usage
-----
python finetune.py -d v2 -m qwen7b -o qwen7b-run1
"""

import os
import pathlib
import argparse
import logging
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# --------------------------------------------------------------------
# 0.  Logging
# --------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)

# --------------------------------------------------------------------
# 1.  Paths & model / data registries
# --------------------------------------------------------------------
DATA_SOURCES = {
    "v1": "finetuning_data_pure_reasoning_v1.jsonl",
    "v2": "finetuning_data_add_full_response_v2.jsonl",
    "v3": "finetuning_data_add_conclusion_response_v3.jsonl",
    "v4": "finetuning_data_add_full_response_simple_prompt_v4.jsonl",
    "v5": "finetuning_data_add_conclusion_response_simple_prompt_v5.jsonl",
}

MODELS = {
    "mini":"Phi-3-mini-128k-instruct",
    "medium":"Phi-3-medium-128k-instruct",
    "small": "Phi-3-small-128k-instruct",
    "qcode": "qwen2.5-Coder-3B-Instruct"

}
current_location = pathlib.Path(__file__).parent.resolve()
upper_level_path = current_location.parent

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
MODEL_CACHE   = PROJECT_ROOT / "models"     # optional local mirror
DATA_DIR      = PROJECT_ROOT / "results"    # where *.jsonl files live
SYSTEM_PROMPT = "You are a helpful coding assistant."  # used for Qwen ChatML

# --------------------------------------------------------------------
# 2.  Helpers
# --------------------------------------------------------------------
def is_qwen(model_name_or_path: str) -> bool:
    return "Qwen" in model_name_or_path or "qwen" in model_name_or_path.lower()


def find_all_linear_names(model: nn.Module):
    """Return every nn.Linear name (used for generic Phi LoRA)."""
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]


# --------------------------------------------------------------------
# 3.  Main
# --------------------------------------------------------------------
def main():
    # ----- CLI
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-source", required=True, choices=DATA_SOURCES,
        help="Key of DATA_SOURCES dict."
    )
    parser.add_argument(
        "-m", "--model", required=True, choices=MODELS,
        help="Key of MODELS dict."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Run-name / output folder."
    )
    args = parser.parse_args()

    # ----- Data
    data_path   = DATA_DIR / DATA_SOURCES[args.data_source]
    raw_dataset = load_dataset("json", data_files=str(data_path))["train"]
    dataset     = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = dataset["train"], dataset["test"]
    logging.info("Loaded %d train / %d val samples",
                 len(train_ds), len(val_ds))

    # ----- Tokenizer
    model_id   = MODELS[args.model]
    # cache_path = MODEL_CACHE / pathlib.Path(model_id).name
    # model_path = MODEL_CACHE / model_id 
    # tokenizer  = AutoTokenizer.from_pretrained(
    #     model_id,
    #     cache_dir=cache_path,
    #     trust_remote_code=True,
    # )

    model_path = upper_level_path/"models" / MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Pad-token logic (covers Phi & Qwen)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token if tokenizer.eos_token is not None
            else tokenizer.unk_token
        )
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
            tokenizer.pad_token
        )
    tokenizer.padding_side    = "right"
    tokenizer.model_max_length = 4096     # reasonable default

    # ----- Pre-processing (Phi format vs. Qwen ChatML)
    def preprocess(example):
        text = example["text"]
        if "<|assistant|>" not in text:
            return None  # skip malformed lines

        user_part, answer_part = text.split("<|assistant|>", 1)

        if is_qwen(model_id):
            chatml = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_part.strip()}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer_part.strip()}<|im_end|>"
            )
            ids = tokenizer(
                chatml,
                truncation=True,
                max_length=tokenizer.model_max_length,
                add_special_tokens=False,
            )["input_ids"]
            return {"input_ids": ids, "labels": ids[:]}  # full LM loss
        else:  # Phi-3 format
            prompt = (
                "Conduct code-clone detection using the following criteria:\n"
                + user_part.strip()
                + "<|assistant|>"
            )
            prompt_ids = tokenizer(
                prompt,
                truncation=True,
                max_length=tokenizer.model_max_length,
                add_special_tokens=False,
            )["input_ids"]
            answer_ids = tokenizer(
                answer_part,
                truncation=True,
                max_length=tokenizer.model_max_length,
                add_special_tokens=False,
            )["input_ids"]

            input_ids = prompt_ids + answer_ids
            labels    = [-100] * len(prompt_ids) + answer_ids
            return {"input_ids": input_ids, "labels": labels}

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(preprocess,   remove_columns=val_ds.column_names)

    # ----- Model
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     cache_dir=cache_path,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     rope_scaling={"type": "linear", "factor": 2.0} if is_qwen(model_id) else None,
    #     trust_remote_code=True,
    #     use_cache=False,
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16, 
        use_cache=False, 
        trust_remote_code=True
    )

    # ----- LoRA config (family-specific target layers)
    target_modules = (
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if is_qwen(model_id)
        else find_all_linear_names(model)
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ----- Training args
    output_dir = PROJECT_ROOT / "results" / args.output
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        logging_steps=10,
        logging_strategy="steps",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False,               # using bfloat16 already
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        seed=42,
        report_to="none",
    )

    # ----- Data collator
    # Works for both formats because labels are already aligned
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False
    )

    # ----- Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))


if __name__ == "__main__":
    main()