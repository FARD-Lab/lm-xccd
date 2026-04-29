#!/usr/bin/env python

"""
Unified offline LoRA fine-tuning script for:
- Phi-3 (mini / medium / small)
- Qwen2.5-Coder-3B-Instruct ("qcode")

Reads models from a local ./models directory ONLY (no internet).
Works on Compute Canada GPU nodes (Narval, etc.).
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
)
from peft import LoraConfig, get_peft_model, TaskType

# -------------------------------------------------------------------
# 0. Force HF offline mode (important on Compute Canada)
# -------------------------------------------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)

# -------------------------------------------------------------------
# 1. Paths, data sources, and model registry
# -------------------------------------------------------------------
DATA_SOURCES = {
    "v1": "finetuning_data_pure_reasoning_v1.jsonl",
    "v2": "finetuning_data_add_full_response_v2.jsonl",
    "v3": "finetuning_data_add_conclusion_response_v3.jsonl",
    "v4": "finetuning_data_add_full_response_simple_prompt_v4.jsonl",
    "v5": "finetuning_data_add_conclusion_response_simple_prompt_v5.jsonl",
    "allpairs": "new_finetuning_data_v2_old_reconstructed.jsonl"
}

MODELS = {
    "mini":   "Phi-3-mini-128k-instruct",
    "medium": "Phi-3-medium-128k-instruct",
    "small":  "Phi-3-small-128k-instruct",
    "qcode":  "qwen2.5-Coder-3B-Instruct",
    "qwen-f2": "qwen-coder-pyjava-merged"

}

current_location = pathlib.Path(__file__).parent.resolve()
upper_level_path = current_location.parent

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
MODEL_ROOT   = upper_level_path / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
SYSTEM_PROMPT = "You are a helpful coding assistant."
DATA_DIR      = PROJECT_ROOT / "results" 

def is_qwen(model_name: str) -> bool:
    return "qwen" in model_name.lower()


# -------------------------------------------------------------------
# 2. Helper to find all linear layers (generic for Phi/Qwen)
# -------------------------------------------------------------------
def find_all_linear_names(model: nn.Module):
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]


# -------------------------------------------------------------------
# 3. Custom data collator (pads input_ids, labels, attention_mask)
# -------------------------------------------------------------------
def make_smart_collator(tokenizer):
    def collator(features):
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        labels = []
        attention_mask = []

        for f in features:
            ids = f["input_ids"]
            lbl = f["labels"]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
            labels.append(lbl + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    return collator


# -------------------------------------------------------------------
# 4. Main
# -------------------------------------------------------------------
def main():
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
        "-o", "--output", required=True,
        help="Run-name / output folder under results/."
    )
    args = parser.parse_args()

    # -----------------------------
    # Data
    # -----------------------------
    data_path = RESULTS_DIR / DATA_SOURCES[args.data_source]
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    raw_dataset = load_dataset("json", data_files=str(data_path))["train"]
    dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = dataset["train"], dataset["test"]
    logging.info("Loaded %d train / %d val samples", len(train_ds), len(val_ds))

    # -----------------------------
    # Local model path (no internet)
    # -----------------------------
    model_name = MODELS[args.model]
    model_dir = MODEL_ROOT / model_name
    if not model_dir.exists():
        raise RuntimeError(
            f"Local model directory not found: {model_dir}\n"
            f"Make sure you downloaded the model to this path."
        )

    logging.info(f"Loading model from local directory: {model_dir}")

    # -----------------------------
    # Tokenizer (local only)
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    tokenizer.model_max_length = 4096
    tokenizer.padding_side = "right"

    # -----------------------------
    # Preprocessing
    # -----------------------------
    def preprocess(example):
        text = example["text"]
        if "<|assistant|>" not in text:
            return None

        user_part, answer_part = text.split("<|assistant|>", 1)

        if is_qwen(model_name):
            # QWEN ChatML format (full LM loss on all tokens)
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
            return {"input_ids": ids, "labels": ids[:]}

        else:
            # Phi-style prompt: mask the prompt, train only on answer
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
            labels = [-100] * len(prompt_ids) + answer_ids
            return {"input_ids": input_ids, "labels": labels}

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(preprocess,   remove_columns=val_ds.column_names)

    train_ds = train_ds.filter(lambda x: x is not None)
    val_ds   = val_ds.filter(lambda x: x is not None)

    logging.info(
        "After preprocessing: %d train / %d val examples",
        len(train_ds), len(val_ds)
    )

    # -----------------------------
    # Model (local only, FlashAttention2)
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        local_files_only=True,
        use_cache=False,
    )

    model.to(device)

    # -----------------------------
    # LoRA config
    # -----------------------------
    target_modules = find_all_linear_names(model)

    logging.info(f"LoRA will be applied to {len(target_modules)} linear submodules.")

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

    # -----------------------------
    # Training arguments
    # -----------------------------
    output_dir = RESULTS_DIR / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

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
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        seed=42,
        report_to="none",
    )

    # -----------------------------
    # Trainer with custom collator
    # -----------------------------
    data_collator = make_smart_collator(tokenizer)

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
    logging.info("Training complete. Model saved to %s", output_dir / "final")


if __name__ == "__main__":
    main()