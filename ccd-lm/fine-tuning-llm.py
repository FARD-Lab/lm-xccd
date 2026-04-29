import os
import pathlib
import argparse
import logging
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# Set up logging to console
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
print("Done!")

DATA_SOURCES = {
    "v1":"finetuning_data_pure_reasoning_v1.jsonl",
    "v2":"finetuning_data_add_full_response_v2.jsonl",
    "v3":"finetuning_data_add_conclusion_response_v3.jsonl",
    "v4":"finetuning_data_add_full_response_simple_prompt_v4.jsonl",
    "v5":"finetuning_data_add_conclusion_response_simple_prompt_v5.jsonl",
    "allpairs": "new_finetuning_data_v2_old_reconstructed.jsonl"
}
MODELS = {
    "mini":"Phi-3-mini-128k-instruct",
    "medium":"Phi-3-medium-128k-instruct",
    "small": "Phi-3-small-128k-instruct"
}
current_location = pathlib.Path(__file__).parent.resolve()
upper_level_path = current_location.parent


def main():
    parser = argparse.ArgumentParser(
        description="Simple example showing argparse usage"
    )
    parser.add_argument("-d","--data-source", help="Path to the input data")
    parser.add_argument("-m","--model", help="The model to be fine-tuned")
    parser.add_argument("-o","--output", help="The output directory to save the trained model")
    parser.add_argument("-b","--base_dir", default="results", help="The base directory of the training data")

    
    args = parser.parse_args()
    logging.info(f"the data source is {args.data_source} the model is {args.model}, and the output is {args.output}")

    data_path = current_location / args.base_dir / DATA_SOURCES[args.data_source]

    # Load the dataset from the JSONL file
    dataset = load_dataset("json", data_files=str(data_path))["train"]

    # Split into training and validation sets (e.g., 90% train, 10% val)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    logging.info(f"Loaded dataset with {len(train_dataset)} training examples and {len(val_dataset)} validation examples.")


    model_path = upper_level_path/"models" / MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # If the tokenizer has no pad token (which is common for GPT-style models), define one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token  # use the 'unknown' token as padding [oai_citation:3‡exnrt.com](https://exnrt.com/blog/ai/finetune-phi3-128k-instruct/#:~:text=,padding_side%20%3D%20%27right)
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.model_max_length = 4096  # limit to 2048 tokens for training to avoid very long sequences

    def preprocess_example(example):
        text = example["text"]
        # Split into prompt and answer parts
        if "<|assistant|>" not in text:
            return None  # skip if format is unexpected

        prompt_part, answer_part = text.split("<|assistant|>")
        prompt_part = "Conduct code-clone detection using the following criteria:\n" + prompt_part + "<|assistant|>"  # include the assistant token as end of prompt
        # Tokenize prompt and answer separately (no special bos/eos tokens added)
        prompt_ids = tokenizer(prompt_part, truncation=True, max_length=tokenizer.model_max_length, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_part, truncation=True, max_length=tokenizer.model_max_length, add_special_tokens=False)["input_ids"]

        # Create labels, masking the prompt tokens with -100
        labels = [-100] * len(prompt_ids) + answer_ids[:]
        input_ids = prompt_ids + answer_ids[:]
        return {"input_ids": input_ids, "labels": labels}

    # Apply preprocessing to the train and validation sets
    train_dataset = train_dataset.map(preprocess_example, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_example, remove_columns=val_dataset.column_names)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16, 
        use_cache=False, 
        trust_remote_code=True
    )

    def find_all_linear_names(model):
        return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    linear_layers = find_all_linear_names(base_model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=linear_layers,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    output_dir = os.path.join("results", args.output)  # directory to save model checkpoints and final model

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        logging_steps=10,
        logging_strategy="steps",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        seed=42,
        report_to="none"  # disable default logging to wandb or others for simplicity
    )


    # Define a data collator to pad inputs and labels in each batch
    def data_collator(batch):
        # batch is a list of examples, each a dict with 'input_ids' and 'labels'
        # Find max length in this batch
        max_length = max(len(ex["input_ids"]) for ex in batch)
        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []
        for ex in batch:
            seq = ex["input_ids"]
            lbl = ex["labels"]
            # Pad sequences
            padding_length = max_length - len(seq)
            input_ids_batch.append(seq + [tokenizer.pad_token_id] * padding_length)
            # Pad labels with -100
            labels_batch.append(lbl + [-100] * padding_length)
            # Attention mask (1 for real tokens, 0 for padded)
            attention_mask_batch.append([1] * len(seq) + [0] * padding_length)
        # Convert to tensors
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        return {"input_ids": input_ids_batch, "labels": labels_batch, "attention_mask": attention_mask_batch}

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,         # attach tokenizer (helps with model internals like auto-padding if needed)
        data_collator=data_collator  # our custom collator for padding
    )

    # Start fine-tuning
    train_result = trainer.train()

if __name__ == "__main__":
    main()