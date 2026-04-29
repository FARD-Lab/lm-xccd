import json
import csv
import os
import pathlib
from collections import defaultdict
from statistics import mean

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Your local model mapping
local_models = {
    "mini": "Phi-3-mini-128k-instruct",
    "qwen": "qwen2.5-Coder-3B-Instruct",
}


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_number}: {e}")
    return data


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def get_embedding(text, tokenizer, model, device, max_length=2048):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    # For causal LMs, use the final hidden state
    last_hidden_state = outputs.hidden_states[-1]
    embedding = mean_pool(last_hidden_state, inputs["attention_mask"])

    return embedding.squeeze(0)


def cosine_similarity_between_codes(code1, code2, tokenizer, model, device, max_length=2048):
    emb1 = get_embedding(code1, tokenizer, model, device, max_length=max_length)
    emb2 = get_embedding(code2, tokenizer, model, device, max_length=max_length)
    sim = F.cosine_similarity(emb1, emb2, dim=0).item()
    return sim


def compute_grouped_similarity(data, tokenizer, model, device, max_length=2048):
    grouped_sims = defaultdict(list)

    for item in data:
        pair_type = item.get("pair_type")
        code1 = item.get("code1", "")
        code2 = item.get("code2", "")

        if pair_type is None:
            pair_type = "Java-Python"

        try:
            sim = cosine_similarity_between_codes(
                code1, code2, tokenizer, model, device, max_length=max_length
            )
            grouped_sims[pair_type].append(sim)
        except Exception as e:
            print(f"Skipping sample due to error: {e}")
            continue

    stats = {}
    for pair_type, sims in grouped_sims.items():
        stats[pair_type] = {
            "samples": len(sims),
            "average_similarity": mean(sims)
        }

    return stats


def main():
    current_location = pathlib.Path(__file__).parent.resolve()
    upper_level_path = current_location.parent

    jsonl_path = os.path.join(
        current_location,
        "extended-experiments/train-test-data/train_data_clones.jsonl"
    )
    output_csv = os.path.join(
        current_location,
        "extended-experiments/train-test-data/",
        "cosine_similarity_by_pair_type.csv"
    )

    phi_path = upper_level_path / "models" / local_models["mini"]
    qwen_path = upper_level_path / "models" / local_models["qwen"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Tokenizers
    phi_tokenizer = AutoTokenizer.from_pretrained(phi_path, trust_remote_code=True)
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)

    # Some causal LMs need a pad token set
    if phi_tokenizer.pad_token is None:
        phi_tokenizer.pad_token = phi_tokenizer.eos_token
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

    # Models
    phi_model = AutoModelForCausalLM.from_pretrained(
        phi_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    phi_model.eval()

    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwen_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    qwen_model.eval()

    data = load_jsonl(jsonl_path)
    print(f"Loaded {len(data)} samples.")

    phi_stats = compute_grouped_similarity(
        data, phi_tokenizer, phi_model, device, max_length=2048
    )
    qwen_stats = compute_grouped_similarity(
        data, qwen_tokenizer, qwen_model, device, max_length=2048
    )

    all_pair_types = sorted(set(phi_stats.keys()) | set(qwen_stats.keys()))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_type",
            "samples",
            "phi3_avg_cosine_similarity",
            "qwen_avg_cosine_similarity"
        ])

        for pair_type in all_pair_types:
            phi = phi_stats.get(pair_type, {})
            qwen = qwen_stats.get(pair_type, {})

            writer.writerow([
                pair_type,
                phi.get("samples", qwen.get("samples", 0)),
                f"{phi.get('average_similarity', 0):.6f}",
                f"{qwen.get('average_similarity', 0):.6f}",
            ])

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()