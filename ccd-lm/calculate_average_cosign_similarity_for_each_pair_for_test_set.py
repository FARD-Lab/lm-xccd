import json
import csv
import os
import pathlib
from collections import defaultdict
from statistics import mean, stdev

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
                print(f"Skipping invalid JSON in {path} on line {line_number}: {e}")
    return data


def parse_filename(filename):
    """
    Expected filenames like:
      test_same_Java_Python.jsonl
      test_different_Rust_Java.jsonl

    Returns:
      group_type: 'same' or 'different'
      lang_pair: 'Java-Python', 'Rust-Java', etc.
    """
    name = filename.removesuffix(".jsonl")
    parts = name.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename}")

    group_type = parts[-3]
    lang1 = parts[-2]
    lang2 = parts[-1]
    lang_pair = f"{lang1}-{lang2}"

    if group_type not in {"same", "different"}:
        raise ValueError(f"Unexpected group type in filename: {filename}")

    return group_type, lang_pair


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

    last_hidden_state = outputs.hidden_states[-1]
    embedding = mean_pool(last_hidden_state, inputs["attention_mask"])

    return embedding.squeeze(0)


def cosine_similarity_between_codes(code1, code2, tokenizer, model, device, max_length=2048):
    emb1 = get_embedding(code1, tokenizer, model, device, max_length=max_length)
    emb2 = get_embedding(code2, tokenizer, model, device, max_length=max_length)
    sim = F.cosine_similarity(emb1, emb2, dim=0).item()
    return sim


def safe_mean(values):
    return mean(values) if values else 0


def safe_std(values):
    return stdev(values) if len(values) > 1 else 0


def compute_similarity_from_files(data_dir, tokenizer, model, device, max_length=2048):
    """
    Returns stats in the form:
    {
        "Java-Python": {
            "same_samples": ...,
            "same_average_similarity": ...,
            "same_std_similarity": ...,
            "different_samples": ...,
            "different_average_similarity": ...,
            "different_std_similarity": ...,
            "combined_samples": ...,
            "combined_average_similarity": ...,
            "combined_std_similarity": ...
        },
        ...
    }
    """
    grouped_sims = defaultdict(lambda: {
        "same": [],
        "different": []
    })

    for filename in os.listdir(data_dir):
        if not filename.endswith(".jsonl"):
            continue

        file_path = os.path.join(data_dir, filename)

        try:
            group_type, lang_pair = parse_filename(filename)
        except ValueError as e:
            print(f"Skipping file: {e}")
            continue

        data = load_jsonl(file_path)
        print(f"Loaded {len(data)} samples from {filename}")

        for item in data:
            code1 = item.get("code1", "")
            code2 = item.get("code2", "")

            try:
                sim = cosine_similarity_between_codes(
                    code1, code2, tokenizer, model, device, max_length=max_length
                )
                grouped_sims[lang_pair][group_type].append(sim)
            except Exception as e:
                print(f"Skipping sample in {filename} due to error: {e}")
                continue

    stats = {}
    for lang_pair, sim_groups in grouped_sims.items():
        same_sims = sim_groups["same"]
        different_sims = sim_groups["different"]
        combined_sims = same_sims + different_sims

        stats[lang_pair] = {
            "same_samples": len(same_sims),
            "same_average_similarity": safe_mean(same_sims),
            "same_std_similarity": safe_std(same_sims),

            "different_samples": len(different_sims),
            "different_average_similarity": safe_mean(different_sims),
            "different_std_similarity": safe_std(different_sims),

            "combined_samples": len(combined_sims),
            "combined_average_similarity": safe_mean(combined_sims),
            "combined_std_similarity": safe_std(combined_sims),
        }

    return stats


def main():
    current_location = pathlib.Path(__file__).parent.resolve()
    upper_level_path = current_location.parent

    data_dir = os.path.join(current_location, "extended-experiments/test_file_feeds")
    output_csv = os.path.join(current_location, "cosine_similarity_by_language_pair_test_set.csv")

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

    phi_stats = compute_similarity_from_files(
        data_dir, phi_tokenizer, phi_model, device, max_length=2048
    )
    qwen_stats = compute_similarity_from_files(
        data_dir, qwen_tokenizer, qwen_model, device, max_length=2048
    )

    all_lang_pairs = sorted(set(phi_stats.keys()) | set(qwen_stats.keys()))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "language_pair",

            "same_samples",
            "phi3_same_avg_cosine_similarity",
            "phi3_same_std_cosine_similarity",
            "qwen_same_avg_cosine_similarity",
            "qwen_same_std_cosine_similarity",

            "different_samples",
            "phi3_different_avg_cosine_similarity",
            "phi3_different_std_cosine_similarity",
            "qwen_different_avg_cosine_similarity",
            "qwen_different_std_cosine_similarity",

            "combined_samples",
            "phi3_combined_avg_cosine_similarity",
            "phi3_combined_std_cosine_similarity",
            "qwen_combined_avg_cosine_similarity",
            "qwen_combined_std_cosine_similarity",
        ])

        for lang_pair in all_lang_pairs:
            phi = phi_stats.get(lang_pair, {})
            qwen = qwen_stats.get(lang_pair, {})

            writer.writerow([
                lang_pair,

                phi.get("same_samples", qwen.get("same_samples", 0)),
                f"{phi.get('same_average_similarity', 0):.6f}",
                f"{phi.get('same_std_similarity', 0):.6f}",
                f"{qwen.get('same_average_similarity', 0):.6f}",
                f"{qwen.get('same_std_similarity', 0):.6f}",

                phi.get("different_samples", qwen.get("different_samples", 0)),
                f"{phi.get('different_average_similarity', 0):.6f}",
                f"{phi.get('different_std_similarity', 0):.6f}",
                f"{qwen.get('different_average_similarity', 0):.6f}",
                f"{qwen.get('different_std_similarity', 0):.6f}",

                phi.get("combined_samples", qwen.get("combined_samples", 0)),
                f"{phi.get('combined_average_similarity', 0):.6f}",
                f"{phi.get('combined_std_similarity', 0):.6f}",
                f"{qwen.get('combined_average_similarity', 0):.6f}",
                f"{qwen.get('combined_std_similarity', 0):.6f}",
            ])

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()