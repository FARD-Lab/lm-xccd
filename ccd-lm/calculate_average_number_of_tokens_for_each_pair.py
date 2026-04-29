import json
import csv
from collections import defaultdict
from statistics import mean
from transformers import AutoTokenizer

# 🔹 Your local model mapping
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
    return data [8000:]


def compute_grouped_token_stats(data, tokenizer):
    grouped_counts = defaultdict(list)

    for item in data:
        pair_type = item.get("pair_type")
        code1 = item.get("code1", "")
        code2 = item.get("code2", "")

        if pair_type is None:
            pair_type = "Java-Python"

        combined_text = code1 + "\n" + code2

        token_ids = tokenizer.encode(
            combined_text,
            add_special_tokens=False
        )

        grouped_counts[pair_type].append(len(token_ids))

    stats = {}
    for pair_type in grouped_counts:
        counts = grouped_counts[pair_type]
        stats[pair_type] = {
            "samples": len(counts),
            "average_tokens": mean(counts)
        }

    return stats


def main():

    import os 
    import pathlib

    current_location =  pathlib.Path(__file__).parent.resolve()
    upper_level_path = current_location.parent

    jsonl_path = os.path.join(current_location, "extended-experiments/train-test-data/train_data_clones.jsonl")
    output_csv = os.path.join(current_location, "extended-experiments/train-test-data/", "token_stats_by_pair_type.csv")

    phi_path = upper_level_path/"models" / local_models["mini"]
    qwen_path = upper_level_path/"models" / local_models["qwen"]
    # 🔹 Load tokenizers from local directories
    phi3_tokenizer = AutoTokenizer.from_pretrained(
        phi_path,
        trust_remote_code=True
    )

    qwen_tokenizer = AutoTokenizer.from_pretrained(
        qwen_path,
        trust_remote_code=True
    )

    data = load_jsonl(jsonl_path)
    print(f"Loaded {len(data)} samples.")

    # Compute stats
    phi3_stats = compute_grouped_token_stats(data, phi3_tokenizer)
    qwen_stats = compute_grouped_token_stats(data, qwen_tokenizer)

    # Combine results and write CSV
    all_pair_types = sorted(set(phi3_stats.keys()) | set(qwen_stats.keys()))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "pair_type",
            "samples",
            "phi3_avg_tokens",
            "qwen_avg_tokens"
        ])

        for pair_type in all_pair_types:
            phi3 = phi3_stats.get(pair_type, {})
            qwen = qwen_stats.get(pair_type, {})

            writer.writerow([
                pair_type,
                phi3.get("samples", qwen.get("samples", 0)),
                f"{phi3.get('average_tokens', 0):.2f}",
                f"{qwen.get('average_tokens', 0):.2f}",
            ])

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()