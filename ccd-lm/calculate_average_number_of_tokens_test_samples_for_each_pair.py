import json
import csv
from collections import defaultdict
from statistics import mean, stdev
from transformers import AutoTokenizer


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

    group_type = parts[-3]          # same / different
    lang1 = parts[-2]
    lang2 = parts[-1]
    lang_pair = f"{lang1}-{lang2}"

    if group_type not in {"same", "different"}:
        raise ValueError(f"Unexpected group type in filename: {filename}")

    return group_type, lang_pair


def compute_token_counts_from_files(data_dir, tokenizer):
    """
    Returns stats in the form:
    {
        "Java-Python": {
            "same_samples": ...,
            "same_average_tokens": ...,
            "different_samples": ...,
            "different_average_tokens": ...,
            "combined_samples": ...,
            "combined_average_tokens": ...
        },
        ...
    }
    """
    grouped_counts = defaultdict(lambda: {
        "same": [],
        "different": []
    })

    import os

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

            combined_text = code1 + "\n" + code2

            token_ids = tokenizer.encode(
                combined_text,
                add_special_tokens=False
            )

            grouped_counts[lang_pair][group_type].append(len(token_ids))

    stats = {}
    for lang_pair, pair_data in grouped_counts.items():
        same_counts = pair_data["same"]
        different_counts = pair_data["different"]
        combined_counts = same_counts + different_counts

        stats[lang_pair] = {
            "same_samples": len(same_counts),
            "same_average_tokens": mean(same_counts) if same_counts else 0,
            "same_standard_dev_tokens": stdev(same_counts) if same_counts else 0,
            "different_samples": len(different_counts),
            "different_average_tokens": mean(different_counts) if different_counts else 0,
            "different_standard_dev_tokens": stdev(different_counts) if same_counts else 0,
            "combined_samples": len(combined_counts),
            "combined_average_tokens": mean(combined_counts) if combined_counts else 0,
            "combined_standard_dev_tokens": stdev(combined_counts) if same_counts else 0,
        }

    return stats


def main():
    import os
    import pathlib

    current_location = pathlib.Path(__file__).parent.resolve()
    upper_level_path = current_location.parent

    data_dir = os.path.join(current_location, "extended-experiments/test_file_feeds")
    output_csv = os.path.join(current_location, "token_stats_by_language_pair_for_test_sets.csv")

    phi_path = upper_level_path / "models" / local_models["mini"]
    qwen_path = upper_level_path / "models" / local_models["qwen"]

    # Load tokenizers from local directories
    phi3_tokenizer = AutoTokenizer.from_pretrained(
        phi_path,
        trust_remote_code=True
    )

    qwen_tokenizer = AutoTokenizer.from_pretrained(
        qwen_path,
        trust_remote_code=True
    )

    # Compute stats
    phi3_stats = compute_token_counts_from_files(data_dir, phi3_tokenizer)
    qwen_stats = compute_token_counts_from_files(data_dir, qwen_tokenizer)

    all_lang_pairs = sorted(set(phi3_stats.keys()) | set(qwen_stats.keys()))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "language_pair",

            "same_samples",
            "phi3_same_avg_tokens",
            "qwen_same_avg_tokens",

            "different_samples",
            "phi3_different_avg_tokens",
            "qwen_different_avg_tokens",

            "combined_samples",
            "phi3_combined_avg_tokens",
            "qwen_combined_avg_tokens",
        ])

        for lang_pair in all_lang_pairs:
            phi3 = phi3_stats.get(lang_pair, {})
            qwen = qwen_stats.get(lang_pair, {})

            writer.writerow([
                lang_pair,

                phi3.get("same_samples", qwen.get("same_samples", 0)),
                f"{phi3.get('same_average_tokens', 0):.2f}",
                f"{qwen.get('same_average_tokens', 0):.2f}",

                phi3.get("different_samples", qwen.get("different_samples", 0)),
                f"{phi3.get('different_average_tokens', 0):.2f}",
                f"{qwen.get('different_average_tokens', 0):.2f}",

                phi3.get("combined_samples", qwen.get("combined_samples", 0)),
                f"{phi3.get('combined_average_tokens', 0):.2f}",
                f"{qwen.get('combined_average_tokens', 0):.2f}",
            ])

    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()