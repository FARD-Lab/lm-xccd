import argparse, json, pathlib, statistics
from transformers import AutoTokenizer

def read_data(data_file):
        with open(data_file, "r") as f:
            return [json.loads(line) for line in f]


if __name__ == "__main__":
    current_location = pathlib.Path(__file__).parent.resolve()
    upper_level_path = current_location.parent


    data_path = "results/finetuning_data_add_full_response_v2.jsonl"
    model_path = upper_level_path/"models" / "Phi-3-mini-128k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,  # required for Phi-3
    )

    # 2️⃣ iterate over JSONL, count tokens
    lengths = []
    data = read_data(data_path)
    
    token_length = [len(tokenizer.encode(d["text"], add_special_tokens=False)) for d in data]
    import pudb;pu.db
    assert 1 == 1
    # lengths.append(len(token_ids))
            
    # avg_len = statistics.mean(lengths) if lengths else 0
    # print(f"Samples:      {len(lengths)}")
    # print(f"Avg tokens:   {avg_len:.2f}")
    # print(f"Min tokens:   {min(lengths)}")
    # print(f"Max tokens:   {max(lengths)}")
    