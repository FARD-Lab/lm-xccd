# ──────────────────────────────────────────────────────────────────────────
# async_classifier_inference.py
# ──────────────────────────────────────────────────────────────────────────

import os
import json
import torch
import pathlib
import argparse
from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import sigmoid

# Classifier head imports
import torch.nn as nn


current_location = pathlib.Path(__file__).parent.resolve()
upper_level_path = current_location.parent


local_models = {
    "binary": "clone_head_binary_ckpt",
    "contrastive": "clone_head_contrastive_ckpt"
}

# ============================================================================
# Classifier Head Model (MUST MATCH TRAINING DEFINITION)
# ============================================================================
class CloneHeadModel(nn.Module):
    def __init__(self, base_model, hidden_size: int, use_contrastive: bool = False):
        super().__init__()
        self.base_model = base_model
        self.use_contrastive = use_contrastive

        # Freeze LLM
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(hidden_size, hidden_size)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        hidden = outputs.hidden_states[-1]   # final layer
        mask = attention_mask.unsqueeze(-1)

        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        z = self.act(self.proj(pooled))
        z = self.dropout(z)
        logits = self.classifier(z).squeeze(-1)

        return logits


# ============================================================================
# Dataset-style prompt builder (same as your training)
# ============================================================================
def build_prompt(code1: str, code2: str) -> str:
    prompt = (
        "<|user|>\n"
        "Compare the following two code snippets with regard to:\n"
        "1. Functionality comparison\n"
        "2. Mathematical logic comparison\n"
        "3. Structural differences\n"
        "4. Similarity analysis\n"
        "5. Conclusion on clone status.\n\n"
        "In the conclusion, clearly state Yes for clones and No for non-clones.\n\n"
        "Code1:\n"
        f"{code1}\n\n"
        "Code2:\n"
        f"{code2}\n\n"
        "<|assistant|>\n"
    )
    return prompt



# ============================================================================
# Inference class — Mirrors your OfflineRequest but uses classifier head
# ============================================================================
class ClassifierInference:
    def __init__(self, ckpt_dir: str, device="cuda"):
        """
        ckpt_dir must contain:
            head.pt
            config.json   (with hidden_size, model_path)
            tokenizer files
        """
        self.device = device

        # Load config.json
        cfg_path = os.path.join(ckpt_dir, "config.json")
        with open(cfg_path, "r") as f:
            meta = json.load(f)

        model_path = meta["model_path"]
        hidden = meta["hidden_size"]

        print(f"Loading base model from: {model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)

        print(f"Loading tokenizer from checkpoint directory…")
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Building classifier model…")
        self.model = CloneHeadModel(base_model, hidden).to(device)

        print(f"Loading classifier weights: {ckpt_dir}/head.pt")
        state_dict = torch.load(os.path.join(ckpt_dir, "head.pt"), map_location=device)
        self.model.load_state_dict(state_dict, strict=False)
        hidden_dtype = base_model.dtype
        self.model.proj = self.model.proj.to(hidden_dtype)
        self.model.classifier = self.model.classifier.to(hidden_dtype)

        self.model.eval()

    # -------------------------------------------------------------
    def classify(self, prompt: str) -> str:
        """
        Returns "yes" or "no".
        """
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, mask)
            prob = sigmoid(logits).item()

        return "yes" if prob >= 0.5 else "no", prob

    # -------------------------------------------------------------
    def run(self, data_file: str, output_file: str):
        with open(data_file, "r") as f:
            data = [json.loads(l) for l in f]

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "a", encoding="utf-8") as fout:
            for row in data:
                sample_id = row["index"]
                code1 = row["code1"]
                code2 = row["code2"]

                prompt = build_prompt(code1, code2)
                label, prob = self.classify(prompt)

                entry = {
                    "idx": sample_id,
                    "text": f"|user|\n{prompt}\n|assistant|\n(Classifier inference)",
                    "final_conclusion": label,
                    "probability": prob,
                }

                json.dump(entry, fout, ensure_ascii=False)
                fout.write("\n")

                print(f"[{sample_id}] → {label} (p={prob:.4f})")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Binary classifier head inference")
    parser.add_argument("--model", required=True,
                        help="Directory containing head.pt and config.json")
    parser.add_argument("--datafile", required=True,
                        help="Input JSONL with index/code1/code2")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")

    args = parser.parse_args()

    inf = ClassifierInference(str(upper_level_path/"models" /local_models[args.model]))
    inf.run(
        os.path.join(current_location,args.datafile), 
        os.path.join(current_location, "offline_results", args.output)
    )


if __name__ == "__main__":
    main()