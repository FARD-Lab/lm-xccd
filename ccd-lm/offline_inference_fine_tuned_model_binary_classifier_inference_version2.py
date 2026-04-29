# ──────────────────────────────────────────────────────────────────────────
# async_classifier_inference.py
# ──────────────────────────────────────────────────────────────────────────

import os
import json
import torch
import pathlib
import argparse
from typing import Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import sigmoid
import torch.nn as nn


from analyse_force_conclusion import Analyser  # currently unused here, but kept



# ---------------------------------------------------------------------
# GLOBALS / PATHS
# ---------------------------------------------------------------------
current_location = pathlib.Path(__file__).parent.resolve()
upper_level_path = current_location.parent

local_models = {
    "binary": "clone_head_binary_ckpt",
    "contrastive": "clone_head_contrastive_ckpt",
    "binary-f2": "clone_head_binary_fine_tuned_ckpt",
    "contrastive-f2": "clone_head_binary_fine_tuned_ckpt",
    "bqwen": "clone_head_binary_qwen_ckpt",
    "cqwen": "clone_head_contrastive__qwen_ckpt",
    "bqwen-f": "clone_head_binary_qwen_fine_tuned_ckpt",
    "cqwen-f": "clone_head_contrastive_qwen_fine_tuned_ckpt"
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

        hidden = outputs.hidden_states[-1]  # final layer
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
# Inference class — Uses classifier head
# ============================================================================
class ClassifierInference:
    def __init__(self, ckpt_dir: str, device: str = "cuda"):
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

        print("Loading tokenizer from checkpoint directory…")
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Building classifier model…")
        self.model = CloneHeadModel(base_model, hidden).to(device)

        print(f"Loading classifier weights: {ckpt_dir}/head.pt")
        state_dict = torch.load(os.path.join(ckpt_dir, "head.pt"), map_location=device)
        self.model.load_state_dict(state_dict, strict=False)

        # Ensure head layers match base dtype
        hidden_dtype = base_model.dtype
        self.model.proj = self.model.proj.to(hidden_dtype)
        self.model.classifier = self.model.classifier.to(hidden_dtype)

        self.model.eval()

    # -------------------------------------------------------------
    def classify(self, prompt: str) -> Tuple[str, float]:
        """
        Returns ("yes"|"no", probability)
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

        return ("yes" if prob >= 0.5 else "no"), prob

    # -------------------------------------------------------------
    def run(self, data_file: str, output_file: str):
        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(l) for l in f]

        # Make sure output directory exists
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
# MAIN (output path auto-generated like your async_chatgpt.py)
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Binary classifier head inference")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(local_models.keys()),
        help="Which classifier checkpoint to use (binary|contrastive).",
    )
    parser.add_argument(
        "--datafile",
        required=True,
        help="Path (relative to this script directory) to input JSONL with index/code1/code2.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (cuda or cpu). Default: cuda",
    )

    args = parser.parse_args()

    # Input data path (same pattern as your reference code)
    data_file_path = os.path.join(current_location, args.datafile)

    # Output path auto-generated based on input data path and model (similar to your async_chatgpt.py)
    # NOTE: if args.datafile includes subfolders (e.g., data/foo.jsonl),
    # this will create subfolders under current_location automatically.
    output = os.path.join(
        current_location,
        f"{args.datafile}_{args.model}_classifier_inference_result.jsonl",
    )

    # Ensure the output parent dirs exist (important when args.datafile contains subfolders)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # Model checkpoint directory
    ckpt_dir = str(upper_level_path / "models" / local_models[args.model])

    print(
        f"Classifier inference info:\n"
        f"  data_path : {data_file_path}\n"
        f"  model_ckpt : {ckpt_dir}\n"
        f"  output     : {output}\n"
        f"  device     : {args.device}\n"
    )

    inf = ClassifierInference(ckpt_dir, device=args.device)
    inf.run(data_file_path, output)

    analyser = Analyser(
        data_file_path,
        output
    )
    metric_description_name = args.datafile.split("/")[-1]
    analyser.compute_metrics(
        output_dir= os.path.join(current_location, "extended-experiments/test_files"),
        description=f"{metric_description_name}_binary_classifier_{args.model}_evaluation_result",
        save_to_file=True
    )
    analyser.compute_missing_samples(type=data_file_path)


if __name__ == "__main__":
    main()