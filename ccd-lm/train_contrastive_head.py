import os
import json
import argparse
import pathlib

from tqdm import tqdm

from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup


current_location = pathlib.Path(__file__).parent.resolve()

# ---------------------------------------------------------------------
# Loading tiktoken ofline for phi3-small
# ---------------------------------------------------------------------
upper_level_path = current_location.parent
tiktoken_cache_dir = upper_level_path /"tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

print(tiktoken_cache_dir)
# validate
assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))


# ============================================================
# Local model name → directory mapping
# ============================================================

local_models = {
    "mini-f2": "phi3-mini-v2-merged",
    "mini-f3": "phi3-mini-v3-merged",
    "mini": "Phi-3-mini-128k-instruct",
    "qwen": "qwen2.5-Coder-3B-Instruct",
    "qf": "qwen-coder-pyjava-merged"
}


# ============================================================
# Config Structure
# ============================================================

@dataclass
class Config:
    model_path: str
    train_file: str
    val_file: Optional[str]
    max_length: int
    batch_size: int
    num_epochs: int
    lr: float
    warmup_ratio: float
    weight_decay: float
    use_contrastive: bool
    contrastive_weight: float
    temperature: float
    num_workers: int
    device: str


# ============================================================
# Dataset
# ============================================================
torch.cuda.empty_cache()
torch.cuda.synchronize()


class CloneDataset(Dataset):
    """
    Expected JSONL format per line:

    {
        "index": ...,
        "code1": "...",
        "code2": "...",
        "label": 0 or 1,
        ...
    }
    """

    def __init__(self, path: str, tokenizer: AutoTokenizer, max_length: int):
        self.samples: List[Dict] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading dataset: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples.\n")

    def __len__(self):
        return len(self.samples)

    def build_prompt(self, code1: str, code2: str) -> str:
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

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        prompt = self.build_prompt(s["code1"], s["code2"])

        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(int(s["label"]), dtype=torch.float),
        }


# ============================================================
# Frozen LLM + Trainable Head
# ============================================================

class CloneHeadModel(nn.Module):
    def __init__(self, base_model: AutoModelForCausalLM, hidden_size: int, use_contrastive: bool = False):
        super().__init__()
        self.base_model = base_model
        self.use_contrastive = use_contrastive

        # Freeze LLM weights
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

        hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1)

        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        z = self.act(self.proj(pooled))
        z = self.dropout(z) 
        logits = self.classifier(z).squeeze(-1)

        return logits, z


# ============================================================
# Contrastive Loss
# ============================================================

def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float):
    device = embeddings.device
    embeddings = nn.functional.normalize(embeddings, dim=1)

    sim = embeddings @ embeddings.T / temperature

    batch_size = labels.size(0)
    mask = torch.eye(batch_size, device=device).bool()
    sim = sim.masked_fill(mask, -1e9)

    labels = labels.view(-1, 1)
    matches = (labels == labels.T).float().masked_fill(mask, 0)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_log_prob = (log_prob * matches).sum(dim=1) / matches.sum(dim=1).clamp(min=1)

    return -pos_log_prob.mean()


# ============================================================
# Training / Eval
# ============================================================

def train_epoch(model, dataloader, optimizer, scheduler, cfg):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total = 0
    pbar = tqdm(dataloader, desc="Training", dynamic_ncols=True)

    for batch in pbar:
        input_ids = batch["input_ids"].to(cfg.device)
        mask = batch["attention_mask"].to(cfg.device)
        labels = batch["label"].to(cfg.device)

        optimizer.zero_grad()

        logits, emb = model(input_ids, mask)
        cls_loss = bce(logits, labels)

        loss = cls_loss
        if cfg.use_contrastive:
            c_loss = supervised_contrastive_loss(emb, labels, cfg.temperature)
            loss = loss + cfg.contrastive_weight * c_loss

        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scheduler.step()

        total += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total / len(dataloader)


@torch.no_grad()
def eval_epoch(model, dataloader, cfg):
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc="Evaluating", dynamic_ncols=True)

    for batch in pbar:
        input_ids = batch["input_ids"].to(cfg.device)
        mask = batch["attention_mask"].to(cfg.device)
        labels = batch["label"].to(cfg.device)

        logits, _ = model(input_ids, mask)
        loss = bce(logits, labels)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == labels).float().mean().item()

        total_loss += loss.item()
        correct += (preds == labels).sum().item()
        total += labels.numel()

        pbar.set_postfix(loss=loss.item(), acc=acc)

    return total_loss / len(dataloader), correct / total


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train binary head on frozen LLM")
    parser.add_argument("--model", type=str, required=True,
                        help="mini | mini-f2 | mini-f3 | qwen")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_path", type=str, default="clone_head_ckpt")
    parser.add_argument("--use_contrastive", type=bool, default=False)

    args = parser.parse_args()

    # Resolve model dir
    if args.model not in local_models:
        raise ValueError(f"Unknown model key: {args.model}")

    model_path = upper_level_path/"models" / local_models[args.model]
    print(f"Loading model: {args.model} → {model_path}\n")

    # Build config
    cfg = Config(
        model_path=model_path,
        train_file=args.train_file,
        val_file=args.val_file,
        max_length=2048,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.0,
        use_contrastive=args.use_contrastive,
        contrastive_weight=0.1,
        temperature=0.07,
        num_workers=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    ).to(cfg.device)

    hidden = base_model.config.hidden_size
    model = CloneHeadModel(base_model, hidden, cfg.use_contrastive).to(cfg.device)
    model.proj = model.proj.to(torch.bfloat16)
    model.classifier = model.classifier.to(torch.bfloat16)
    # Load data
    train_ds = CloneDataset(cfg.train_file, tokenizer, cfg.max_length)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers)

    val_dl = None
    if cfg.val_file:
        val_ds = CloneDataset(cfg.val_file, tokenizer, cfg.max_length)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)

    total_steps = len(train_dl) * cfg.num_epochs
    warm = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm, num_training_steps=total_steps
    )

    # Train
    print("Begin training")
    for e in range(1, cfg.num_epochs + 1):
        loss = train_epoch(model, train_dl, optimizer, scheduler, cfg)
        print(f"[Epoch {e}] Train Loss = {loss:.4f}")

        if val_dl:
            v_loss, v_acc = eval_epoch(model, val_dl, cfg)
            print(f"[Epoch {e}] Val Loss = {v_loss:.4f} | Val Acc = {v_acc:.4f}")

    # Save
    save_dir = str(upper_level_path/"models" /args.output_path)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "head.pt"))
    tokenizer.save_pretrained(save_dir)

    meta = {
        "hidden_size": hidden,
        "model_path": str(cfg.model_path),
        "use_contrastive": cfg.use_contrastive,
        "contrastive_weight": cfg.contrastive_weight,
        "temperature": cfg.temperature
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved classifier head + tokenizer → {save_dir}/\n")


if __name__ == "__main__":
    main()