# ─────────────────────────────────────────────────────────────────────────────
# batched_async_chatgpt.py — Optimized for H100 + 4-bit + FlashAttention-2
# ─────────────────────────────────────────────────────────────────────────────
import os, json, pathlib, torch
import argparse
from typing import List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------------------
current_location = pathlib.Path(__file__).parent.resolve()
upper_level_path = current_location.parent

system_prompt = (
    "You are a code analysis assistant for cross-language clone detection. "
    "Analyze the user's code snippets and respond *only* in JSON format."
)

local_models = {
    "mini-f2": "phi3-mini-v2-merged",
    "mini-f3": "phi3-mini-v3-merged",
    "mini": "Phi-3-mini-128k-instruct",
    "qwen": "qwen2.5-Coder-3B-Instruct",
}

# ---------------------------------------------------------------------
# QUANTIZATION CONFIG
# ---------------------------------------------------------------------
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# ---------------------------------------------------------------------
# MODEL WRAPPER (BATCHED)
# ---------------------------------------------------------------------
class OfflineRequest:
    def __init__(self, model="Phi-3-mini-128k-instruct", batch_size=4):
        self.model_path = upper_level_path / "models" / model
        self.batch_size = batch_size

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        # Load model with FlashAttention-2
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.model.eval()

    # -----------------------------------------------------------------
    # BATCHED GENERATION
    # -----------------------------------------------------------------
    @torch.no_grad()
    def generate_batch(self, batch: List[Tuple[int, str]]):
        """batch = [(id, prompt_text), ...]"""
        if not batch:
            return []

        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]

        # Tokenize batched
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16384,
        ).to("cuda", non_blocking=True)

        input_len = encoded["input_ids"].shape[1]

        # High-speed FlashAttention-based inference
        outputs = self.model.generate(
            **encoded,
            max_new_tokens=1500,
            temperature=0.5,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            use_cache=False,
            no_repeat_ngram_size=4,
        )

        # Decode only new tokens
        generated = outputs[:, input_len:]

        results = []
        for sid, gen in zip(ids, generated):
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
            print(f"[{sid}] {text}\n")
            results.append((sid, text))

        return results

    # -----------------------------------------------------------------
    # PROCESS ALL PROMPTS IN BATCHES
    # -----------------------------------------------------------------
    def process_prompts(self, prompts, output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "a", encoding="utf-8") as fout:
            for batch in batchify(prompts, self.batch_size):
                results = self.generate_batch(batch)
                for sid, response in results:
                    original_prompt = next(p[1] for p in batch if p[0] == sid)
                    entry = {
                        "idx": sid,
                        "text": f"|user|\n{original_prompt}\n|assistant|\n{response.strip()}"
                    }
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------
# FINETUNED VERSION
# ---------------------------------------------------------------------
class FineTunedModelInference(OfflineRequest):
    def __init__(self, lora_path, model="Phi-3-mini-128k-instruct", batch_size=4):
        base_path = upper_level_path / "models" / model
        peft_path = upper_level_path / "results" / lora_path

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Merge LoRA
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": "cpu"},
        )
        model = PeftModel.from_pretrained(base, peft_path).merge_and_unload()

        merged_dir = upper_level_path / "merged_model"
        merged_dir.mkdir(exist_ok=True)
        model.save_pretrained(merged_dir)

        # Load quantized merged model on GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.model.eval()
        self.batch_size = batch_size

# ---------------------------------------------------------------------
# CCD PIPELINE
# ---------------------------------------------------------------------
class CodeCloneDetection:
    def __init__(self, data_file, model, output_file, batch_size=4):
        self.model_name = model
        self.output_file = output_file
        self.data = self._read_jsonl(data_file)

        # Avoid recomputation
        done = set()
        if os.path.exists(output_file):
            existing = self._read_jsonl(output_file)
            done = {d["idx"] for d in existing}

        # Only missing samples
        self.prompts = [
            self._mk_prompt(d["index"], d["code1"], d["code2"])
            for d in self.data
            if d["index"] not in done
        ]

        self.gpt = OfflineRequest(model=model, batch_size=batch_size)

    @staticmethod
    def _read_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _mk_prompt(idx, c1, c2):
        text = (
            "Conduct code-clone detection using the following criteria:\n"
            "1. Functionality comparison\n"
            "2. Mathematical logic comparison\n"
            "3. Structural differences\n"
            "4. Similarity analysis\n"
            "5. Conclusion (Yes/No) in pure JSON.\n\n"
            f"Code1:\n{c1}\n\nCode2:\n{c2}"
        )
        return (idx, text)

    def run(self):
        self.gpt.process_prompts(self.prompts, self.output_file)

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datafile", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-b", "--batch", type=int, default=4)
    args = parser.parse_args()

    data_path = os.path.join(current_location, args.datafile)
    output_path = os.path.join(current_location, "offline_results", args.output)

    ccd = CodeCloneDetection(
        data_file=data_path,
        model=local_models[args.model],
        output_file=output_path,
        batch_size=args.batch,
    )
    ccd.run()

if __name__ == "__main__":
    main()