# async_chatgpt.py
import os, json, pathlib, argparse
import torch
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

current_location = pathlib.Path(__file__).parent.resolve()

# ---------------------------------------------------------------------
# Load tiktoken offline
# ---------------------------------------------------------------------
upper_level_path = current_location.parent
tiktoken_cache_dir = upper_level_path / "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)
assert os.path.exists(os.path.join(tiktoken_cache_dir,
    "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

# ---------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# ---------------------------------------------------------------------
# Offline model wrapper
# ---------------------------------------------------------------------
class OfflineRequest:
    def __init__(self, model="Phi-3-medium-128k-instruct", stream=False):
        self.model_path = upper_level_path / "models" / model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.stream = stream

    def _send_request(self, prompt_id, prompt):
        torch.cuda.empty_cache()

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        input_len = inputs.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=1500,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                no_repeat_ngram_size=4,
                do_sample=True
            )

        response = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=False)
        print(f"[✓] Completed prompt {prompt_id}")
        return prompt_id, response

    @staticmethod
    def _post_process_response(result, prompt):
        sample_id, model_answer = result
        conversation = (
            f"|user|\n{prompt}\n|assistant|\n"
            f"{model_answer.strip()}"
        )
        return sample_id, conversation

    def process_prompts(self, prompts, output_file):
        for pid, prompt in prompts:
            result = self._send_request(pid, prompt)
            idx, text = self._post_process_response(result, prompt)

            with open(output_file, "a", encoding="utf-8") as fout:
                json.dump({"idx": idx, "text": text}, fout, ensure_ascii=False)
                fout.write("\n")

# ---------------------------------------------------------------------
# CCD Core Class
# ---------------------------------------------------------------------
class CodeCloneDetection:
    def __init__(self, data_file, output_file, model="Phi-3-medium-128k-instruct"):
        self.data_file = data_file
        self.output_file = output_file
        self.data = self._read_data(data_file)
        self.model = model

        # Auto-generate requested_ids list file
        self.req_ids_file = self._auto_ids_filename(data_file)

        if os.path.exists(output_file):
            existing = self._read_data(output_file)
            done_ids = {x["idx"] for x in existing}
        else:
            done_ids = set()

        # Build prompts for missing samples
        self.prompts = [
            self._make_prompt(x["index"], x["code1"], x["code2"])
            for x in self.data if x["index"] not in done_ids
        ]

        self.gpt = OfflineRequest(model=self.model)

    @staticmethod
    def _read_data(file_path):
        with open(file_path, "r") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _auto_ids_filename(datafile):
        """
        If datafile = 'python_java_clones.jsonl'
        return          'python_java_clones_ids.txt'
        """
        base = os.path.splitext(datafile)[0]
        return base + "_ids.txt"

    @staticmethod
    def _make_prompt(idx, code1, code2):
        user_prompt = (
            "Compare the following two code snippets with regard to:\n"
            "1. Functionality comparison\n"
            "2. Mathematical logic comparison\n"
            "3. Structural differences\n"
            "4. Similarity analysis\n"
            "5. Conclusion on clone status (codes may be in different languages).\n\n"
            "In the conclusion, clearly state Yes for code clones and No for non-clones "
            "before writing the rest of the conclusion.\n\n"
            "Do not include any explanation outside the JSON.\n\n"
            f"Code1:\n{code1}\n\nCode2:\n{code2}"
        )
        return idx, user_prompt

    def run(self):
        print(f"[INFO] Writing results to → {self.output_file}")
        self.gpt.process_prompts(self.prompts, self.output_file)
        print("[DONE] Processing complete.")

# ---------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file", required=True, help="Path to seed jsonl file")
    parser.add_argument("--output_file", required=True, help="Where to save results")
    parser.add_argument("--model", default="Phi-3-mini-128k-instruct")

    return parser.parse_args()

def main():
    args = parse_args()

    ccd = CodeCloneDetection(
        data_file=args.data_file,
        output_file=args.output_file,
        model=args.model
    )
    ccd.run()

if __name__ == "__main__":
    main()