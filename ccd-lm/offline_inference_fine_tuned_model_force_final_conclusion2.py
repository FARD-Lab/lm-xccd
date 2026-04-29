# ──────────────────────────────────────────────────────────────────────────
# async_chatgpt.py
# ──────────────────────────────────────────────────────────────────────────
import os
import json
import pathlib
import argparse
from typing import List, Tuple, Any

import asyncio  # kept for future async extensions
import aiohttp  # kept for future async extensions
import tiktoken  # offline tokenizer cache validation

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from analyse_force_conclusion import Analyser  # currently unused here, but kept

# ---------------------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------------------
current_location = pathlib.Path(__file__).parent.resolve()

system_prompt = (
    "You are a code analysis assistant for cross-language clone detection. "
    "Analyze the user's code snippets and respond *only* in JSON format."
)

# ---------------------------------------------------------------------
# Loading tiktoken offline for phi3-small
# ---------------------------------------------------------------------
upper_level_path = current_location.parent
tiktoken_cache_dir = upper_level_path / "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

# validate cache (as in your original code)
assert os.path.exists(
    os.path.join(
        tiktoken_cache_dir,
        "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
    )
)

local_models = {
    "mini-f2": "phi3-mini-v2-merged",
    "mini-f3": "phi3-mini-v3-merged",
    "mini": "Phi-3-mini-128k-instruct",
    "qwen": "qwen2.5-Coder-3B-Instruct",
}

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


# ---------------------------------------------------------------------
# OfflineRequest: base inference class
# ---------------------------------------------------------------------
class OfflineRequest:
    def __init__(self, model="Phi-3-medium-128k-instruct", stream=False):
        self.model_path = upper_level_path / "models" / model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.stream = stream
        self.results = []

        self.model.to("cuda")
        self.model.eval()

    # ---------------------------------------------------------------
    # Compute probability of a target word as the next token
    # (Option A: compare next-token probabilities for "yes" vs "no")
    # ---------------------------------------------------------------
    def _compute_word_probability(self, text: str, target_word: str) -> float:
        """
        Compute the probability of `target_word` as the NEXT token
        after the given text.

        We:
        - run the model on `text`
        - take logits for the last position
        - compute softmax over vocab
        - extract probability of the token corresponding to " target_word"
        """
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        enc = {k: v.to("cuda") for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits[:, -1, :]  # last token logits
            probs = F.softmax(logits, dim=-1)

        # Try encoding " yes" / " no" as a single token; if multiple, use last token ID.
        token_ids = self.tokenizer.encode(" " + target_word, add_special_tokens=False)
        if len(token_ids) == 0:
            # fallback: try without leading space
            token_ids = self.tokenizer.encode(target_word, add_special_tokens=False)

        token_id = token_ids[-1]
        return probs[0, token_id].item()

    # ---------------------------------------------------------------
    # Force final conclusion based on probabilities of yes/no
    # ---------------------------------------------------------------
    def _force_final_conclusion(self, full_response: str) -> str:
        """
        Second step:
        - Prompt the model with the *analysis* (`full_response`)
        - Ask it conceptually to decide Yes/No
        - But instead of sampling/generating, we compute:
              P(next_token = " yes") vs P(next_token = " no")
        - Return "yes" or "no" depending on which is higher.
        """

        second_prompt = (
            "Based on the following analysis, determine the final clone conclusion.\n"
            "You must decide whether the two codes are clones.\n"
            "Do not repeat the analysis. Just think internally and decide.\n"
            "We will infer your answer from the probabilities of the next token.\n\n"
            f"{full_response}\n\n"
            "Final Answer (Yes or No):"
        )

        p_yes = self._compute_word_probability(second_prompt, "yes")
        p_no = self._compute_word_probability(second_prompt, "no")

        final_label = "yes" if p_yes >= p_no else "no"

        print(
            f"[DEBUG] Probabilities given analysis:\n"
            f"  P(yes) = {p_yes:.6f}\n"
            f"  P(no)  = {p_no:.6f}\n"
            f"  => final_conclusion = {final_label}"
        )

        return final_label

    # ---------------------------------------------------------------
    # First-step reasoning generation
    # ---------------------------------------------------------------
    def _send_request(self, prompt_id: int, prompt: str):
        """
        First pass:
        - Generate full reasoning answer (JSON or explanation)
        Second pass:
        - Compute probabilities for "yes" vs "no" as next token
          on a separate classification prompt, and pick the higher one.
        """
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16384,
        )
        encoded = {k: v.to("cuda") for k, v in encoded.items()}

        input_len = encoded["input_ids"].shape[1]

        # FIRST PASS – generate full reasoning
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=3000,  # long enough for detailed reasoning
                temperature=0.5,
                top_k=50,
                top_p=0.9,
                no_repeat_ngram_size=4,
                do_sample=True,
                use_cache=True,
            )

        full_response = self.tokenizer.decode(
            outputs[0, input_len:],
            skip_special_tokens=False,
        )

        # print(f"[{prompt_id}] FIRST RESPONSE:\n{full_response}\n")

        # SECOND PASS – force Yes/No via next-token probabilities
        final_conclusion = self._force_final_conclusion(full_response)
        print(f"[{prompt_id}] FINAL FORCED CONCLUSION = {final_conclusion}\n")

        # Return all pieces
        return prompt_id, full_response, final_conclusion

    def process_prompts(self, prompts, requested_samples_file, output_file):
        """
        Iterate over prompts, run the two-step inference, and append results
        to the given JSONL output file.

        Each line of the output has:
            {
              "idx": <id>,
              "text": "<|user|...|assistant|...full reasoning...>",
              "final_conclusion": "yes" | "no"
            }
        """
        for prompt in prompts:
            result = self._send_request(prompt[0], prompt[1])
            sample_id, conversation = self._post_process_response(result, prompt)
            final_conclusion = result[2]

            entry = {
                "idx": sample_id,
                "text": conversation,
                "final_conclusion": final_conclusion,
            }

            with open(output_file, "a", encoding="utf-8") as fout:
                json.dump(entry, fout, ensure_ascii=False)
                fout.write("\n")

    @staticmethod
    def _post_process_response(result: tuple, prompt: Tuple[int, str]) -> Tuple[int, str]:
        """
        result = (id, full_response, final_conclusion)
        prompt = (id, user_prompt)
        """
        sample_id = result[0]
        full_response = result[1]

        conversation = (
            f"|user|\n{prompt[1]}\n|assistant|\n"
            f"{full_response.strip()}"
        )

        return sample_id, conversation


# ---------------------------------------------------------------------
# Fine-tuned model inference wrapper (LoRA merged)
# ---------------------------------------------------------------------
class FineTunedModelInference(OfflineRequest):
    # Complete inheritance, but override model loading
    def __init__(self, lora_path, model="Phi-3-medium-128k-instruct", stream=False):
        # Do not call parent __init__ model loading; we override the sequence
        self.model_path = upper_level_path / "models" / model
        self.peft_path = upper_level_path / "results" / lora_path

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Load LoRA and merge
        peft_model = PeftModel.from_pretrained(base_model, self.peft_path)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained("merged_model")

        # Reload merged model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            "merged_model",
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.model.to("cuda")
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.stream = stream
        self.results = []


# ---------------------------------------------------------------------
# CodeCloneDetection: dataset + prompt building + orchestration
# ---------------------------------------------------------------------
class CodeCloneDetection:
    def __init__(
        self,
        data_file: str,
        temperature: float = 0.7,
        model: str = "Phi-3-medium-128k-instruct",
        output_file: str | None = None,
    ):
        self.model = model
        self.data_file = data_file
        self.data = self._read_data(data_file)
        self.output_file = output_file

        requested_indices = None

        # If output_file exists, collect already-processed indices to skip
        if self.output_file and os.path.exists(self.output_file) and self.output_file.endswith(".jsonl"):
            output_data = self._read_data(self.output_file)
            requested_indices = [d["idx"] for d in output_data if "idx" in d]

        if requested_indices is None:
            self.prompts = [
                self._make_prompt(d["index"], d["code1"], d["code2"]) for d in self.data
            ]
        else:
            self.prompts = [
                self._make_prompt(d["index"], d["code1"], d["code2"])
                for d in self.data
                if d["index"] not in requested_indices
            ]

        self.gpt = OfflineRequest(model=self.model, stream=False)

    # ---------- data helpers -----------------------------------------
    @staticmethod
    def _read_data(data_file: str) -> List[dict]:
        with open(data_file, "r") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _make_prompt(sample_id: int, code1: str, code2: str) -> Tuple[int, str]:
        user_prompt = (
            "Conduct code-clone detection using the following criteria:\n"
            "Compare the following two code snippets with regard to:\n"
            "1. Functionality comparison\n"
            "2. Mathematical logic comparison\n"
            "3. Structural differences\n"
            "4. Similarity analysis\n"
            "5. Conclusion on clone status (codes may be in different languages).\n\n"
            "In the conclusion, clearly state Yes for code clones and No for non-clones.\n"
            "Do not include any explanation outside the JSON.\n\n"
            f"Code1:\n{code1}\n\nCode2:\n{code2}"
        )
        return sample_id, user_prompt

    # ---------- main entry -------------------------------------------
    def run_processing(self, requested_samples_file: str, output_file: str):
        self.output_file = output_file
        self.gpt.process_prompts(self.prompts, requested_samples_file, output_file)
        return self


class CodeCloneDetectionFineTuned(CodeCloneDetection):
    def __init__(
        self,
        lora_path: str,
        data_file: str,
        temperature: float = 0.7,
        model: str = "Phi-3-medium-128k-instruct",
        output_file: str | None = None,
    ):
        super().__init__(data_file, temperature, model, output_file)
        self.gpt = FineTunedModelInference(lora_path, model=self.model, stream=False)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    checkpoint_versions = {
        "v2": "mini-v2-fine-tuned/checkpoint-3340",
    }

    parser = argparse.ArgumentParser(
        description="Offline code-clone detection inference with two-stage reasoning + forced Yes/No."
    )

    parser.add_argument(
        "-d",
        "--datafile",
        help="Path to the JSONL input dataset",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Key of the model to be loaded (from local_models dict)",
        choices=list(local_models.keys()),
        required=True,
    )
    # parser.add_argument(
    #     "-o",
    #     "--output",
    #     help="Name of the inference output results file (stored in offline_results/)",
    #     required=True,
    # )
    parser.add_argument(
        "-f",
        "--finetuned",
        action="store_true",
        help="Use fine-tuned LoRA-merged model instead of base one",
    )
    parser.add_argument(
        "--checkpoint",
        help="Fine-tuned checkpoint key (for --finetuned). Must be one of: "
             + ", ".join(checkpoint_versions.keys()),
        choices=list(checkpoint_versions.keys()),
    )

    args = parser.parse_args()

    data_file_path = os.path.join(current_location, args.datafile)
    output = os.path.join(current_location, f"{args.datafile}_{args.model}_force_conclusion_inference_result.jsonl")

    # output = os.path.join(current_location, "offline_results", args.output)
    data_stem = os.path.splitext(data_file_path)[0]
    ids_filename = f"{data_stem}_ids.txt"

    print(
        f"Code input info:\n"
        f"  output path: {output}\n"
        f"  data path:   {data_file_path}\n"
        f"  ids file:    {ids_filename}"
    )

    if args.finetuned:
        if not args.checkpoint:
            raise ValueError("You must provide --checkpoint when using --finetuned.")
        print("Running fine-tuned version")
        ccd = CodeCloneDetectionFineTuned(
            lora_path=checkpoint_versions[args.checkpoint],
            model="Phi-3-mini-128k-instruct",
            data_file=data_file_path,
            output_file=output,
        )
    else:
        print("Running original version")
        ccd = CodeCloneDetection(
            model=local_models[args.model],
            data_file=data_file_path,
            output_file=output,
        )

    # Note: fixed typo "offlone_results" -> "offline_results"
    # requested_samples_file_path = os.path.join(
    #     current_location, "offline_results", ids_filename
    # )
    ccd.run_processing(
        requested_samples_file=os.path.join(
            current_location, "offlone_results", ids_filename
        ),
        output_file= output
    )

    analyser = Analyser(
        data_file_path,
        output
    )
    metric_description_name = args.datafile.split("/")[-1]
    analyser.compute_metrics(
        output_dir= os.path.join(current_location, "extended-experiments/test_files"),
        description=f"{metric_description_name}_force_conclusion_{args.model}_evaluation_result",
        save_to_file=True
    )
    analyser.compute_missing_samples(type=data_file_path)
    # analyser.compute_missing_samples(type=data_file_path)


if __name__ == "__main__":
    main()