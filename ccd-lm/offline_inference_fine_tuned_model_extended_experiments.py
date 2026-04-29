# ──────────────────────────────────────────────────────────────────────────
# async_chatgpt.py
# ──────────────────────────────────────────────────────────────────────────
import os, json, pathlib, asyncio, aiohttp, tiktoken, torch
import argparse
from typing import List, Tuple, Any

from analyse_reasoning import Analyser
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from analyse_reasoning_new import Analyser

# ---------------------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------------------
current_location = pathlib.Path(__file__).parent.resolve()

system_prompt = (
    "You are a code analysis assistant for cross-language clone detection. "
    "Analyze the user's code snippets and respond *only* in JSON format."
)


# ---------------------------------------------------------------------
# Loading tiktoken ofline for phi3-small
# ---------------------------------------------------------------------
upper_level_path = current_location.parent
tiktoken_cache_dir = upper_level_path /"tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

# validate
assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))



local_models = {
    "mini-f2-allpairs": "mini-v2-fine-tuned-allpairs-merged",
    "mini-f2-pj": "mini-v2-fine-tuned-python-java-merged",
    "mini-f2": "phi3-mini-v2-merged", 
    "mini-f3":"phi3-mini-v3-merged",
    "mini":"Phi-3-mini-128k-instruct",
    "qwen":"qwen2.5-Coder-3B-Instruct",
    "qwen-f2": "qwen-coder-pyjava-merged"

}


# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_threshold=6.0,
#     llm_int8_skip_modules=None,
# )
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)


class OfflineRequest:
    def __init__(self,  model="Phi-3-medium-128k-instruct", stream=False):
        self.model_path = upper_level_path/"models" / model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config = quantization_config,    
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.stream = stream
        
        self.results = []

    def _send_request(self, prompt_id, prompt):
        def build_prompt_text(tokenizer, system_prompt, user_prompt):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Phi-3-style fallback
            return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

        prompt_text = build_prompt_text(self.tokenizer, system_prompt, prompt)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_max = getattr(self.tokenizer, "model_max_length", 16384)
        if model_max is None or model_max > 200_000:
            model_max = 16384
        max_length = model_max  # or min(model_max, 131072) if you want a cap

        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
        input_len = encoded["input_ids"].shape[1]

        # EOS / terminators
        eos_ids = [self.tokenizer.eos_token_id]
        for tok in ["<|end|>", "<|endoftext|>", "<|eot_id|>"]:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                eos_ids.append(tid)
        eos_ids = sorted({i for i in eos_ids if i is not None})

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=3000,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_ids,
                use_cache=True,
            )

        response = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        print(f"[{prompt_id}] {response}\n")
        return prompt_id, response
            
    def process_prompts(self, prompts, requested_samples_file, output_file):
        # requested_ids = self._get_requested_ids(requested_samples_file)
        for prompt in prompts:                
            result = self._send_request(prompt[0], prompt[1])
            sample_id, conversation = self._post_process_response(result, prompt)

            entry = {"idx": sample_id, "text": conversation}
            
            with open(output_file, "a", encoding="utf-8") as fout:
                json.dump(entry, fout, ensure_ascii=False)
                fout.write("\n")



    @staticmethod
    def _post_process_response(result: dict, prompt: str) -> str:
        return result[0], (
            f"|user|\n{prompt}\n|assistant|\n"
            f"{result[1].strip()}"
        )
    

class QwenOfflineRequest(OfflineRequest):
    print("Inferencing using Qwen Coder Model. Begins .......")
    def _send_request(self, prompt_id, prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16384,
        )
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
        input_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=3000,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,                            
            )

            response = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
            print(f"[{prompt_id}] {response}\n")
            return prompt_id, response


class FineTunedModelInference(OfflineRequest):
    # Complete inheritance
    def __init__(self, lora_path, model="Phi-3-medium-128k-instruct", stream=False):
        super().__init__(model=model, stream=stream)
        self.model_path = upper_level_path/"models" / model
        self.peft_path = upper_level_path / "results" / lora_path
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(self.base_model, self.peft_path)
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained("merged_model")
        self.model = AutoModelForCausalLM.from_pretrained(
            "merged_model",
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.eval()
    



class CodeCloneDetection:
    def __init__(
        self,
        data_file: str,
        temperature: float = 0.7,
        model: str = "Phi-3-medium-128k-instruct",
        output_file: str = None
    ):
        self.model = model
        self.data_file = data_file
        self.data = self._read_data(data_file)
        self.output_file = output_file 
        requested_indices = None
        # write a code to check if self.output_file exists and the extension is .jsonl. If the condition satisties, make a list of all all indexes
        if os.path.exists(self.output_file) and self.output_file.endswith(".jsonl"):
            output = self._read_data(self.output_file)
            requested_indices = [d["idx"] for d in output]

            
        if requested_indices is None:
            self.prompts = [
            self._make_prompt(d["index"], d["code1"], d["code2"]) for d in self.data
            ]
            assert 1==1
        else:
            self.prompts = [
                self._make_prompt(d["index"], d["code1"], d["code2"])
                for d in self.data if d["index"] not in requested_indices
            ]
        
        self.gpt = OfflineRequest(model=self.model, stream=False)


#     # ---------- data helpers -----------------------------------------
    @staticmethod
    def _read_data(data_file: str) -> List[dict]:
        with open(data_file, "r") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _make_prompt(sample_id: int, code1: str, code2: str) -> Tuple[int, str]:
        # user_prompt = (
        #     "Conduct code-clone detection using the following criteria:\n"
        #     "Compare the following two code snippets with regard to:\n"
        #     "1. Functionality comparison\n"
        #     "2. Mathematical logic comparison\n"
        #     "3. Similarity analysis\n"
        #     "4. Code written in different languages can still be clones, so do not take programming-language differences into account in your comparison\n"
        #     "4. Conclusion on clone status (codes may be in different languages).\n\n"
        #     "In the conclusion, clearly state Yes for code clones and No for non-clones "
        #     "before writing the rest of the conclusion.\n\n"
        #     "Do not include any explanation outside the JSON.\n\n"
        #     f"Code1:\n{code1}\n\nCode2:\n{code2}"
        # )
        user_prompt = (
            "Conduct code-clone detection using the following criteria:\n"
            "Compare the following two code snippets with regard to:\n"
            "1. Functionality comparison\n"
            "2. Mathematical logic comparison\n"
            "3. Structural differences\n"
            "4. Similarity analysis\n"
            "5. Conclusion on clone status (codes may be in different languages).\n\n"
            "In the conclusion, clearly state Yes for code clones and No for non-clones\n"
            "Do not include any explanation outside the JSON.\n\n"
            f"Code1:\n{code1}\n\nCode2:\n{code2}"
        )
        return sample_id, user_prompt

#     # ---------- main entry -------------------------------------------
#     def run_processing(self, requested_file: str, output_file: str):
#         asyncio.run(
#             self.gpt.process_prompts_async(self.prompts, requested_file, output_file)
#         )
    def run_processing(self, requested_samples_file, output_file):
        self.output_file = output_file
        self.gpt.process_prompts(self.prompts, requested_samples_file, output_file)
        return self


class CodeCloneDetectionFineTuned(CodeCloneDetection):
    def __init__(
        self,
        lora_path,
        data_file: str,
        temperature: float = 0.7,
        model: str = "Phi-3-medium-128k-instruct",
        output_file: str = None,
    ):
        super().__init__(data_file, temperature, model, output_file)
        self.gpt = FineTunedModelInference(lora_path, model=self.model, stream=False)


class CodeCloneDetectionQwenCoder(CodeCloneDetection):
    def __init__(
        self,
        data_file: str,
        temperature: float = 0.7,
        model: str = "qwen2.5-Coder-3B-Instruct",
        output_file: str = None,
    ):
        super().__init__(data_file, temperature, model, output_file)
        self.gpt = QwenOfflineRequest(model=self.model, stream=False)


# # ---------------------------------------------------------------------
# # MAIN
# # ---------------------------------------------------------------------
def main():

    checkpoint_versions = {
        "v2" : "mini-v2-fine-tuned/checkpoint-3340"
    }


    parser = argparse.ArgumentParser(
        description="Simple example showing argparse usage"
    )
    parser.add_argument(
        "-d", "--datafile",
        help="Path to the JSONL input dataset",
        required=True
    )
    parser.add_argument("-m","--model", help="Model to be loaded")
    # parser.add_argument("-o","--output", help="The inference output results")
    parser.add_argument("-f", "--finetuned", help="Choose normal model or fine-tuned one")

    
    args = parser.parse_args()

    # output = os.path.join(current_location, "offline_results", args.output)
    data_file_path = os.path.join(current_location, args.datafile)
    output = os.path.join(current_location, f"{args.datafile}_{args.model}_inference_result.jsonl")
    data_stem = os.path.splitext(data_file_path)[0]    
    ids_filename = f"{data_stem}_ids.txt" 

    print(f"code input infor: the output path:{output} \n data_path:{data_file_path} \n ids_file:{ids_filename}")
    if args.finetuned:
        print("Running fine-tuned version")
        ccd = CodeCloneDetectionFineTuned(
            lora_path=checkpoint_versions[args.checkpoint],
            model="Phi-3-mini-128k-instruct",
            data_file=data_file_path,
            output_file= output
        )
    elif "qwen" in args.model:
        print("Running Qwen Coder version")
        ccd = CodeCloneDetectionQwenCoder(
            model=local_models[args.model],
            data_file=data_file_path,
            output_file= output
        )
    else:
        print("Running original version")
        ccd = CodeCloneDetection(
            model=local_models[args.model],
            data_file=data_file_path,
            output_file= output
        )
    

    ccd.run_processing(
        requested_samples_file=os.path.join(
            current_location, "offlone_results", ids_filename
        ),
        output_file= output
    )
    # analyser = Analyser(
    #     os.path.join(current_location, 'java_test_clone_2.jsonl'),
    #     os.path.join(current_location, 'results', 'results_for_java2.txt')
    # )

    analyser = Analyser(
        data_file_path,
        output
    )
    metric_description_name = args.datafile.split("/")[-1]
    analyser.compute_metrics(
        ouput_dir= os.path.join(current_location, "extended-experiments/test_files"),
        description=f"{metric_description_name}_{args.model}_evaluation_result",
        save_to_file=True
    )
    analyser.compute_missing_samples(type=data_file_path)


if __name__ == "__main__":
    main()
