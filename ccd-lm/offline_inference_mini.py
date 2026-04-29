# ──────────────────────────────────────────────────────────────────────────
# async_chatgpt.py
# ──────────────────────────────────────────────────────────────────────────
import os, json, pathlib, asyncio, aiohttp, tiktoken, torch
from typing import List, Tuple, Any

from analyse_reasoning import Analyser
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------------------
current_location = pathlib.Path(__file__).parent.resolve()

system_prompt = (
    "You are a code analysis assistant for cross-language clone detection. "
    "Analyze the user's code snippets and respond *only* in JSON format."
)


user_prompt ={ "v1": (
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
        ),
        "v2": (
            ""
        )
        
}

# ---------------------------------------------------------------------
# Loading tiktoken ofline for phi3-small
# ---------------------------------------------------------------------
upper_level_path = current_location.parent
tiktoken_cache_dir = upper_level_path /"tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_cache_dir)

# validate
assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))



quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # یا float16
    bnb_4bit_quant_type="nf4"
)


class OfflineRequest:
    def __init__(self,  model="Phi-3-medium-128k-instruct", stream=False):
        self.model_path = upper_level_path/"models" / model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config = quantization_config,
            device_map="auto",      
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.stream = stream
        
        self.results = []

    def _send_request(self, prompt_id, prompt):
        torch.cuda.empty_cache()
        pad_token_id = self.tokenizer.eos_token_id
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", padding=True)
        inputs = inputs.to('cuda')
        input_ids_length = inputs.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                max_new_tokens=1500, 
                temperature=0.7, 
                top_k=50,
                top_p=0.9, 
                no_repeat_ngram_size=4,
                do_sample=True )
            
        print(f"The request status for prompt id {prompt_id} is {self.tokenizer.decode(outputs[0, input_ids_length:], skip_special_tokens=False)}\n")
        return prompt_id, self.tokenizer.decode(outputs[0, input_ids_length:], skip_special_tokens=False)

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


# # ---------------------------------------------------------------------
# # MAIN
# # ---------------------------------------------------------------------
def main():
    output = os.path.join(current_location, "offline_results", "python_java_results.jsonl")
    ccd = CodeCloneDetection(
        model="Phi-3-mini-128k-instruct",
        data_file=os.path.join(current_location, "python_java_clones.jsonl"),
        output_file= output
    )
    

    ccd.run_processing(
        requested_samples_file=os.path.join(
            current_location, "offlone_results", "python_java_request_ids.txt"
        ),
        output_file= output
    )
    # analyser = Analyser(
    #     os.path.join(current_location, 'java_test_clone_2.jsonl'),
    #     os.path.join(current_location, 'results', 'results_for_java2.txt')
    # )


if __name__ == "__main__":
    main()