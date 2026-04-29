import os
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Any

import requests

# ---------------------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------------------
current_location = pathlib.Path(__file__).parent.resolve()

system_prompt = (
    "You are a code analysis assistant for cross-language clone detection. "
    "Analyze the user's code snippets and respond *only* in JSON format."
)

# ---------------------------------------------------------------------
# API WRAPPER WITH CONCURRENCY
# ---------------------------------------------------------------------
class ChatGPTRequest:
    def __init__(
        self,
        temperature: float = 0.3,
        model: str = "deepseek-reasoner",
        max_concurrent: int = 10,               # <── choose your level of parallelism
    ):
        self.temperature = temperature
        self.model = model
        self.max_concurrent = max_concurrent
        self.api_key = self._read_api_key()
        self.url = "https://api.deepseek.com/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    # ----- helpers ----------------------------------------------------
    def _read_api_key(self) -> str:
        with open("/Users/MK/workspace/key.txt", "r") as file:
            return file.readline().strip()

    def _single_call(
        self, prompt_id: Any, prompt: str
    ) -> Tuple[Any, str, dict]:
        """
        ONE blocking HTTP request. Runs inside a thread.
        Returns (prompt_id, prompt, response_json or {'error':...})
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(
                self.url, headers=self.headers, json=payload, timeout=60
            )
            print(f"[{prompt_id}] HTTP {resp.status_code}")
            resp.raise_for_status()
            return prompt_id, prompt, resp.json()
        except requests.RequestException as e:
            # keep shape consistent, capture the exception text
            print(f"[{prompt_id}] ERROR → {e}")
            return prompt_id, prompt, {"error": str(e)}

    # ----- public workhorse ------------------------------------------
    def process_prompts(
        self,
        prompts: List[Tuple[int, str]],
        requested_samples_file: str,
        output_file: str,
    ) -> None:
        """
        Process prompts in rounds of 6 concurrent requests,
        wait 4 s between rounds, and append results to file.
        """
        import time
        from itertools import islice

        # 1️⃣ Already–requested ids
        requested_ids = self._get_requested_ids(requested_samples_file)

        # 2️⃣  Build list of (id, prompt) still needing work
        to_request: List[Tuple[int, str]] = []
        with open(requested_samples_file, "a") as f_seen:
            for sid, prm in prompts:
                if sid in requested_ids:
                    continue
                to_request.append((sid, prm))
                f_seen.write(f"{sid}\n")          # mark as requested

        # if not to_request:
        #     print("Nothing new to request.")
        #     return

        # 3️⃣  Helper to iterate in chunks of 6
        def batched(iterable, n=6):
            it = iter(iterable)
            while True:
                batch = list(islice(it, n))
                if not batch:
                    break
                yield batch

        # 4️⃣  Process each batch
        for batch in batched(to_request, n=10):
            results = []
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
                futs = [pool.submit(self._single_call, pid, prm) for pid, prm in batch]
                for fut in as_completed(futs):
                    results.append(fut.result())

            # ----- write results for this batch -----
            for sample_id, prompt, result in results:
                if not isinstance(result, dict) or "choices" not in result:
                    continue  # skip failures / malformed

                conversation = self._post_process_response(result, prompt)

                entry = {"idx": sample_id, "text": conversation}
                reasoning_entry = {
                    "idx": sample_id,
                    "reasoning": result["choices"][0]["message"]
                                .get("reasoning_content", "")
                                .strip(),
                }

                with open(output_file, "a", encoding="utf-8") as fout:
                    json.dump(entry, fout, ensure_ascii=False)
                    fout.write("\n")

                results_path = os.path.join(
                    current_location, "results", "python_java_results_deepseek_reasoning.jsonl"
                )
                with open(results_path, "a", encoding="utf-8") as fr:
                    json.dump(reasoning_entry, fr, ensure_ascii=False)
                    fr.write("\n")

            # 5️⃣  Pause 4 seconds before next round
            if len(batch) == 10:
                print(f"Processed {len(batch)} prompts, waiting 15 seconds...")
                time.sleep(6)

    # ----- small utils ------------------------------------------------
    @staticmethod
    def _post_process_response(result: dict, prompt: str) -> str:
        return (
            f"|user|\n{prompt}\n|assistant|\n"
            f"{result['choices'][0]['message']['content'].strip()}"
        )

    @staticmethod
    def _get_requested_ids(file_name: str) -> List[int]:
        if not os.path.exists(file_name):
            return []
        with open(file_name, "r") as file:
            return [int(line.strip()) for line in file]

# ---------------------------------------------------------------------
# DATA-SIDE GLUE (unchanged except for concurrency param pass-through)
# ---------------------------------------------------------------------
class CodeCloneDetection:
    def __init__(
        self,
        data_file: str,
        temperature: float = 0.3,
        model: str = "deepseek-reasoner",
        max_concurrent: int = 10,
        output_file: str = None
    ):
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
        
        self.gpt = ChatGPTRequest(
            temperature=temperature, model=model, max_concurrent=max_concurrent
        )

    # ---------- data helpers -----------------------------------------
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
            "Provide the answer as a JSON object with keys "
            '"functionality_comparison", "mathematical_logic_comparison", '
            '"structural_differences", "similarity_analysis", and "conclusion". '
            "In the conclusion, clearly state Yes for code clones and No for non-clones "
            "before writing the rest of the conclusion.\n\n"
            "Do not include any explanation outside the JSON.\n\n"
            f"Code1:\n{code1}\n\nCode2:\n{code2}"
        )
        return sample_id, user_prompt

    # ---------- main entry -------------------------------------------
    def run_processing(self, requested_file: str, output_file: str):
        self.gpt.process_prompts(self.prompts, requested_file, output_file)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    output = os.path.join(current_location, "results", "python_java_results_deepseek.jsonl")
    ccd = CodeCloneDetection(
        data_file=os.path.join(current_location, "python_java_clones.jsonl"),
        max_concurrent=6,  # <── adjust to suit your bandwidth / quota
        output_file= output
    )

    ccd.run_processing(
        requested_file=os.path.join(
            current_location, "results", "python_java_request_ids.txt"
        ),
        output_file= output
    )


if __name__ == "__main__":
    main()