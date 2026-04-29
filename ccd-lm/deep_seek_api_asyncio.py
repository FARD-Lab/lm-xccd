# ──────────────────────────────────────────────────────────────────────────
# async_chatgpt.py
# ──────────────────────────────────────────────────────────────────────────
import os, json, pathlib, asyncio, aiohttp
from typing import List, Tuple, Any

from analyse_reasoning import Analyser

# ---------------------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------------------
current_location = pathlib.Path(__file__).parent.resolve()

system_prompt = (
    "You are a code analysis assistant for cross-language clone detection. "
    "Analyze the user's code snippets and respond *only* in JSON format."
)

# ---------------------------------------------------------------------
# API WRAPPER (aiohttp version)
# ---------------------------------------------------------------------
class ChatGPTRequest:
    def __init__(
        self,
        temperature: float = 0.3,
        model: str = "deepseek-reasoner",
        max_concurrent: int = 100,   # number of simultaneous HTTP posts
    ):
        self.temperature = temperature
        self.model        = model
        self.max_conc     = max_concurrent
        self.api_key      = self._read_api_key()
        self.url          = "https://api.deepseek.com/chat/completions"
        self.headers      = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    # ---------------------------------------------------------------
    def _read_api_key(self) -> str:
        with open("/Users/MK/workspace/key.txt") as f:
            return f.readline().strip()

    # ---------------------------------------------------------------
    async def _single_call(
        self, session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        prompt_id: Any, prompt: str
    ) -> Tuple[Any, str, dict]:
        """
        ONE non-blocking HTTP request.
        Returns (prompt_id, prompt, response_json | {'error': ...})
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "temperature": self.temperature,
        }

        async with sem:                      # limits concurrency
            try:
                async with session.post(
                    self.url, headers=self.headers, json=payload, timeout=60
                ) as resp:
                    print(f"[{prompt_id}] HTTP {resp.status}")
                    resp.raise_for_status()
                    data = await resp.json()
                    return prompt_id, prompt, data
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"[{prompt_id}] ERROR → {e}")
                return prompt_id, prompt, {"error": str(e)}

    # ---------------------------------------------------------------
    async def process_prompts_async(
        self,
        prompts: List[Tuple[int, str]],
        requested_samples_file: str,
        output_file: str,
    ) -> None:
        """
        Async version of process_prompts:
        • Sends prompts in batches of 10
        • Up to self.max_conc at a time
        • Sleeps 15 s after each batch
        """
        from itertools import islice

        # 1️⃣ already requested ids
        requested_ids = self._get_requested_ids(requested_samples_file)

        # 2️⃣ figure out which prompts to send
        to_request: List[Tuple[int, str]] = []
        with open(requested_samples_file, "a") as f_seen:
            for sid, prm in prompts:
                if sid in requested_ids:
                    continue
                to_request.append((sid, prm))
                f_seen.write(f"{sid}\n")          # mark as requested early

        # helper: yield chunks of n
        def batched(it, n=100):
            it = iter(it)
            while True:
                batch = list(islice(it, n))
                if not batch:
                    break
                yield batch

        # 3️⃣ run batches
        for batch in batched(to_request, n=100):
            sem   = asyncio.Semaphore(self.max_conc)
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._single_call(session, sem, pid, prm) for pid, prm in batch
                ]
                results = await asyncio.gather(*tasks)

            # 4️⃣ write results
            for sample_id, prompt, result in results:
                if not isinstance(result, dict) or "choices" not in result:
                    continue
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

                res_path = current_location / "results" / \
                           "python_java_results_deepseek_reasoning.jsonl"
                with open(res_path, "a", encoding="utf-8") as fr:
                    json.dump(reasoning_entry, fr, ensure_ascii=False)
                    fr.write("\n")

            if len(batch) == 100:
                print(f"Processed {len(batch)} prompts, waiting 15 s …")
                await asyncio.sleep(30)

    # ---------------------------------------------------------------
    @staticmethod
    def _post_process_response(result: dict, prompt: str) -> str:
        return (
            f"|user|\n{prompt}\n|assistant|\n"
            f"{result['choices'][0]['message']['content'].strip()}"
        )

    @staticmethod
    def _get_requested_ids(path: str):
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [int(x.strip()) for x in f]

class CodeCloneDetection:
    def __init__(
        self,
        data_file: str,
        temperature: float = 0.3,
        model: str = "deepseek-reasoner",
        max_concurrent: int = 100,
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
        asyncio.run(
            self.gpt.process_prompts_async(self.prompts, requested_file, output_file)
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    output = os.path.join(current_location, "results", "python_java_results_deepseek.jsonl")
    ccd = CodeCloneDetection(
        data_file=os.path.join(current_location, "python_java_clones.jsonl"),
        max_concurrent=100,  # <── adjust to suit your bandwidth / quota
        output_file= output
    )

    ccd.run_processing(
        requested_file=os.path.join(
            current_location, "results", "python_java_request_ids_async.txt"
        ),
        output_file= output
    )
    analyser = Analyser(
        os.path.join(current_location, 'java_test_clone_2.jsonl'),
        os.path.join(current_location, 'results', 'results_for_java2.txt')
    )


if __name__ == "__main__":
    main()