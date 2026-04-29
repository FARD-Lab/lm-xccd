import pathlib

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

current_location = pathlib.Path(__file__).parent.resolve()
upper_level_path = current_location.parent

BASE_DIR    = upper_level_path/"models"/ "Phi-3-mini-128k-instruct"
# directory where Trainer.save_pretrained() stored the adapter   #
ADAPTER_DIR = upper_level_path / "results" / "mini-v5-fine-tuned" / "checkpoint-501"


tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, trust_remote_code=True)
# ensure there is a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_DIR,
    torch_dtype=torch.float16,
    device_map="auto",              # automatic GPU placement
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()                       # inference mode

def make_prompt(user_msg: str) -> str:
    """
    Create a Phi-3 chat prompt.
    *Leave out the final <|end|> so the model keeps generating.*
    """
    return f"<|user|>\n{user_msg}\n<|end|>\n<|assistant|>"

# ─────────────────────────── run inference ────────────────────
user_question = (
    "Explain fine-tuning as simple as you can"
)

with torch.no_grad():
    inputs = tokenizer(make_prompt(user_question), return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\n=== Assistant answer ===\n")
print(response)