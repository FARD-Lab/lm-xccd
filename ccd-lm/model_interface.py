from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Type
import pathlib

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling


current_location = pathlib.Path(__file__).parent.resolve()
upper_level_path = current_location.parent



PhiMODELS = {
    "mini":"Phi-3-mini-128k-instruct",
    "medium":"Phi-3-medium-128k-instruct",
    "small": "Phi-3-small-128k-instruct",
}
    
QWENMODELS = {
    "qcode": "qwen2.5-Coder-3B-Instruct"
}


class BaseModelInterface(ABC):
    """
    Base interface for LLM wrappers. Subclasses should implement `generate`.
    Use `BaseInterface.load_model(model_name, **kwargs)` to get the right one.
    """


    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model_path = upper_level_path/"models"
        self.config = kwargs  # store anything you passed (device, dtype, etc.)


    @classmethod
    def factory(cls, model_name):
        if model_name in QWENMODELS:
            return QwenInterface(model_name)
        elif model_name in PhiMODELS:
            return PhiInterface(model_name)
        
    def load_model(self, model_name: str, **kwargs):
        return NotImplementedError

    def load_tokenizer(self, model_name):
        return NotImplementedError
    

    def layers_to_train(self, model):
        return NotImplementedError
    
    # def get_model_path(self):
    #     return 


    def load_model_from_path(self, model_name):
        return AutoModelForCausalLM.from_pretrained(
            self.model_path/model_name,
            attn_implementation="sdpa", 
            torch_dtype=torch.float16,
            use_cache=False,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
    def _get_tokenizer(self, model_name, model_max_length=4096):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path/model_name, trust_remote_code=True)
        tokenizer.padding_side = "right"

        # Prefer EOS as PAD for chat LLMs (works well for Qwen and Phi)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        # Keep a reasonable cap; use tokenizer’s own limit if set
        if not hasattr(tokenizer, "model_max_length") or tokenizer.model_max_length > 32760:
            tokenizer.model_max_length = model_max_length

        return tokenizer


class PhiInterface(BaseModelInterface):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        

    def load_model(self, **kwargs):
        return self.load_model_from_path(PhiMODELS[self.model_name])

    def load_tokenizer(self, **kwargs):
        return self._get_tokenizer(PhiMODELS[self.model_name])
    
    def layers_to_train(self, model):
        return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]




class QwenInterface(BaseModelInterface):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

    def load_model(self, **kwargs):
        return self.load_model_from_path(QWENMODELS[self.model_name])
    
    def load_tokenizer(self, **kwargs):
        return self._get_tokenizer(QWENMODELS[self.model_name], 2048)
    
    def layers_to_train(self, model):
        wanted = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
        found = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                leaf = name.split(".")[-1]
                if leaf in wanted:
                    found.add(leaf)
        return sorted(found or wanted)



