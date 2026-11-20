from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

MAX_NEW_TOKENS = 64

MODEL_NAME = "Qwen/Qwen3-8B"

@dataclass
class Generation:
    answer: str
    prompt_tokens: int
    new_tokens: int
    truncated: bool

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(model_name: str = MODEL_NAME, device: str | None = None) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map=device or get_device())
    return model.eval(), tok

def _stop_ids(model: PreTrainedModel) -> set[int]:
    eos = model.generation_config.eos_token_id
    return set(eos) if isinstance(eos, list) else {eos}

@torch.inference_mode()
def generate_answer(model: PreTrainedModel, tok: PreTrainedTokenizerBase, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, **generate_kwargs: Any) -> Generation:
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    n_prompt = enc["input_ids"].shape[1]
    out = model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=max_new_tokens, **generate_kwargs)
    new = out[0, n_prompt:].tolist()
    stopped = bool(new) and new[-1] in _stop_ids(model)
    if stopped:
        new = new[:-1]
    return Generation(answer=tok.decode(new, skip_special_tokens=True).strip(), prompt_tokens=n_prompt, new_tokens=len(new), truncated=not stopped)