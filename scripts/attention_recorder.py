from __future__ import annotations
import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

ATTN_NAME = "recorder"
query_rows: list[int] = []  # rows of the current forward input whose attention is recorded
recorded: dict[int, torch.Tensor] = {}  # layer -> (n_heads, n_rows, kv_len) float32 on the CPU

def record(rows: list[int]) -> None:
    """Set the rows of the next forward pass whose attention is recorded, and clear earlier records."""
    query_rows.clear()
    query_rows.extend(rows)
    recorded.clear()

def get_recorded() -> torch.Tensor:
    """Return the recorded attention of all layers as one (n_layers, n_heads, n_rows, kv_len) tensor."""
    layer_weights = []
    for layer in sorted(recorded):
        layer_weights.append(recorded[layer])
    return torch.stack(layer_weights)

def get_row_weights(module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor | None, scaling: float) -> torch.Tensor:
    """Return softmax(q k^T * scaling + mask) of the recorded rows for every query head, as float32 on the CPU."""
    q_len = query.shape[2]
    kv_len = key.shape[2]
    rows = torch.tensor(query_rows, device=query.device)

    # keys repeated to the query heads, as sdpa does for grouped KV heads
    key = repeat_kv(key, module.num_key_value_groups)
    scores = torch.matmul(query[:, :, rows], key.transpose(2, 3)) * scaling

    if attention_mask is None:
        # sdpa_mask returns None when the mask is purely causal, so the causal mask is built here:
        # row r is at position kv_len - q_len + r and sees the keys up to that position
        row_positions = kv_len - q_len + rows
        key_positions = torch.arange(kv_len, device=query.device)
        allowed = key_positions[None, :] <= row_positions[:, None]
        scores = scores.masked_fill(~allowed, float("-inf"))
    elif attention_mask.dtype == torch.bool:
        scores = scores.masked_fill(~attention_mask[:, :, rows], float("-inf"))
    else:
        scores = scores + attention_mask[:, :, rows]

    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
    return weights[0].cpu()

def recording_attention(module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None, dropout: float = 0.0, scaling: float | None = None, is_causal: bool | None = None, **kwargs) -> tuple[torch.Tensor, None]:
    """Return the output of sdpa attention unchanged, and record the attention of the rows in query_rows."""
    output = sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
    if len(query_rows) > 0:
        recorded[module.layer_idx] = get_row_weights(module, query, key, attention_mask, scaling)
    return output

# a new attention name also needs a mask function, otherwise transformers builds no mask at all
AttentionInterface.register(ATTN_NAME, recording_attention)
AttentionMaskInterface.register(ATTN_NAME, sdpa_mask)

def run_steps(model_name: str, dtype: torch.dtype, attn: str, input_ids: torch.Tensor, steps: list[tuple[str, int, int, list[int]]]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return the logits and the attention of the chosen rows for every step."""
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation=attn).eval()
    cache = DynamicCache()
    logits = []
    weights = []
    for name, start, end, rows in steps:
        if attn == ATTN_NAME:
            record(rows)
        with torch.inference_mode():
            out = model(input_ids=input_ids[:, start:end], past_key_values=cache, output_attentions=(attn == "eager"))
        logits.append(out.logits.float())

        if attn == ATTN_NAME:
            weights.append(get_recorded())
            record([])
        if attn == "eager":
            layer_weights = []
            for layer_attention in out.attentions:
                layer_weights.append(layer_attention[0][:, rows])
            weights.append(torch.stack(layer_weights).float())
    return logits, weights

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(model_name)
    question = "Buddy Hield was named the MVP of the 2015 Diamond Head Classic. Which team does the player named 2015 Diamond Head Classic's MVP play for?"
    messages = [{"role": "user", "content": question}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    n = input_ids.shape[1]
    m = n // 2

    # each step: name, first token, last token + 1, rows to record (relative to the step's input)
    steps = [
        ("prefill", 0, m, [0, 3, m - 1]),
        ("prefill with cache", m, n - 1, [0, n - 2 - m]),
        ("one decoding step", n - 1, n, [0]),
    ]
    print("prompt tokens:", n)

    for dtype in [torch.float32, torch.bfloat16]:
        recorder_logits, recorder_weights = run_steps(model_name, dtype, ATTN_NAME, input_ids, steps)
        eager_logits, eager_weights = run_steps(model_name, dtype, "eager", input_ids, steps)
        sdpa_logits, sdpa_weights = run_steps(model_name, dtype, "sdpa", input_ids, steps)
        print()
        print(dtype)
        for i in range(len(steps)):
            weight_diff = (recorder_weights[i] - eager_weights[i]).abs().max().item()
            logit_diff = (recorder_logits[i] - sdpa_logits[i]).abs().max().item()
            print(steps[i][0], "| recorded", tuple(recorder_weights[i].shape), "| max diff to eager weights", weight_diff, "| max diff to sdpa logits", logit_diff)