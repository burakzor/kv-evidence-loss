"""Answer the 169 HotpotQA questions with a compressed KV cache and record which tokens were kept.

Usage (Colab L4, after: pip install kvpress==0.5.5 transformers==5.2.0 pandas==2.2.3):
    python scripts/run_kvpress.py --method snapkv --budget 0.1 --limit 1   # one question, end to end
    python scripts/run_kvpress.py --method snapkv --budget 0.1             # all 169 questions

For each question:
    1. build the prompt exactly as the full-cache baseline does (prompts.build_prompt);
    2. prefill all prompt tokens except the last inside `with press(model):`, so the press compresses
       the cache of every layer;
    3. record, for every layer and KV head, which prompt positions are still in the cache;
    4. outside the press, feed the last prompt token and decode greedily from the compressed cache;
    5. score the answer exactly as the baseline does (scoring.score_answer);
    6. save the kept mask (.npz) and one row of the answers table, with memory and time.

The choices made here (why the last token is held back, the span checks, method settings, ...)
are explained in docs/runner_notes.md; the comments below refer to them as "notes, decision N".
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from packaging.version import Version
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel, PreTrainedTokenizerBase

# kvpress 0.5.5 only works with transformers>=4.56,<5.3. Fail early with a clear message instead of
# failing (or, worse, silently misbehaving) inside a press.
if Version(version("transformers")) >= Version("5.3"):
    raise SystemExit(f'transformers {version("transformers")} is installed; kvpress 0.5.5 needs <5.3: pip install "transformers<5.3"')

# Importing kvpress also patches transformers' attention functions so that masking presses work.
from kvpress import BasePress, ObservedAttentionPress, RandomPress, ScorerPress, SnapKVPress, StreamingLLMPress
from kvpress.utils import compute_n_kept

# The team's shared code: same model, prompt, stop rule, generation limit and scoring as the baseline.
from generation import MAX_NEW_TOKENS, MODEL_NAME, _stop_ids, get_device
from prompts import build_prompt, load_questions
from run_full_baseline import device_name, print_summary
from scoring import ROOT, load_aliases, score_answer

DATA_DIR = ROOT / "data" / "hotpotqa"
CLEAN_PATH = DATA_DIR / "hotpot_bridge_ab_clean.parquet"
SPANS_PATH = DATA_DIR / "hotpot_token_spans_qwen3_8b.parquet"
ANSWERS_DIR = ROOT / "results" / "answers"
KEPT_DIR = ROOT / "results" / "raw" / "kept"

RANDOM_SEED = 0

# Methods (notes, decision 5). Each entry: how to build the press for a given compression_ratio,
# the attention implementation it needs, and how its kept tokens are recorded:
#   "scorer": the press prunes the cache with scores.topk(...)  -> record the topk indices;
#   "mask":   the press keeps the cache and masks evicted keys  -> read module.masked_key_indices
#             (no method uses this now; kept for DuoAttention, see "Open" in the notes).
# All settings are kvpress defaults, written out so that the config file records them.
# AdaKV was dropped by the team; DuoAttention is waiting for a team decision.
METHODS = {
    # No compression, through this runner's prefill/decode path (the budget-1.0 check run). Not called "full":
    # its output would be full_1.0.parquet and overwrite the baseline it is compared against.
    "nopress": {"press": None, "attn": "sdpa", "kind": None},
    "snapkv": {"press": lambda r: SnapKVPress(compression_ratio=r, window_size=64, kernel_size=5), "attn": "sdpa", "kind": "scorer"},
    "streamingllm": {"press": lambda r: StreamingLLMPress(compression_ratio=r, n_sink=4), "attn": "sdpa", "kind": "scorer"},
    "random": {"press": lambda r: RandomPress(compression_ratio=r, seed=RANDOM_SEED), "attn": "sdpa", "kind": "scorer"},
    # Closest kvpress counterpart of H2O, not H2O itself; needs the attention weights, hence eager.
    "observed_attention": {"press": lambda r: ObservedAttentionPress(compression_ratio=r), "attn": "eager", "kind": "scorer"},
}


# ----------------------------------------------------------------------------------------------
# Budget -> kvpress compression_ratio
# ----------------------------------------------------------------------------------------------

def n_kept_total(n_tokens: int, budget: float) -> int:
    """Positions kept per head out of the whole prompt, the held-back last token included."""
    if budget >= 1.0:
        return n_tokens
    # round() guards against float noise such as 0.1 * 1380 = 137.99999...
    return max(int(round(n_tokens * budget, 6)), 2)


def press_ratio(n_tokens: int, budget: float) -> float:
    """compression_ratio that makes the press keep exactly n_kept_total - 1 of the n_tokens - 1 tokens it sees.

    The press only sees the first n_tokens - 1 tokens (the last one is held back, notes decision 1) and the
    held-back token counts as kept, so the press must keep one position less than the budget allows.
    kvpress keeps int(k_len * (1 - ratio)) tokens (kvpress.utils.compute_n_kept, and the same formula in
    StreamingLLMPress); aiming half a token above the target makes int() land exactly on it.
    """
    if budget >= 1.0:
        return 0.0
    k_len, target = n_tokens - 1, n_kept_total(n_tokens, budget) - 1
    ratio = 1 - (target + 0.5) / k_len
    assert compute_n_kept(k_len, ratio) == target, (n_tokens, budget, ratio)
    return ratio


# ----------------------------------------------------------------------------------------------
# Recording kept tokens (notes, decision 4)
# ----------------------------------------------------------------------------------------------

def attach_scorer_recorder(press: ScorerPress, store: dict[int, torch.Tensor]) -> None:
    """Make a scorer press also write its kept positions into store[layer_idx] as a (n_kv_heads, k_len) bool tensor.

    kvpress calls press.compress(...) once per layer, right after the layer's attention during prefill. The
    function below is ScorerPress.compress from kvpress 0.5.5 line by line, plus the two lines marked RECORD.
    """
    if type(press).compress is not ScorerPress.compress:
        # This press has its own compress(); the copy below would not be the code that normally runs.
        raise TypeError(f"{type(press).__name__} overrides compress(); kept tokens cannot be recorded this way")

    def compress(module, hidden_states, keys, values, attentions, kwargs):
        k_len = keys.shape[2]
        if press.compression_ratio == 0:
            store[module.layer_idx] = torch.ones(keys.shape[1], k_len, dtype=torch.bool)  # RECORD: everything kept
            return keys, values
        scores = press.score(module, hidden_states, keys, values, attentions, kwargs)  # (batch, n_kv_heads, k_len)
        n_kept = compute_n_kept(k_len, press.compression_ratio)
        indices = scores.topk(n_kept, dim=-1).indices  # the positions that stay in the cache
        store[module.layer_idx] = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)[0].cpu()  # RECORD
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values

    # An attribute on the instance takes precedence over the class method, so kvpress's hook
    # (BasePress.forward_hook calls self.compress) now runs the function above.
    press.compress = compress


def read_masked_kept(model: PreTrainedModel, n_kv_heads: int, k_len: int) -> list[torch.Tensor]:
    """Kept positions of a masking press (e.g. DuoAttention; none is in METHODS now), per layer, as (n_kv_heads, k_len) bool tensors.

    The cache is not pruned: kvpress stores the evicted positions in module.masked_key_indices as
    (batch_idx, head_idx, position) and hides those keys at decoding time (kvpress/attention_patch.py).
    It resets the attribute to None at the start of each prefill, so stale masks from an earlier question
    cannot leak in; None after prefill means nothing was evicted.
    """
    kept = []
    for layer in model.model.layers:
        mask = torch.ones(n_kv_heads, k_len, dtype=torch.bool)
        masked = getattr(layer.self_attn, "masked_key_indices", None)
        if masked is not None:
            _, heads, positions = masked
            mask[heads.cpu(), positions.cpu()] = False
        kept.append(mask)
    return kept


# ----------------------------------------------------------------------------------------------
# Prompt and span checks
# ----------------------------------------------------------------------------------------------

def load_spans() -> pd.DataFrame:
    """Spans file with the sentence text attached (for checking).

    token_spans.py builds the prompt with prompts.build_text, the same code the runner uses, so the spans
    index the run prompt directly (tok_end inclusive). check_spans verifies this for every question.
    """
    spans = pd.read_parquet(SPANS_PATH).reset_index()
    sents = pd.read_parquet(CLEAN_PATH).reset_index()[["id", "title", "sent_id", "sentences"]]
    spans = spans.merge(sents, on=["id", "title", "sent_id"], how="left", validate="one_to_one")
    return spans.set_index("id")


def check_spans(qid: str, input_ids: list[int], tok: PreTrainedTokenizerBase, spans: pd.DataFrame) -> None:
    """Stop the run if the prompt no longer lines up with the spans file (e.g. the prompt changed but the spans were not rebuilt)."""
    rows = spans.loc[[qid]]
    if len(input_ids) != rows["n_tokens"].iat[0]:
        raise RuntimeError(f"{qid}: prompt has {len(input_ids)} tokens, spans file expects {rows['n_tokens'].iat[0]}")
    for r in rows.itertuples():
        decoded = tok.decode(input_ids[r.tok_start : r.tok_end + 1])
        if r.sentences.strip() not in decoded:
            raise RuntimeError(f"{qid}: span {r.tok_start}-{r.tok_end} decodes to {decoded!r}, expected {r.sentences!r}")


def inspect_kept(qid: str, input_ids: list[int], kept: np.ndarray, layer: int, tok: PreTrainedTokenizerBase, spans: pd.DataFrame, head: int = 0) -> None:
    """Print the kept tokens of one layer and head, and how much of the A and B sentences survived there."""
    row = kept[layer, head]
    print(f"\n--- kept tokens, question {qid}, layer {layer}, head {head}: {row.sum()}/{len(row)} ---")
    pieces, run = [], []
    for pos, keep in enumerate(row):  # join consecutive kept tokens, mark gaps with " ┃ "
        if keep:
            run.append(input_ids[pos])
        elif run:
            pieces.append(tok.decode(run))
            run = []
    if run:
        pieces.append(tok.decode(run))
    print(" ┃ ".join(repr(p) for p in pieces))
    for tag in ("A", "B"):
        positions = [p for r in spans.loc[[qid]].itertuples() if r.sentence_tag == tag for p in range(r.tok_start, r.tok_end + 1)]
        all_heads = kept[:, :, positions].mean()
        print(f"{tag}: {row[positions].sum()}/{len(positions)} tokens kept in this layer/head; {all_heads:.1%} over all layers and heads")


# ----------------------------------------------------------------------------------------------
# One question
# ----------------------------------------------------------------------------------------------

def sync(device: torch.device) -> None:
    """Wait for the GPU to finish, so that the clock measures the work and not just the launch."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def answer_one(model: PreTrainedModel, tok: PreTrainedTokenizerBase, input_ids: torch.Tensor, press: BasePress | None, kind: str | None, store: dict[int, torch.Tensor], stop_ids: set[int], max_new_tokens: int) -> dict:
    device = model.device
    n = input_ids.shape[1]
    n_layers, n_kv_heads = model.config.num_hidden_layers, model.config.num_key_value_heads
    cache = DynamicCache()
    store.clear()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    t0 = time.perf_counter()

    # --- Prefill: all tokens except the last, with the press active (notes, decisions 1 and 3) -------------
    # Same call as kvpress's pipeline: the model body without the LM head, since no logits are needed here.
    # Each layer's hook compresses that layer's cache as soon as the layer has processed the prompt.
    with press(model) if press is not None else contextlib.nullcontext():
        model.model(input_ids=input_ids[:, :-1], past_key_values=cache)

    sync(device)
    t1 = time.perf_counter()

    # --- Kept mask over the whole prompt: (n_layers, n_kv_heads, n) -------------------------------------------
    if kind == "scorer":
        per_layer = [store[i] for i in range(n_layers)]
    elif kind == "mask":
        per_layer = read_masked_kept(model, n_kv_heads, n - 1)
    else:  # no press: everything kept
        per_layer = [torch.ones(n_kv_heads, n - 1, dtype=torch.bool)] * n_layers
    kept = torch.stack(per_layer).numpy()
    kept = np.concatenate([kept, np.ones((n_layers, n_kv_heads, 1), dtype=bool)], axis=2)  # held-back token: kept
    kv_cache_bytes = sum(layer.keys.nbytes + layer.values.nbytes for layer in cache.layers)

    # --- Decode, press inactive: greedy, from the compressed cache ---------------------------------------------
    # Explicit position_ids as in kvpress's pipeline: after pruning, the cache is shorter than the prompt, so
    # the model cannot infer the positions from the cache length. The first pass feeds the held-back last
    # prompt token and predicts the first answer token.
    next_input, position, new = input_ids[:, -1:], n - 1, []
    for _ in range(max_new_tokens):
        out = model(input_ids=next_input, past_key_values=cache, position_ids=torch.tensor([[position]], device=device), logits_to_keep=1)
        token = out.logits[0, -1].argmax()
        new.append(token.item())
        if new[-1] in stop_ids:
            break
        next_input, position = token.view(1, 1), position + 1

    sync(device)
    t2 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else float("nan")

    # Same post-processing as generation.generate_answer: drop the stop token, decode, strip.
    stopped = bool(new) and new[-1] in stop_ids
    answer_ids = new[:-1] if stopped else new
    return {
        "answer": tok.decode(answer_ids, skip_special_tokens=True).strip(),
        "prompt_tokens": n,
        "new_tokens": len(answer_ids),
        "truncated": not stopped,
        "n_kept": int(kept.sum(axis=2).mean()),  # per head (mean over layers and heads)
        "kept_fraction": float(kept.mean()),
        "kv_cache_mb": kv_cache_bytes / 1024**2,
        "prefill_s": t1 - t0,
        "decode_s_per_token": (t2 - t1) / len(new),  # decoding forward passes, stop token included
        "peak_mem_gb": peak,
        "kept": kept,
    }


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------

def load_model(attn: str, device: str | None) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """generation.load_model, plus the attention implementation the press needs."""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map=device or get_device(), attn_implementation=attn)
    return model.eval(), tok


def budget_name(budget: float) -> str:
    return str(float(budget))  # 0.1 -> "0.1", 1 -> "1.0", matching full_1.0.parquet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--budget", type=float, required=True, help="share of the prompt's tokens kept, e.g. 0.2, 0.1, 0.05; 1.0 = nothing removed")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (for testing); the answers file gets a _limitN suffix")
    parser.add_argument("--device", default=None)
    parser.add_argument("--inspect-layer", type=int, default=None, help="print the kept tokens of the first question in this layer")
    args = parser.parse_args()

    if not 0 < args.budget <= 1:
        parser.error("--budget must be in (0, 1]")
    if args.method == "nopress" and args.budget != 1.0:
        parser.error("--method nopress only makes sense with --budget 1.0")
    spec = METHODS[args.method]

    ids = pd.read_parquet(CLEAN_PATH).index.unique()[: args.limit]
    questions = load_questions(ids)
    aliases = load_aliases()
    spans = load_spans()

    model, tok = load_model(spec["attn"], args.device)
    stop_ids = _stop_ids(model)

    # One press object for the whole run; its compression_ratio is set per question (see press_ratio).
    press = spec["press"](1 - args.budget) if spec["press"] is not None else None
    press_settings = repr(press)  # before the per-question ratio overwrites compression_ratio
    store: dict[int, torch.Tensor] = {}
    if spec["kind"] == "scorer":
        attach_scorer_recorder(press, store)

    name = f"{args.method}_{budget_name(args.budget)}" + (f"_limit{args.limit}" if args.limit else "")
    kept_dir = KEPT_DIR / args.method / budget_name(args.budget)
    kept_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (qid, q) in enumerate(tqdm(questions.iterrows(), total=len(questions))):
        prompt = build_prompt(tok, q["context"], q["question"])
        input_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        check_spans(qid, input_ids[0].tolist(), tok, spans)
        if press is not None:
            press.compression_ratio = press_ratio(input_ids.shape[1], args.budget)

        result = answer_one(model, tok, input_ids, press, spec["kind"], store, stop_ids, MAX_NEW_TOKENS)
        kept = result.pop("kept")
        np.savez_compressed(kept_dir / f"{qid}.npz", kept=kept)
        if i == 0 and args.inspect_layer is not None:
            inspect_kept(qid, input_ids[0].tolist(), kept, args.inspect_layer, tok, spans)

        score = score_answer(result["answer"], q["answer"], aliases.get(qid, []))
        rows.append({"id": qid, "question": q["question"], "gold": q["answer"], "answer": result.pop("answer"),
                     "contains": score.contains, "contains_alias": score.contains_alias, "f1": score.f1, **result})

    # Same layout as full_1.0.parquet (results/README.md), with the extra cost and kept columns at the end.
    df = pd.DataFrame(rows).set_index("id")
    df.insert(0, "method", args.method)
    df.insert(1, "budget", float(args.budget))
    df.insert(2, "model", MODEL_NAME)
    df.insert(3, "device", device_name(model))
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    out = ANSWERS_DIR / f"{name}.parquet"
    df.to_parquet(out)

    config = {
        "method": args.method, "budget": args.budget, "press": press_settings, "attn_implementation": spec["attn"],
        "compression_ratio": "set per question so that int(budget * prompt_tokens) positions per head are kept, held-back token included",
        "held_back_last_token": True,"max_new_tokens": MAX_NEW_TOKENS, "n_questions": len(df),
        "device": device_name(model), "versions": {p: version(p) for p in ("kvpress", "transformers", "torch")},
    }
    (ANSWERS_DIR / f"{name}_config.json").write_text(json.dumps(config, indent=2))

    print(f"saved {out} and {len(df)} kept masks in {kept_dir}")
    print_summary(df)
    print(f"mean kept fraction: {df['kept_fraction'].mean():.4f} (budget {args.budget}) | mean KV cache: {df['kv_cache_mb'].mean():.1f} MB")
    print(f"mean prefill: {df['prefill_s'].mean():.3f} s | mean decode: {1000 * df['decode_s_per_token'].mean():.1f} ms/token | max peak memory: {df['peak_mem_gb'].max():.2f} GB")


if __name__ == "__main__":
    main()
