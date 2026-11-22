# Scoring adapted from github.com/princeton-pli/QRHead at commit 7a668e7 (MIT License, Copyright (c) 2025 Princeton Language and Intelligence)
from __future__ import annotations
from importlib.metadata import version
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel, PreTrainedTokenizerBase
import attention_recorder
from generation import MAX_NEW_TOKENS, generate_answer, get_device
from icr import select_by_budget
from prompts import build_prompt, load_questions
from qrhead_detect import NULL_QUERY, compose_scoring_prompt, score_per_token_attention_to_query
from run_full_baseline import device_name
from scoring import ROOT, load_aliases, score_answer

MODEL_NAME = "Qwen/Qwen3-8B"
DATA_DIR = ROOT / "data" / "hotpotqa"
HEADS_PATH = DATA_DIR / "qrhead_heads_qwen3_8b.txt"
CLEAN_PATH = DATA_DIR / "hotpot_bridge_ab_clean.parquet"
SPANS_PATH = DATA_DIR / "hotpot_token_spans_qwen3_8b.parquet"
ANSWERS_DIR = ROOT / "results" / "answers"
KEPT_DIR = ROOT / "results" / "raw" / "kept" / "qrhead"
N_QUESTIONS = 169
BUDGETS = [0.2, 0.1]

def sync(device: torch.device) -> None:
    """Wait for the GPU to finish before the clock is read."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def score_docs(model: PreTrainedModel, tok: PreTrainedTokenizerBase, question: str, context: dict, attn_head_set: str) -> tuple[dict[str, float], list[int], dict[str, tuple[int, int]]]:
    """Return the calibrated QRHead score of every paragraph (keyed by title), the prompt token ids and the paragraph token spans."""
    input_ids, query_span, doc_spans = compose_scoring_prompt(tok, context, question)
    null_input_ids, null_query_span, _ = compose_scoring_prompt(tok, context, NULL_QUERY)

    assert null_input_ids[: query_span[0]] == input_ids[: query_span[0]]
    assert query_span[0] == null_query_span[0], "Query start indices do not match between query and null query."

    cache = DynamicCache()
    per_token_scores = score_per_token_attention_to_query(model, input_ids, query_span, cache, 0)

    cache.crop(query_span[0])
    start_idx = query_span[0]
    null_per_token_scores = score_per_token_attention_to_query(model, null_input_ids, null_query_span, cache, start_idx)

    min_length = min(per_token_scores.shape[-1], null_per_token_scores.shape[-1])
    per_token_scores_CAL = per_token_scores[:, :, :min_length] - null_per_token_scores[:, :, :min_length]

    head_set = attn_head_set.split(",")
    head_set = [tuple(map(int, h.split("-"))) for h in head_set]
    indices = torch.tensor(head_set)
    layers = indices[:, 0]
    heads = indices[:, 1]
    per_token_scores_CAL = per_token_scores_CAL[layers, heads]
    per_token_scores_CAL = per_token_scores_CAL.sum(0)

    doc_scores = {}
    for title, doc_span in doc_spans.items():
        curr_doc_per_tok_scores_CAL = per_token_scores_CAL[doc_span[0] : doc_span[1] + 1]
        threshold = curr_doc_per_tok_scores_CAL.mean() - 2 * curr_doc_per_tok_scores_CAL.std()
        tok_mask = curr_doc_per_tok_scores_CAL > threshold
        doc_scores[title] = (curr_doc_per_tok_scores_CAL * tok_mask).sum().item()
    return doc_scores, input_ids, doc_spans

def rank_titles(doc_scores: dict[str, float]) -> list[str]:
    """Return the titles from the highest score to the lowest, ties in prompt order."""
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    titles = []
    for title, score in sorted_docs:
        titles.append(title)
    return titles

def keep_context(context: dict, kept_titles: list[str]) -> dict:
    """Return the context with only the kept paragraphs, in their original order."""
    titles = []
    sentences = []
    for title, sents in zip(context["title"], context["sentences"]):
        if title in kept_titles:
            titles.append(title)
            sentences.append(sents)
    return {"title": titles, "sentences": sentences}

def get_kept_mask(n_layers: int, n_kv_heads: int, n_tokens: int, doc_spans: dict[str, tuple[int, int]], kept_titles: list[str]) -> np.ndarray:
    """Return the (n_layers, n_kv_heads, n_tokens) mask, False on the tokens of dropped paragraphs."""
    kept = np.ones((n_layers, n_kv_heads, n_tokens), dtype=bool)
    for title, (tok_start, tok_end) in doc_spans.items():
        if title not in kept_titles:
            kept[:, :, tok_start : tok_end + 1] = False
    return kept

def get_ab_titles() -> pd.DataFrame:
    """Return the titles of paragraph A and paragraph B of every clean question."""
    clean_df = pd.read_parquet(CLEAN_PATH).reset_index()
    ab_df = clean_df.groupby(["id", "sentence_tag"])["title"].first().unstack()
    return ab_df[["A", "B"]]

def print_budget_summary(df: pd.DataFrame) -> None:
    """Print accuracy, paragraphs kept, kept fraction and A/B kept shares of one budget."""
    both_kept = df["a_kept"] & df["b_kept"]
    print()
    print("budget:", df["budget"].iat[0], "| questions:", len(df), "| device:", df["device"].iat[0])
    print("contains_alias:", df["contains_alias"].sum(), "/", len(df), round(df["contains_alias"].mean(), 3))
    print("paragraphs kept:", round(df["n_paragraphs_kept"].mean(), 2), "| prompt kept fraction:", round(df["prompt_kept_fraction"].mean(), 3))
    print("A kept:", round(df["a_kept"].mean(), 3), "| B kept:", round(df["b_kept"].mean(), 3), "| both kept:", round(both_kept.mean(), 3))
    print("rank_s:", round(df["rank_s"].mean(), 3), "| generate_s:", round(df["generate_s"].mean(), 3), "| peak_mem_gb:", round(df["peak_mem_gb"].max(), 2))

def run(model_name: str, heads_path: Path, n_questions: int, answers_dir: Path, kept_dir: Path) -> dict[float, pd.DataFrame]:
    """Return the answers table of every budget, and write the tables and the kept masks."""
    attn_head_set = heads_path.read_text().strip()
    ids = pd.read_parquet(CLEAN_PATH).index.unique()[:n_questions]
    questions = load_questions(ids)
    aliases = load_aliases()
    ab_df = get_ab_titles()
    spans_n_tokens = pd.read_parquet(SPANS_PATH).groupby(level=0)["n_tokens"].first()

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map=get_device(), attn_implementation=attention_recorder.ATTN_NAME).eval()
    device = model.device
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads

    rows = {}
    for budget in BUDGETS:
        rows[budget] = []
        (kept_dir / str(budget)).mkdir(parents=True, exist_ok=True)

    for qid, q in tqdm(questions.iterrows(), total=len(questions)):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        t0 = time.perf_counter()
        doc_scores, input_ids, doc_spans = score_docs(model, tok, q["question"], q["context"], attn_head_set)
        sync(device)
        rank_s = time.perf_counter() - t0

        n_tokens = len(input_ids)
        if n_tokens != spans_n_tokens[qid]:
            raise RuntimeError(f"{qid}: prompt has {n_tokens} tokens, the spans file has {spans_n_tokens[qid]}")

        ranked_titles = rank_titles(doc_scores)

        for budget in BUDGETS:
            # the same paragraph budget rule as ICR: only paragraph tokens count against budget * n_tokens
            kept_titles = select_by_budget(ranked_titles, doc_spans, budget * n_tokens)
            prompt = build_prompt(tok, keep_context(q["context"], kept_titles), q["question"])

            sync(device)
            t0 = time.perf_counter()
            gen = generate_answer(model, tok, prompt)
            sync(device)
            generate_s = time.perf_counter() - t0
            if device.type == "cuda":
                peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            else:
                peak_mem_gb = float("nan")

            kept = get_kept_mask(n_layers, n_kv_heads, n_tokens, doc_spans, kept_titles)
            np.savez_compressed(kept_dir / str(budget) / f"{qid}.npz", kept=kept)

            score = score_answer(gen.answer, q["answer"], aliases.get(qid, []))
            rows[budget].append({
                "id": qid,
                "question": q["question"],
                "gold": q["answer"],
                "answer": gen.answer,
                "contains": score.contains,
                "contains_alias": score.contains_alias,
                "f1": score.f1,
                "prompt_tokens": n_tokens,
                "new_tokens": gen.new_tokens,
                "truncated": gen.truncated,
                "peak_mem_gb": peak_mem_gb,
                "prefill_s": float("nan"),
                "decode_s_per_token": float("nan"),
                "rank_s": rank_s,
                "generate_s": generate_s,
                "kept_titles": kept_titles,
                "n_paragraphs_kept": len(kept_titles),
                "prompt_kept_fraction": float(kept.mean()),
                "short_prompt_tokens": gen.prompt_tokens,
                "a_kept": ab_df.loc[qid, "A"] in kept_titles,
                "b_kept": ab_df.loc[qid, "B"] in kept_titles,
            })

    answers_dir.mkdir(parents=True, exist_ok=True)
    dfs = {}
    for budget in BUDGETS:
        df = pd.DataFrame(rows[budget]).set_index("id")
        df.insert(0, "method", "qrhead")
        df.insert(1, "budget", budget)
        df.insert(2, "model", model_name)
        df.insert(3, "device", device_name(model))
        df.to_parquet(answers_dir / f"qrhead_{budget}.parquet")
        config = {
            "method": "qrhead", "budget": budget, "model": model_name, "heads": attn_head_set, "heads_file": heads_path.name,
            "null_query": NULL_QUERY, "selection": "icr.select_by_budget, kept paragraphs in prompt order",
            "attn_implementation": attention_recorder.ATTN_NAME, "max_new_tokens": MAX_NEW_TOKENS, "n_questions": len(df),
            "device": device_name(model), "versions": {p: version(p) for p in ("transformers", "torch")},
        }
        (answers_dir / f"qrhead_{budget}_config.json").write_text(json.dumps(config, indent=2))
        dfs[budget] = df
        print_budget_summary(df)
    print()
    print("answers written to", answers_dir, "| masks written to", kept_dir)
    return dfs

if __name__ == "__main__":
    run(MODEL_NAME, HEADS_PATH, N_QUESTIONS, ANSWERS_DIR, KEPT_DIR)