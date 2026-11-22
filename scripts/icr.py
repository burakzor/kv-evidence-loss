# Paragraph scoring adapted from github.com/OSU-NLP-Group/In-Context-Reranking at commit 76d7dd8 (MIT License, Copyright (c) 2024 OSU Natural Language Processing)
from __future__ import annotations
import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel, PreTrainedTokenizerBase
import attention_recorder
from generation import MAX_NEW_TOKENS, MODEL_NAME, generate_answer, get_device
from model_filter import load_ab_titles, select_paragraphs
from prompts import Context, build_prompt, load_questions
from qrhead_detect import NULL_QUERY, compose_scoring_prompt, score_per_token_attention_to_query
from run_full_baseline import CLEAN_PATH, device_name, print_summary
from scoring import ROOT, load_aliases, score_answer

SPANS_PATH = ROOT / "data" / "hotpotqa" / "hotpot_token_spans_qwen3_8b.parquet"
FILTER_PATH = ROOT / "data" / "hotpotqa" / "filter_qwen3_8b.parquet"
RESULTS_DIR = ROOT / "results"
BUDGETS = [0.2, 0.1]

DocSpans = dict[str, tuple[int, int]]  # title -> (tok_start, tok_end), end inclusive

def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def icr_scores(model: PreTrainedModel, tok: PreTrainedTokenizerBase, context: Context, question: str) -> tuple[dict[str, float], list[int], DocSpans]:
    input_ids, query_span, doc_spans = compose_scoring_prompt(tok, context, question)
    null_input_ids, null_query_span, _ = compose_scoring_prompt(tok, context, NULL_QUERY)
    assert null_input_ids[: query_span[0]] == input_ids[: query_span[0]]
    assert null_query_span[0] == query_span[0]

    cache = DynamicCache()
    # ICR sums the query-averaged attention over all layers and heads before calibrating: (n_tokens,)
    query_scores = score_per_token_attention_to_query(model, input_ids, query_span, cache, 0).sum((0, 1))
    cache.crop(query_span[0])
    null_scores = score_per_token_attention_to_query(model, null_input_ids, null_query_span, cache, query_span[0]).sum((0, 1))

    scores = {}
    for title, (start, end) in doc_spans.items():
        calibrated = query_scores[start : end + 1] - null_scores[start : end + 1]
        tok_mask = calibrated > calibrated.mean() - 2 * calibrated.std()
        scores[title] = calibrated[tok_mask].sum().item()
    return scores, input_ids, doc_spans

def select_by_budget(ranked: list[str], doc_spans: DocSpans, target: float) -> list[str]:
    # walk down the ranking while the next paragraph brings the kept paragraph tokens closer to the target; at least one
    lengths = {title: end - start + 1 for title, (start, end) in doc_spans.items()}
    kept, total = [ranked[0]], lengths[ranked[0]]
    for title in ranked[1:]:
        if abs(total + lengths[title] - target) > abs(total - target):
            break
        kept.append(title)
        total += lengths[title]
    return kept

def paragraph_mask(n_tokens: int, doc_spans: DocSpans, kept_titles: list[str], n_layers: int, n_kv_heads: int) -> np.ndarray:
    kept = np.ones(n_tokens, dtype=bool)
    for title, (start, end) in doc_spans.items():
        if title not in kept_titles:
            kept[start : end + 1] = False
    # a boundary token shared with a kept paragraph stays kept
    for title in kept_titles:
        start, end = doc_spans[title]
        kept[start : end + 1] = True
    return np.broadcast_to(kept, (n_layers, n_kv_heads, n_tokens)).copy()

def check_prompt(qid: str, tok: PreTrainedTokenizerBase, context: Context, question: str, input_ids: list[int], doc_spans: DocSpans, spans: pd.DataFrame) -> None:
    prompt_ids = tok(build_prompt(tok, context, question), add_special_tokens=False)["input_ids"]
    if prompt_ids != input_ids:
        raise RuntimeError(f"{qid}: scoring prompt differs from the answering prompt")
    rows = spans.loc[[qid]]
    if len(input_ids) != rows["n_tokens"].iat[0]:
        raise RuntimeError(f"{qid}: prompt has {len(input_ids)} tokens, spans file expects {rows['n_tokens'].iat[0]}")
    for r in rows.itertuples():
        start, end = doc_spans[r.title]
        if not start <= r.tok_start <= r.tok_end <= end:
            raise RuntimeError(f"{qid}: {r.sentence_tag} sentence {r.tok_start}-{r.tok_end} is outside its paragraph {start}-{end}")

def budget_name(budget: float) -> str:
    return str(float(budget))

def print_icr_summary(df: pd.DataFrame, keep: pd.Series | None) -> None:
    print_summary(df)
    print(f"paragraphs kept: mean {df['n_paragraphs_kept'].mean():.2f} | {df['n_paragraphs_kept'].value_counts().sort_index().to_dict()}")
    print(f"mean prompt kept fraction: {df['prompt_kept_fraction'].mean():.4f} (budget {df['budget'].iat[0]}) | paragraph tokens / target: {(df['paragraph_tokens_kept'] / df['target_tokens']).mean():.3f}")
    both = df["a_kept"] & df["b_kept"]
    print(f"A kept: {df['a_kept'].mean():.3f} | B kept: {df['b_kept'].mean():.3f} | both: {both.mean():.3f}")
    if keep is not None:
        sub = df[keep.reindex(df.index, fill_value=False)]
        print(f"on model-filter keep questions ({len(sub)}): contains_alias {sub['contains_alias'].mean():.3f}")
    print(f"mean rank time: {df['rank_s'].mean():.3f} s | mean generate time: {df['generate_s'].mean():.3f} s | max peak memory: {df['peak_mem_gb'].max():.2f} GB")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", type=float, nargs="+", default=BUDGETS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    if not all(0 < b <= 1 for b in args.budgets):
        parser.error("budgets must be in (0, 1]")

    ids = pd.read_parquet(CLEAN_PATH).index.unique()[: args.limit]
    questions = load_questions(ids)
    titles = load_ab_titles(ids)
    aliases = load_aliases()
    spans = pd.read_parquet(SPANS_PATH)
    keep = pd.read_parquet(FILTER_PATH)["keep"] if FILTER_PATH.exists() else None

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device or get_device(), attn_implementation=attention_recorder.ATTN_NAME).eval()
    n_layers, n_kv_heads = model.config.num_hidden_layers, model.config.num_key_value_heads
    device = model.device

    suffix = f"_limit{args.limit}" if args.limit else ""
    results_dir = args.results_dir.resolve()
    answers_dir = results_dir / "answers"
    kept_root = results_dir / "raw" / f"kept{suffix}" / "icr"
    for b in args.budgets:
        (kept_root / budget_name(b)).mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)

    rows: dict[float, list[dict]] = {b: [] for b in args.budgets}
    score_rows = []
    for qid, q in tqdm(questions.iterrows(), total=len(questions)):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        t0 = time.perf_counter()
        scores, input_ids, doc_spans = icr_scores(model, tok, q["context"], q["question"])
        sync(device)
        rank_s = time.perf_counter() - t0
        check_prompt(qid, tok, q["context"], q["question"], input_ids, doc_spans, spans)

        n_tokens = len(input_ids)
        a_title, b_title = titles.at[qid, "A"], titles.at[qid, "B"]
        ranked = sorted(scores, key=scores.get, reverse=True)
        for rank, title in enumerate(ranked, 1):
            start, end = doc_spans[title]
            score_rows.append({"id": qid, "title": title, "score": scores[title], "rank": rank, "tok_start": start, "tok_end": end, "is_a": title == a_title, "is_b": title == b_title})

        for b in args.budgets:
            target = b * n_tokens
            kept_titles = select_by_budget(ranked, doc_spans, target)
            kept = paragraph_mask(n_tokens, doc_spans, kept_titles, n_layers, n_kv_heads)
            np.savez_compressed(kept_root / budget_name(b) / f"{qid}.npz", kept=kept)

            short_context = select_paragraphs(q["context"], kept_titles)
            sync(device)
            t1 = time.perf_counter()
            gen = generate_answer(model, tok, build_prompt(tok, short_context, q["question"]), MAX_NEW_TOKENS)
            sync(device)
            generate_s = time.perf_counter() - t1
            peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else float("nan")

            score = score_answer(gen.answer, q["answer"], aliases.get(qid, []))
            rows[b].append({
                "id": qid, "question": q["question"], "gold": q["answer"], "answer": gen.answer,
                "contains": score.contains, "contains_alias": score.contains_alias, "f1": score.f1,
                "prompt_tokens": n_tokens, "new_tokens": gen.new_tokens, "truncated": gen.truncated,
                "short_prompt_tokens": gen.prompt_tokens, "n_paragraphs_kept": len(kept_titles), "kept_titles": short_context["title"],
                "target_tokens": target, "paragraph_tokens_kept": sum(doc_spans[t][1] - doc_spans[t][0] + 1 for t in kept_titles),
                "a_kept": a_title in kept_titles, "b_kept": b_title in kept_titles, "prompt_kept_fraction": float(kept.mean()),
                "rank_s": rank_s, "generate_s": generate_s, "peak_mem_gb": peak,
            })

    scores_df = pd.DataFrame(score_rows).set_index("id")
    scores_path = results_dir / f"icr_scores{suffix}.parquet"
    scores_df.to_parquet(scores_path)
    ab = scores_df[scores_df["is_a"] | scores_df["is_b"]]
    print(f"saved {scores_path}")
    print(f"mean rank of A: {ab.loc[ab['is_a'], 'rank'].mean():.2f} | of B: {ab.loc[ab['is_b'], 'rank'].mean():.2f} (of 10)")

    config = {
        "method": "icr", "model": args.model, "attn_implementation": attention_recorder.ATTN_NAME,
        "scoring": "ICR masked_NA_calibration: query-averaged attention from the query span, summed over all layers and heads; per paragraph, calibrated = query - null query, tokens below mean - 2 std of calibrated dropped, score = sum of the rest",
        "query_span": "from 'Question:' to the end of the prompt, chat template suffix included", "null_query": NULL_QUERY,
        "budget_rule": "target = budget * prompt_tokens; paragraphs in ICR rank order, the next one added while it brings the kept paragraph tokens closer to the target; at least one; kept paragraphs in their original order",
        "mask": "True everywhere except the tokens of dropped paragraphs, in all layers and KV heads",
        "max_new_tokens": MAX_NEW_TOKENS, "n_questions": len(questions), "device": device_name(model),
        "versions": {p: version(p) for p in ("transformers", "torch")},
    }
    for b in args.budgets:
        df = pd.DataFrame(rows[b]).set_index("id")
        df.insert(0, "method", "icr")
        df.insert(1, "budget", float(b))
        df.insert(2, "model", args.model)
        df.insert(3, "device", device_name(model))
        name = f"icr_{budget_name(b)}{suffix}"
        out = answers_dir / f"{name}.parquet"
        df.to_parquet(out)
        (answers_dir / f"{name}_config.json").write_text(json.dumps({**config, "budget": b}, indent=2))
        print(f"\nsaved {out} and {len(df)} kept masks in {kept_root / budget_name(b)}")
        print_icr_summary(df, keep)

if __name__ == "__main__":
    main()
