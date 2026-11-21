from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from generation import MODEL_NAME, load_model
from prompts import Context, load_questions
from run_full_baseline import ANSWERS_DIR, CLEAN_PATH, answer_questions
from scoring import ROOT, load_aliases

FILTER_PATH = ROOT / "data" / "hotpotqa" / "filter_qwen3_8b.parquet"

FULL_PATH = ANSWERS_DIR / "full_1.0.parquet"

CONDITIONS = ["a_only", "b_only", "no_ctx"]

# b_only is recorded but not filtered on: with B as the only paragraph the model answers 156/169 (single-paragraph shortcut)
KEEP_CONDITIONS = ["a_only", "no_ctx"]

MIN_KEEP = 100

def load_ab_titles(ids: pd.Index) -> pd.DataFrame:
    clean = pd.read_parquet(CLEAN_PATH).reset_index()
    titles = clean.groupby(["id", "sentence_tag"])["title"].agg(set).unstack()
    if titles.map(len).ne(1).any().any():
        raise ValueError("every question needs exactly one A and one B paragraph")
    return titles.map(lambda s: next(iter(s))).loc[ids]

def condition_titles(titles: pd.DataFrame, qid: str, condition: str) -> list[str]:
    return {"a_only": [titles.at[qid, "A"]], "b_only": [titles.at[qid, "B"]], "no_ctx": []}[condition]

def select_paragraphs(context: Context, keep: list[str]) -> Context:
    idx = [i for i, t in enumerate(context["title"]) if t in keep]
    found = [context["title"][i] for i in idx]
    if sorted(found) != sorted(keep):
        raise ValueError(f"expected paragraphs {keep}, found {found}")
    return {"title": found, "sentences": [context["sentences"][i] for i in idx]}

def run_condition(model: PreTrainedModel, tok: PreTrainedTokenizerBase, questions: pd.DataFrame, titles: pd.DataFrame, aliases: dict[str, list[str]], condition: str) -> pd.DataFrame:
    q = questions.copy()
    q["context"] = [select_paragraphs(ctx, condition_titles(titles, qid, condition)) for qid, ctx in q["context"].items()]
    print(f"\n--- condition: {condition} ---")
    return answer_questions(model, tok, q, aliases)

def keep_mask(df: pd.DataFrame, conditions: list[str] = KEEP_CONDITIONS, suffix: str = "") -> pd.Series:
    return df[f"full_contains{suffix}"] & ~df[[f"{c}_contains{suffix}" for c in conditions]].any(axis=1)

def build_filter(full: pd.DataFrame, runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_runs = {"full": full, **runs}
    df = pd.DataFrame({f"{name}_contains": r["contains_alias"] for name, r in all_runs.items()})
    df["keep"] = keep_mask(df)
    for name, r in all_runs.items():
        df[f"{name}_contains_strict"] = r["contains"]
    for name, r in all_runs.items():
        df[f"{name}_answer"] = r["answer"]
    df.index.name = "id"
    return df

def print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\nquestions: {n}")
    for name in ["full", *CONDITIONS]:
        print(f"{name:>7} correct: contains_alias {df[f'{name}_contains'].sum():>3}/{n} | contains {df[f'{name}_contains_strict'].sum():>3}/{n}")
    full_ok = df[df["full_contains"]]
    print(f"\nof {len(full_ok)} answered correctly with the full context, also correct with:")
    for c in CONDITIONS:
        print(f"{c:>7}: {full_ok[f'{c}_contains'].sum()}")
    print(f"\nkept (full correct, {' and '.join(KEEP_CONDITIONS)} wrong): {df['keep'].sum()}/{n} | scored with contains: {keep_mask(df, suffix='_strict').sum()}/{n}")
    print(f"kept questions that b_only also answers: {df.loc[df['keep'], 'b_only_contains'].sum()} | kept if b_only were filtered on too: {keep_mask(df, CONDITIONS).sum()}")
    if df["keep"].sum() < MIN_KEEP:
        print(f"WARNING: fewer than {MIN_KEEP} questions remain; tell the team")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.rebuild:
        out = (args.out or FILTER_PATH).resolve()
        df = pd.read_parquet(out)
        df["keep"] = keep_mask(df)
        df.to_parquet(out)
        print(f"rebuilt keep in {out}")
        print_summary(df)
        return
    ids = pd.read_parquet(CLEAN_PATH).index.unique()[: args.limit]
    full = pd.read_parquet(FULL_PATH).loc[ids]
    if not full["model"].eq(MODEL_NAME).all():
        raise ValueError(f"{FULL_PATH} was not generated with {MODEL_NAME}")
    titles = load_ab_titles(ids)
    questions = load_questions(ids)
    aliases = load_aliases()
    model, tok = load_model(MODEL_NAME, args.device)
    runs = {c: run_condition(model, tok, questions, titles, aliases, c) for c in CONDITIONS}
    df = build_filter(full, runs)
    out = (args.out or FILTER_PATH.with_stem(FILTER_PATH.stem + (f"_limit{args.limit}" if args.limit else ""))).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"saved {out}")
    print_summary(df)

if __name__ == "__main__":
    main()