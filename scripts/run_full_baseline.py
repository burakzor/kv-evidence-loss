from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from generation import MAX_NEW_TOKENS, MODEL_NAME, generate_answer, load_model
from prompts import build_prompt, load_questions
from scoring import score_answer

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "hotpotqa" / "hotpot_bridge_ab_clean.parquet"
ANSWERS_DIR = ROOT / "results" / "answers"

def device_name(model: PreTrainedModel) -> str:
    if model.device.type == "cuda":
        return torch.cuda.get_device_name(model.device)
    return model.device.type

def answer_questions(model: PreTrainedModel, tok: PreTrainedTokenizerBase, questions: pd.DataFrame, max_new_tokens: int = MAX_NEW_TOKENS) -> pd.DataFrame:
    rows = []
    for qid, q in tqdm(questions.iterrows(), total=len(questions)):
        gen = generate_answer(model, tok, build_prompt(tok, q["context"], q["question"]), max_new_tokens)
        score = score_answer(gen.answer, q["answer"])
        rows.append({"id": qid, "question": q["question"], "gold": q["answer"], "answer": gen.answer, "contains": score.contains, "f1": score.f1, "prompt_tokens": gen.prompt_tokens, "new_tokens": gen.new_tokens, "truncated": gen.truncated})
    return pd.DataFrame(rows).set_index("id")

def print_summary(df: pd.DataFrame) -> None:
    print(f"\nquestions: {len(df)} | model: {df['model'].iat[0]} | device: {df['device'].iat[0]}")
    print(f"contains accuracy: {df['contains'].mean():.3f} ({df['contains'].sum()}/{len(df)})")
    print(f"mean f1: {df['f1'].mean():.3f}")
    print(f"truncated answers: {df['truncated'].sum()} | mean new tokens: {df['new_tokens'].mean():.1f}")

def write_check(df: pd.DataFrame, path: Path, n: int, seed: int) -> None:
    check = df.sample(min(n, len(df)), random_state=seed)[["question", "gold", "answer", "contains", "f1"]]
    check["human_correct"] = pd.NA
    check.to_csv(path)
    print(f"\n--- sanity check: {len(check)} random answers (also in {path}) ---")
    for qid, r in check.iterrows():
        print(f"\n[{qid}] contains={r['contains']} f1={r['f1']:.2f}\n  Q:      {r['question']}\n  gold:   {r['gold']}\n  answer: {r['answer']}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ANSWERS_DIR / "full_1.0.parquet")
    parser.add_argument("--n-check", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ids = pd.read_parquet(CLEAN_PATH).index.unique()[: args.limit]
    questions = load_questions(ids)
    model, tok = load_model(args.model, args.device)
    df = answer_questions(model, tok, questions)
    df.insert(0, "method", "full")
    df.insert(1, "budget", 1.0)
    df.insert(2, "model", args.model)
    df.insert(3, "device", device_name(model))
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"saved {out}")
    print_summary(df)
    write_check(df, out.with_name(out.stem + "_check.csv"), args.n_check, args.seed)

if __name__ == "__main__":
    main()