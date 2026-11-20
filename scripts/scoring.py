from __future__ import annotations
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import string
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ALIASES_PATH = ROOT / "data" / "hotpotqa" / "answer_aliases.csv"

PUNCTUATION = set(string.punctuation)

@dataclass
class Score:
    contains: bool
    contains_alias: bool
    f1: float

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in PUNCTUATION)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def contains_score(prediction: str, gold: str) -> bool:
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    if not ref:
        return pred == ref
    return f" {ref} " in f" {pred} "

def f1_score(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    if (pred in ("yes", "no", "noanswer") or ref in ("yes", "no", "noanswer")) and pred != ref:
        return 0.0
    pred_tokens, ref_tokens = pred.split(), ref.split()
    num_same = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

def load_aliases(path: Path = ALIASES_PATH) -> dict[str, list[str]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df.groupby("id")["alias"].apply(list).to_dict()

def score_answer(prediction: str, gold: str, aliases: Sequence[str] = ()) -> Score:
    contains = contains_score(prediction, gold)
    return Score(contains=contains, contains_alias=contains or any(contains_score(prediction, a) for a in aliases), f1=f1_score(prediction, gold))