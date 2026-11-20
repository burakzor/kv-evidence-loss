from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import re
import string

PUNCTUATION = set(string.punctuation)

@dataclass
class Score:
    contains: bool
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

def score_answer(prediction: str, gold: str) -> Score:
    return Score(contains=contains_score(prediction, gold), f1=f1_score(prediction, gold))