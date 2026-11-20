from __future__ import annotations
import argparse
import re
from collections.abc import Callable
import pandas as pd
from scoring import ALIASES_PATH, ROOT, normalize_answer

CLEAN_PATH = ROOT / "data" / "hotpotqa" / "hotpot_bridge_ab_clean.parquet"

MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"

DATE = re.compile(rf"^(?:\d{{1,2}}\s+)?(?:{MONTHS})(?:\s+\d{{1,2}})?,?\s+(\d{{4}})$", re.I)

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]

WORD_TO_DIGIT = {w: str(i) for i, w in enumerate(NUMBER_WORDS)}

DIGIT_TO_WORD = {d: w for w, d in WORD_TO_DIGIT.items()}

UNITS = {"ft": "feet", "feet": "ft", "mi": "miles", "miles": "mi", "km": "kilometres", "kilometres": "km", "kilometers": "km"}

COUNT = re.compile(rf"^((?:\d[\d,.]*|{'|'.join(NUMBER_WORDS)})(?:\s+(?:thousand|million|billion))?)(\s+.+)?$", re.I)

LEADING = re.compile(r"^(?:from|in|at|on|about|approximately|around|over|nearly|almost|more than|up to)\s+", re.I)

TRAILING = re.compile(r"\s+above sea level$", re.I)

COMPANY = re.compile(r",?\s+(?:Corporation|Corp\.?|Inc\.?|Ltd\.?|Limited|Company|Co\.)$")

TITLE = re.compile(r"^(?:President|King|Queen|Prince|Princess|Sir|Dame|Dr\.?|Saint)\s+")

PLACE_QUESTION = re.compile(r"\b(?:where|city|town|located|headquartered|based|from what|place)\b", re.I)

STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "and", "or", "to", "for", "from", "by", "with", "i"}

def r1_count(gold: str, question: str) -> list[str]:
    m = COUNT.match(gold.strip())
    if m and m.group(2) and re.search(r"\b(?:how|this|that) (?:many|much)\b", question, re.I):
        return [m.group(1)]
    return []

def asked_nouns(question: str) -> set[str]:
    spans = re.findall(r"\b(?:what|which|how many|how much)\s+((?:\S+\s+){0,4}\S+)", question, re.I)
    return {w.rstrip("s") for span in spans for w in normalize_answer(span).split()}

def r1_repeated_noun(gold: str, question: str) -> list[str]:
    if "," in gold or re.search(r"\band\b", gold):
        return []
    asked = asked_nouns(question)
    words = gold.split()
    while len(words) > 1 and words[-1].lower() not in STOPWORDS and normalize_answer(words[-1]).rstrip("s") in asked:
        words = words[:-1]
    if len(words) == len(gold.split()) or words[-1].lower() in STOPWORDS:
        return []
    return [" ".join(words)]

def r2_year(gold: str, question: str) -> list[str]:
    m = DATE.match(gold.strip())
    asks_year = re.search(r"\b(?:what|which) year\b", question, re.I) and not re.search(r"\b(?:day|date)\b", question, re.I)
    return [m.group(1)] if m and asks_year else []

def r3_city(gold: str, question: str) -> list[str]:
    if "," not in gold or re.search(r"\d|\band\b", gold) or not PLACE_QUESTION.search(question):
        return []
    return [gold.split(",")[0].strip()]

def r4_leading_trailing(gold: str, question: str) -> list[str]:
    if not gold[:1].islower():
        return []
    s = gold
    while (t := TRAILING.sub("", LEADING.sub("", s))) != s:
        s = t
    return [s] if s != gold else []

def r5_suffix_title(gold: str, question: str) -> list[str]:
    s = TITLE.sub("", COMPANY.sub("", gold)).strip()
    return [s] if s and s != gold else []

def r6_spelling(text: str) -> list[str]:
    words = text.split()
    if not words or DATE.match(text.strip()):
        return []
    swap = WORD_TO_DIGIT.get(words[0].lower()) or DIGIT_TO_WORD.get(words[0])
    if swap and (len(words) == 1 or words[1].lower() != "of"):
        words = [swap] + words[1:]
    s = " ".join(UNITS.get(w.lower(), w) for w in words)
    return [s] if s != text else []

RULES: list[tuple[str, Callable[[str, str], list[str]]]] = [("R1", r1_count), ("R1", r1_repeated_noun), ("R2", r2_year), ("R3", r3_city), ("R4", r4_leading_trailing), ("R5", r5_suffix_title)]

def make_aliases(gold: str, question: str) -> list[tuple[str, str]]:
    found = [(alias, rule) for rule, fn in RULES for alias in fn(gold, question)]
    found += [(alias, "R6") for text in [gold] + [a for a, _ in found] for alias in r6_spelling(text)]
    seen = {normalize_answer(gold)}
    out = []
    for alias, rule in found:
        key = normalize_answer(alias)
        if key and key not in seen:
            seen.add(key)
            out.append((alias, rule))
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if ALIASES_PATH.exists() and not args.force:
        raise SystemExit(f"{ALIASES_PATH} exists (it may be hand-reviewed); use --force to overwrite")
    questions = pd.read_parquet(CLEAN_PATH).groupby(level=0)[["question", "answer"]].first()
    rows = [{"id": qid, "question": q["question"], "gold": q["answer"], "alias": alias, "rule": rule} for qid, q in questions.iterrows() for alias, rule in make_aliases(q["answer"], q["question"])]
    df = pd.DataFrame(rows, columns=["id", "question", "gold", "alias", "rule"])
    df.to_csv(ALIASES_PATH, index=False)
    print(f"saved {ALIASES_PATH}: {len(df)} aliases for {df['id'].nunique()}/{len(questions)} questions")
    print(df["rule"].value_counts().sort_index().to_string())

if __name__ == "__main__":
    main()