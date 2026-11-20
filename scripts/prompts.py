from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import TypedDict
from datasets import load_dataset
import pandas as pd
from transformers import PreTrainedTokenizerBase

INSTRUCTION = "Answer the question with a short phrase from the paragraphs. Do not explain."

class Context(TypedDict):
    title: Sequence[str]
    sentences: Sequence[Sequence[str]]

# (title, sent_id) -> (char_start, char_end), end exclusive
Spans = dict[tuple[str, int], tuple[int, int]]

def load_questions(ids: Iterable[str] | None = None) -> pd.DataFrame:
    hotpotqa = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    df = pd.DataFrame(hotpotqa).set_index("id")[["question", "answer", "context"]]
    if ids is not None:
        df = df.loc[pd.Index(ids).unique()]
    return df

def build_text(context: Context, question: str, instruction: str | None = INSTRUCTION) -> tuple[str, Spans]:
    text = instruction + "\n\n" if instruction else ""
    spans: Spans = {}
    for title, sents in zip(context["title"], context["sentences"]):
        for sent_id, sent in enumerate(sents):
            start = len(text)
            text += sent
            spans[(title, sent_id)] = (start, len(text))
        text += "\n\n"
    text += "Question: " + question
    return text, spans

def get_prompt(tok: PreTrainedTokenizerBase, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def build_prompt(tok: PreTrainedTokenizerBase, context: Context, question: str, instruction: str | None = INSTRUCTION) -> str:
    text, _ = build_text(context, question, instruction)
    return get_prompt(tok, text)