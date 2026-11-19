from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

def build_text(c, question):
    """Return the prompt text and character spans in the sentences."""
    text = ""
    spans = {}
    for title, sents in zip(c["title"], c["sentences"]):
        for sent_id, sent in enumerate(sents):
            start = len(text)
            text += sent
            end = len(text)
            spans[(title, sent_id)] = (start, end)
        text += "\n\n"
    text += "Question: " + question
    return text, spans

def get_prompt(tok, text):
    """Return the chat prompt to give to the model for a text."""
    messages = [{"role": "user", "content": text}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def get_token_span(offsets, start, end):
    """Return the starting and ending token of a sentence."""
    tok_start = None
    for i, (s, e) in enumerate(offsets):
        if tok_start is None and e > start:
            tok_start = i
        if s < end:
            tok_end = i
    return tok_start, tok_end

if __name__ == "__main__":
    hotpotqa = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    hotpotqa_df = pd.DataFrame(hotpotqa).set_index("id")

    data_dir = Path(__file__).parent.parent / "data" / "hotpotqa"
    ab_df = pd.read_parquet(data_dir.joinpath("hotpot_bridge_ab_clean.parquet"))

    context_df = hotpotqa_df.loc[ab_df.index.unique(), ["question", "context"]]

    ctx = context_df["context"].to_dict()
    id_list = ab_df.index.unique().to_list()

    texts = {}
    all_spans = {}

    for qid in id_list:
        texts[qid], all_spans[qid] = build_text(ctx[qid], context_df.loc[qid, "question"])
        

    model_name = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(model_name)

    prompt_df = pd.DataFrame({"text": texts})
    prompt_df["prompt"] = prompt_df["text"].apply(lambda x: get_prompt(tok, x))
    prompt_df["offset"] = prompt_df.apply(lambda x: x["prompt"].find(x["text"]), axis="columns")
    prompt_df["enc"] = prompt_df["prompt"].apply(lambda x: tok(x, add_special_tokens=False, return_offsets_mapping=True))
    prompt_df["n_tokens"] = prompt_df["enc"].apply(lambda x: len(x["input_ids"]))

    spans_df = ab_df.join(prompt_df[["offset", "enc", "n_tokens"]])
    spans_df["char_start"] = spans_df.apply(lambda x: all_spans[x.name][(x["title"], x["sent_id"])][0] + x["offset"], axis="columns")
    spans_df["char_end"] = spans_df.apply(lambda x: all_spans[x.name][(x["title"], x["sent_id"])][1] + x["offset"], axis="columns")
    spans_df["tok_start"] = spans_df.apply(lambda x: get_token_span(x["enc"]["offset_mapping"], x["char_start"], x["char_end"])[0], axis="columns")
    spans_df["tok_end"] = spans_df.apply(lambda x: get_token_span(x["enc"]["offset_mapping"], x["char_start"], x["char_end"])[1], axis="columns")
    spans_df["match"] = spans_df.apply(lambda x: x["sentences"].strip() in tok.decode(x["enc"]["input_ids"][x["tok_start"] : x["tok_end"] + 1]), axis="columns")
    spans_df["model"] = model_name

    spans_df = spans_df[["title", "sent_id", "sentence_tag", "model", "tok_start", "tok_end", "n_tokens", "match"]]

    print(len(spans_df), spans_df["match"].sum())
    spans_df.to_parquet(data_dir.joinpath("hotpot_token_spans_qwen3_8b.parquet"))
