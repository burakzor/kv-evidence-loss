# Scoring adapted from github.com/princeton-pli/QRHead at commit 7a668e7 (MIT License, Copyright (c) 2025 Princeton Language and Intelligence)
from __future__ import annotations
from pathlib import Path
from itertools import product
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel, PreTrainedTokenizerBase
import attention_recorder
from generation import get_device
from prompts import build_text, get_prompt, load_questions
from token_spans import get_token_span

MODEL_NAME = "Qwen/Qwen3-8B"
N_QUESTIONS = 128
SEED = 0
N_TOP_HEADS = 16  # QRHead uses 16 heads for models under 10B parameters
NULL_QUERY = "N/A"
DATA_DIR = Path(__file__).parent.parent / "data" / "hotpotqa"

def get_detection_questions(n_questions: int) -> pd.DataFrame:
    """Return n_questions bridge questions outside the clean set, with their two gold paragraph titles."""
    ab_df = pd.read_parquet(DATA_DIR / "hotpot_bridge_ab.parquet")
    clean_df = pd.read_parquet(DATA_DIR / "hotpot_bridge_ab_clean.parquet")
    clean_filt = ab_df.index.isin(clean_df.index)
    candidate_ids = sorted(ab_df.loc[~clean_filt].index.unique())
    rng = np.random.default_rng(SEED)
    ids = sorted(str(qid) for qid in rng.choice(candidate_ids, n_questions, replace=False))
    questions = load_questions(ids)
    questions["gold_titles"] = ab_df.loc[ids].groupby(level=0)["title"].apply(lambda x: sorted(set(x)))
    return questions

def compose_scoring_prompt(tok: PreTrainedTokenizerBase, context: dict, question: str) -> tuple[list[int], tuple[int, int], dict[str, tuple[int, int]]]:
    """Return the prompt token ids, the query span and the token span of every paragraph (keyed by title)."""
    text, spans = build_text(context, question)
    prompt = get_prompt(tok, text)
    offset = prompt.find(text)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    doc_spans = {}
    for title, sents in zip(context["title"], context["sentences"]):
        start = spans[(title, 0)][0] + offset
        end = spans[(title, len(sents) - 1)][1] + offset
        doc_spans[title] = get_token_span(enc["offset_mapping"], start, end)
    # As in QRHead, the query span runs from the query line to the end of the prompt, chat template suffix included.
    # The query line is the last block of the text, after the last blank line.
    query_start = offset + text.rfind("\n\n") + 2
    query_span = get_token_span(enc["offset_mapping"], query_start, len(prompt))
    return enc["input_ids"], query_span, doc_spans

def score_per_token_attention_to_query(model: PreTrainedModel, input_ids: list[int], query_span: tuple[int, int], cache: DynamicCache, start_idx: int) -> torch.Tensor:
    """Return the attention of the query tokens to every token, averaged over query tokens: (n_layers, n_heads, n_tokens)."""
    attention_recorder.record(range(query_span[0] - start_idx, query_span[1] - start_idx + 1))
    with torch.inference_mode():
        # the model body without the LM head, since no logits are needed
        model.model(input_ids=torch.tensor([input_ids[start_idx:]], device=model.device), past_key_values=cache)
    per_token_scores = attention_recorder.get_recorded().mean(2)  # average over query tokens
    attention_recorder.record([])
    return per_token_scores

def score_docs_per_head_for_detection(model: PreTrainedModel, tok: PreTrainedTokenizerBase, question: str, context: dict) -> dict[str, torch.Tensor]:
    """Return the calibrated score of every paragraph for every head: title -> (n_layers, n_heads) tensor."""
    input_ids, query_span, doc_spans = compose_scoring_prompt(tok, context, question)
    null_input_ids, null_query_span, _ = compose_scoring_prompt(tok, context, NULL_QUERY)

    # up to query prompt the tokenized_prompt is the same
    assert null_input_ids[: query_span[0]] == input_ids[: query_span[0]]
    assert query_span[0] == null_query_span[0], "Query start indices do not match between query and null query."

    # scoring with actual query
    cache = DynamicCache()
    per_token_scores = score_per_token_attention_to_query(model, input_ids, query_span, cache, 0)

    # use kv_cache from first query to speed up forward() for the calibration query.
    cache.crop(query_span[0])
    start_idx = query_span[0]
    null_per_token_scores = score_per_token_attention_to_query(model, null_input_ids, null_query_span, cache, start_idx)

    min_length = min(per_token_scores.shape[-1], null_per_token_scores.shape[-1])
    per_token_scores_CAL = per_token_scores[:, :, :min_length] - null_per_token_scores[:, :, :min_length]  # shape: (n_layers, n_heads, n_tok)

    # remove abnormally calibrated scores and aggregate scores across tokens to get per-document scores tensor (n_layers, n_heads)
    per_doc_score_tensors = {}
    for title, doc_span in doc_spans.items():
        curr_doc_per_tok_scores_CAL = per_token_scores_CAL[:, :, doc_span[0] : doc_span[1] + 1]  # shape: (n_layers, n_heads, n_tok)
        threshold = curr_doc_per_tok_scores_CAL.mean(dim=-1) - 2 * curr_doc_per_tok_scores_CAL.std(dim=-1)  # shape: (n_layers, n_heads)
        # broadcast threshold over tokens: compare (n_layers, n_heads, n_tok) > (n_layers, n_heads, 1)
        tok_mask = curr_doc_per_tok_scores_CAL > threshold.unsqueeze(-1)
        # zero out tokens below threshold and sum over tokens -> (n_layers, n_heads)
        masked_scores = curr_doc_per_tok_scores_CAL.masked_fill(~tok_mask, 0.0)
        per_doc_score_tensors[title] = masked_scores.sum(dim=-1)
    return per_doc_score_tensors

def lme_eval(retrieval_results: dict[str, dict[str, float]], data_instances: list[dict]) -> float:
    """Return the QRScore of one head: the summed score of the gold paragraphs, averaged over questions."""
    all_score_over_gold = []
    for data in data_instances:
        qid = data["idx"]
        gt_docs = data["gt_docs"]  # a list of doc ids
        doc_id2score = retrieval_results[qid]  # doc_id -> score
        if len(gt_docs) == 0:
            score_over_gold = 0
        else:
            score_over_gold = np.sum([doc_id2score[doc_id] for doc_id in gt_docs])
        all_score_over_gold.append(score_over_gold)
    return np.mean(all_score_over_gold)

def score_heads(doc_scores_per_head: dict[str, dict[str, torch.Tensor]], data_instances: list[dict]) -> list[tuple[str, float]]:
    """Return (layer-head, QRScore) for every head, sorted from the highest score."""
    first_qid = next(iter(doc_scores_per_head))
    first_doc_id = next(iter(doc_scores_per_head[first_qid]))
    num_layers, num_heads = doc_scores_per_head[first_qid][first_doc_id].shape

    head_scores = {}
    for layer, head in tqdm(product(range(num_layers), range(num_heads)), total=num_layers * num_heads):
        retrieval_results = {}  # qid -> {doc_id -> score} for this head
        for qid, per_doc_score_tensors in doc_scores_per_head.items():
            doc_id2score = {}
            for doc_id, score_tensor in per_doc_score_tensors.items():
                doc_id2score[doc_id] = score_tensor[layer][head].item()
            retrieval_results[qid] = doc_id2score
        head_scores[(layer, head)] = lme_eval(retrieval_results, data_instances)

    head_scores_list = [(f"{layer}-{head}", score) for (layer, head), score in head_scores.items()]
    head_scores_list.sort(key=lambda x: x[1], reverse=True)
    return head_scores_list

def detect(model_name: str, n_questions: int, out_dir: Path) -> list[tuple[str, float]]:
    """Return the QRScore of every head, and write the scores, the best heads and the question ids to out_dir."""
    questions = get_detection_questions(n_questions)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map=get_device(), attn_implementation=attention_recorder.ATTN_NAME).eval()

    doc_scores_per_head = {}  # qid -> {title -> (n_layers, n_heads) tensor}
    data_instances = []
    for qid, q in tqdm(questions.iterrows(), total=len(questions)):
        doc_scores_per_head[qid] = score_docs_per_head_for_detection(model, tok, q["question"], q["context"])
        data_instances.append({"idx": qid, "gt_docs": q["gold_titles"]})
    head_scores_list = score_heads(doc_scores_per_head, data_instances)

    model_tag = model_name.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    top_heads = [head for head, _ in head_scores_list[:N_TOP_HEADS]]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"qrhead_scores_{model_tag}.json").write_text(json.dumps(dict(head_scores_list), indent=1))
    (out_dir / f"qrhead_heads_{model_tag}.txt").write_text(",".join(top_heads) + "\n")
    (out_dir / "qrhead_detection_ids.txt").write_text("\n".join(questions.index) + "\n")

    n_gold = questions["gold_titles"].apply(len)
    print("questions:", len(questions), "| gold paragraphs per question:", n_gold.min(), "to", n_gold.max(), "| heads scored:", len(head_scores_list))
    print("written to", out_dir)
    print(f"top {N_TOP_HEADS} heads:", ",".join(top_heads))
    for head, score in head_scores_list[:N_TOP_HEADS]:
        print(f"  {head:6} {score:.4f}")
    return head_scores_list

if __name__ == "__main__":
    detect(MODEL_NAME, N_QUESTIONS, DATA_DIR)