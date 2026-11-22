# QRHead output formats

Files written by `scripts/qrhead_detect.py` and `scripts/qrhead_run.py`. The scoring is adapted from github.com/princeton-pli/QRHead at commit 7a668e7 (MIT License, Copyright (c) 2025 Princeton Language and Intelligence). The licence is in `LICENSES/QRHead_LICENSE`.

## What QRHead does here

QRHead (query-focused retrieval heads) ranks the paragraphs of a prompt by the attention that a few selected heads pay from the question to each paragraph. Like ICR, it shortens the prompt instead of compressing the KV cache. It runs in two steps.

1. Head detection (`scripts/qrhead_detect.py`, run once). 128 bridge questions are drawn from `data/hotpotqa/hotpot_bridge_ab.parquet` with seed 0, none of them in the clean set. Their two gold paragraphs are the paragraphs to be found. For every question, Qwen3-8B reads the prompt once with the real question and once with the content-free question `N/A`. For every head, the attention from the question tokens to each token, averaged over the question tokens, is taken with the question minus with `N/A`. This is the calibrated score of the token. In each paragraph, tokens more than two standard deviations below the paragraph's mean are dropped and the rest are summed, which gives the head's score for that paragraph. A head's QRScore is the summed score of the two gold paragraphs, averaged over the 128 questions. All 1,152 query heads (36 layers × 32 heads) are scored, and the 16 best are kept, the number QRHead uses for models under 10B parameters.

2. Paragraph selection and answering (`scripts/qrhead_run.py`). For every clean question, the calibrated per-token scores of the 16 heads are added up, the same mean minus two standard deviations rule is applied inside each paragraph, and the remaining tokens are summed into one score per paragraph. This is done once per question. The 10 paragraphs are ranked from the highest score to the lowest, ties keeping their prompt order. For each budget, `select_paragraphs` chooses how many of the top paragraphs to keep (see below). The kept paragraphs, in their original order, go into a new prompt built by `scripts/prompts.py`, and Qwen3-8B answers from it with `scripts/generation.py`. The answer is scored with `scripts/scoring.py`.

Both steps use the prompt the model answers from, built by `scripts/prompts.py`, so the paragraph positions match `data/hotpotqa/hotpot_token_spans_qwen3_8b.parquet`. The question part of the prompt runs from `Question:` to the end of the chat template. A paragraph is the tokens of its sentences. When the tokenizer merges a paragraph's last character with the blank line after it, that token belongs to the paragraph.

## Budgets and paragraph selection

QRHead runs at budgets `0.2` and `0.1` only. One paragraph is about a tenth of the prompt, so at `0.05` the target would be about half a paragraph.

`select_paragraphs` in `scripts/paragraph_budget.py` takes the paragraphs in rank order and keeps the top k, where k brings the kept tokens closest to `budget * prompt_tokens`. The kept tokens are the k paragraphs plus every token outside the paragraphs (the instruction, the question and the chat template), which are always kept. k is at least 1, and a tie goes to the smaller k. Because the target counts the whole prompt, `prompt_kept_fraction` is comparable with the kvpress methods, which keep the same share of every layer and head.

## `data/hotpotqa/qrhead_scores_qwen3_8b.json`

The QRScore of every head of Qwen3-8B. A JSON object with one entry per head, sorted from the highest score to the lowest. The key is `"layer-head"` (for example `"23-10"`, layer 23, query head 10, both counted from 0) and the value is the QRScore as a float.

## `data/hotpotqa/qrhead_heads_qwen3_8b.txt`

The 16 heads with the highest QRScore, best first, on one line separated by commas, in the same `"layer-head"` form. `scripts/qrhead_run.py` reads this file. Head numbers count the 32 query heads of a layer, not the 8 KV heads.

## `data/hotpotqa/qrhead_detection_ids.txt`

The HotpotQA `id` of the 128 detection questions, one per line.

## `data/hotpotqa/qrhead_detect_env.txt`

GPU, driver, Python, torch and CUDA versions and `pip freeze` of the detection run, written by `notebooks/run_qrhead_detect.ipynb`.

## `results/answers/qrhead_{budget}.parquet`

One row per question, indexed by the HotpotQA `id`. Budgets are `0.2` and `0.1`. Both files come from the same run.

These columns are shared with all methods (`results/README.md`):

| column | type | meaning |
|---|---|---|
| `method` | str | always `qrhead` |
| `budget` | float | `0.2` or `0.1` |
| `model` | str | e.g. `Qwen/Qwen3-8B` |
| `device` | str | e.g. `NVIDIA L4` |
| `question` | str | the question |
| `gold` | str | HotpotQA gold answer |
| `answer` | str | model answer generated from the shortened prompt, stripped, without the stop token |
| `contains` | bool | the normalized gold appears in the normalized answer as whole words |
| `contains_alias` | bool | the same, with the accepted aliases. This is the main metric. |
| `f1` | float | token-level F1 against the gold, from 0 to 1 |
| `prompt_tokens` | int | length of the original prompt in tokens. It equals `n_tokens` of the mask. |
| `new_tokens` | int | length of the answer in tokens, without the stop token |
| `truncated` | bool | generation stopped at the 64-token limit |
| `peak_mem_gb` | float | peak GPU memory in GiB (2^30 bytes), model weights included, from the start of the scoring pass to the end of the answer. NaN off CUDA. |
| `prefill_s` | float | always NaN |
| `decode_s_per_token` | float | always NaN |

`prefill_s` and `decode_s_per_token` are NaN because the answer comes from a single `generate()` call in `scripts/generation.py`, which cannot be timed in parts. `generate_s` holds the time of the whole call.

This column is shared with all compressed runs:

| column | type | meaning |
|---|---|---|
| `prompt_kept_fraction` | float | mean of the mask, the share of the original prompt that was kept, instruction, question and template included |

These columns are QRHead's own:

| column | type | meaning |
|---|---|---|
| `rank_s` | float | seconds for scoring the 10 paragraphs (the passes with the question and with `N/A`). It is measured once per question, so it is the same in both budget files. |
| `generate_s` | float | seconds for answering from the shortened prompt, the whole `generate()` call |
| `kept_titles` | list[str] | titles of the kept paragraphs, in prompt order. It is read back from parquet as a NumPy array of str. |
| `n_paragraphs_kept` | int | number of kept paragraphs, 1 to 10 |
| `short_prompt_tokens` | int | length in tokens of the shortened prompt the answer was generated from |
| `a_kept` | bool | paragraph A was kept |
| `b_kept` | bool | paragraph B was kept |

All times are taken after `torch.cuda.synchronize()`. The memory counter is reset once per question, before the scoring pass, and read after each answer. The `0.2` answer is generated first, so the `0.1` value of `peak_mem_gb` also covers the `0.2` answer of the same question. The scoring pass reads the full prompt and is the largest part in both.

`prompt_kept_fraction × prompt_tokens` counts the kept tokens in the original prompt, and `short_prompt_tokens` counts the tokens of the shortened prompt. They are equal when the kept paragraphs are tokenized the same way in both prompts.

## `results/raw/kept/qrhead/{budget}/{id}.npz`

One file per question, written with `np.savez_compressed(path, kept=kept)`. It holds a single bool array `kept` of shape `(36 layers, 8 KV heads, prompt_tokens)`. It is True for every token of a kept paragraph and for every token outside the paragraphs (the instruction, the question, the chat template and any blank line that is a token of its own), and False for every token of a dropped paragraph. It is the same in every layer and KV head. No token belongs to two paragraphs. Positions are the same as in `data/hotpotqa/hotpot_token_spans_qwen3_8b.parquet`, and `scripts/qrhead_run.py` stops if the prompt length differs from `n_tokens` there. `scripts/retention.py` reads these masks directly, and QRHead gives every A and B sentence a `kept_fraction` of exactly 0 or 1.

The trial run on 3 questions in `notebooks/run_qrhead.ipynb` writes its masks to `results/raw/kept_test/qrhead/{budget}/` and its answers to `/content/qrhead_test/answers/`, so it never mixes with the full run.

## `results/answers/qrhead_{budget}_config.json`

The settings of the run, one file per budget.

| field | type | meaning |
|---|---|---|
| `method` | str | always `qrhead` |
| `budget` | float | `0.2` or `0.1` |
| `model` | str | e.g. `Qwen/Qwen3-8B` |
| `heads` | str | the 16 heads used, as in `qrhead_heads_qwen3_8b.txt` |
| `heads_file` | str | name of the heads file, e.g. `qrhead_heads_qwen3_8b.txt` |
| `null_query` | str | the calibration question, `N/A` |
| `selection` | str | the paragraph selection rule, `paragraph_budget.select_paragraphs, kept paragraphs in prompt order` |
| `attn_implementation` | str | `recorder`, the attention of `scripts/attention_recorder.py` |
| `max_new_tokens` | int | generation limit, 64 |
| `n_questions` | int | number of questions in the run |
| `device` | str | e.g. `NVIDIA L4` |
| `versions` | dict | `transformers` and `torch` versions |

## `results/answers/qrhead_{budget}_env.txt`

GPU, driver, Python, torch and CUDA versions, the repository commit and `pip freeze` of the run, written by `notebooks/run_qrhead.ipynb`. One run writes both budgets, so the two files are identical.

## Differences from the original QRHead

The scoring follows `score_docs` and `score_docs_per_head_for_detection` in the QRHead code: the calibration with `N/A`, the mean minus two standard deviations rule, the sum over the selected heads and the ranking with ties in document order. The rest differs as follows.

- Attention is read with `scripts/attention_recorder.py`. It computes the softmax of the recorded query rows in float32 on top of sdpa attention, with the model in bfloat16. The original uses its own model code with flash attention in float16.
- The prompt is the project's prompt from `scripts/prompts.py`, the one the model answers from. It has no paragraph titles, no `[i]` numbers and no retrieval instruction, it says `Question:` instead of `Query:`, and it ends with Qwen3's chat template with an empty thinking block. Paragraph text is neither stripped nor truncated.
- The heads are detected on 128 HotpotQA bridge questions with their two gold paragraphs, not on LongMemEval or Natural Questions. Paragraphs are identified by title instead of a document index, and the scores are saved as a JSON object keyed `"layer-head"`.
- The number of kept paragraphs is chosen per question by the budget with `select_paragraphs`, instead of a fixed top k (5 in the original examples).
- The answer is generated by `scripts/generation.py` (greedy, at most 64 new tokens) from the project's prompt with the kept paragraphs in their original order. The original generates with vLLM at temperature 0, up to 500 tokens, with its own answer templates. It also restores the original order of the kept documents in its CLIPPER script.
- The same Qwen3-8B model ranks the paragraphs and answers. In the original, retrieval and generation are separate runs and can use different models.

## Names that differ from `docs/icr_fields.md`

- The share of the original prompt that was kept is `prompt_kept_fraction` in `scripts/qrhead_run.py` and `kept_fraction` in `docs/icr_fields.md`.
- QRHead's answers tables have `prefill_s` and `decode_s_per_token` (always NaN). The ICR tables have neither.
- ICR has `target_tokens` and `paragraph_tokens_kept`, and a per-paragraph score file, `results/icr_scores.parquet`. QRHead has none of these.
- The config files name the paragraph rule `selection` (QRHead) and `budget_rule` (ICR). ICR's config also has `scoring`, `query_span` and `mask`, which QRHead's does not. QRHead's has `heads` and `heads_file`.
- Test masks go to `results/raw/kept_test/qrhead/` for QRHead and to `results/raw/kept_limitN/` for ICR.
