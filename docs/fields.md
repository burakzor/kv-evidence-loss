# ICR output formats

Files written by `scripts/icr.py`. The method, the budget rule and the masks are described in `results/README.md`. Runs with `--limit N` write the same files with a `_limitN` suffix, and their masks go to `results/raw/kept_limitN/`.

## `results/answers/icr_{budget}.parquet`

One row per question, indexed by the HotpotQA `id`. Budgets are `0.2` and `0.1`; `1.0` is only the check run.

These columns are the same as in `full_1.0.parquet`:

| column | type | meaning |
|---|---|---|
| `method` | str | always `icr` |
| `budget` | float | `0.2` or `0.1` |
| `model` | str | e.g. `Qwen/Qwen3-8B` |
| `device` | str | e.g. `NVIDIA L4` |
| `question` | str | the question |
| `gold` | str | HotpotQA gold answer |
| `answer` | str | model answer generated from the shortened prompt, stripped, without the stop token |
| `contains` | bool | normalized gold appears in the normalized answer |
| `contains_alias` | bool | the same, with the accepted aliases; the main metric |
| `f1` | float | token-level F1 against the gold |
| `prompt_tokens` | int | length of the original prompt; equals `n_tokens` of the mask |
| `new_tokens` | int | length of the answer in tokens |
| `truncated` | bool | generation stopped at the 64-token limit |

These columns are added by ICR:

| column | type | meaning |
|---|---|---|
| `short_prompt_tokens` | int | length of the shortened prompt the answer was generated from |
| `n_paragraphs_kept` | int | number of kept paragraphs, 1 to 10 |
| `kept_titles` | list[str] | titles of the kept paragraphs, in prompt order |
| `target_tokens` | float | `budget * prompt_tokens` |
| `paragraph_tokens_kept` | int | total tokens of the kept paragraphs |
| `a_kept` | bool | paragraph A was kept |
| `b_kept` | bool | paragraph B was kept |
| `kept_fraction` | float | mean of the mask, i.e. the share of the original prompt that was kept, instruction, question and template included |
| `rank_s` | float | seconds for the two scoring passes (question and `N/A`) |
| `generate_s` | float | seconds for answering from the shortened prompt |
| `peak_mem_gb` | float | peak GPU memory over scoring and answering; NaN off CUDA |

The rule stops at the first paragraph that would move the total away from the target, so a long paragraph high in the ranking can leave `paragraph_tokens_kept` well below `target_tokens`.

## `results/icr_scores.parquet`

One row per question and paragraph (169 × 10 rows), indexed by `id`. The ranking does not depend on the budget, so this file is written once per run.

| column | type | meaning |
|---|---|---|
| `title` | str | paragraph title |
| `score` | float | calibrated ICR score |
| `rank` | int | 1 is the most relevant, 10 the least |
| `tok_start` | int | first token of the paragraph in the original prompt |
| `tok_end` | int | last token of the paragraph, inclusive |
| `is_a` | bool | this is paragraph A |
| `is_b` | bool | this is paragraph B |

## `results/raw/kept/icr/{budget}/{id}.npz`

One file per question, written with `np.savez_compressed(path, kept=kept)`. It holds a single bool array `kept` of shape `(36 layers, 8 KV heads, prompt_tokens)`. It is False for every token of a dropped paragraph and True everywhere else, and the same in every layer and head. A token shared at the boundary between a dropped and a kept paragraph counts as kept. Positions are the same as in `data/hotpotqa/hotpot_token_spans_qwen3_8b.parquet`, so `scripts/retention.py` reads these masks directly; ICR gives every A and B sentence a `kept_fraction` of exactly 0 or 1.

## `results/answers/icr_{budget}_config.json`

The settings of the run: model, attention implementation, scoring rule, query span, null query, budget rule, mask rule, `max_new_tokens`, number of questions, device, budget, and the `transformers` and `torch` versions.

## `results/answers/icr_{budget}_env.txt`

GPU, driver, Python, CUDA and `pip freeze` of the run, written by `notebooks/run_icr.ipynb`. One run writes both budgets, so the two files are identical.
