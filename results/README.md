# Results

## `answers/{method}_{budget}.parquet`
One file per method and budget, one row per question, index `id`. The full-cache baseline is `full_1.0.parquet` (`scripts/run_full_baseline.py`); compressed runs use the same layout, e.g. `snapkv_0.2.parquet`.

| column | type | meaning |
|---|---|---|
| `id` (index) | str | HotpotQA question id |
| `method` | str | `full`, `snapkv`, `h2o`, ... |
| `budget` | float | share of the prompt's KV cache kept (1.0 = full cache) |
| `model` | str | e.g. `Qwen/Qwen3-8B` |
| `device` | str | GPU name (`NVIDIA A100-SXM4-40GB`, `mps`, ...); compare runs only within the same device |
| `question` | str | question text |
| `gold` | str | HotpotQA gold answer |
| `answer` | str | model output, stripped, stop token removed |
| `contains` | bool | normalized gold appears in normalized answer as whole words (main metric) |
| `f1` | float | token-level F1 as in the official HotpotQA script |
| `prompt_tokens` | int | prompt length in tokens |
| `new_tokens` | int | generated tokens, stop token excluded |
| `truncated` | bool | generation hit `max_new_tokens` before the stop token |

Generation: `scripts/prompts.py` (prompt with instruction, Qwen3 chat template, thinking disabled) and `scripts/generation.py` (bfloat16, greedy, `max_new_tokens=64`). Scoring: `scripts/scoring.py`.

## `answers/{method}_{budget}_check.csv`
Random sample (seed 0) for a manual check of the scores. `human_correct` is filled in by hand (True/False).

## `raw/`
Large outputs (kept token indices, tensors). Ignored by git.
