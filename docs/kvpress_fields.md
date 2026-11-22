# kvpress output formats

Files written by `scripts/run_kvpress.py`. The methods, the budget rule and how kept tokens are recorded are described in `docs/runner_notes.md`. Test runs write the same files with a suffix and keep their masks out of `results/raw/kept/`, so `scripts/retention.py` never reads them. Runs with `--limit N` use the suffix `_limitN` and put their masks in `results/raw/kept_limitN/`. Runs with `--ids` use the suffix `_ids` and put their masks in `results/raw/kept_ids/`.

Methods: `snapkv`, `streamingllm`, `random`, `h2o` (kvpress's `ObservedAttentionPress`, the closest counterpart of H2O) and `nopress`. `nopress` is the check run without compression, at budget `1.0` only, and it writes no masks. DuoAttention and AdaKV are not used. Budgets: `0.2`, `0.1`, `0.05`.

## `results/answers/{method}_{budget}.parquet`

One row per question, indexed by the HotpotQA `id`.

These columns are the same as in `full_1.0.parquet`:

| column | type | meaning |
|---|---|---|
| `method` | str | e.g. `snapkv` |
| `budget` | float | `0.2`, `0.1` or `0.05`, and `1.0` for `nopress` |
| `model` | str | e.g. `Qwen/Qwen3-8B` |
| `device` | str | e.g. `NVIDIA L4` |
| `question` | str | the question |
| `gold` | str | HotpotQA gold answer |
| `answer` | str | model answer generated from the compressed KV cache, stripped, without the stop token |
| `contains` | bool | normalized gold appears in the normalized answer |
| `contains_alias` | bool | the same, with the accepted aliases (the main metric) |
| `f1` | float | token-level F1 against the gold |
| `prompt_tokens` | int | length of the prompt, equal to `n_tokens` of the mask |
| `new_tokens` | int | length of the answer in tokens |
| `truncated` | bool | generation stopped at the 64-token limit |

These columns are added by the kvpress runner:

| column | type | meaning |
|---|---|---|
| `n_kept` | int | prompt positions kept per layer and KV head, the held-back last token included, equal to `int(budget * prompt_tokens)` |
| `prompt_kept_fraction` | float | mean of the mask, i.e. the share of the whole prompt still in the cache after prefill |
| `kv_cache_mb` | float | size of the KV cache (keys and values, all layers) after prefill |
| `prefill_s` | float | seconds for the prefill under the press, scoring and eviction included |
| `decode_s_per_token` | float | decoding seconds divided by the number of decoding forward passes (the held-back token's pass and the stop token included) |
| `peak_mem_gb` | float | peak GPU memory over prefill and decoding, model weights included, NaN off CUDA |
| `compression_ratio` | float | the `compression_ratio` given to the press for this question, `0.0` for `nopress` |

`compression_ratio` is set per question so that exactly `n_kept` positions are kept. It is therefore slightly above `1 - budget` (for example `0.90047` instead of `0.9`). The press scores prompt tokens separately in every layer and KV head and keeps exactly `n_kept` positions in each, so different heads keep different tokens. The masks are recorded outside the timed part, so `prefill_s` and `decode_s_per_token` measure only the model's work.

## `results/raw/kept/{method}/{budget}/{id}.npz`

One file per question, written with `np.savez_compressed(path, kept=kept)`. It holds a single bool array `kept` of shape `(36 layers, 8 KV heads, prompt_tokens)`. `True` means that prompt position is still in that layer's and head's cache after prefill. Every layer and head has exactly `n_kept` `True` values, and they usually differ between layers and heads. The last position (the held-back chat template token) is always `True`. Positions are the same as in `data/hotpotqa/hotpot_token_spans_qwen3_8b.parquet`, and the runner checks this for every question. `scripts/retention.py` therefore reads these masks directly. Unlike ICR, an A or B sentence can be partly kept, and its kept fraction can differ by layer and head.

## `results/answers/{method}_{budget}_config.json`

The settings of the run: method, budget, model, the press with its settings (e.g. `SnapKVPress(compression_ratio=0.9, window_size=64, kernel_size=5)`), attention implementation (`sdpa`, or `eager` for `h2o`), the per-question compression-ratio rule, the held-back last token, `max_new_tokens`, number of questions, device, and the `kvpress`, `transformers` and `torch` versions.

## `results/answers/{method}_{budget}_env.txt`

The commit, GPU, driver, Python, CUDA and `pip freeze` of the run, written by `notebooks/run_kvpress.ipynb`.
