# kvpress runner: notes and decisions

Choices in `scripts/run_kvpress.py` that the report must state. Checked against the installed `kvpress==0.5.5` source.

## Run (Colab L4, the team's GPU)
```bash
pip install -q -r requirements.txt
pip install -q kvpress==0.5.5 transformers==5.2.0 pandas==2.2.3   # same pins as notebooks/run_full_baseline.ipynb
python scripts/run_kvpress.py --method snapkv --budget 0.1 --limit 1   # test: first question only
python scripts/run_kvpress.py --method snapkv --budget 0.1             # all 169
```
Methods: `nopress` (budget 1.0, no compression; not named `full` so it cannot overwrite the baseline's `full_1.0.parquet`), `snapkv`, `streamingllm`, `random`, `observed_attention`. AdaKV was dropped by the team. Budget = share of prompt tokens kept.

Outputs: `results/answers/{method}_{budget}.parquet` (`_limitN` suffix for test runs), a `_config.json` with settings and versions, and `results/raw/kept/{method}/{budget}/{id}.npz` with `kept`, bool `(36 layers, 8 KV heads, prompt_tokens)`. Extra columns: `n_kept`, `kept_fraction`, `kv_cache_mb`, `prefill_s`, `decode_s_per_token`, `peak_mem_gb`.

## Decisions
1. **Last prompt token held back.** kvpress compresses a layer's cache after that layer's attention, so the prefill pass itself still sees the full prompt. The runner prefills tokens `0..n-2` inside the press and feeds token `n-1` (the chat template's final newline) outside it, so every answer token is predicted from the compressed cache. The held-back token counts toward the budget: `int(budget * n)` positions per head are kept in total, and the last position in `kept` is always `True`.
2. **Spans match the run prompt.** `token_spans.py` builds the prompt with `prompts.build_text`, the code the runner uses, instruction line included, so `tok_start` and `tok_end` index the run prompt directly (`tok_end` inclusive). An older spans file was built without the instruction and was 15 tokens off; it has been regenerated. For every question the runner checks that the prompt length equals `n_tokens` and that each span decodes to its sentence, and stops otherwise.
3. **Whole prompt under the press.** kvpress's `pipeline` is not used, because it adds the question after compression. Prefill and greedy decoding copy the pipeline's steps; the stop rule, the 64-token limit and scoring are the baseline's.
4. **Recording kept tokens.** For scorer presses, the runner uses a copy of `ScorerPress.compress` that also stores the `scores.topk(n_kept)` indices per layer and head. Masking presses (only DuoAttention, if it is added) do not prune; there, kept = all positions minus `module.masked_key_indices`, and the runner already has this path.
5. **Settings (kvpress defaults).** SnapKV `window_size=64`, `kernel_size=5`. StreamingLLM `n_sink=4`. Random `seed=0`. ObservedAttention uses eager attention (all others sdpa); it is the closest kvpress counterpart of H2O, not H2O.

## For the report
- SnapKV always keeps its last 64 tokens (question and template). At 5% of a median prompt (1,380 tokens, 69 kept), only 4 are chosen by attention score.
- Peak memory includes the ~16 GB of weights; `kv_cache_mb` shows the cache alone.
- Random re-seeds per call, so every layer and head gets the same random pattern.
- `decode_s_per_token` counts every decoding forward pass, stop token included.

## Checks before the full runs
1. `--limit 1 --inspect-layer 15`: runs end to end and prints kept tokens and A/B survival.
2. `--method nopress --budget 1.0`: answers equal `full_1.0.parquet` (same L4 and pins as the baseline).
3. `--method random --budget 0.1`: mean `kept_fraction` ≈ 0.1.
4. `observed_attention` on the longest prompt (2,683 tokens) fits in the L4's 24 GB.

## Open
- **DuoAttention (team decision).** There is no Qwen3 pattern (it would need the experimental `on_the_fly_scoring`), and its 128 sink + 256 recent tokens keep at least ~28%, so the 20/10/5% budgets are out of reach with defaults.
- `results/README.md` mentions `scripts/rescore.py`, which does not exist, and uses `h2o` as a method name; the runner uses `observed_attention`.
