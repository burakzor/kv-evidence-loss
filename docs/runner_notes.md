# kvpress runner: notes and decisions

Choices in `scripts/run_kvpress.py` that the report must state. Checked against the installed `kvpress==0.5.5` source.

## Run (Colab L4, the team's GPU)
```bash
pip install -q -r requirements.txt   # pins kvpress==0.5.5, transformers==5.2.0, pandas==2.2.3
python scripts/run_kvpress.py --method snapkv --budget 0.1 --limit 1   # test: first question only
python scripts/run_kvpress.py --method snapkv --budget 0.1             # all 169
```
Methods: `nopress`, `snapkv`, `streamingllm`, `random` and `h2o`, named as in `results/README.md`. `nopress` is the budget-1.0 check without compression. It is not named `full`, so it cannot overwrite the baseline's `full_1.0.parquet`. DuoAttention and AdaKV are not used. The budget is the share of prompt tokens kept.

Outputs are listed in `docs/kvpress_fields.md`. `scripts/retention.py` reads every mask under `results/raw/kept/`. Test runs therefore write their masks elsewhere: `--limit N` to `results/raw/kept_limitN/` (as ICR does) and `--ids` to `results/raw/kept_ids/`. `nopress` writes no masks.

## Decisions
1. **Last prompt token held back.** kvpress compresses a layer's cache after that layer's attention, so the prefill pass itself still sees the full prompt. The runner prefills tokens `0..n-2` inside the press and feeds token `n-1` (the chat template's final newline) outside it. Every answer token is therefore predicted from the compressed cache. The held-back token counts toward the budget: `int(budget * n)` positions per head are kept in total, and the last position in `kept` is always `True`. The press gets a per-question `compression_ratio` (saved in the column of that name) that makes it keep exactly one position less.
2. **Spans match the run prompt.** `token_spans.py` builds the prompt with `prompts.build_text`, the code the runner uses, instruction line included. `tok_start` and `tok_end` therefore index the run prompt directly (`tok_end` inclusive). An older spans file was built without the instruction and was 15 tokens off. It has been regenerated. For every question the runner checks that the prompt length equals `n_tokens` and that each span decodes to its sentence, and it stops otherwise.
3. **Whole prompt under the press.** kvpress's `pipeline` is not used, because it adds the question after compression. Prefill and greedy decoding copy the pipeline's steps. The stop rule, the 64-token limit and scoring are the baseline's.
4. **Recording kept tokens.** For scorer presses, the runner uses a copy of `ScorerPress.compress` that also stores the `scores.topk(n_kept)` indices per layer and head. Masking presses (HeadKV, run through the runner as `kind "mask"`) do not prune. For them, kept = all positions minus `module.masked_key_indices`. In both cases the masks stay on the GPU until decoding has finished and are moved to the CPU after the timed part.
5. **Settings (kvpress defaults).** SnapKV `window_size=64`, `kernel_size=5`. StreamingLLM `n_sink=4`. Random `seed=0`. `h2o` is kvpress's `ObservedAttentionPress` with eager attention (all others use sdpa). It is the closest kvpress counterpart of H2O, not H2O itself. It scores tokens by the average attention they receive during prefill and evicts once, with no recency window and no eviction during decoding. The report must describe it that way.

## For the report
- SnapKV always keeps its last 64 tokens (question and template). At 5% of a median prompt (1,380 tokens, 69 kept), only 4 are chosen by attention score.
- Peak memory includes the ~16 GB of weights. `kv_cache_mb` shows the cache alone.
- kvpress `RandomPress` re-seeds on every call, so every layer and head gets the same random pattern.
- `decode_s_per_token` counts every decoding forward pass, stop token included.
- `prefill_s` and `decode_s_per_token` measure only the model's work. The mask recording is not timed.

## Checks before the full runs
1. `--limit 1 --inspect-layer 15`: runs end to end and prints kept tokens and A/B survival.
2. `--method nopress --budget 1.0`: answers equal `full_1.0.parquet` (same L4 and pins as the baseline).
3. `--method random --budget 0.1`: mean `prompt_kept_fraction` ≈ 0.1.
4. `h2o` with `--ids` on the longest prompt (2,698 tokens) fits in the L4's 24 GB.
5. `streamingllm` at 0.1 on all questions, with the masks on Google Drive for the team's mask comparison.
