# results

This folder holds the outputs of the experiments: the model's answers for every method and memory budget, the kept-token masks that show what each compression method left in the KV cache, and the retention table computed from them. All runs use Qwen3-8B on an NVIDIA L4 with the package versions in `requirements.txt`.

## what's in here

```
results/
  answers/
    {method}_{budget}.parquet        answers and costs, one row per question
    {method}_{budget}_env.txt        GPU, driver and package versions of the run
    {method}_{budget}_check.csv      20 answers whose scores were checked by hand
  raw/kept/{method}/{budget}/
    {id}.npz                         kept-token mask of one question
  retention.parquet                  share of each A/B sentence that was kept
  retention_by_layer.parquet         the same share for every layer separately
```

`raw/` is too large for git and is kept on the team's Google Drive. Everything else is committed.

The methods are `full` (no compression), `snapkv`, `h2o`, `streamingllm`, `random`, `compresskv`, `headkv`, `icr` and `qrhead`, always spelled like this. The budget is the share of the prompt's KV cache that is kept: `1.0` is the full cache, `0.2`, `0.1` and `0.05` keep a fifth, a tenth and a twentieth of it. kvpress describes the same thing the other way round, as the fraction removed, so a budget of 0.1 is `compression_ratio=0.9`.

## answers

`answers/full_1.0.parquet` is the full-cache baseline from `scripts/run_full_baseline.py`. Compressed runs have the same layout, e.g. `answers/snapkv_0.1.parquet`. Each file has one row per question, indexed by the HotpotQA `id`, with these columns:

- `method`, `budget`: as in the file name
- `model`, `device`: e.g. `Qwen/Qwen3-8B` and `NVIDIA L4`
- `question`, `gold`: the question and its HotpotQA gold answer
- `answer`: what the model wrote, stripped, without the stop token
- `contains`: the normalized gold answer appears in the normalized model answer as whole words
- `contains_alias`: the same, but a shorter accepted form of the gold (see aliases below) also counts. This is the main metric.
- `f1`: token-level F1 against the gold, as in the official HotpotQA evaluation script
- `prompt_tokens`, `new_tokens`: length of the prompt and of the generated answer, in tokens
- `truncated`: generation stopped at the 64-token limit before the model finished
- `peak_mem_gb`: peak GPU memory while answering, from `torch.cuda.max_memory_allocated()`
- `prefill_s`: seconds spent processing the prompt
- `decode_s_per_token`: average seconds per generated token

Timings are taken after `torch.cuda.synchronize()`, so they include the GPU work and not just the time to queue it. Prompts are built by `scripts/prompts.py`: an instruction to answer with a short phrase, the ten HotpotQA paragraphs, then the question, wrapped in the Qwen3 chat template with thinking disabled. Answers are generated greedily in bfloat16 by `scripts/generation.py` and scored by `scripts/scoring.py`.

Results from different GPUs or package versions can differ slightly, which is why every run comes with its `_env.txt`. Rerunning the full-cache baseline with the pinned versions gave a byte-identical answers file.

## aliases

Some HotpotQA gold answers are more specific than the question asks for. The question asks for a year, the gold says "June 2003", and a model that answers "2003" fails `contains`. `data/hotpotqa/answer_aliases.csv` lists shorter forms of the gold that still fully answer the question. `scripts/make_aliases.py` derives them from the question and the gold answer only, never from model outputs, and every one was checked by hand. The same aliases apply to every method. The rules:

1. Drop a noun the question already asks for: "43,000 marchers" (how many marchers) → "43,000".
2. Year only, when the question asks for a year and not a day or date: "June 2003" → "2003".
3. City only, for "City, Region/Country" answers, never the region alone: "Knoxville, Tennessee" → "Knoxville".
4. Drop leading prepositions and hedges, and "above sea level": "from 1989 until 1994" → "1989 until 1994".
5. Drop company suffixes and titles: "President Bill Clinton" → "Bill Clinton".
6. Spelling only: number words up to twenty and digits, unit abbreviations: "ten" ↔ "10", "ft" ↔ "feet".

Other names, synonyms, abbreviations that aren't in the gold, partial lists and vaguer answers are never accepted. The rules were written after a manual check of the baseline showed correct short answers failing `contains`. On the 20 hand-checked baseline answers, `contains_alias` agreed with the human verdict 20 times and `contains` 18 times.

## model filter

`data/hotpotqa/filter_qwen3_8b.parquet` marks the questions that Qwen3-8B answers only with the full context. `scripts/model_filter.py` asks every question again with only paragraph A, only paragraph B, and no paragraph, using the same prompt template, generation and scoring as the baseline; the full-context results are read from `answers/full_1.0.parquet`. It has one row per question, indexed by `id`, with these columns:

- `full_contains`, `a_only_contains`, `b_only_contains`, `no_ctx_contains`: `contains_alias` of the answer in each condition
- `keep`: the full-context answer is correct, and the A-only and no-paragraph answers are not. RQ2 runs on these questions.
- `*_contains_strict`: the same with `contains`
- `*_answer`: the model's answer in each condition

| condition | correct (`contains_alias`) |
|---|---|
| full context | 131 / 169 |
| A only | 18 / 169 |
| B only | 156 / 169 |
| no paragraph | 11 / 169 |

111 questions are kept (106 with `contains`). B only is recorded but not used in `keep`: with B as the only paragraph the model takes the one answer-shaped span and does not need A, so it answers more questions than with the full context, and filtering on it would leave 5 questions. The labeling already removed questions whose wording names B's entity. Rerunning with `--rebuild` recomputes `keep` from the saved columns without the model. The environment of the run is in `data/hotpotqa/filter_qwen3_8b_env.txt`.

## kept tokens

For every compressed run there is one mask per question in `raw/kept/{method}/{budget}/{id}.npz`. It holds a single boolean array `kept` of shape `(n_layers, n_kv_heads, n_tokens)`. `kept[l, h, t]` is True if prompt token `t` was still in the cache of layer `l` and KV head `h` once the prompt had been processed. `n_tokens` is the prompt length and always equals `prompt_tokens` in the answers file. ICR and QRHead drop whole paragraphs rather than single tokens, so their masks are True for every token of a kept paragraph in all layers and heads.

Token positions are the same as in `data/hotpotqa/hotpot_token_spans_qwen3_8b.parquet`, which gives the first and last token (`tok_start`, `tok_end`, both inclusive) of every A and B sentence. A is the supporting sentence that names the bridge entity, B the one that contains the answer. From the repository root, this is how much of one question's B sentence SnapKV kept at a budget of 0.1:

```python
import numpy as np
import pandas as pd

qid = "5a85f5115542991dd0999ed9"
spans = pd.read_parquet("data/hotpotqa/hotpot_token_spans_qwen3_8b.parquet").loc[[qid]]
kept = np.load(f"results/raw/kept/snapkv/0.1/{qid}.npz")["kept"]

b = spans[spans["sentence_tag"] == "B"].iloc[0]
print(kept[:, :, b["tok_start"] : b["tok_end"] + 1].mean())
```

The masks are written with `np.savez_compressed(path, kept=kept)`.

## retention

`retention.parquet` does the calculation above for every question, method, budget and A/B sentence. It is produced by `scripts/retention.py` and has the columns `id`, `method`, `budget`, `title` and `sent_id` (which sentence, as in the token spans file), `sentence_tag` (A or B) and `kept_fraction`, the share of the sentence's tokens that were kept, averaged over all layers and KV heads. Comparing A with B, and both with `random`, answers RQ1: which supporting sentence a compression method keeps.

`retention_by_layer.parquet` comes from the same run and has the same columns plus `layer` (0 to 35 for Qwen3-8B), with one row per sentence and layer. Its `kept_fraction` is averaged over the KV heads of that layer only, so it shows in which layers a sentence was dropped. For a method that keeps the same tokens in every layer, such as `streamingllm`, all layers give the same value.

`scripts/retention.py` reads the masks in `raw/kept/`. To read them from somewhere else, for example from Google Drive, change `kept_dir` at the top of the main block. Masks whose `n_tokens` differs from the token spans of their question are skipped and listed at the end.