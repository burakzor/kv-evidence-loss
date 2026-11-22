from pathlib import Path
import numpy as np
import pandas as pd

def get_retention(kept, spans):
    """Return the spans rows with the kept fraction of each sentence, averaged over all layers and KV heads."""
    retention_df = spans.copy()
    # tok_end is inclusive
    retention_df["kept_fraction"] = retention_df.apply(lambda x: kept[:, :, x["tok_start"] : x["tok_end"] + 1].mean(), axis="columns")
    return retention_df

def get_retention_by_layer(kept, spans):
    """Return the spans rows once per layer with the kept fraction of each sentence, averaged over KV heads."""
    layer_dfs = []
    for layer in range(kept.shape[0]):
        layer_df = spans.copy()
        layer_df["layer"] = layer
        layer_df["kept_fraction"] = layer_df.apply(lambda x: kept[layer, :, x["tok_start"] : x["tok_end"] + 1].mean(), axis="columns")
        layer_dfs.append(layer_df)
    return pd.concat(layer_dfs)

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data" / "hotpotqa"
    results_dir = Path(__file__).parent.parent / "results"
    kept_dir = results_dir / "raw" / "kept"

    spans_df = pd.read_parquet(data_dir.joinpath("hotpot_token_spans_qwen3_8b.parquet"))

    retention_dfs = []
    by_layer_dfs = []
    paths = sorted(kept_dir.glob("*/*/*.npz"))
    wrong_length = []
    for path in paths:
        qid = path.stem
        method = path.parent.parent.name
        budget = float(path.parent.name)
        kept = np.load(path)["kept"]
        spans = spans_df.loc[[qid]]

        # the mask must cover the same prompt as the token spans
        if kept.shape[2] != spans["n_tokens"].iloc[0]:
            wrong_length.append(path)
            continue

        retention_df = get_retention(kept, spans)
        retention_df["method"] = method
        retention_df["budget"] = budget
        retention_dfs.append(retention_df)

        by_layer_df = get_retention_by_layer(kept, spans)
        by_layer_df["method"] = method
        by_layer_df["budget"] = budget
        by_layer_dfs.append(by_layer_df)

    retention_df = pd.concat(retention_dfs).reset_index()
    retention_df = retention_df[["id", "method", "budget", "title", "sent_id", "sentence_tag", "kept_fraction"]]
    by_layer_df = pd.concat(by_layer_dfs).reset_index()
    by_layer_df = by_layer_df[["id", "method", "budget", "title", "sent_id", "sentence_tag", "layer", "kept_fraction"]]

    retention_df.to_parquet(results_dir.joinpath("retention.parquet"), index=False)
    by_layer_df.to_parquet(results_dir.joinpath("retention_by_layer.parquet"), index=False)

    print("files read:", len(paths))
    print("files with wrong length:", len(wrong_length), wrong_length)
    print("rows:", len(retention_df), "rows by layer:", len(by_layer_df))
    print(retention_df.groupby(["method", "budget", "sentence_tag"])["kept_fraction"].mean().unstack())