from pathlib import Path
import pandas as pd
from scipy.stats import mannwhitneyu

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    retention = pd.read_parquet(root / "results" / "retention.parquet")
    keep = pd.read_parquet(root / "data" / "hotpotqa" / "filter_qwen3_8b.parquet")["keep"]
    keep_ids = set(keep.index[keep])

    # mean kept fraction of the B sentences of each question
    b = (retention[retention["sentence_tag"] == "B"]
         .groupby(["method", "budget", "id"])["kept_fraction"].mean())

    rows = []
    for (method, budget), b_kept in b.groupby(level=[0, 1]):
        f = root / "results" / "answers" / f"{method}_{budget}.parquet"
        if not f.exists():
            continue
        correct = pd.read_parquet(f)["contains_alias"].astype(bool)
        d = pd.DataFrame({"b_kept": b_kept.droplevel([0, 1]), "correct": correct}).dropna()
        d = d[d.index.isin(keep_ids)]  # the 111 questions whose answer needs the paragraphs
        hi = d["b_kept"] >= d["b_kept"].median()
        ok, bad = d.loc[d["correct"], "b_kept"], d.loc[~d["correct"], "b_kept"]
        p = mannwhitneyu(ok, bad).pvalue if len(ok) and len(bad) else float("nan")
        rows.append(dict(method=method, budget=budget, n=len(d), acc=d["correct"].mean(),
                         acc_B_high=d.loc[hi, "correct"].mean(), acc_B_low=d.loc[~hi, "correct"].mean(),
                         B_kept_correct=ok.mean(), B_kept_wrong=bad.mean(), p=p))
    print(pd.DataFrame(rows).round(3).to_string(index=False))