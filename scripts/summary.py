from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon

if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "results"
    retention = pd.read_parquet(results_dir / "retention.parquet")

    per_question = (retention.groupby(["method", "budget", "id", "sentence_tag"])["kept_fraction"]
                    .mean().unstack("sentence_tag").dropna())

    rows = []
    for (method, budget), g in per_question.groupby(level=[0, 1]):
        p = wilcoxon(g["A"], g["B"]).pvalue if (g["A"] != g["B"]).any() else 1.0
        rows.append(dict(method=method, budget=budget, n=len(g),
                         A_kept=g["A"].mean(), B_kept=g["B"].mean(),
                         B_less=(g["B"] < g["A"]).sum(), A_less=(g["A"] < g["B"]).sum(), p=p))
    print(pd.DataFrame(rows).round(3).to_string(index=False))