from pathlib import Path
from datasets import load_dataset
import pandas as pd

data_dir = Path(__file__).parent.parent / "data" / "hotpotqa"

hotpotqa = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
hotpotqa_df = pd.DataFrame(hotpotqa).set_index("id")

type_filt = (hotpotqa_df["type"] == "bridge")
hotpotqa_df = hotpotqa_df.loc[type_filt]

title_df = pd.DataFrame(hotpotqa_df["supporting_facts"].to_list(), index=hotpotqa_df.index)
title_df = title_df.explode(["title", "sent_id"])
title_df["sent_id"] = title_df["sent_id"].astype(int)

context_df = pd.DataFrame(hotpotqa_df["context"].to_list(), index=hotpotqa_df.index)
exploded_c_df = context_df.explode(["title", "sentences"]).explode("sentences")
exploded_c_df["sent_id"] = exploded_c_df.groupby(["id", "title"]).cumcount()

merged_df = title_df.merge(exploded_c_df, how="left", on=["id", "sent_id", "title"])

qa_df = merged_df.merge(hotpotqa_df[["question", "answer"]], on="id")

na_filt = qa_df["sentences"].isna()
na_index = qa_df.loc[na_filt].index
qa_df = qa_df.drop(na_index)
qa_df["contains_answer"] = qa_df.apply(lambda row: row["answer"].lower() in row["sentences"].lower(), axis=1)

para_contains_answer = qa_df.groupby(["id","title"])["contains_answer"].any()

sum_para_contains_answer = para_contains_answer.groupby("id").sum()

para_filt = (sum_para_contains_answer == 1)
usable_df = qa_df.loc[para_filt].copy()

usable_df["sentence_tag"] = usable_df.groupby(["id", "title"])["contains_answer"].transform(lambda x: x.any()).map({True: "B", False: "A"})

usable_df.to_parquet(data_dir.joinpath("hotpot_bridge_ab.parquet"))

print("bridge questions:", len(hotpotqa_df))
print("paragraphs containing the answer per question:")
print(sum_para_contains_answer.value_counts())
print("output rows:", len(usable_df))
print("output questions:", usable_df.index.nunique())