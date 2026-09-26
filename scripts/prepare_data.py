"""Build the two data files used by every run from HF openai/gsm8k (main):
  data/gsm8k_train.jsonl  — ms-swift SFT format (user = question, assistant = reference solution)
  data/verl/{train,test}.parquet — verl RL format (prompt = question only, ground truth = final number)
"""
import json, os
import pandas as pd
from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")
os.makedirs("data/verl", exist_ok=True)

with open("data/gsm8k_train.jsonl", "w") as f:
    for r in ds["train"]:
        f.write(json.dumps({"messages": [{"role": "user", "content": r["question"]},
                                         {"role": "assistant", "content": r["answer"]}]}, ensure_ascii=False) + "\n")

for split in ("train", "test"):
    rows = [{
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": r["question"]}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": r["answer"].split("####")[-1].strip().replace(",", "")},
        "extra_info": {"split": split, "index": i},
    } for i, r in enumerate(ds[split])]
    pd.DataFrame(rows).to_parquet(f"data/verl/{split}.parquet")
