import json
import random
import pandas as pd
from pathlib import Path
import math

# 128 synthetic prompts from 6 harmful categories:
    # 1. Disinformation and deception
    # 2. Hate and harassment
    # 3. Illegal goods and services
    # 4. Non-violent crimes
    # 5. Sexual content
    # 6. Violence
# 40 strongreject prompts from each category
# Test set: 10 strongreject prompts and 32 synthetic prompts

# 500 prompt each from two 2 benign datasets
    # Alpaca
    # Lmsys

# TODO: Test for overrefusal


all = []
categories = ["disinformation_deception", "hate_harassment", "illegal_goods_services", "nonviolent_crimes", "sexual_content", "violence"]

for category in categories:
    in_path  = f"models/harmful_refusal_llama/{category}_refusals.json"
    out_path = f"models/harmful_refusal_llama/{category}_test.json"

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = len(data)
    k = max(1, math.floor(0.20 * n)) if n else 0

    idx = list(range(n))
    random.shuffle(idx)
    test_idx = set(idx[:k])

    test  = [data[i] for i in test_idx]
    train = [data[i] for i in range(n) if i not in test_idx]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    all.extend(train)


with open ("models/harmful_refusal_llama/refusals2.json", "r") as f:
        data = json.load(f)
        all.extend(data)
        print(f"number of prompts from first three categories: {len(data)}")

with open ("models/harmful_refusal_llama/refusals1.json", "r") as f:
        data = json.load(f)
        all.extend(data)
        print(f"number of prompts from last three categories: {len(data)}")


#sample 500 random each from Alpaca and Lmsys
alpaca_df = pd.read_parquet("models/harmful_refusal_llama/alpaca.parquet", engine="pyarrow")
alpaca_sampled = alpaca_df.sample(n=750, random_state=42)

formatted = [
{
    "user": row["instruction"],
    "assistant": row["output"],
    "category": "Helpful"
}
for _, row in alpaca_sampled.iterrows()
]

with open("sampled_alpaca.json", "w", encoding="utf-8") as f:
    json.dump(formatted, f, ensure_ascii=False, indent=2)
all.extend(formatted)


lmsys_df = pd.read_parquet("models/harmful_refusal_llama/lmsys-chat-train-00000-of-00006-4feeb3f83346a0e9.parquet", engine="pyarrow")
lmsys_sampled = lmsys_df.sample(n=750, random_state=42)

formatted = [
{
    "user": row["conversation"][0]['content'],
    "assistant": row["conversation"][1]['content'],
    "category": "Helpful"
}
for _, row in lmsys_sampled.iterrows()
]

with open("sampled_lmsys.json", "w", encoding="utf-8") as f:
    json.dump(formatted, f, ensure_ascii=False, indent=2)
all.extend(formatted)

random.shuffle(all)

with open('models/harmful_refusal_llama/final_dataset_1500_harmless.json', "w", encoding='utf-8') as f:
    json.dump(all, f, ensure_ascii=False)