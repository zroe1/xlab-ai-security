import json
import random
import pandas as pd
from pathlib import Path

# 200 prompts from 6 harmful categories:
# Some prompts from StrongREJECT 
    # 1. Disinformation and deception
    # 2. Hate and harassment
    # 3. Illegal goods and services
    # 4. Non-violent crimes
    # 5. Sexual content
    # 6. Violence
# 2 benign datasets
    # Alpaca
    # Lmsys

all = []
categories = ["disinformation_and_deception", "hate_harassment", "illegal_goods_services", "non_violent_crimes", "sexual_content", "violence"]

for category in categories:
    with open (f"models/harmful_refusal_llama/{category}_refusals.json", "r") as f:
        all.extend(json.load(f))

with open ("models/harmful_refusal_llama/refusals1.json", "r") as f:
    all.extend(json.load(f))

with open ("models/harmful_refusal_llama/refusals2.json", "r") as f:
    all.extend(json.load(f))


#sample 500 random each from Alpaca and Lmsys
alpaca_df = pd.read_parquet("models/harmful_refusal_llama/alpaca.parquet", engine="pyarrow")
alpaca_sampled = alpaca_df.sample(n=500, random_state=42)

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
lmsys_sampled = lmsys_df.sample(n=500, random_state=42)

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

with open('models/harmful_refusal_llama/final_dataset.json', "w", encoding='utf-8') as f:
    json.dump(all, f, ensure_ascii=False)