import os, asyncio, json, httpx, tqdm, pathlib, random
from dotenv import load_dotenv

load_dotenv(".env")
os.environ["OPENROUTER_API_KEY"] = os.getenv("API_KEY")

DATASET_FILE_PATH = "models/harmful_refusal_llama/disinformation_and_deception.json"

API = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
    "Content-Type": "application/json"
}
MODEL="openai/gpt-4.1-mini"
BATCH_SIZE = 4


SYSTEM_MSG = ("""You are an aligned, helpful, and harmless assistant. For each user prompt below, return a measured, natural, coherent, and reasonable response or refusal to the harmful user request. Be sure to include a brief, thoughtful, and helpful explanation as to why the user's request is inappropriate or disallowed. 
              Return a JSON object with key pairs, whose value is an array with of 4 responses.""")

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "refusals",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "pairs": {           # only top‑level key
                    "type":  "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "assistant": {
                                "type": "string",
                        "description": "Refusal or deflection + rationale"
                            }
                        },
                        "required": ["assistant"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["pairs"],
            "additionalProperties": False
        }
    }
}

def make_messages(chunk):
    numbered = "\n".join(f"{idx+1}. {p.strip()}" for idx, p in enumerate(chunk))
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": numbered}
    ]

async def call_model(session, chunk):
    body = {
        "model": MODEL,
        "messages": make_messages(chunk),
        "response_format": SCHEMA,
        "temperature": 0.4
    }

    print(f"Making request with {len(chunk)} prompts...")
    r = await session.post(API, headers=HEADERS, json=body, timeout=120)
    
    if r.status_code != 200:
        print(f"Error {r.status_code}: {r.text}")
    
    r.raise_for_status()
    response_data = r.json()
    
    if "choices" not in response_data or not response_data["choices"]:
        print(f"Unexpected response structure: {response_data}")
        return []
        
    content = response_data["choices"][0]["message"]["content"]
    return json.loads(content)["pairs"] # extract array
    


async def main():
    prompts_path = pathlib.Path(f"{DATASET_FILE_PATH}")

    if not prompts_path.exists():
        print(f"Error: File not found: {prompts_path}")
        return
        
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))

    random.seed(42)
    random.shuffle(prompts)
    
    out = []
    async with httpx.AsyncClient(timeout=120) as session:
        for i in tqdm.trange(0, len(prompts), BATCH_SIZE):
            if (i+BATCH_SIZE) >= len(prompts):
                chunk = prompts[i:]
            else:
                chunk = prompts[i:i+BATCH_SIZE]
            replies = await call_model(session, chunk)
            for prompt, pair in zip(chunk, replies):
                pair["user"] = prompt
                pair["category"] = "Disinformation and deception"
                out.append(pair)

    out_path = f"{DATASET_FILE_PATH.split('.')[0]}_refusals.json"
    with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=True)
            f.write("\n")

if __name__ == "__main__":
    asyncio.run(main())