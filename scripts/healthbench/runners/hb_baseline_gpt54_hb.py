import os, json, time, requests, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from requests.exceptions import RequestException

RESUME = True
AUTOSAVE_EVERY = 25
CONDITION = "baseline"
PREFIX = "gpt54_baseline_hb"
OUTPUT_PATH = Path("results_raw_healthbench") / f"{PREFIX}_hb.csv"
DATA_PATH = Path("data/healthbench/healthbench_subset_combined.jsonl")

load_dotenv()
KEY = os.getenv("AIHUB_API_KEY") or os.getenv("SECUREGPT_API_KEY")
if not KEY:
    print("❌ AIHUB_API_KEY not found in .env")
    raise SystemExit

HEADERS = {"api-key": KEY, "Content-Type": "application/json"}

GPT5_URL = (
    "https://aihubapi.stanfordhealthcare.org/azure-openai/deployments/gpt-5-4/"
    "chat/completions?api-version=2025-04-01-preview"
)

INSTRUCTIONS = """You are an experienced physician. Answer the following patient question as you would in a clinical setting. Provide a clear, direct, and clinically appropriate response."""


def post_with_retries(url, headers, json_data, timeout=90, max_retries=8):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=json_data, timeout=timeout)
            if resp.status_code == 429:
                wait = min(60 * attempt, 300)
                print(f"    ⏳ Rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            if 500 <= resp.status_code < 600:
                time.sleep(10 * attempt)
                continue
            return resp
        except RequestException as e:
            last_err = e
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)
    raise last_err if last_err else RuntimeError("Unknown error")


def call_model(prompt_turns: list) -> str:
    data = {
        "messages": [{"role": "system", "content": INSTRUCTIONS}] + [
            {"role": t["role"], "content": t["content"]} for t in prompt_turns
        ],
        "model": "chat",
        "max_completion_tokens": 5000,
    }
    try:
        resp = post_with_retries(GPT5_URL, headers=HEADERS, json_data=data)
    except Exception as e:
        return f"ERROR: {e}"
    if resp.status_code != 200:
        return f"ERROR: {resp.status_code} {resp.text}"
    return resp.json()["choices"][0]["message"]["content"]


def is_result_ok(row_dict) -> bool:
    raw = row_dict.get(f"{PREFIX}_raw")
    return isinstance(raw, str) and raw.strip() and not raw.startswith("ERROR:")


DATA_PATH = DATA_PATH.resolve()
print(f"📂 Loading from {DATA_PATH}")
if not DATA_PATH.exists():
    print(f"❌ Not found: {DATA_PATH}")
    raise SystemExit

rows_in = []
with open(DATA_PATH) as f:
    for line in f:
        r = json.loads(line)
        prompt_turns = r.get("prompt", [])
        rows_in.append({
            "prompt_id": r["prompt_id"],
            "question": prompt_turns[-1]["content"] if prompt_turns else "",
            "prompt_turns": json.dumps(prompt_turns),
            "subset": r.get("subset", ""),
            "rubrics_json": json.dumps(r.get("rubrics", [])),
        })

print(f"  {len(rows_in)} questions loaded")

prev_map = {}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
if RESUME and OUTPUT_PATH.exists():
    prev_df = pd.read_csv(OUTPUT_PATH)
    if "prompt_id" in prev_df.columns:
        prev_map = {row["prompt_id"]: row.to_dict() for _, row in prev_df.iterrows()}
    print(f"🔁 Resuming — {len(prev_map)} previous rows loaded")

rows_out = []
total = len(rows_in)
print(f"\n🚀 {PREFIX} — {total} questions\n")

for i, row in enumerate(rows_in):
    qid = row["prompt_id"]
    prompt_turns = json.loads(row["prompt_turns"])
    prev_row = prev_map.get(qid)
    if prev_row and is_result_ok(prev_row):
        raw = prev_row[f"{PREFIX}_raw"]
        reused = True
    else:
        raw = call_model(prompt_turns)
        reused = False

    rows_out.append({
        "prompt_id": qid,
        "question": row["question"],
        "subset": row["subset"],
        "rubrics_json": row["rubrics_json"],
        f"{PREFIX}_raw": raw,
    })

    status = "[reused]" if reused else ("❌ ERROR" if isinstance(raw, str) and raw.startswith("ERROR:") else "")
    print(f"  {i+1}/{total} [{qid[:12]}] {status}")

    if len(rows_out) % AUTOSAVE_EVERY == 0:
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
        print(f"  💾 Autosaved")

    if not reused:
        time.sleep(0.1)

pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
errors = sum(1 for r in rows_out if isinstance(r.get(f"{PREFIX}_raw"), str) and r[f"{PREFIX}_raw"].startswith("ERROR:"))
print(f"\n✅ Done — {total} questions, {errors} errors → {OUTPUT_PATH}\n")
