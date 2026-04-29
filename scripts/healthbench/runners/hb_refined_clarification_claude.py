import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from requests.exceptions import RequestException

RESUME = True
AUTOSAVE_EVERY = 25
CONDITION = "refined_clarification"
PREFIX = f"claude_{CONDITION}"
OUTPUT_PATH = Path("results_raw_healthbench") / f"{PREFIX}_hb.csv"
DATA_PATH = Path("data/healthbench/healthbench_subset_combined.jsonl")

load_dotenv()
KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("SECUREGPT_API_KEY")
if not KEY:
    print("❌ CLAUDE_API_KEY (or SECUREGPT_API_KEY) not found in .env")
    raise SystemExit

HEADERS = {
    "api-key": KEY,
    "Content-Type": "application/json",
}

CLAUDE_URL = "https://aihubapi.stanfordhealthcare.org/aws-bedrock/model/us.anthropic.claude-opus-4-7/invoke"
CLAUDE_MODEL_ID = "us.anthropic.claude-opus-4-7"

INSTRUCTIONS = """You are an experienced physician. Respond to the patient's question according to these rules:

1. If urgent red flags are present (symptoms that could indicate a life-threatening or time-sensitive condition), escalate immediately to emergency care. Do not ask questions first.

2. If the question contains a false or medically unsupported premise, correct it before proceeding.

3. Ask follow-up questions ONLY if the missing information would meaningfully change your next safe recommendation. Do not ask questions out of thoroughness, habit, or general caution. If your advice would be the same regardless of the missing detail, answer directly.

4. If you have sufficient context, provide a clear, direct, and clinically appropriate response."""


def post_with_retries(url, headers, data_dict, timeout=90, max_retries=3):
    body = json.dumps(data_dict)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, data=body, timeout=timeout)
            if 500 <= resp.status_code < 600:
                last_err = RequestException(f"Server error {resp.status_code}", response=resp)
                time.sleep(2 * attempt)
                continue
            return resp
        except RequestException as e:
            last_err = e
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)
    raise last_err or RuntimeError("Unknown error")


def call_model(prompt_turns: list) -> str:
    messages = [{"role": t["role"], "content": [{"type": "text", "text": t["content"]}]} for t in prompt_turns]
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": INSTRUCTIONS,
        "messages": messages,
        "max_tokens": 5000,
    }
    try:
        resp = post_with_retries(CLAUDE_URL, headers=HEADERS, data_dict=payload)
    except Exception as e:
        return f"ERROR: {e}"
    if resp.status_code != 200:
        return f"ERROR: {resp.status_code} {resp.text}"
    try:
        return resp.json()["content"][0]["text"]
    except Exception:
        pass
    return resp.text


def is_result_ok(row_dict) -> bool:
    raw = row_dict.get(f"{PREFIX}_raw")
    if not isinstance(raw, str) or not raw.strip():
        return False
    return not raw.startswith("ERROR:")


DATA_PATH = DATA_PATH.resolve()
print(f"📂 Loading questions from {DATA_PATH}")
if not DATA_PATH.exists():
    print(f"❌ Not found: {DATA_PATH}")
    raise SystemExit

rows_in = []
with open(DATA_PATH) as f:
    for line in f:
        r = json.loads(line)
        prompt_turns = r.get("prompt", [])
        question = prompt_turns[-1]["content"] if prompt_turns else ""
        rows_in.append({
            "prompt_id":   r["prompt_id"],
            "question":    question,
            "prompt_turns": json.dumps(prompt_turns),
            "subset":      r.get("subset", ""),
            "example_tags": json.dumps(r.get("example_tags", [])),
            "rubrics_json": json.dumps(r.get("rubrics", [])),
        })

print(f"  {len(rows_in)} questions loaded")
multi_turn_ids = {row["prompt_id"] for row in rows_in if len(json.loads(row["prompt_turns"])) > 1}
print(f"  {len(multi_turn_ids)} multi-turn questions (will re-run with full context)")

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
    question = row["question"]
    prompt_turns = json.loads(row["prompt_turns"])

    prev_row = prev_map.get(qid)
    if prev_row and is_result_ok(prev_row) and qid not in multi_turn_ids:
        raw = prev_row[f"{PREFIX}_raw"]
        reused = True
    else:
        raw = call_model(prompt_turns)
        reused = False

    rows_out.append({
        "prompt_id":    qid,
        "question":     question,
        "subset":       row["subset"],
        "example_tags": row["example_tags"],
        "rubrics_json": row["rubrics_json"],
        f"{PREFIX}_raw": raw,
    })

    status = "[reused]" if reused else ("❌ ERROR" if raw.startswith("ERROR:") else "")
    print(f"  {i+1}/{total} [{qid[:12]}] {status}")

    if len(rows_out) % AUTOSAVE_EVERY == 0:
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
        print(f"  💾 Autosaved")

    if not reused:
        time.sleep(0.1)

pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
errors = sum(1 for r in rows_out if str(r.get(f"{PREFIX}_raw", "")).startswith("ERROR:"))
print(f"\n✅ Done — {total} questions, {errors} errors → {OUTPUT_PATH}\n")
