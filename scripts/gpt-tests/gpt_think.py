import os
import re
import json
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from requests.exceptions import RequestException

# ------------------------------------------------------
# 0. Config toggles
# ------------------------------------------------------
RESUME = True
AUTOSAVE_EVERY = 25

# Input / output
INPUT_CSV = "medqa_nato_results_gpt copy.csv"   # change if needed
OUTPUT_PATH = "medqa_nato_judge_1-26.csv"

# Column to judge
PROMPT_COL = "noto_prompt"

# ------------------------------------------------------
# 1. Load API key from .env
# ------------------------------------------------------
load_dotenv()
KEY = os.getenv("SECUREGPT_API_KEY")
if not KEY:
    print("❌ SECUREGPT_API_KEY not found in .env")
    raise SystemExit

HEADERS = {
    "Ocp-Apim-Subscription-Key": KEY,
    "Content-Type": "application/json",
}

# ------------------------------------------------------
# 2. GPT-5 endpoint (your existing APIM endpoint)
# ------------------------------------------------------
GPT5_URL = (
    "https://apim.stanfordhealthcare.org/openai-eastus2/"
    "deployments/gpt-5/chat/completions?api-version=2024-12-01-preview"
)

# ------------------------------------------------------
# 3. Judge system prompt (your requested instructions)
# ------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are an impartial judge.

Your task is NOT to answer the medical question.
Your task is to determine whether this exam question is a TRUE-NOTA question.

Definition:
A question is TRUE-NOTA if none of the listed answer choices are medically correct.

Rules:
- Do NOT answer the question.
- Do NOT choose an option.
- ONLY classify whether it is TRUE-NOTA.
- Provide a brief rationale (1–3 sentences) explaining why the question is or is not TRUE-NOTA.

Return JSON ONLY:
{
  "is_true_nota": true or false,
  "rationale": "1-3 sentence explanation"
}
"""

# ------------------------------------------------------
# 4. Robust JSON-ish parser for judge output
# ------------------------------------------------------
def safe_parse_judge_json(text: str):
    default = {
        "is_true_nota": None,
        "rationale": "Model did not respond in the requested JSON format."
    }

    if not isinstance(text, str) or not text.strip():
        return default

    cleaned = text.strip()

    # Remove <think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)

    # Remove markdown fences
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # 1) Try full JSON
    obj = None
    try:
        obj = json.loads(cleaned)
    except Exception:
        obj = None

    # 2) Try substring JSON
    if obj is None:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end+1]
            try:
                obj = json.loads(candidate)
            except Exception:
                obj = None

    if isinstance(obj, dict):
        is_true = obj.get("is_true_nota")
        rat = obj.get("rationale")

        # Normalize boolean-ish values
        if isinstance(is_true, str):
            if is_true.strip().lower() in {"true", "yes"}:
                is_true = True
            elif is_true.strip().lower() in {"false", "no"}:
                is_true = False
            else:
                is_true = None

        if not isinstance(is_true, bool):
            is_true = None

        if not isinstance(rat, str) or not rat.strip():
            rat = default["rationale"]

        return {"is_true_nota": is_true, "rationale": rat.strip()}

    # 3) Regex fallback
    flat = " ".join(cleaned.split())
    m_bool = re.search(r'["\']?\s*is_true_nota\s*["\']?\s*:\s*(true|false)', flat, re.IGNORECASE)
    m_rat = re.search(r'["\']?\s*rationale\s*["\']?\s*:\s*["\'](.*?)["\']', cleaned, re.DOTALL | re.IGNORECASE)

    is_true = None
    if m_bool:
        is_true = True if m_bool.group(1).lower() == "true" else False

    rat = m_rat.group(1).strip() if m_rat else default["rationale"]

    if is_true is None and rat == default["rationale"]:
        return default

    return {"is_true_nota": is_true, "rationale": rat}

# ------------------------------------------------------
# 5. POST helper w/ retries
# ------------------------------------------------------
def post_with_retries(url, headers, json_data, timeout=90, max_retries=3):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=json_data, timeout=timeout)
            if 500 <= resp.status_code < 600:
                last_err = RequestException(f"Server error {resp.status_code}", response=resp)
                raise last_err
            return resp
        except RequestException as e:
            last_err = e
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)
    raise last_err if last_err else RuntimeError("Unknown error in post_with_retries")

# ------------------------------------------------------
# 6. Call GPT-5 judge
# ------------------------------------------------------
def call_gpt5_judge(item_prompt: str) -> str:
    data = {
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": item_prompt},
        ],
        "temperature": 0,
        "max_completion_tokens": 600,
    }

    try:
        resp = post_with_retries(GPT5_URL, headers=HEADERS, json_data=data)
    except Exception as e:
        return f"ERROR: {e}"

    if resp.status_code != 200:
        return f"ERROR: {resp.status_code} {resp.text}"

    return resp.json()["choices"][0]["message"]["content"]

# ------------------------------------------------------
# 7. Resume helper
# ------------------------------------------------------
def is_prev_judge_ok(prev_row: dict) -> bool:
    raw = prev_row.get("judge_raw")
    err = prev_row.get("judge_error")

    if isinstance(err, str) and err.strip():
        return False
    if not isinstance(raw, str) or not raw.strip():
        return False
    if raw.startswith("ERROR:"):
        return False

    # If parsing previously failed (None), rerun
    prev_val = prev_row.get("judge_is_true_nota")
    if pd.isna(prev_val):
        return False

    return True

# ------------------------------------------------------
# 8. Load input CSV
# ------------------------------------------------------
csv_path = Path(INPUT_CSV).resolve()
print(f"📂 Input CSV: {csv_path}")
if not csv_path.exists():
    print(f"❌ Input CSV not found: {csv_path}")
    raise SystemExit

df = pd.read_csv(csv_path)
if PROMPT_COL not in df.columns:
    raise SystemExit(f"❌ Missing required column: {PROMPT_COL}")

# Prefer an ID column if present
id_col = None
for candidate in ["question_id", "id", "qid", "QID", "row_index"]:
    if candidate in df.columns:
        id_col = candidate
        break

print("Detected ID column:", id_col if id_col else "(none, will use row index)")

# ------------------------------------------------------
# 9. Load resume data
# ------------------------------------------------------
prev_map = None
prev_key_col = None

if RESUME and os.path.exists(OUTPUT_PATH):
    print(f"\n🔁 Resuming from {OUTPUT_PATH}")
    prev_df = pd.read_csv(OUTPUT_PATH)

    if id_col and id_col in prev_df.columns:
        prev_key_col = id_col
    elif "row_index" in prev_df.columns:
        prev_key_col = "row_index"

    if prev_key_col:
        prev_map = {row[prev_key_col]: row.to_dict() for _, row in prev_df.iterrows()}
        print(f"Loaded {len(prev_map)} previous rows.")

# ------------------------------------------------------
# 10. Run judge (Option A: only noto_prompt)
# ------------------------------------------------------
rows_out = []
total = len(df)
print(f"\n🚀 Starting TRUE-NOTA judge on {total} rows...\n")

for idx, row in df.iterrows():
    key = row[id_col] if id_col else (idx + 1)

    prev_row = prev_map.get(key) if prev_map else None
    reused = False

    if prev_row and is_prev_judge_ok(prev_row):
        judge_raw = prev_row.get("judge_raw", "")
        judge_is_true = prev_row.get("judge_is_true_nota")
        judge_rationale = prev_row.get("judge_rationale", "")
        judge_error = prev_row.get("judge_error", "")
        reused = True
    else:
        item_prompt = str(row[PROMPT_COL])
        judge_raw = call_gpt5_judge(item_prompt)

        if isinstance(judge_raw, str) and judge_raw.startswith("ERROR:"):
            judge_is_true = None
            judge_rationale = ""
            judge_error = judge_raw
        else:
            parsed = safe_parse_judge_json(judge_raw)
            judge_is_true = parsed["is_true_nota"]
            judge_rationale = parsed["rationale"]
            judge_error = ""

    out_row = row.to_dict()
    out_row["row_index"] = idx + 1  # always add a stable index
    out_row["judge_raw"] = judge_raw
    out_row["judge_is_true_nota"] = judge_is_true
    out_row["judge_rationale"] = judge_rationale
    out_row["judge_error"] = judge_error

    rows_out.append(out_row)

    note = " [reused]" if reused else ""
    print(f"Processed {idx+1}/{total}{note}")

    if AUTOSAVE_EVERY and (len(rows_out) % AUTOSAVE_EVERY == 0):
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
        print(f"   💾 Autosaved to {OUTPUT_PATH}")

    time.sleep(0.1)

# ------------------------------------------------------
# 11. Save final
# ------------------------------------------------------
pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Done! Saved to {OUTPUT_PATH}\n")
