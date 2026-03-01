#!/usr/bin/env python3
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
OUTPUT_PATH = "claude_baseline_afrimedqa.csv"

EXPECTED_TOTAL = 490

# ------------------------------------------------------
# 1. Load API key from .env
# ------------------------------------------------------
load_dotenv()

KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("SECUREGPT_API_KEY")
if not KEY:
    print("❌ CLAUDE_API_KEY (or SECUREGPT_API_KEY) not found in .env")
    raise SystemExit

HEADERS = {
    "Ocp-Apim-Subscription-Key": KEY,
    "Content-Type": "application/json",
}

# ------------------------------------------------------
# 2. Claude endpoint & model name
# ------------------------------------------------------
CLAUDE_URL = "https://apim.stanfordhealthcare.org/aws-claude4-sonnet/aws-claude4-sonnet"
CLAUDE_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

# ------------------------------------------------------
# 3. JSON-only instructions
# ------------------------------------------------------
INSTRUCTIONS = """
You’re a careful clinician. When you answer, you should only give one short, structured reply — not a paragraph, not a sentence, not an explanation. Just a small bundle of information with four parts:

- choice – Pick A, B, C, or D if you’re sure about the answer. If you’re not sure, leave this blank (write "null").
- abstain_code – If you didn’t pick an answer, say why.
- confidence – Number between 0 and 1.
- rationale – One to three sentences.

Respond ONLY in this JSON format:

{
  "choice": "A/B/C/D or null",
  "abstain_code": "NO_VALID_OPTION/INSUFFICIENT_INFO/NO_ACTION_NEEDED or null",
  "confidence": 0.0,
  "rationale": "one to three sentences"
}
"""

# ------------------------------------------------------
# 4. Robust JSON parser
# ------------------------------------------------------
def safe_parse_json(text: str):
    default = {
        "choice": None,
        "abstain_code": "PARSING_ERROR",
        "confidence": 0.0,
        "rationale": "Could not parse JSON",
    }

    if not isinstance(text, str) or not text.strip():
        return default

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"{.*}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

    return default

# ------------------------------------------------------
# 5. POST helper with retries
# ------------------------------------------------------
def post_with_retries(url, headers, data_dict, timeout=90, max_retries=3):
    last_err = None
    body = json.dumps(data_dict)

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, data=body, timeout=timeout)
            if 500 <= resp.status_code < 600:
                last_err = RequestException(f"Server error {resp.status_code}", response=resp)
                raise last_err
            return resp

        except RequestException as e:
            last_err = e
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)

    raise last_err or RuntimeError("Unknown error in post_with_retries")

# ------------------------------------------------------
# 6. Claude call helper
# ------------------------------------------------------
def call_claude(user_prompt: str) -> str:
    full_prompt = f"{INSTRUCTIONS.strip()}\n\nQuestion:\n{user_prompt}"

    payload = {
        "model_id": CLAUDE_MODEL_ID,
        "prompt_text": full_prompt,
        "max_tokens": 5000,
    }

    try:
        resp = post_with_retries(CLAUDE_URL, headers=HEADERS, data_dict=payload)
    except Exception as e:
        return f"ERROR: {e}"

    if resp.status_code != 200:
        return f"ERROR: {resp.status_code} {resp.text}"

    try:
        content = resp.json().get("content", [])
        if content and isinstance(content, list) and "text" in content[0]:
            return content[0]["text"]
    except Exception:
        pass

    return resp.text

# ------------------------------------------------------
# 7. Resume helper
# ------------------------------------------------------
def is_model_result_ok(row_dict, prefix: str) -> bool:
    choice = row_dict.get(f"{prefix}_choice")
    abst = row_dict.get(f"{prefix}_abstain_code")

    if choice in {"A", "B", "C", "D"}:
        return True
    if isinstance(abst, str) and abst not in {"", "API_ERROR", "PARSING_ERROR"}:
        return True
    return False

# ------------------------------------------------------
# 8. Load questions CSV
# ------------------------------------------------------
CSV_PATH = Path("data/afrimedqa_questions.csv").resolve()

print("📂 Using questions CSV from fixed path...")
print(f"➡️ Resolved CSV path: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required_cols = ["stem", "option_A", "option_B", "option_C", "option_D", "question_id"]
for col in required_cols:
    if col not in df.columns:
        print(f"❌ Missing required column: {col}")
        raise SystemExit

# ✅ enforce row count
if len(df) != EXPECTED_TOTAL:
    raise ValueError(f"Expected {EXPECTED_TOTAL} questions, but found {len(df)} rows in {CSV_PATH}")

# ✅ CRITICAL FIX: question_id is string (it can be hash-like)
df["question_id"] = df["question_id"].astype(str)

print("Using ID column: question_id (as string)")

# ------------------------------------------------------
# 9. Load previous results if resume
# ------------------------------------------------------
prev_map = None

if RESUME and os.path.exists(OUTPUT_PATH):
    prev_df = pd.read_csv(OUTPUT_PATH)

    if "question_id" in prev_df.columns:
        # ✅ ensure same type as df
        prev_df["question_id"] = prev_df["question_id"].astype(str)

        prev_map = {
            row["question_id"]: row.to_dict()
            for _, row in prev_df.iterrows()
        }

# ------------------------------------------------------
# 10. Run model
# ------------------------------------------------------
rows_out = []
total = len(df)

print(f"\n🚀 Starting Claude evaluation on {total} questions...\n")

for idx, row in df.iterrows():
    question_id = str(row["question_id"])  # ✅ no int()

    stem = row["stem"]
    opts = (
        f"A. {row['option_A']}\n"
        f"B. {row['option_B']}\n"
        f"C. {row['option_C']}\n"
        f"D. {row['option_D']}"
    )
    user_prompt = f"{stem}\n\nOptions:\n{opts}"

    prev_row = prev_map.get(question_id) if prev_map else None

    if prev_row and is_model_result_ok(prev_row, "claude"):
        claude_raw = prev_row.get("claude_raw", "")
        claude_parsed = {
            "choice": prev_row.get("claude_choice"),
            "abstain_code": prev_row.get("claude_abstain_code"),
            "confidence": prev_row.get("claude_confidence"),
            "rationale": prev_row.get("claude_rationale"),
        }
        reused = True
    else:
        reused = False
        claude_raw = call_claude(user_prompt)
        claude_parsed = safe_parse_json(claude_raw)

    out_row = {
        "row_index": idx + 1,
        "question_id": question_id,
        "stem": stem,
        "option_A": row["option_A"],
        "option_B": row["option_B"],
        "option_C": row["option_C"],
        "option_D": row["option_D"],

        "claude_raw": claude_raw,
        "claude_choice": claude_parsed.get("choice"),
        "claude_abstain_code": claude_parsed.get("abstain_code"),
        "claude_confidence": claude_parsed.get("confidence"),
        "claude_rationale": claude_parsed.get("rationale"),
    }

    rows_out.append(out_row)

    print(f"Processed {idx+1}/{total} (question_id={question_id}){' [reused]' if reused else ''}")

    if AUTOSAVE_EVERY and (len(rows_out) % AUTOSAVE_EVERY == 0):
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)

    time.sleep(0.1)

# ------------------------------------------------------
# 11. Final save
# ------------------------------------------------------
pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Done! Results saved to {OUTPUT_PATH}\n")
