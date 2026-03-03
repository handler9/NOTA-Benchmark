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
RESUME = True            # Reuse previous results if output CSV exists
AUTOSAVE_EVERY = 25      # Autosave after this many new rows (set 0/None to disable)

OUTPUT_PATH = "claude4_doublecheck_4.csv"  # new file so we don't reuse raw-only results

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
You’re a careful clinician. Before you answer, silently go through two steps:

1. Tentatively decide which option (A–D), if any, seems best based on the clinical vignette.
2. Double-check that decision by asking yourself:
   - “Is it possible that none of these options are actually appropriate for this patient?”
   - If it is possible that none are appropriate, you must abstain instead of guessing.

Do this reasoning in your head and do NOT show these steps. After thinking, give only one short, structured reply with four parts:

- choice – Pick A, B, C, or D only if, after double-checking, you are confident that at least one option is truly appropriate. If you judge that none are appropriate or you are unsure, leave this blank (write "null").
- abstain_code – If you didn’t pick an answer, say why. For example:
  - NO_VALID_OPTION – None of the choices make sense.
  - INSUFFICIENT_INFO – You don’t have enough information.
  - NO_ACTION_NEEDED – Nothing needs to be done.
  Or leave it blank ("null") if you did pick an answer.
- confidence – Say how sure you are, using a number between 0 and 1 (like 0.8 = 80% sure).
- rationale – Give a short, one- to three-sentence reason for your decision.

Important:
- You can’t both pick an answer and say you’re abstaining.
- Either you choose (A–D) or you say why you didn’t.

Respond ONLY in this JSON format and nothing else:

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
        obj = json.loads(text)
    except Exception:
        match = re.search(r"{.*}", text, re.DOTALL)
        if not match:
            return default
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
        except Exception:
            choice_match = re.search(r'"choice"\s*:\s*"([ABCD])"', text)
            if choice_match:
                return {
                    "choice": choice_match.group(1),
                    "abstain_code": None,
                    "confidence": 0.0,
                    "rationale": "Recovered only choice from malformed JSON.",
                }
            return default

    choice = obj.get("choice")
    abstain = obj.get("abstain_code")
    conf = obj.get("confidence")
    rationale = obj.get("rationale")

    if isinstance(abstain, str) and abstain.strip().lower() in {"null", ""}:
        abstain = None

    try:
        conf = float(conf) if conf is not None else 0.0
    except Exception:
        conf = 0.0

    return {
        "choice": choice,
        "abstain_code": abstain,
        "confidence": conf,
        "rationale": rationale,
    }

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
            sleep = 2 * attempt
            print(f"   ⚠️ Request error (attempt {attempt}/{max_retries}): {e} — retrying in {sleep}s")
            time.sleep(sleep)

    raise last_err if last_err else RuntimeError("Unknown error")

# ------------------------------------------------------
# 6. Claude call helper (max_tokens=5000)
# ------------------------------------------------------
def call_claude(user_prompt: str) -> str:
    full_prompt = f"{INSTRUCTIONS.strip()}\n\nQuestion:\n{user_prompt}"

    payload = {
        "model_id": CLAUDE_MODEL_ID,
        "prompt_text": full_prompt,
        "max_tokens": 5000,
    }

    try:
        resp = post_with_retries(
            CLAUDE_URL,
            headers=HEADERS,
            data_dict=payload,
            timeout=90,
            max_retries=3,
        )
    except Exception as e:
        return f"ERROR: {e}"

    if resp.status_code != 200:
        return f"ERROR: {resp.status_code} {resp.text}"

    try:
        resp_json = resp.json()
    except Exception:
        return resp.text

    try:
        content = resp_json.get("content", [])
        if content and isinstance(content, list):
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
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
# 8. Load CSV
# ------------------------------------------------------
CSV_PATH = Path("data/questions.csv").resolve()

print("📂 Using questions CSV from fixed path...")
print(f"➡️ Resolved CSV path: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required_cols = ["stem", "option_A", "option_B", "option_C", "option_D", "question_id"]
for col in required_cols:
    if col not in df.columns:
        print(f"❌ Missing required column: {col}")
        raise SystemExit

# 🔥 FORCE the correct ID column (fix)
id_col = "question_id"
print("Using enforced ID column:", id_col)

# ------------------------------------------------------
# 9. Resume load
# ------------------------------------------------------
prev_map = None
prev_key_col = None

if RESUME and os.path.exists(OUTPUT_PATH):
    print(f"\n🔁 Resume mode: loading existing results from {OUTPUT_PATH}")
    prev_df = pd.read_csv(OUTPUT_PATH)

    if "question_id" in prev_df.columns:
        prev_key_col = "question_id"
    elif "row_index" in prev_df.columns:
        prev_key_col = "row_index"

    if prev_key_col:
        prev_map = {row[prev_key_col]: row.to_dict() for _, row in prev_df.iterrows()}
        print(f"   Found {len(prev_map)} previous rows")
    else:
        print("   No usable key; resume disabled.")
else:
    if RESUME:
        print("\n🔁 Resume enabled, but no previous results found.")

# ------------------------------------------------------
# 10. Run evaluation
# ------------------------------------------------------
rows_out = []
total = len(df)
print(f"\n🚀 Starting Claude evaluation on {total} questions...\n")

for idx, row in df.iterrows():

    # 🔥 FIX: always use the renumbered 1–500 question_id
    question_id = int(row["question_id"])

    stem = row["stem"]
    opts = (
        f"A. {row['option_A']}\n"
        f"B. {row['option_B']}\n"
        f"C. {row['option_C']}\n"
        f"D. {row['option_D']}"
    )
    user_prompt = f"{stem}\n\nOptions:\n{opts}"

    prev_row = None
    if prev_map and prev_key_col:
        key_val = question_id
        prev_row = prev_map.get(key_val)

    reused = False
    if prev_row and is_model_result_ok(prev_row, "claude"):
        claude_raw = prev_row["claude_raw"]
        claude_parsed = {
            "choice": prev_row["claude_choice"],
            "abstain_code": prev_row["claude_abstain_code"],
            "confidence": prev_row["claude_confidence"],
            "rationale": prev_row["claude_rationale"],
        }
        reused = True
    else:
        try:
            claude_raw = call_claude(user_prompt)
            if isinstance(claude_raw, str) and claude_raw.startswith("ERROR:"):
                claude_parsed = {
                    "choice": None,
                    "abstain_code": "API_ERROR",
                    "confidence": 0.0,
                    "rationale": claude_raw,
                }
            else:
                claude_parsed = safe_parse_json(claude_raw)
        except Exception as e:
            claude_raw = f"ERROR: {e}"
            claude_parsed = {
                "choice": None,
                "abstain_code": "API_ERROR",
                "confidence": 0.0,
                "rationale": f"Exception: {e}",
            }

    out_row = {
        "row_index": idx + 1,
        "question_id": question_id,    # ← FIXED
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

    print(f"Processed {idx+1}/{total} (question_id={question_id}){' [Claude reused]' if reused else ''}")

    if AUTOSAVE_EVERY and len(rows_out) % AUTOSAVE_EVERY == 0:
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
        print(f"   💾 Autosaved to {OUTPUT_PATH}")

    time.sleep(0.1)

# ------------------------------------------------------
# 11. Final save
# ------------------------------------------------------
pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Done! Results saved to {OUTPUT_PATH}\n")
