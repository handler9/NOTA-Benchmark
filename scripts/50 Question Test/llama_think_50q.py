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
OUTPUT_PATH = "llama_think_judge.csv"

# ------------------------------------------------------
# 1. Load API key
# ------------------------------------------------------
load_dotenv()

KEY = (
    os.getenv("LLAMA_API_KEY")
    or os.getenv("SECUREGPT_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

if not KEY:
    print("❌ No API key found. Set LLAMA_API_KEY or SECUREGPT_API_KEY.")
    raise SystemExit

HEADERS = {
    "Ocp-Apim-Subscription-Key": KEY,
    "Content-Type": "application/json",
}

# ------------------------------------------------------
# 2. Llama endpoint & model
# ------------------------------------------------------
LLAMA_URL = "https://apim.stanfordhealthcare.org/llama4-maverick/v1/chat/completions"
LLAMA_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"
# ------------------------------------------------------
# 3. Prompt instructions
# ------------------------------------------------------
INSTRUCTIONS = """
You’re a careful clinician. Before you answer, silently go through three steps:

1. Identify the key clinical findings and what the correct diagnosis or management would be in general.
2. Check whether any of the four options (A–D) truly match that correct diagnosis or management.
3. If none of the options are appropriate, you must abstain instead of guessing.

Do this reasoning in your head and do NOT show these steps. After thinking, give only one short, structured reply with four parts:

- choice – Pick A, B, C, or D if, after checking, you are sure one option is appropriate. If none are appropriate or you are unsure, leave this blank (write "null").
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
# 4. Super-simple regex-based parser (no json.loads)
# ------------------------------------------------------
def safe_parse_json(text: str):
    """
    Extracts choice, abstain_code, confidence, rationale from
    whatever the model returned. Does NOT assume valid JSON.
    """

    default = {
        "choice": None,
        "abstain_code": "PARSING_ERROR",
        "confidence": 0.0,
        "rationale": "Could not parse model output",
    }

    if not isinstance(text, str) or not text.strip():
        return default

    # Strip ```json fences and <think> blocks
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()

    # ---------- choice ----------
    choice = None
    m = re.search(r'["\']?choice["\']?\s*:\s*["\']?([^",}\n]+)["\']?', t, flags=re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        raw_up = raw.upper()
        if raw_up in {"A", "B", "C", "D"}:
            choice = raw_up
        elif raw_up == "NULL":
            choice = None
        else:
            m2 = re.search(r"[ABCD]", raw_up)
            choice = m2.group(0) if m2 else None

    # ---------- abstain_code ----------
    abstain = None
    m = re.search(r'["\']?abstain_code["\']?\s*:\s*["\']?([^",}\n]+)["\']?', t, flags=re.IGNORECASE)
    if m:
        raw = m.group(1).strip().upper()
        if raw in {"NO_VALID_OPTION", "INSUFFICIENT_INFO", "NO_ACTION_NEEDED"}:
            abstain = raw
        elif raw in {"NULL", "NONE", ""}:
            abstain = None
        else:
            abstain = None

    # If there is a definite choice, force abstain_code to None
    if choice in {"A", "B", "C", "D"}:
        abstain = None

    # ---------- confidence ----------
    conf = 0.0
    m = re.search(r'["\']?confidence["\']?\s*:\s*([-+]?[0-9]*\.?[0-9]+)', t, flags=re.IGNORECASE)
    if m:
        try:
            conf = float(m.group(1))
        except Exception:
            conf = 0.0

    # ---------- rationale ----------
    rationale = None
    m = re.search(
        r'["\']?rationale["\']?\s*:\s*["\'](.+?)["\']\s*[},\n]',
        t,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        rationale = m.group(1).strip()

    # If we got *nothing* useful, fall back to default
    if choice is None and abstain is None and rationale is None and conf == 0.0:
        return default

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
                raise RequestException(f"Server error {resp.status_code}", response=resp)
            return resp
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)

    raise last_err

# ------------------------------------------------------
# 6. Llama call (5k tokens, with fallback)
# ------------------------------------------------------
def call_llama(prompt: str):
    base_payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }

    # First try with max_tokens = 5000
    payload = dict(base_payload)
    payload["max_tokens"] = 5000

    resp = post_with_retries(LLAMA_URL, HEADERS, payload)

    # If the API complains about max_tokens, retry without it
    if resp.status_code == 400:
        try:
            err = resp.json().get("error", {})
            msg = (err.get("message") or "").lower()
            code = (err.get("code") or "").lower()
        except Exception:
            msg = ""
            code = ""

        if "unsupported parameter" in msg or "unsupported_parameter" in code:
            payload = dict(base_payload)
            resp = post_with_retries(LLAMA_URL, HEADERS, payload)

    if resp.status_code != 200:
        return f"ERROR: {resp.status_code} {resp.text}"

    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        # If response JSON is weird, just return raw text
        return resp.text

# ------------------------------------------------------
# 7. Load questions CSV from fixed path (data/questions.csv)
# ------------------------------------------------------
# Assumes working directory is repo root, e.g.:
#   /Users/handler9/Desktop/LLM NOTA Benchmark copy
# and CSV at:
#   data/questions.csv

CSV_PATH = Path("data/50questions.csv").resolve()

print("📂 Using questions CSV from fixed path...")
print(f"➡️ Resolved CSV path: {CSV_PATH}")

if not CSV_PATH.exists():
    print(f"❌ CSV file not found at {CSV_PATH}")
    raise SystemExit

df = pd.read_csv(CSV_PATH)

required_cols = ["stem", "option_A", "option_B", "option_C", "option_D"]
for col in required_cols:
    if col not in df.columns:
        print(f"❌ Missing required column: {col}")
        raise SystemExit

# Prefer question_id if available
id_col = None
for candidate in ["question_id", "id", "QID", "qid", "QuestionID"]:
    if candidate in df.columns:
        id_col = candidate
        break

print("ID column:", id_col or "(none, will use row index)")

# ------------------------------------------------------
# 8. Resume handling
# ------------------------------------------------------
prev_map = None
prev_key_col = None

if RESUME and os.path.exists(OUTPUT_PATH):
    prev_df = pd.read_csv(OUTPUT_PATH)

    if "question_id" in prev_df.columns:
        prev_key_col = "question_id"
    elif "row_index" in prev_df.columns:
        prev_key_col = "row_index"

    if prev_key_col:
        prev_map = {
            row[prev_key_col]: row.to_dict()
            for _, row in prev_df.iterrows()
        }
        print(f"🔁 Loaded {len(prev_map)} previous rows using key '{prev_key_col}'.")
    else:
        print("⚠️ No question_id/row_index in previous results; resume disabled.")

def is_ok(prev, prefix="llama"):
    c = prev.get(f"{prefix}_choice")
    a = prev.get(f"{prefix}_abstain_code")
    if c in {"A", "B", "C", "D"}:
        return True
    if isinstance(a, str) and a not in {"API_ERROR", "PARSING_ERROR"}:
        return True
    return False

# ------------------------------------------------------
# 9. Evaluation loop (LLAMA ONLY)
# ------------------------------------------------------
rows_out = []
total = len(df)
print(f"\n🚀 Running Llama evaluation on {total} questions...\n")

for idx, row in df.iterrows():
    stem = row["stem"]
    opts = (
        f"A. {row['option_A']}\n"
        f"B. {row['option_B']}\n"
        f"C. {row['option_C']}\n"
        f"D. {row['option_D']}"
    )
    prompt = f"{stem}\n\nOptions:\n{opts}"

    # Use question_id if present, else 1..N
    qid = row[id_col] if id_col else idx + 1

    # RESUME
    prev = None
    if prev_map and prev_key_col:
        key = qid if prev_key_col == "question_id" else (idx + 1)
        prev = prev_map.get(key)

    reused = False
    if prev and is_ok(prev, "llama"):
        llama_raw = prev["llama_raw"]
        llama_parsed = {
            "choice": prev["llama_choice"],
            "abstain_code": prev["llama_abstain_code"],
            "confidence": prev["llama_confidence"],
            "rationale": prev["llama_rationale"],
        }
        reused = True
    else:
        llama_raw = call_llama(prompt)
        if isinstance(llama_raw, str) and llama_raw.startswith("ERROR:"):
            llama_parsed = {
                "choice": None,
                "abstain_code": "API_ERROR",
                "confidence": 0.0,
                "rationale": llama_raw,
            }
        else:
            llama_parsed = safe_parse_json(llama_raw)

    out = {
        "row_index": idx + 1,
        "question_id": qid,
        "stem": stem,
        "option_A": row["option_A"],
        "option_B": row["option_B"],
        "option_C": row["option_C"],
        "option_D": row["option_D"],
        "llama_raw": llama_raw,
        "llama_choice": llama_parsed["choice"],
        "llama_abstain_code": llama_parsed["abstain_code"],
        "llama_confidence": llama_parsed["confidence"],
        "llama_rationale": llama_parsed["rationale"],
    }

    rows_out.append(out)

    print(f"Processed {idx+1}/{total} (question_id={qid}){' [reused]' if reused else ''}")

    if AUTOSAVE_EVERY and len(rows_out) % AUTOSAVE_EVERY == 0:
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
        print(f"💾 Autosaved to {OUTPUT_PATH}")

# ------------------------------------------------------
# 10. Final save
# ------------------------------------------------------
pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ DONE! Saved to {OUTPUT_PATH}\n")
