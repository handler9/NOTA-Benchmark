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
RESUME = True          # ✅ so we can reuse good rows and rerun only bad ones
AUTOSAVE_EVERY = 25

OUTPUT_PATH = "gpt_doublecheck.csv"

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
# 2. GPT-5 endpoint
# ------------------------------------------------------
GPT5_URL = (
    "https://apim.stanfordhealthcare.org/openai-eastus2/"
    "deployments/gpt-5/chat/completions?api-version=2024-12-01-preview"
)

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
# 4. Robust JSON-ish parser
# ------------------------------------------------------
def safe_parse_json(text: str):
    """
    Try very hard to recover a structured answer from GPT-5 output.
    """
    default = {
        "choice": None,
        "abstain_code": "INSUFFICIENT_INFO",
        "confidence": 0.0,
        "rationale": "Model did not respond in the requested JSON-like format; treating as abstention.",
    }

    if not isinstance(text, str) or not text.strip():
        return default

    # Remove <think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove markdown code fences (``` or ```json)
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()

    # ---------- 1) Try full JSON ----------
    obj = None
    try:
        obj = json.loads(cleaned)
    except Exception:
        obj = None

    # ---------- 2) Try substring JSON ----------
    if obj is None:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                obj = json.loads(candidate)
            except Exception:
                obj = None

    # ---------- 3) If JSON parsing succeeded ----------
    if isinstance(obj, dict):
        choice = obj.get("choice")
        abstain = obj.get("abstain_code")
        conf = obj.get("confidence")
        rationale = obj.get("rationale")

        # Normalize abstain_code "null" / "" -> None
        if isinstance(abstain, str) and abstain.strip().lower() in {"null", ""}:
            abstain = None

        # Normalize confidence
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

    # ---------- 4) Regex-based extraction fallback ----------
    flat = " ".join(cleaned.split())

    # choice
    choice = None
    m_choice = re.search(
        r'["\']?\s*choice\s*["\']?\s*:\s*["\']?([ABCD]|null)["\']?',
        flat,
        re.IGNORECASE,
    )
    if m_choice:
        raw_choice = m_choice.group(1)
        choice = None if raw_choice.lower() == "null" else raw_choice.upper()
    else:
        m_choice2 = re.search(r'\bchoice\b\s*[:=]\s*([ABCD])', flat, re.IGNORECASE)
        if m_choice2:
            choice = m_choice2.group(1).upper()

    # abstain_code
    abstain = None
    m_abstain = re.search(
        r'["\']?\s*abstain_code\s*["\']?\s*:\s*["\']?([A-Z_]+|null)["\']?',
        flat,
        re.IGNORECASE,
    )
    if m_abstain:
        raw_abstain = m_abstain.group(1)
        abstain = None if raw_abstain.lower() == "null" else raw_abstain.upper()

    # confidence
    conf = 0.0
    m_conf = re.search(
        r'["\']?\s*confidence\s*["\']?\s*:\s*([0-9]*\.?[0-9]+)',
        flat,
        re.IGNORECASE,
    )
    if m_conf:
        try:
            conf = float(m_conf.group(1))
        except Exception:
            conf = 0.0

    # rationale (supports single/double quotes)
    rationale = None
    m_rat = re.search(
        r'["\']?\s*rationale\s*["\']?\s*:\s*["\'](.*?)["\']',
        cleaned,
        re.DOTALL | re.IGNORECASE,
    )
    if m_rat:
        rationale = m_rat.group(1).strip()

    # If we got something, return it; else default abstention
    if choice is not None or abstain is not None or rationale is not None or conf != 0.0:
        return {
            "choice": choice,
            "abstain_code": abstain,
            "confidence": conf,
            "rationale": rationale or "Recovered fields from malformed JSON-like output.",
        }

    return default

# ------------------------------------------------------
# 5. POST helper
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
# 6. Call GPT-5
# ------------------------------------------------------
def call_gpt5(user_prompt: str) -> str:
    data = {
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": 5000,
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
def is_model_result_ok(row_dict, prefix: str) -> bool:
    """
    Decide if we should trust & reuse an existing result for a given model.

    We want to *rerun* rows where:
    - abstain_code == "INSUFFICIENT_INFO"
    - and the rationale contains our default JSON-format message.
    """
    raw = row_dict.get(f"{prefix}_raw")
    abst = row_dict.get(f"{prefix}_abstain_code")
    rationale = row_dict.get(f"{prefix}_rationale")

    if not isinstance(raw, str):
        return False
    if raw.startswith("ERROR:"):
        return False
    if abst == "API_ERROR":
        return False

    # 🔁 Force re-run of the "JSON-like format" abstentions
    if (
        abst == "INSUFFICIENT_INFO"
        and isinstance(rationale, str)
        and "JSON-like format; treating as abstention" in rationale
    ):
        return False

    return True

# ------------------------------------------------------
# 8. Load input CSV from fixed path (data/questions.csv)
# ------------------------------------------------------
# Assumes working directory is repo root:
#   /Users/handler9/Desktop/LLM NOTA Benchmark copy
# and CSV at:
#   data/afrimedqa_questions_FALSE-NOTA-REMOVED

CSV_PATH = Path("data/afrimedqa_questions_FALSE-NOTA-REMOVED.csv").resolve()

print("📂 Using questions CSV from fixed path...")
print(f"➡️ Resolved CSV path: {CSV_PATH}")

if not CSV_PATH.exists():
    print(f"❌ CSV file not found at {CSV_PATH}")
    raise SystemExit

df = pd.read_csv(CSV_PATH)

required_cols = ["stem", "option_A", "option_B", "option_C", "option_D"]
for col in required_cols:
    if col not in df.columns:
        raise SystemExit(f"❌ Missing column: {col}")

# Prefer 'question_id' if it exists
id_col = None
for candidate in ["question_id", "id", "qid", "QID", "QuestionID"]:
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

    if "question_id" in prev_df.columns:
        prev_key_col = "question_id"
    elif "row_index" in prev_df.columns:
        prev_key_col = "row_index"

    if prev_key_col:
        prev_map = {row[prev_key_col]: row.to_dict() for _, row in prev_df.iterrows()}
        print(f"Loaded {len(prev_map)} previous rows.")

# ------------------------------------------------------
# 10. Run evaluation
# ------------------------------------------------------
rows_out = []
total = len(df)
print(f"\n🚀 Starting evaluation on {total} questions...\n")

for idx, row in df.iterrows():
    stem = row["stem"]
    opts = (
        f"A. {row['option_A']}\n"
        f"B. {row['option_B']}\n"
        f"C. {row['option_C']}\n"
        f"D. {row['option_D']}"
    )
    user_prompt = f"{stem}\n\nOptions:\n{opts}"

    # use existing question_id column if present, else 1..N
    question_id = row[id_col] if id_col else idx + 1

    prev_row = None
    if prev_map and prev_key_col:
        key = question_id if prev_key_col == "question_id" else idx + 1
        prev_row = prev_map.get(key)

    if prev_row and is_model_result_ok(prev_row, "gpt5"):
        gpt5_raw = prev_row["gpt5_raw"]
        gpt5_parsed = {
            "choice": prev_row["gpt5_choice"],
            "abstain_code": prev_row["gpt5_abstain_code"],
            "confidence": prev_row["gpt5_confidence"],
            "rationale": prev_row["gpt5_rationale"],
        }
        reused = True
    else:
        reused = False
        gpt5_raw = call_gpt5(user_prompt)
        if isinstance(gpt5_raw, str) and gpt5_raw.startswith("ERROR:"):
            gpt5_parsed = {
                "choice": None,
                "abstain_code": "API_ERROR",
                "confidence": 0.0,
                "rationale": gpt5_raw,
            }
        else:
            gpt5_parsed = safe_parse_json(gpt5_raw)

    out_row = {
        "row_index": idx + 1,
        "question_id": question_id,
        "stem": stem,
        "option_A": row["option_A"],
        "option_B": row["option_B"],
        "option_C": row["option_C"],
        "option_D": row["option_D"],
        "gpt5_raw": gpt5_raw,
        "gpt5_choice": gpt5_parsed["choice"],
        "gpt5_abstain_code": gpt5_parsed["abstain_code"],
        "gpt5_confidence": gpt5_parsed["confidence"],
        "gpt5_rationale": gpt5_parsed["rationale"],
    }

    rows_out.append(out_row)

    note = " [reused]" if reused else ""
    print(f"Processed {idx+1}/{total}{note}")

    if AUTOSAVE_EVERY and len(rows_out) % AUTOSAVE_EVERY == 0:
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
        print(f"   💾 Autosaved to {OUTPUT_PATH}")

    time.sleep(0.1)

# ------------------------------------------------------
# 11. Save final
# ------------------------------------------------------
pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Done! Saved to {OUTPUT_PATH}\n")
