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

# READ from your existing DeepSeek results
PREV_RESULTS_PATH = "deepseek_doublecheck_judge.csv"

# WRITE to a new file so original is untouched
OUTPUT_PATH = "deepseek_doublecheckfixed_judge.csv"

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
# 2. DeepSeek endpoint & model
# ------------------------------------------------------
DEEPSEEK_URL = "https://apim.stanfordhealthcare.org/deepseekr1/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

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
    """
    Try to recover a valid JSON object from a model response.

    Handles:
    - DeepSeek <think>...</think> blocks
    - Extra text before/after JSON
    - "null"/"" abstain_code normalization
    - Non-float confidence
    """
    default = {
        "choice": None,
        "abstain_code": "PARSING_ERROR",
        "confidence": 0.0,
        "rationale": "Could not parse JSON",
    }

    if not isinstance(text, str) or not text.strip():
        return default

    # Strip DeepSeek-style <think> blocks, just in case
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 1) Try full string
    try:
        obj = json.loads(text)
    except Exception:
        # 2) Try first {...} block
        match = re.search(r"{.*}", text, re.DOTALL)
        if not match:
            return default
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
        except Exception:
            # 3) Last resort: try to recover just the choice letter
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

# ------------------------------------------------------
# 5. POST helper with retries
# ------------------------------------------------------
def post_with_retries(url, headers, json_data, timeout=90, max_retries=3):
    """
    POST with retries for:
    - Connection/DNS issues
    - Timeouts
    - 5xx server errors
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=json_data, timeout=timeout)

            # Retry on 5xx
            if 500 <= resp.status_code < 600:
                last_err = RequestException(f"Server error {resp.status_code}", response=resp)
                raise last_err

            return resp

        except RequestException as e:
            last_err = e
            if attempt == max_retries:
                raise
            sleep_sec = 2 * attempt
            print(f"   ⚠️ Request error (attempt {attempt}/{max_retries}): {e} — retrying in {sleep_sec}s")
            time.sleep(sleep_sec)

    raise last_err if last_err else RuntimeError("Unknown error in post_with_retries")

# ------------------------------------------------------
# 6. Helper function for DeepSeek
# ------------------------------------------------------
def call_deepseek(user_prompt: str) -> str:
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": 5000,
    }
    try:
        resp = post_with_retries(
            DEEPSEEK_URL,
            headers=HEADERS,
            json_data=data,
            timeout=90,
            max_retries=3,
        )
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
    prefix: "deepseek"
    """
    raw = row_dict.get(f"{prefix}_raw")
    abst = row_dict.get(f"{prefix}_abstain_code")

    if not isinstance(raw, str):
        return False
    if raw.startswith("ERROR:"):
        return False
    if abst in {"API_ERROR", "PARSING_ERROR"}:
        return False

    return True

# ------------------------------------------------------
# 8. Load questions CSV from fixed path & enforce question_id
# ------------------------------------------------------
# Assumes working directory is your repo root, e.g.:
#   /Users/handler9/Desktop/LLM NOTA Benchmark copy
# and CSV is at:
#   data/questions.csv

CSV_PATH = Path("data/50questions.csv").resolve()

print("📂 Using questions CSV from fixed path...")
print(f"➡️ Resolved CSV path: {CSV_PATH}")

if not CSV_PATH.exists():
    print(f"❌ CSV file not found at {CSV_PATH}")
    raise SystemExit

df = pd.read_csv(CSV_PATH)

required_cols = ["stem", "option_A", "option_B", "option_C", "option_D", "question_id"]
for col in required_cols:
    if col not in df.columns:
        print(f"❌ Missing required column: {col}")
        raise SystemExit

# 🔒 Force using our renumbered question_id
id_col = "question_id"
print("Using enforced ID column:", id_col)

# ------------------------------------------------------
# 9. Load previous DeepSeek results (resume mode)
# ------------------------------------------------------
prev_map = None
prev_key_col = None

if RESUME and os.path.exists(PREV_RESULTS_PATH):
    print(f"\n🔁 Resume mode: loading existing DeepSeek results from {PREV_RESULTS_PATH}")
    prev_df = pd.read_csv(PREV_RESULTS_PATH)

    if "question_id" in prev_df.columns:
        prev_key_col = "question_id"
    elif "row_index" in prev_df.columns:
        prev_key_col = "row_index"

    if prev_key_col:
        prev_map = {
            row[prev_key_col]: row.to_dict()
            for _, row in prev_df.iterrows()
        }
        print(f"   Found {len(prev_map)} previous rows with key '{prev_key_col}'")
    else:
        print("   Existing DeepSeek CSV has no question_id/row_index; resume disabled.")
else:
    if RESUME:
        print(f"\n🔁 Resume enabled, but no existing results file found at {PREV_RESULTS_PATH}.")

# ------------------------------------------------------
# 10. Run evaluation (DeepSeek-only)
# ------------------------------------------------------
rows_out = []
total = len(df)
print(f"\n🚀 Starting DeepSeek evaluation on {total} questions...\n")

for idx, row in df.iterrows():
    stem = row["stem"]
    opts = (
        f"A. {row['option_A']}\n"
        f"B. {row['option_B']}\n"
        f"C. {row['option_C']}\n"
        f"D. {row['option_D']}"
    )
    user_prompt = f"{stem}\n\nOptions:\n{opts}"

    # 🔒 Always use internal question_id (1–500)
    question_id = int(row["question_id"])

    # Try to reuse previous DeepSeek results if available
    prev_row = None
    if prev_map is not None and prev_key_col is not None:
        key_val = question_id if prev_key_col == "question_id" else (idx + 1)
        prev_row = prev_map.get(key_val)

    deepseek_raw = None
    deepseek_parsed = None
    deepseek_reused_flag = False

    # ----- DeepSeek -----
    if prev_row and is_model_result_ok(prev_row, "deepseek"):
        deepseek_raw = prev_row.get("deepseek_raw")
        deepseek_parsed = {
            "choice": prev_row.get("deepseek_choice"),
            "abstain_code": prev_row.get("deepseek_abstain_code"),
            "confidence": prev_row.get("deepseek_confidence"),
            "rationale": prev_row.get("deepseek_rationale"),
        }
        deepseek_reused_flag = True
    else:
        try:
            deepseek_raw = call_deepseek(user_prompt)
            if deepseek_raw.startswith("ERROR:"):
                deepseek_parsed = {
                    "choice": None,
                    "abstain_code": "API_ERROR",
                    "confidence": 0.0,
                    "rationale": deepseek_raw,
                }
            else:
                deepseek_parsed = safe_parse_json(deepseek_raw)
        except Exception as e:
            deepseek_raw = f"ERROR: {e}"
            deepseek_parsed = {
                "choice": None,
                "abstain_code": "API_ERROR",
                "confidence": 0.0,
                "rationale": f"Exception: {e}",
            }

    out_row = {
        "row_index": idx + 1,
        "question_id": question_id,
        "stem": stem,
        "option_A": row["option_A"],
        "option_B": row["option_B"],
        "option_C": row["option_C"],
        "option_D": row["option_D"],

        # DeepSeek
        "deepseek_raw": deepseek_raw,
        "deepseek_choice": deepseek_parsed.get("choice"),
        "deepseek_abstain_code": deepseek_parsed.get("abstain_code"),
        "deepseek_confidence": deepseek_parsed.get("confidence"),
        "deepseek_rationale": deepseek_parsed.get("rationale"),
    }

    rows_out.append(out_row)

    reuse_note = " [DeepSeek reused]" if deepseek_reused_flag else ""
    print(f"Processed {idx+1}/{total} (question_id={question_id}){reuse_note}")

    # Autosave periodically
    if AUTOSAVE_EVERY and (len(rows_out) % AUTOSAVE_EVERY == 0):
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
        print(f"   💾 Autosaved partial results to {OUTPUT_PATH}")

    time.sleep(0.1)  # gentle pacing

# ------------------------------------------------------
# 11. Final save
# ------------------------------------------------------
pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Done! DeepSeek results saved to {OUTPUT_PATH}\n")
