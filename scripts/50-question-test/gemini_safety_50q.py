#!/usr/bin/env python3
import os
import re
import json
import time
import random
import math
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from requests.exceptions import RequestException

# ------------------------------------------------------
# 0. Config
# ------------------------------------------------------
RESUME = True
RETRY_BROKEN_ROWS = True
AUTOSAVE_EVERY = 25

INPUT_CSV = "data/50questions.csv"
OUTPUT_PATH = "gemini_doublecheck_judge.csv"

SLEEP_BETWEEN_CALLS = 0.15
MAX_RETRIES = 3

MAX_OUT_TOKENS_GEMINI = 5000
THINKING_BUDGET = 128
TEMPERATURE = 0.0

SAVE_HTTP_DEBUG = True
HTTP_DEBUG_TRUNCATE = 6000

# ------------------------------------------------------
# 1. Load API key
# ------------------------------------------------------
load_dotenv()
KEY = os.getenv("SECUREGPT_API_KEY")
if not KEY:
    print("❌ SECUREGPT_API_KEY not found in .env")
    raise SystemExit(1)

HEADERS = {
    "Ocp-Apim-Subscription-Key": KEY,
    "Content-Type": "application/json",
}
session = requests.Session()

# ------------------------------------------------------
# 2. Endpoint (ONLY the one that works)
# ------------------------------------------------------
GEMINI_MODE = "native_contents"
GEMINI_URL = "https://apim.stanfordhealthcare.org/gemini-25-pro/gemini-25-pro"

# ------------------------------------------------------
# 3. Instructions (UPDATED)
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


""".strip()

# ------------------------------------------------------
# 4. Helpers
# ------------------------------------------------------
def strip_fences(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE)
    s = s.replace("```", "")
    return s.strip()

def safe_parse_json(text: str):
    default = {
        "choice": None,
        "abstain_code": "INSUFFICIENT_INFO",
        "confidence": 0.0,
        "rationale": "Model did not respond in the requested JSON; treating as abstention.",
    }
    if not isinstance(text, str) or not text.strip():
        return default

    cleaned = strip_fences(text)

    # Try direct JSON parse
    try:
        obj = json.loads(cleaned)
    except Exception:
        obj = None

    # Try extracting first {...} block
    if obj is None:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(cleaned[start:end+1])
            except Exception:
                obj = None

    if not isinstance(obj, dict):
        return default

    choice = obj.get("choice")
    abstain = obj.get("abstain_code")
    conf = obj.get("confidence")
    rationale = obj.get("rationale")

    # Normalize choice
    if isinstance(choice, str) and choice.strip().lower() == "null":
        choice = None
    if isinstance(choice, str):
        choice = choice.strip().upper()
        if choice not in {"A", "B", "C", "D"}:
            choice = None
    elif choice is not None:
        # If it came as a non-string (e.g., null/None), keep as None
        choice = None

    # Normalize abstain_code
    if isinstance(abstain, str) and abstain.strip().lower() in {"null", ""}:
        abstain = None
    if isinstance(abstain, str):
        abstain = abstain.strip().upper()
        if abstain not in {"NO_VALID_OPTION", "INSUFFICIENT_INFO", "NO_ACTION_NEEDED", "API_ERROR"}:
            # Keep unknown codes as-is but uppercase; or set None if you prefer strictness
            pass

    # Normalize confidence
    try:
        conf = float(conf) if conf is not None else 0.0
    except Exception:
        conf = 0.0
    if math.isnan(conf) or math.isinf(conf):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    # Enforce mutual exclusivity
    if choice in {"A", "B", "C", "D"}:
        abstain = None
    else:
        # If no choice, ensure abstain_code present
        if abstain is None:
            abstain = "INSUFFICIENT_INFO"

    # Rationale
    if not isinstance(rationale, str):
        rationale = ""

    return {
        "choice": choice,
        "abstain_code": abstain,
        "confidence": conf,
        "rationale": rationale.strip(),
    }

def post_with_retries(url, headers, payload_dict, timeout=120, max_retries=MAX_RETRIES):
    last_err = None
    last_resp = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(url, headers=headers, json=payload_dict, timeout=timeout)
            last_resp = resp

            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                sleep_s = float(ra) if ra and ra.replace(".", "", 1).isdigit() else (2**attempt) + random.random()
                print(f"⚠️ 429 rate-limited. Sleeping {sleep_s:.1f}s then retrying ({attempt}/{max_retries})")
                time.sleep(sleep_s)
                continue

            if 500 <= resp.status_code < 600:
                sleep_s = (2**attempt) + random.random()
                print(f"⚠️ {resp.status_code} server error. Sleeping {sleep_s:.1f}s then retrying ({attempt}/{max_retries})")
                time.sleep(sleep_s)
                continue

            return resp

        except RequestException as e:
            last_err = e
            sleep_s = (2**attempt) + random.random()
            print(f"⚠️ Network error: {e}. Sleeping {sleep_s:.1f}s then retrying ({attempt}/{max_retries})")
            time.sleep(sleep_s)

    if last_resp is not None:
        return last_resp
    raise last_err if last_err else RuntimeError("Unknown error")

def extract_text_from_native_contents(data) -> str:
    """
    Stanford APIM returns a LIST of chunk objects.
    We concatenate text across ALL chunks (in order).
    """
    texts = []

    items = data if isinstance(data, list) else [data]

    for item in items:
        if not isinstance(item, dict):
            continue
        candidates = item.get("candidates") or []
        if not candidates:
            continue
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        for p in parts:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                texts.append(p["text"])

    return "".join(texts).strip()

def looks_truncated_or_broken(raw: str) -> bool:
    if not isinstance(raw, str) or not raw.strip():
        return True
    s = raw.strip()
    if s.startswith("ERROR:"):
        return True
    if len(s) < 40:
        return True
    if "{" in s and "}" not in s:
        return True
    return False

def is_model_result_ok(row_dict, prefix: str) -> bool:
    raw = row_dict.get(f"{prefix}_raw")
    if looks_truncated_or_broken(raw):
        return False
    abst = row_dict.get(f"{prefix}_abstain_code")
    if isinstance(abst, str) and abst.strip().upper() == "API_ERROR":
        return False
    return True

def to_csv_null(v):
    return "null" if v is None else v

def norm_key(v):
    if v is None:
        return None
    try:
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
    except Exception:
        pass
    return str(v).strip()

# ------------------------------------------------------
# 5. Gemini call
# ------------------------------------------------------
def call_gemini(user_prompt: str):
    full_text = f"{INSTRUCTIONS}\n\nQuestion:\n{user_prompt}"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": full_text}]}],
        "generationConfig": {
            "maxOutputTokens": MAX_OUT_TOKENS_GEMINI,
            "temperature": TEMPERATURE,
            "thinkingConfig": {"thinkingBudget": THINKING_BUDGET},
        },
    }

    try:
        resp = post_with_retries(GEMINI_URL, HEADERS, payload)
    except Exception as e:
        return f"ERROR: {e}", None

    http_debug = None
    if SAVE_HTTP_DEBUG:
        http_debug = f"status={resp.status_code}\n{resp.text[:HTTP_DEBUG_TRUNCATE]}"

    if resp.status_code != 200:
        return f"ERROR: {resp.status_code} {resp.text[:1500]}", http_debug

    try:
        data = resp.json()
    except Exception:
        return f"ERROR: Could not decode JSON: {resp.text[:1500]}", http_debug

    try:
        text = extract_text_from_native_contents(data)
        if not text:
            return "ERROR: Empty extracted text", http_debug
        return text, http_debug
    except Exception as e:
        return f"ERROR: Extract failed: {e}", http_debug

# ------------------------------------------------------
# 6. Load input CSV
# ------------------------------------------------------
CSV_PATH = Path(INPUT_CSV).resolve()
print("📂 Using questions CSV:", CSV_PATH)
if not CSV_PATH.exists():
    print(f"❌ CSV not found: {CSV_PATH}")
    raise SystemExit(1)

df = pd.read_csv(CSV_PATH)
required_cols = ["stem", "option_A", "option_B", "option_C", "option_D"]
for c in required_cols:
    if c not in df.columns:
        raise SystemExit(f"❌ Missing column: {c}")

id_col = None
for candidate in ["question_id", "id", "qid", "QID", "QuestionID"]:
    if candidate in df.columns:
        id_col = candidate
        break
print("Detected ID column:", id_col if id_col else "(none; using row index)")

# ------------------------------------------------------
# 7. Resume from output (optional)
# ------------------------------------------------------
prev_map = None
prev_key_col = None
if RESUME and os.path.exists(OUTPUT_PATH):
    prev_df = pd.read_csv(OUTPUT_PATH)
    prev_key_col = "question_id" if "question_id" in prev_df.columns else ("row_index" if "row_index" in prev_df.columns else None)
    if prev_key_col:
        prev_map = {norm_key(r[prev_key_col]): r.to_dict() for _, r in prev_df.iterrows()}
        print(f"↩️ Resume enabled: loaded {len(prev_map)} prior rows from {OUTPUT_PATH}")

# ------------------------------------------------------
# 8. Run
# ------------------------------------------------------
rows_out = []
total = len(df)
print(f"\n🚀 Running Gemini judge on {total} questions")
print(f"➡️ Output: {OUTPUT_PATH}")
print(f"➡️ maxOutputTokens={MAX_OUT_TOKENS_GEMINI}, thinkingBudget={THINKING_BUDGET}, temperature={TEMPERATURE}\n")

for idx, row in df.iterrows():
    stem = str(row["stem"])
    opts = (
        f"A. {row['option_A']}\n"
        f"B. {row['option_B']}\n"
        f"C. {row['option_C']}\n"
        f"D. {row['option_D']}"
    )
    user_prompt = f"{stem}\n\nOptions:\n{opts}"

    qid = row[id_col] if id_col else (idx + 1)

    prev_row = None
    if prev_map and prev_key_col:
        key = norm_key(qid if prev_key_col == "question_id" else (idx + 1))
        prev_row = prev_map.get(key)

    reused = False
    gemini_http_raw = None

    # Reuse only if row looks OK; otherwise re-run if RETRY_BROKEN_ROWS
    can_reuse = bool(prev_row and is_model_result_ok(prev_row, "gemini"))
    if can_reuse:
        gemini_raw = prev_row.get("gemini_raw", "")
        gemini_parsed = {
            "choice": None if prev_row.get("gemini_choice") in {"null", None} else prev_row.get("gemini_choice"),
            "abstain_code": None if prev_row.get("gemini_abstain_code") in {"null", None} else prev_row.get("gemini_abstain_code"),
            "confidence": prev_row.get("gemini_confidence", 0.0),
            "rationale": "" if prev_row.get("gemini_rationale") in {"null", None} else prev_row.get("gemini_rationale"),
        }
        gemini_http_raw = prev_row.get("gemini_http_raw", "") if SAVE_HTTP_DEBUG else None
        reused = True
    else:
        if prev_row and (not RETRY_BROKEN_ROWS):
            # Keep the broken previous row as-is (no retry)
            gemini_raw = prev_row.get("gemini_raw", "")
            gemini_parsed = {
                "choice": None if prev_row.get("gemini_choice") in {"null", None} else prev_row.get("gemini_choice"),
                "abstain_code": None if prev_row.get("gemini_abstain_code") in {"null", None} else prev_row.get("gemini_abstain_code"),
                "confidence": prev_row.get("gemini_confidence", 0.0),
                "rationale": "" if prev_row.get("gemini_rationale") in {"null", None} else prev_row.get("gemini_rationale"),
            }
            gemini_http_raw = prev_row.get("gemini_http_raw", "") if SAVE_HTTP_DEBUG else None
            reused = True
        else:
            gemini_raw, gemini_http_raw = call_gemini(user_prompt)
            if isinstance(gemini_raw, str) and gemini_raw.startswith("ERROR:"):
                gemini_parsed = {
                    "choice": None,
                    "abstain_code": "API_ERROR",
                    "confidence": 0.0,
                    "rationale": gemini_raw,
                }
            else:
                gemini_parsed = safe_parse_json(gemini_raw)

    out_row = {
        "row_index": idx + 1,
        "question_id": qid,
        "stem": stem,
        "option_A": row["option_A"],
        "option_B": row["option_B"],
        "option_C": row["option_C"],
        "option_D": row["option_D"],
        "gemini_raw": gemini_raw,
        "gemini_choice": to_csv_null(gemini_parsed["choice"]),
        "gemini_abstain_code": to_csv_null(gemini_parsed["abstain_code"]),
        "gemini_confidence": float(gemini_parsed["confidence"] or 0.0),
        "gemini_rationale": to_csv_null(gemini_parsed["rationale"]),
    }
    if SAVE_HTTP_DEBUG:
        out_row["gemini_http_raw"] = gemini_http_raw if gemini_http_raw is not None else ""

    rows_out.append(out_row)

    note = " [reused]" if reused else ""
    print(f"Processed {idx+1}/{total} (question_id={qid}){note}")

    if AUTOSAVE_EVERY and (len(rows_out) % AUTOSAVE_EVERY == 0):
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
        print(f"💾 Autosaved to {OUTPUT_PATH}")

    time.sleep(SLEEP_BETWEEN_CALLS)

pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Done! Saved to {OUTPUT_PATH}\n")
