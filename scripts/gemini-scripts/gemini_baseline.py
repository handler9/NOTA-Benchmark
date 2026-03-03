#!/usr/bin/env python3
import os
import re
import json
import time
import random
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
OUTPUT_PATH = "gemini_baseline_7.csv"

SLEEP_BETWEEN_CALLS = 0.15
MAX_RETRIES = 3

# Keep generations small for stability (your output is tiny JSON)
MAX_OUT_TOKENS_GEMINI = 256

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

session = requests.Session()

# ------------------------------------------------------
# 2. Gemini endpoints to try (AUTODETECT)
# ------------------------------------------------------
# We will try these in order until one returns HTTP 200 consistently.
# (Stanford APIM sometimes exposes Gemini via different front doors.)
GEMINI_CANDIDATES = [
    # Your current one (often Google-native)
    ("native_contents", "https://apim.stanfordhealthcare.org/gemini-25-pro/gemini-25-pro"),

    # Common OpenAI-compatible front doors (if enabled in your APIM)
    ("chat_completions_v1", "https://apim.stanfordhealthcare.org/gemini-25-pro/v1/chat/completions"),
    ("chat_completions", "https://apim.stanfordhealthcare.org/gemini-25-pro/chat/completions"),
]

# Cache the working candidate after first success
_WORKING_GEMINI = {"mode": None, "url": None}

# ------------------------------------------------------
# 3. JSON-only instructions
# ------------------------------------------------------
INSTRUCTIONS = """
You’re a careful clinician. When you answer, you should only give one short, structured reply — not a paragraph, not a sentence, not an explanation. Just a small bundle of information with four parts:

- choice – Pick A, B, C, or D if you’re sure about the answer. If you’re not sure, leave this blank (write "null").
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
# 4. Robust JSON-ish parser (your original, kept)
# ------------------------------------------------------
def safe_parse_json(text: str):
    default = {
        "choice": None,
        "abstain_code": "INSUFFICIENT_INFO",
        "confidence": 0.0,
        "rationale": "Model did not respond in the requested JSON-like format; treating as abstention.",
    }

    if not isinstance(text, str) or not text.strip():
        return default

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    obj = None
    try:
        obj = json.loads(cleaned)
    except Exception:
        obj = None

    if obj is None:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                obj = json.loads(candidate)
            except Exception:
                obj = None

    if isinstance(obj, dict):
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

        # normalize choice "null"
        if isinstance(choice, str) and choice.strip().lower() == "null":
            choice = None
        if isinstance(choice, str):
            choice = choice.strip().upper()
            if choice not in {"A", "B", "C", "D"}:
                # allow malformed
                choice = None

        # enforce mutual exclusion
        if choice in {"A", "B", "C", "D"}:
            abstain = None

        return {
            "choice": choice,
            "abstain_code": abstain,
            "confidence": conf,
            "rationale": rationale if isinstance(rationale, str) else "",
        }

    flat = " ".join(cleaned.split())

    choice = None
    m_choice = re.search(r'["\']?\s*choice\s*["\']?\s*:\s*["\']?([ABCD]|null)["\']?', flat, re.IGNORECASE)
    if m_choice:
        raw_choice = m_choice.group(1)
        choice = None if raw_choice.lower() == "null" else raw_choice.upper()

    abstain = None
    m_abstain = re.search(r'["\']?\s*abstain_code\s*["\']?\s*:\s*["\']?([A-Z_]+|null)["\']?', flat, re.IGNORECASE)
    if m_abstain:
        raw_abstain = m_abstain.group(1)
        abstain = None if raw_abstain.lower() == "null" else raw_abstain.upper()

    conf = 0.0
    m_conf = re.search(r'["\']?\s*confidence\s*["\']?\s*:\s*([0-9]*\.?[0-9]+)', flat, re.IGNORECASE)
    if m_conf:
        try:
            conf = float(m_conf.group(1))
        except Exception:
            conf = 0.0

    rationale = None
    m_rat = re.search(r'["\']?\s*rationale\s*["\']?\s*:\s*["\'](.*?)["\']', cleaned, re.DOTALL | re.IGNORECASE)
    if m_rat:
        rationale = m_rat.group(1).strip()

    if choice is not None or abstain is not None or rationale is not None or conf != 0.0:
        if choice in {"A", "B", "C", "D"}:
            abstain = None
        return {
            "choice": choice,
            "abstain_code": abstain,
            "confidence": conf,
            "rationale": rationale or "Recovered fields from malformed JSON-like output.",
        }

    return default

# ------------------------------------------------------
# 5. POST helper with retries
# ------------------------------------------------------
def post_with_retries(url, headers, payload_dict, timeout=90, max_retries=MAX_RETRIES):
    payload_str = json.dumps(payload_dict)
    last_err = None
    last_resp = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(url, headers=headers, data=payload_str, timeout=timeout)
            last_resp = resp

            # 429 backoff
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                sleep_s = float(ra) if ra and ra.replace(".", "", 1).isdigit() else (2 ** attempt) + random.random()
                print(f"⚠️ 429 rate-limited. Sleeping {sleep_s:.1f}s then retrying (attempt {attempt}/{max_retries})")
                time.sleep(sleep_s)
                continue

            # 5xx retry (small number)
            if 500 <= resp.status_code < 600:
                sleep_s = (2 ** attempt) + random.random()
                print(f"⚠️ {resp.status_code} server error. Sleeping {sleep_s:.1f}s then retrying (attempt {attempt}/{max_retries})")
                time.sleep(sleep_s)
                continue

            return resp

        except RequestException as e:
            last_err = e
            sleep_s = (2 ** attempt) + random.random()
            print(f"⚠️ Network error: {e}. Sleeping {sleep_s:.1f}s then retrying (attempt {attempt}/{max_retries})")
            time.sleep(sleep_s)

    # return last response if we have it; otherwise raise
    if last_resp is not None:
        return last_resp
    raise last_err if last_err else RuntimeError("Unknown error in post_with_retries")

# ------------------------------------------------------
# 6. Build payloads for each candidate mode
# ------------------------------------------------------
def build_payload(mode: str, full_text: str):
    """
    Return (payload_dict, response_extractor_fn)
    extractor takes response_json -> text (or raises)
    """
    if mode == "native_contents":
        payload = {
            "contents": [{"role": "user", "parts": [{"text": full_text}]}],
            "generation_config": {"maxOutputTokens": MAX_OUT_TOKENS_GEMINI},
        }

        def extract(data):
            # Stanford sometimes returns list; sometimes dict
            items = data if isinstance(data, list) else [data]
            texts = []
            for item in items:
                candidates = item.get("candidates") or []
                if not candidates:
                    continue
                content = candidates[0].get("content", {}) or {}
                parts = content.get("parts", []) or []
                for p in parts:
                    if isinstance(p, dict) and isinstance(p.get("text"), str):
                        texts.append(p["text"])
            out = "".join(texts).strip()
            if not out:
                raise ValueError("Empty text from native_contents")
            return out

        return payload, extract

    # OpenAI-compatible chat/completions style
    payload = {
        "model": "gemini-2.5-pro",
        "messages": [
            {"role": "user", "content": full_text},
        ],
        "max_tokens": MAX_OUT_TOKENS_GEMINI,
        "temperature": 0.0,
    }

    def extract(data):
        return data["choices"][0]["message"]["content"]

    return payload, extract

# ------------------------------------------------------
# 7. Autodetect Gemini route once
# ------------------------------------------------------
def autodetect_gemini():
    if _WORKING_GEMINI["url"] and _WORKING_GEMINI["mode"]:
        return _WORKING_GEMINI["mode"], _WORKING_GEMINI["url"]

    probe_text = (
        f"{INSTRUCTIONS}\n\n"
        f"Question:\nTEST\n\nOptions:\n"
        f"A. a\nB. b\nC. c\nD. d\n"
    )

    for mode, url in GEMINI_CANDIDATES:
        payload, extract = build_payload(mode, probe_text)
        try:
            resp = post_with_retries(url, HEADERS, payload, timeout=60, max_retries=2)
        except Exception as e:
            print(f"🔎 Gemini probe {mode} @ {url} -> EXCEPTION: {e}")
            continue

        if resp.status_code != 200:
            print(f"🔎 Gemini probe {mode} @ {url} -> {resp.status_code}: {resp.text[:200]}")
            continue

        try:
            data = resp.json()
            _ = extract(data)
        except Exception as e:
            print(f"🔎 Gemini probe {mode} @ {url} -> 200 but could not parse response: {e}")
            continue

        print(f"✅ Gemini route detected: mode={mode} url={url}")
        _WORKING_GEMINI["mode"] = mode
        _WORKING_GEMINI["url"] = url
        return mode, url

    # If nothing works, hard fail with guidance
    raise RuntimeError(
        "❌ Could not detect a working Gemini route. All candidates failed.\n"
        "This usually means the APIM Gemini deployment is down or your subscription key lacks access.\n"
        "Next step: paste one full 500 response body + any request-id/trace headers to APIM support."
    )

# ------------------------------------------------------
# 8. Gemini call (uses detected route)
# ------------------------------------------------------
def call_gemini(user_prompt: str) -> str:
    mode, url = autodetect_gemini()
    full_text = f"{INSTRUCTIONS}\n\nQuestion:\n{user_prompt}"

    payload, extract = build_payload(mode, full_text)

    try:
        resp = post_with_retries(url, HEADERS, payload)
    except Exception as e:
        return f"ERROR: {e}"

    if resp.status_code != 200:
        return f"ERROR: {resp.status_code} {resp.text}"

    try:
        data = resp.json()
    except Exception:
        return f"ERROR: Could not decode JSON: {resp.text[:300]}"

    try:
        return extract(data)
    except Exception as e:
        return f"ERROR: Could not extract text: {e} | raw={resp.text[:300]}"

# ------------------------------------------------------
# 9. Resume helper
# ------------------------------------------------------
def is_model_result_ok(row_dict, prefix: str) -> bool:
    raw = row_dict.get(f"{prefix}_raw")
    abst = row_dict.get(f"{prefix}_abstain_code")
    if not isinstance(raw, str):
        return False
    if raw.startswith("ERROR:"):
        return False
    if abst == "API_ERROR":
        return False
    return True

def to_csv_null(v):
    return "null" if v is None else v

# ------------------------------------------------------
# 10. Load input CSV (data/questions.csv)
# ------------------------------------------------------
CSV_PATH = Path("data/questions.csv").resolve()
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

id_col = None
for candidate in ["question_id", "id", "qid", "QID", "QuestionID"]:
    if candidate in df.columns:
        id_col = candidate
        break
print("Detected ID column:", id_col if id_col else "(none, will use row index)")

# ------------------------------------------------------
# 11. Load resume data
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
# 12. Run evaluation
# ------------------------------------------------------
rows_out = []
total = len(df)
print(f"\n🚀 Starting Gemini evaluation on {total} questions...\n")

for idx, row in df.iterrows():
    stem = str(row["stem"])
    opts = (
        f"A. {row['option_A']}\n"
        f"B. {row['option_B']}\n"
        f"C. {row['option_C']}\n"
        f"D. {row['option_D']}"
    )
    user_prompt = f"{stem}\n\nOptions:\n{opts}"

    question_id = row[id_col] if id_col else idx + 1

    prev_row = None
    if prev_map and prev_key_col:
        key = question_id if prev_key_col == "question_id" else idx + 1
        prev_row = prev_map.get(key)

    reused = False
    if prev_row and is_model_result_ok(prev_row, "gemini"):
        gemini_raw = prev_row["gemini_raw"]
        gemini_parsed = {
            "choice": None if prev_row["gemini_choice"] in {"null", None} else prev_row["gemini_choice"],
            "abstain_code": None if prev_row["gemini_abstain_code"] in {"null", None} else prev_row["gemini_abstain_code"],
            "confidence": prev_row.get("gemini_confidence", 0.0),
            "rationale": prev_row.get("gemini_rationale", ""),
        }
        reused = True
    else:
        gemini_raw = call_gemini(user_prompt)
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
        "question_id": question_id,
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

    rows_out.append(out_row)
    note = " [reused]" if reused else ""
    print(f"Processed {idx+1}/{total} (question_id={question_id}){note}")

    if AUTOSAVE_EVERY and (len(rows_out) % AUTOSAVE_EVERY == 0):
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
        print(f"💾 Autosaved to {OUTPUT_PATH}")

    time.sleep(SLEEP_BETWEEN_CALLS)

pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Done! Saved to {OUTPUT_PATH}\n")
