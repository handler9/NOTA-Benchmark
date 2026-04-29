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
# 0. Config
# ------------------------------------------------------
RESUME = True
AUTOSAVE_EVERY = 25
OUTPUT_PATH = "gpt5_triage_with_labs_prompt2.csv"

# ------------------------------------------------------
# 1. Load API key from .env
# ------------------------------------------------------
load_dotenv()
KEY = os.getenv("AIHUB_API_KEY") or os.getenv("SECUREGPT_API_KEY")
if not KEY:
    print("❌ SECUREGPT_API_KEY not found in .env")
    raise SystemExit

HEADERS = {
    "api-key": KEY,
    "Content-Type": "application/json",
}

# ------------------------------------------------------
# 2. GPT-5 endpoint
# ------------------------------------------------------
GPT5_URL = (
    "https://aihubapi.stanfordhealthcare.org/azure-openai/deployments/gpt-5-4/chat/completions?api-version=2025-04-01-preview"
)

# ------------------------------------------------------
# 3. Think-Then-Decide prompt (JSON output)
# ------------------------------------------------------
PROMPT_TEMPLATE = """
You’re a careful clinician evaluating patient-triage scenarios.

Before answering, silently go through three steps:

1. Identify the key clinical findings and determine the most appropriate level of urgency for this patient.
2. Check whether any of the four options (A–D) actually match that correct urgency level.
3. If none of the options are appropriate or you are unsure, you must abstain instead of guessing.

Do this reasoning in your head and do NOT show these steps.

After thinking, give only one short structured reply with four parts:

- choice – Pick A, B, C, or D if, after checking, you are confident one option is appropriate. If none are appropriate or you are unsure, write null.
- abstain_code – If you didn’t pick an answer, say why:
  - NO_VALID_OPTION – None of the choices make sense.
  - INSUFFICIENT_INFO – You don’t have enough information.
  - NO_ACTION_NEEDED – Nothing needs to be done.
  Otherwise write null.
- confidence – A number between 0 and 1 (for example 0.8 = 80% sure).
- rationale – A short 1–3 sentence explanation.

Important rules:
- You cannot both pick an answer and abstain.
- Either choose A–D or provide an abstain_code.
- Base your answer only on the information provided.
- If unsure between choosing and abstaining, abstain.

Respond ONLY in this JSON format and nothing else:

{{
  "choice": "A/B/C/D or null",
  "abstain_code": "NO_VALID_OPTION/INSUFFICIENT_INFO/NO_ACTION_NEEDED or null",
  "confidence": 0.0,
  "rationale": "one to three sentences"
}}

Clinical vignette:

{vignette}
"""

# ------------------------------------------------------
# 4. Parse GPT-5 JSON output
# ------------------------------------------------------
def safe_parse_triage_output(text: str):
    default = {
        "choice": None,
        "abstain_code": "INSUFFICIENT_INFO",
        "confidence": 0.0,
        "rationale": "Model did not respond in the requested JSON format.",
        "parse_status": "FAILED_PARSE",
    }

    if not isinstance(text, str) or not text.strip():
        return default

    cleaned = text.strip()

    # Remove markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # Remove <think> blocks if present
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    obj = None

    # 1) Try direct JSON parse
    try:
        obj = json.loads(cleaned)
    except Exception:
        obj = None

    # 2) Try extracting JSON substring
    if obj is None:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                obj = json.loads(candidate)
            except Exception:
                obj = None

    # 3) Parsed dict path
    if isinstance(obj, dict):
        choice = obj.get("choice")
        abstain_code = obj.get("abstain_code")
        confidence = obj.get("confidence")
        rationale = obj.get("rationale")

        if isinstance(choice, str):
            if choice.strip().lower() == "null":
                choice = None
            else:
                choice = choice.strip().upper()

        if isinstance(abstain_code, str):
            if abstain_code.strip().lower() == "null":
                abstain_code = None
            else:
                abstain_code = abstain_code.strip().upper()

        try:
            confidence = float(confidence) if confidence is not None else 0.0
        except Exception:
            confidence = 0.0

        rationale = str(rationale).strip() if rationale is not None else ""

        return {
            "choice": choice,
            "abstain_code": abstain_code,
            "confidence": confidence,
            "rationale": rationale,
            "parse_status": "OK",
        }

    # 4) Regex fallback
    flat = " ".join(cleaned.split())

    choice = None
    abstain_code = None
    confidence = 0.0
    rationale = None

    m_choice = re.search(
        r'["\']?\s*choice\s*["\']?\s*:\s*["\']?([ABCD]|null)["\']?',
        flat,
        flags=re.IGNORECASE,
    )
    if m_choice:
        raw_choice = m_choice.group(1)
        choice = None if raw_choice.lower() == "null" else raw_choice.upper()

    m_abstain = re.search(
        r'["\']?\s*abstain_code\s*["\']?\s*:\s*["\']?([A-Z_]+|null)["\']?',
        flat,
        flags=re.IGNORECASE,
    )
    if m_abstain:
        raw_abstain = m_abstain.group(1)
        abstain_code = None if raw_abstain.lower() == "null" else raw_abstain.upper()

    m_conf = re.search(
        r'["\']?\s*confidence\s*["\']?\s*:\s*([0-9]*\.?[0-9]+)',
        flat,
        flags=re.IGNORECASE,
    )
    if m_conf:
        try:
            confidence = float(m_conf.group(1))
        except Exception:
            confidence = 0.0

    m_rat = re.search(
        r'["\']?\s*rationale\s*["\']?\s*:\s*["\'](.*?)["\']',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m_rat:
        rationale = m_rat.group(1).strip()

    if choice is not None or abstain_code is not None or rationale is not None:
        return {
            "choice": choice,
            "abstain_code": abstain_code,
            "confidence": confidence,
            "rationale": rationale or "Recovered fields from malformed JSON-like output.",
            "parse_status": "RECOVERED",
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
        "model": "chat",
        "messages": [
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

    try:
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return f"ERROR: Unexpected response format: {resp.text}"

# ------------------------------------------------------
# 7. Resume helper
# ------------------------------------------------------
def is_model_result_ok(row_dict, prefix: str) -> bool:
    raw = row_dict.get(f"{prefix}_raw")
    parse_status = row_dict.get(f"{prefix}_parse_status")

    if not isinstance(raw, str):
        return False
    if raw.startswith("ERROR:"):
        return False
    if parse_status in {"FAILED_PARSE", "API_ERROR"}:
        return False

    return True

# ------------------------------------------------------
# 8. Load input CSV
# ------------------------------------------------------
CSV_PATH = Path("data/triageQA_with_labs_questions.csv").resolve()

print("📂 Using triage CSV...")
print(f"➡️ Resolved CSV path: {CSV_PATH}")

if not CSV_PATH.exists():
    print(f"❌ CSV file not found at {CSV_PATH}")
    raise SystemExit

df = pd.read_csv(CSV_PATH)

required_cols = [
    "question_id",
    "stem",
    "option_A",
    "option_B",
    "option_C",
    "option_D",
]
for col in required_cols:
    if col not in df.columns:
        raise SystemExit(f"❌ Missing column: {col}")

print(f"Loaded {len(df)} rows.")

# ------------------------------------------------------
# 9. Load resume data
# ------------------------------------------------------
prev_map = None
if RESUME and os.path.exists(OUTPUT_PATH):
    print(f"\n🔁 Resuming from {OUTPUT_PATH}")
    prev_df = pd.read_csv(OUTPUT_PATH)
    if "question_id" in prev_df.columns:
        prev_map = {row["question_id"]: row.to_dict() for _, row in prev_df.iterrows()}
        print(f"Loaded {len(prev_map)} previous rows.")

# ------------------------------------------------------
# 10. Run evaluation
# ------------------------------------------------------
rows_out = []
total = len(df)
print(f"\n🚀 Starting evaluation on {total} with-lab questions...\n")

for idx, row in df.iterrows():
    question_id = row["question_id"]
    prev_row = prev_map.get(question_id) if prev_map else None

    if prev_row and is_model_result_ok(prev_row, "gpt5"):
        gpt5_raw = prev_row["gpt5_raw"]
        gpt5_parsed = {
            "choice": prev_row.get("gpt5_choice"),
            "abstain_code": prev_row.get("gpt5_abstain_code"),
            "confidence": prev_row.get("gpt5_confidence"),
            "rationale": prev_row.get("gpt5_rationale"),
            "parse_status": prev_row.get("gpt5_parse_status"),
        }
        user_prompt = prev_row.get("user_prompt", "")
        reused = True
    else:
        reused = False

        vignette = (
            f"{row['stem']}\n\n"
            f"Options:\n"
            f"A. {row['option_A']}\n"
            f"B. {row['option_B']}\n"
            f"C. {row['option_C']}\n"
            f"D. {row['option_D']}"
        )
        user_prompt = PROMPT_TEMPLATE.format(vignette=vignette)

        gpt5_raw = call_gpt5(user_prompt)

        if isinstance(gpt5_raw, str) and gpt5_raw.startswith("ERROR:"):
            gpt5_parsed = {
                "choice": None,
                "abstain_code": "API_ERROR",
                "confidence": 0.0,
                "rationale": gpt5_raw,
                "parse_status": "API_ERROR",
            }
        else:
            gpt5_parsed = safe_parse_triage_output(gpt5_raw)

    out_row = {
        "row_index": idx + 1,
        "question_id": row["question_id"],
        "correct_choice": row["correct_choice"] if "correct_choice" in df.columns else None,
        "stem": row["stem"],
        "option_A": row["option_A"],
        "option_B": row["option_B"],
        "option_C": row["option_C"],
        "option_D": row["option_D"],
        "prompt_used": "think_then_decide_prompt2_with_labs",
        "user_prompt": user_prompt,
        "gpt5_raw": gpt5_raw,
        "gpt5_choice": gpt5_parsed["choice"],
        "gpt5_abstain_code": gpt5_parsed["abstain_code"],
        "gpt5_confidence": gpt5_parsed["confidence"],
        "gpt5_rationale": gpt5_parsed["rationale"],
        "gpt5_parse_status": gpt5_parsed["parse_status"],
    }

    rows_out.append(out_row)

    note = " [reused]" if reused else ""
    print(f"Processed {idx + 1}/{total}{note}")

    if AUTOSAVE_EVERY and len(rows_out) % AUTOSAVE_EVERY == 0:
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
        print(f"   💾 Autosaved to {OUTPUT_PATH}")

    time.sleep(0.1)

# ------------------------------------------------------
# 11. Save final
# ------------------------------------------------------
pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Done! Saved to {OUTPUT_PATH}\n")