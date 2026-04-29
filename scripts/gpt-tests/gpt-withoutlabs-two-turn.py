import os
import re
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
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = str(REPO_ROOT / "results_raw_withoutlabs_triage" / "gpt5_triage_without_labs_two-turn.csv")

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
# 3. Prompts
# ------------------------------------------------------
SYSTEM_MESSAGE = (
    "You are an experienced emergency medicine physician with 20 years of clinical "
    "triage experience. You are precise, systematic, and never guess when you can "
    "reason from evidence."
)

TURN_1_TEMPLATE = """A patient presents without lab results available. Before I show you any answer options, analyze this case carefully.

Work through the following:
1. What are the key clinical findings and time course?
2. Are there any red-flag features (e.g. rest pain, neurological deficits, haemodynamic instability, signs of end-organ involvement)?
3. What is the most appropriate urgency level for this patient — and why?

Be specific. Commit to an urgency level before seeing any options.

Clinical vignette:

{stem}
"""

TURN_2_TEMPLATE = """Based on your analysis, select the option that best matches the urgency level you identified.

The four options follow this scale:
- A = Monitor at home (nonurgent) — stable, low-risk, no red flags
- B = See a doctor within weeks (semiurgent) — non-urgent but needs evaluation
- C = See a doctor within 24–48 hours (urgent outpatient) — time-sensitive but not immediately life-threatening
- D = Go to the emergency department now — potentially life-threatening or rapidly evolving

Options for this case:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}

Choose the option that best matches your urgency assessment. Only write null if you genuinely cannot determine urgency from the information given.

Respond ONLY in this JSON format and nothing else:

{{
  "choice": "A/B/C/D or null",
  "abstain_code": "NO_VALID_OPTION/INSUFFICIENT_INFO/NO_ACTION_NEEDED or null",
  "confidence": 0.0,
  "rationale": "one to three sentences"
}}
"""

# ------------------------------------------------------
# 4. Parse GPT-5 output
# ------------------------------------------------------
def safe_parse_triage_output(text: str):
    import json

    default = {
        "explanation": None,
        "triage": None,
        "confidence": None,
        "parse_status": "FAILED_PARSE",
    }

    if not isinstance(text, str) or not text.strip():
        return default

    cleaned = text.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "").strip()

    # Extract JSON object even if there is surrounding text
    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not json_match:
        return default
    cleaned = json_match.group(0)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return default

    choice = data.get("choice")
    triage = choice if isinstance(choice, str) and choice.upper() in {"A", "B", "C", "D"} else None
    confidence = data.get("confidence")
    explanation = data.get("rationale")

    if triage is None and confidence is None and explanation is None:
        return default

    return {
        "explanation": explanation,
        "triage": triage,
        "confidence": confidence,
        "parse_status": "OK",
    }

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
# 6. Two-turn call
# ------------------------------------------------------
def call_gpt5_two_turn(stem: str, option_A: str, option_B: str, option_C: str, option_D: str):
    turn1_user = TURN_1_TEMPLATE.format(stem=stem)
    turn2_user = TURN_2_TEMPLATE.format(
        option_A=option_A,
        option_B=option_B,
        option_C=option_C,
        option_D=option_D,
    )

    # --- Turn 1: reasoning without options ---
    data_turn1 = {
        "model": "chat",
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": turn1_user},
        ],
        "max_completion_tokens": 1000,
    }

    try:
        resp1 = post_with_retries(GPT5_URL, headers=HEADERS, json_data=data_turn1)
    except Exception as e:
        return f"ERROR: {e}", None

    if resp1.status_code != 200:
        return f"ERROR: {resp1.status_code} {resp1.text}", None

    try:
        turn1_reply = resp1.json()["choices"][0]["message"]["content"]
    except Exception:
        return f"ERROR: Unexpected response format: {resp1.text}", None

    # --- Turn 2: map reasoning to options ---
    data_turn2 = {
        "model": "chat",
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": turn1_user},
            {"role": "assistant", "content": turn1_reply},
            {"role": "user", "content": turn2_user},
        ],
        "max_completion_tokens": 500,
    }

    try:
        resp2 = post_with_retries(GPT5_URL, headers=HEADERS, json_data=data_turn2)
    except Exception as e:
        return f"ERROR: {e}", turn1_reply

    if resp2.status_code != 200:
        return f"ERROR: {resp2.status_code} {resp2.text}", turn1_reply

    try:
        turn2_reply = resp2.json()["choices"][0]["message"]["content"]
    except Exception:
        return f"ERROR: Unexpected response format: {resp2.text}", turn1_reply

    return turn2_reply, turn1_reply

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
# 8. Load input CSV (without-labs sheet)
# ------------------------------------------------------
CSV_PATH = REPO_ROOT / "data" / "withoutlabs_triageqa-questions.csv"

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
print(f"\n🚀 Starting two-turn evaluation on {total} without-lab questions...\n")

for idx, row in df.iterrows():
    question_id = row["question_id"]
    prev_row = prev_map.get(question_id) if prev_map else None

    if prev_row and is_model_result_ok(prev_row, "gpt5"):
        gpt5_raw = prev_row["gpt5_raw"]
        gpt5_reasoning = prev_row.get("gpt5_reasoning", "")
        gpt5_parsed = {
            "explanation": prev_row.get("gpt5_explanation"),
            "triage": prev_row.get("gpt5_triage"),
            "confidence": prev_row.get("gpt5_confidence"),
            "parse_status": prev_row.get("gpt5_parse_status"),
        }
        reused = True
    else:
        reused = False

        gpt5_raw, gpt5_reasoning = call_gpt5_two_turn(
            stem=row["stem"],
            option_A=row["option_A"],
            option_B=row["option_B"],
            option_C=row["option_C"],
            option_D=row["option_D"],
        )

        if isinstance(gpt5_raw, str) and gpt5_raw.startswith("ERROR:"):
            gpt5_parsed = {
                "explanation": None,
                "triage": None,
                "confidence": None,
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
        "prompt_used": "two-turn",
        "gpt5_reasoning": gpt5_reasoning,
        "gpt5_raw": gpt5_raw,
        "gpt5_explanation": gpt5_parsed["explanation"],
        "gpt5_triage": gpt5_parsed["triage"],
        "gpt5_confidence": gpt5_parsed["confidence"],
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
