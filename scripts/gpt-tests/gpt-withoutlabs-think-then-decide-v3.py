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
OUTPUT_PATH = "gpt5_triage_without_labs_think-then-decide-v3.csv"

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

PROMPT_TEMPLATE = """Evaluate the following patient-triage scenario and determine the most appropriate urgency level.

Work through these four steps before answering:

1. Extract the key clinical signal: chief complaint, time course, severity, and any red-flag features
   (e.g. rest pain, neurological deficits, haemodynamic instability, signs of end-organ involvement).
2. Determine the most appropriate urgency level based on those findings — independently of the answer options.
3. Map your urgency level to the option that best matches, using these clinical anchors:
   - A = Monitor at home (nonurgent) — stable, low-risk, no red flags (e.g. mild viral illness, minor sprain)
   - B = See a doctor within weeks (semiurgent) — non-urgent but needs evaluation (e.g. new rash, stable chronic symptom change)
   - C = See a doctor within 24–48 hours (urgent outpatient) — time-sensitive but not immediately life-threatening (e.g. suspected infection with fever, new exertional symptoms that resolve)
   - D = Go to the emergency department now — potentially life-threatening or rapidly evolving (e.g. stroke symptoms, severe chest pain at rest, acute breathing difficulty)
4. Challenge your answer: is there a red-flag feature you may have underweighted that would push urgency higher?

After working through these steps, give one short structured reply with four parts:

- choice – Pick A, B, C, or D. Write the reasoning that led to this choice in the rationale field. Only write null if you genuinely cannot distinguish between urgency levels from the information given.
- abstain_code – If you wrote null for choice, say why:
  - NO_VALID_OPTION – None of the options match the appropriate urgency level.
  - INSUFFICIENT_INFO – The vignette lacks enough information to determine urgency.
  - NO_ACTION_NEEDED – No triage action is required.
  Otherwise write null.
- confidence – A number between 0 and 1 (e.g. 0.8 = 80% sure).
- rationale – Write your key clinical reasoning here: the finding that drove your urgency decision and why it maps to your chosen option. 1–3 sentences.

Important rules:
- You cannot both pick an answer and abstain.
- Either choose A–D or provide an abstain_code.
- Base your answer only on the information provided.
- If you can rank the options by fit, always choose the best one. Only abstain if confidence is below 0.4 or you truly cannot distinguish urgency levels.

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
# 6. Call GPT-5
# ------------------------------------------------------
def call_gpt5(user_prompt: str) -> str:
    data = {
        "model": "chat",
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
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
# 8. Load input CSV (without-labs sheet)
# ------------------------------------------------------
CSV_PATH = Path("data/withoutlabs_triageqa-questions.csv").resolve()

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
print(f"\n🚀 Starting evaluation on {total} without-lab questions...\n")

for idx, row in df.iterrows():
    question_id = row["question_id"]
    prev_row = prev_map.get(question_id) if prev_map else None

    if prev_row and is_model_result_ok(prev_row, "gpt5"):
        gpt5_raw = prev_row["gpt5_raw"]
        gpt5_parsed = {
            "explanation": prev_row.get("gpt5_explanation"),
            "triage": prev_row.get("gpt5_triage"),
            "confidence": prev_row.get("gpt5_confidence"),
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
        "prompt_used": "think-then-decide-v3",
        "user_prompt": user_prompt,
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
