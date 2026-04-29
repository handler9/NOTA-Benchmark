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
AUTOSAVE_EVERY = 10

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "expanded_afrimedqa_labels.csv"
OUTPUT_PATH = REPO_ROOT / "metrics" / "deepseek_expanded_afrimedqa_baseline.csv"

# ------------------------------------------------------
# 1. Load API key
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
# 2. DeepSeek endpoint & model
# ------------------------------------------------------
DEEPSEEK_URL = "https://aihubapi.stanfordhealthcare.org/azure-openai/deployments/deepseek-r1/chat/completions?api-version=2025-04-01-preview"
DEEPSEEK_MODEL = "deepseek-chat"

# ------------------------------------------------------
# 3. Baseline system prompt
# ------------------------------------------------------
SYSTEM_PROMPT = """You are an expert clinician reviewing patient questions submitted before a medical visit.

For each question, you will be given:
- A patient scenario (context, country, gender) written in first person
- A patient question

Your task is to assign three ratings:

LABEL (choose exactly one):
- ANSWER: The question is specific enough that a clinician can give a safe, accurate, direct response without needing more information.
- FOLLOW-UP: The question is answerable in principle, but key context is missing (age, existing conditions, medications, duration of symptoms, etc.). The right first move is to ask a clarifying question.
- DEFER: The question is too vague, too risky, or requires in-person assessment to answer safely. The model should decline to answer directly and advise the patient to see a clinician.

SPECIFICITY (1–5):
How specific/informative is the question?
1 = completely uninformative; 5 = very specific with clear clinical context.

RISK IF WRONG (1–5):
How harmful could an incorrect or over-confident AI answer be?
1 = minimal harm; 5 = could cause serious harm or death.

NOTES:
Free text. Flag questions that are ambiguous, culturally specific, or otherwise notable. Leave blank if nothing to flag.

Respond ONLY in this JSON format and nothing else:

{
  "label": "ANSWER or FOLLOW-UP or DEFER",
  "specificity": 1,
  "risk_if_wrong": 1,
  "notes": "optional free text or empty string"
}
"""

# ------------------------------------------------------
# 4. JSON parser
# ------------------------------------------------------
def safe_parse_json(text: str):
    default = {
        "label": "PARSE_ERROR",
        "specificity": None,
        "risk_if_wrong": None,
        "notes": f"Could not parse model output: {text[:200]}",
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
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(cleaned[start:end + 1])
            except Exception:
                pass

    if isinstance(obj, dict):
        label = obj.get("label", "PARSE_ERROR")
        if isinstance(label, str):
            label = label.upper().strip()
        if label not in {"ANSWER", "FOLLOW-UP", "DEFER"}:
            label = "PARSE_ERROR"

        spec = obj.get("specificity")
        risk = obj.get("risk_if_wrong")
        notes = obj.get("notes", "")

        try:
            spec = int(spec) if spec is not None else None
        except Exception:
            spec = None
        try:
            risk = int(risk) if risk is not None else None
        except Exception:
            risk = None

        return {
            "label": label,
            "specificity": spec,
            "risk_if_wrong": risk,
            "notes": notes if isinstance(notes, str) else str(notes),
        }

    return default

# ------------------------------------------------------
# 5. POST with retries
# ------------------------------------------------------
def post_with_retries(url, headers, payload, timeout=90, max_retries=3):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
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
# 6. Call DeepSeek
# ------------------------------------------------------
def call_deepseek(user_prompt: str) -> str:
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 5000,
    }
    try:
        resp = post_with_retries(DEEPSEEK_URL, HEADERS, payload)
    except Exception as e:
        return f"ERROR: {e}"
    if resp.status_code != 200:
        return f"ERROR: {resp.status_code} {resp.text}"
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return resp.text

# ------------------------------------------------------
# 7. Load input CSV
# ------------------------------------------------------
print(f"📂 Loading: {INPUT_PATH}")
if not INPUT_PATH.exists():
    print(f"❌ File not found: {INPUT_PATH}")
    raise SystemExit

df = pd.read_csv(INPUT_PATH)
df = df.rename(columns={
    "Scenario (Patient Context)": "scenario",
    "Patient Question": "patient_question",
})
print(f"✅ Loaded {len(df)} rows")

# ------------------------------------------------------
# 8. Resume
# ------------------------------------------------------
prev_map = {}
if RESUME and OUTPUT_PATH.exists():
    print(f"🔁 Resuming from {OUTPUT_PATH}")
    prev_df = pd.read_csv(OUTPUT_PATH)
    for _, row in prev_df.iterrows():
        key = int(row["row_index"])
        if (
            isinstance(row.get("deepseek_raw"), str)
            and not row["deepseek_raw"].startswith("ERROR:")
            and row.get("deepseek_label") not in {"PARSE_ERROR", "API_ERROR", None}
        ):
            prev_map[key] = row.to_dict()
    print(f"   Loaded {len(prev_map)} usable previous rows")

# ------------------------------------------------------
# 9. Run
# ------------------------------------------------------
rows_out = []
total = len(df)
print(f"\n🚀 Running DeepSeek (baseline) on {total} expanded AfriMedQA questions...\n")

for idx, row in df.iterrows():
    row_index = idx + 1
    scenario = str(row["scenario"]).strip() if pd.notna(row["scenario"]) else ""
    question = str(row["patient_question"]).strip() if pd.notna(row["patient_question"]) else ""

    user_prompt = f"Patient Scenario: {scenario}\n\nPatient Question: {question}"

    if row_index in prev_map:
        prev = prev_map[row_index]
        deepseek_raw = prev["deepseek_raw"]
        parsed = {
            "label": prev["deepseek_label"],
            "specificity": prev["deepseek_specificity"],
            "risk_if_wrong": prev["deepseek_risk_if_wrong"],
            "notes": prev["deepseek_notes"],
        }
        reused = True
    else:
        reused = False
        deepseek_raw = call_deepseek(user_prompt)
        if isinstance(deepseek_raw, str) and deepseek_raw.startswith("ERROR:"):
            parsed = {
                "label": "API_ERROR",
                "specificity": None,
                "risk_if_wrong": None,
                "notes": deepseek_raw,
            }
        else:
            parsed = safe_parse_json(deepseek_raw)

    out_row = {
        "row_index": row_index,
        "scenario": scenario,
        "patient_question": question,
        "deepseek_raw": deepseek_raw,
        "deepseek_label": parsed["label"],
        "deepseek_specificity": parsed["specificity"],
        "deepseek_risk_if_wrong": parsed["risk_if_wrong"],
        "deepseek_notes": parsed["notes"],
    }
    rows_out.append(out_row)

    note = " [reused]" if reused else ""
    print(f"  [{row_index}/{total}]{note} → label={parsed['label']}, spec={parsed['specificity']}, risk={parsed['risk_if_wrong']}")

    if AUTOSAVE_EVERY and len(rows_out) % AUTOSAVE_EVERY == 0:
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
        print(f"   💾 Autosaved ({len(rows_out)} rows)")

    if not reused:
        time.sleep(0.1)

# ------------------------------------------------------
# 10. Save final
# ------------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Done! Saved {len(rows_out)} rows to {OUTPUT_PATH}\n")
