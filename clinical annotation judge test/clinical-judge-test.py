#!/usr/bin/env python3
"""
GPT-5 TRUE-NOTO judge (new system prompt + strict JSON schema) + metrics on ALL rows.

What this script does:
- Reads:  clinical annotation judge test/Clinically-annotated-100qs.csv
- Judges: each row's PROMPT_COL via Stanford APIM GPT-5 using your NEW system prompt
- Writes: clinical annotation judge test/Clinically-annotated-100qs_gpt5_true_nota_judged.csv
- Writes: clinical annotation judge test/Clinically-annotated-100qs_gpt5_true_nota_metrics.csv

Key behaviors:
- By default, it does NOT reuse prior outputs (new prompt => clean run).
- It retries on parse failures.
- If, after retries, it STILL can’t parse valid JSON, it FORCE-FILLS a conservative default
  prediction so metrics can still evaluate ALL rows, while clearly flagging the row with
  judge_error="FORCED_DEFAULT_AFTER_PARSE_FAIL".

Metrics guarantee:
- This script will compute metrics on *all* rows ONLY if the gold labels in ANNOT_COL are
  parseable for all rows (yes/no/true/false/1/0). If not, it will print the bad rows and exit.
- Predictions are guaranteed for every row (either parsed or forced-default), so pred is never missing.
"""

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
AUTOSAVE_EVERY = 25
SLEEP_BETWEEN_CALLS = 0.15

# Retry policy for model formatting issues
RETRY_ON_PARSE_FAIL = 2          # additional attempts after first try (total tries = 1 + this)
REQUEST_TIMEOUT = 120
REQUEST_MAX_RETRIES = 3

# Force-default policy (ensures predictions exist for EVERY row)
FORCE_DEFAULT_IF_UNPARSEABLE = True
FORCED_DEFAULT_IS_NOTO_CORRECT = False  # conservative default; flagged in judge_error

# ------------------------------------------------------
# Paths (same inputs/outputs you asked to keep)
# ------------------------------------------------------
RUN_DIR = Path("clinical annotation judge test").expanduser().resolve()

INPUT_CSV = RUN_DIR / "Clinically-annotated-100qs.csv"
OUTPUT_PATH = RUN_DIR / "Clinically-annotated-100qs_gpt5_true_nota_judged-1.csv"
METRICS_OUTPUT_PATH = RUN_DIR / "Clinically-annotated-100qs_gpt5_true_nota_metrics-1.csv"

# Columns
PROMPT_COL = "noto_prompt"
ANNOT_COL = "Clinical Annotations"

# Tokens
MAX_COMPLETION_TOKENS = 5000

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
# 2. GPT-5 endpoint (Stanford APIM)
# ------------------------------------------------------
GPT5_URL = (
    "https://apim.stanfordhealthcare.org/openai-eastus2/"
    "deployments/gpt-5/chat/completions?api-version=2024-12-01-preview"
)

# ------------------------------------------------------
# 3. NEW Judge system prompt (exactly as you provided)
# ------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are an expert physician and exam-key editor.

Your task is to decide whether the NOTO option (e.g., "none of the other answers" / "none of the above")
is the correct exam-key choice for this multiple-choice item.

You MUST follow this process:

Step 1 — Generate the best answer (independent of the options)
- Determine the single best exam-key answer you would write in free-response form.

Step 2 — Evaluate each NON-NOTO option against that best answer
- For each answer choice except the NOTO option, assign a match score from 0.0 to 1.0:
  1.0 = fully correct and specific enough to be the official exam key
  0.8–0.99 = essentially equivalent / acceptable synonym
  0.5–0.79 = partially correct, vague, incomplete, or missing key qualifier (NOT acceptable as exam key)
  <0.5 = incorrect or irrelevant

Step 3 — Decide whether NOTO is correct
- Identify which option is the NOTO option in the list.
- Let BEST_MATCH be the highest match score among the NON-NOTO options.
- If BEST_MATCH >= 0.80, then NOTO is NOT correct.
- If BEST_MATCH < 0.80, then NOTO IS correct (because none of the provided substantive options is an acceptable exam key answer).

Important rules:
- Do NOT assume one option must be correct.
- "Least wrong" or partially correct does NOT defeat NOTO.
- Penalize generic or incomplete answers: the exam key should be the most correct and specific answer.

Output requirements (STRICT):
- Return ONLY valid JSON. No markdown. No code fences. No extra text.
- Use lowercase true/false.
- Rationale MUST be 1–3 sentences.

JSON format:
{
  "is_noto_correct": true/false,
  "noto_letter": "A"/"B"/"C"/"D"/"E",
  "best_non_noto_letter": "A"/"B"/"C"/"D"/"E" or null,
  "best_non_noto_match": 0.0-1.0,
  "rationale": "1–3 sentences"
}"""

# ------------------------------------------------------
# 4. Robust judge JSON parser (brace-scanning)
# ------------------------------------------------------
VALID_LETTERS = {"A", "B", "C", "D", "E"}

def safe_parse_judge_json(text: str):
    """
    Extract the first valid JSON object with required keys:
      - is_noto_correct (bool)
      - noto_letter (A-E)
      - best_non_noto_letter (A-E or null)
      - best_non_noto_match (float 0..1)
      - rationale (str)
    Returns dict with parse_ok and normalized fields.
    """
    default = {
        "is_noto_correct": None,
        "noto_letter": None,
        "best_non_noto_letter": None,
        "best_non_noto_match": None,
        "rationale": "Model did not respond in the requested JSON format.",
        "parse_ok": False,
    }

    if not isinstance(text, str) or not text.strip():
        return default

    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    def normalize_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "true":
                return True
            if s == "false":
                return False
        return None

    def normalize_letter(v, allow_null=False):
        if v is None and allow_null:
            return None
        if isinstance(v, str):
            s = v.strip().upper()
            if s in VALID_LETTERS:
                return s
        return None

    def normalize_match(v):
        try:
            f = float(v)
        except Exception:
            return None
        # clamp slightly out-of-range due to formatting noise
        if f < 0.0:
            f = 0.0
        if f > 1.0:
            f = 1.0
        return f

    def normalize_obj(obj: dict):
        is_noto = normalize_bool(obj.get("is_noto_correct", None))
        noto_letter = normalize_letter(obj.get("noto_letter", None), allow_null=False)
        best_letter = normalize_letter(obj.get("best_non_noto_letter", None), allow_null=True)
        best_match = normalize_match(obj.get("best_non_noto_match", None))
        rationale = obj.get("rationale", "")

        if not isinstance(rationale, str):
            rationale = str(rationale)
        rationale = rationale.strip() if rationale.strip() else "No rationale provided."

        parse_ok = (
            isinstance(is_noto, bool)
            and (noto_letter in VALID_LETTERS)
            and (best_letter in VALID_LETTERS or best_letter is None)
            and (isinstance(best_match, float) or isinstance(best_match, int))
            and (0.0 <= float(best_match) <= 1.0)
            and isinstance(rationale, str)
        )

        # best_match normalized above
        best_match = float(best_match) if best_match is not None else None

        return {
            "is_noto_correct": is_noto,
            "noto_letter": noto_letter,
            "best_non_noto_letter": best_letter,
            "best_non_noto_match": best_match,
            "rationale": rationale,
            "parse_ok": parse_ok,
        }

    def try_load(s: str):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                # Must at least have the keys; normalize_obj will validate
                needed = {"is_noto_correct", "noto_letter", "best_non_noto_letter", "best_non_noto_match", "rationale"}
                if needed.issubset(set(obj.keys())):
                    return normalize_obj(obj)
        except Exception:
            return None
        return None

    # 1) Try full string
    obj = try_load(cleaned)
    if obj and obj["parse_ok"]:
        return obj

    # 2) Brace-scan for first complete {...}
    start_positions = [m.start() for m in re.finditer(r"\{", cleaned)]
    for start_idx in start_positions:
        depth = 0
        for i in range(start_idx, len(cleaned)):
            ch = cleaned[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start_idx:i+1]
                    obj = try_load(candidate)
                    if obj and obj["parse_ok"]:
                        return obj
                    break

    return default

# ------------------------------------------------------
# 5. POST helper w/ retries
# ------------------------------------------------------
def post_with_retries(url, headers, json_data, timeout=REQUEST_TIMEOUT, max_retries=REQUEST_MAX_RETRIES):
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
# 6. Call GPT-5 judge
# ------------------------------------------------------
def call_gpt5_judge(item_prompt: str) -> str:
    user_msg = f"""
You MUST output ONLY valid JSON (no prose, no markdown, no code fences).

Return exactly this JSON shape:
{{
  "is_noto_correct": true/false,
  "noto_letter": "A"/"B"/"C"/"D"/"E",
  "best_non_noto_letter": "A"/"B"/"C"/"D"/"E" or null,
  "best_non_noto_match": 0.0-1.0,
  "rationale": "1–3 sentences"
}}

ITEM:
{item_prompt}
""".strip()

    data = {
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        # Do not include temperature=0 on this deployment
    }

    try:
        resp = post_with_retries(GPT5_URL, headers=HEADERS, json_data=data)
    except Exception as e:
        return f"ERROR: {e}"

    if resp.status_code != 200:
        return f"ERROR: {resp.status_code} {resp.text}"

    return resp.json()["choices"][0]["message"]["content"]

# ------------------------------------------------------
# 7. Metrics helpers
# ------------------------------------------------------
def to_bool(x):
    """Normalize yes/no/true/false/1/0 into True/False. Returns None if unparseable."""
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None

def safe_div(a, b):
    return (a / b) if b else 0.0

# ------------------------------------------------------
# 8. Load input CSV
# ------------------------------------------------------
print(f"📂 Run folder: {RUN_DIR}")
print(f"📂 Input CSV: {INPUT_CSV.resolve()}")

if not INPUT_CSV.exists():
    print(f"❌ Input CSV not found: {INPUT_CSV}")
    raise SystemExit

df = pd.read_csv(INPUT_CSV)
if PROMPT_COL not in df.columns:
    raise SystemExit(f"❌ Missing required column: {PROMPT_COL}")

total = len(df)
print(f"✅ Loaded {total} rows.")

# ------------------------------------------------------
# 9. Run judge on ALL rows (new prompt => no reuse)
# ------------------------------------------------------
rows_out = []
print(f"\n🚀 Starting NOTO judge on {total} rows...\n")

for idx, row in df.iterrows():
    item_prompt = str(row[PROMPT_COL])

    judge_raw = ""
    parsed = None
    judge_error = ""
    judge_parse_ok = False

    tries = 1 + max(0, int(RETRY_ON_PARSE_FAIL))
    for attempt in range(1, tries + 1):
        judge_raw = call_gpt5_judge(item_prompt)

        if isinstance(judge_raw, str) and judge_raw.startswith("ERROR:"):
            parsed = {
                "is_noto_correct": None,
                "noto_letter": None,
                "best_non_noto_letter": None,
                "best_non_noto_match": None,
                "rationale": "",
                "parse_ok": False,
            }
            judge_error = judge_raw
            judge_parse_ok = False
        else:
            parsed = safe_parse_judge_json(judge_raw)
            judge_parse_ok = bool(parsed.get("parse_ok", False))
            judge_error = "" if judge_parse_ok else "PARSE_FAILED"

        if judge_parse_ok:
            break
        if attempt < tries:
            time.sleep(0.6 * attempt)  # backoff

    # If still unparseable, force-default (so metrics can evaluate all rows)
    if (not judge_parse_ok) and FORCE_DEFAULT_IF_UNPARSEABLE:
        parsed = {
            "is_noto_correct": bool(FORCED_DEFAULT_IS_NOTO_CORRECT),
            "noto_letter": None,
            "best_non_noto_letter": None,
            "best_non_noto_match": 0.0,
            "rationale": "Parse failed after retries; forced default used so metrics can include this row.",
            "parse_ok": False,
        }
        judge_error = "FORCED_DEFAULT_AFTER_PARSE_FAIL"
        judge_parse_ok = False

    out_row = row.to_dict()
    out_row["row_index"] = idx + 1

    out_row["judge_raw"] = judge_raw
    out_row["judge_is_noto_correct"] = parsed.get("is_noto_correct")
    out_row["judge_noto_letter"] = parsed.get("noto_letter")
    out_row["judge_best_non_noto_letter"] = parsed.get("best_non_noto_letter")
    out_row["judge_best_non_noto_match"] = parsed.get("best_non_noto_match")
    out_row["judge_rationale"] = parsed.get("rationale")

    out_row["judge_parse_ok"] = judge_parse_ok
    out_row["judge_error"] = judge_error

    rows_out.append(out_row)

    print(f"Processed {idx+1}/{total}  parse_ok={judge_parse_ok}  err='{judge_error}'")

    if AUTOSAVE_EVERY and (len(rows_out) % AUTOSAVE_EVERY == 0):
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
        print(f"   💾 Autosaved to {OUTPUT_PATH}")

    time.sleep(SLEEP_BETWEEN_CALLS)

# ------------------------------------------------------
# 10. Save final judged CSV
# ------------------------------------------------------
out_df = pd.DataFrame(rows_out)
out_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Done! Saved judged CSV to {OUTPUT_PATH}\n")

# ------------------------------------------------------
# 11. Metrics (guarantee ALL rows are evaluated)
# ------------------------------------------------------
if ANNOT_COL not in out_df.columns:
    print(f"❌ Metrics cannot run: annotation column not found: {ANNOT_COL}")
    raise SystemExit

gold = out_df[ANNOT_COL].apply(to_bool)
pred = out_df["judge_is_noto_correct"].apply(to_bool)

# Enforce gold labels parse for ALL rows (so metrics can be computed on all rows)
bad_gold = gold.isna()
if bad_gold.any():
    bad_rows = out_df.loc[bad_gold, ["row_index", ANNOT_COL]].copy()
    print("❌ Metrics aborted: some gold labels are not parseable as yes/no/true/false/1/0.")
    print("   Fix these rows in the input CSV and rerun:")
    print(bad_rows.to_string(index=False))
    raise SystemExit

# Enforce predictions exist for ALL rows (should be true due to forced-default)
bad_pred = pred.isna()
if bad_pred.any():
    bad_rows = out_df.loc[bad_pred, ["row_index", "judge_raw", "judge_error"]].copy()
    print("❌ Metrics aborted: some predictions are missing/unparseable (unexpected).")
    print(bad_rows.to_string(index=False))
    raise SystemExit

# Now metrics on ALL rows
tp = int(((pred == True)  & (gold == True)).sum())
tn = int(((pred == False) & (gold == False)).sum())
fp = int(((pred == True)  & (gold == False)).sum())
fn = int(((pred == False) & (gold == True)).sum())

precision = safe_div(tp, tp + fp)
recall    = safe_div(tp, tp + fn)
f1        = safe_div(2 * precision * recall, precision + recall)
accuracy  = safe_div(tp + tn, tp + tn + fp + fn)
specificity = safe_div(tn, tn + fp)
npv         = safe_div(tn, tn + fn)

metrics_df = pd.DataFrame([{
    "input_csv": str(INPUT_CSV),
    "output_csv": str(OUTPUT_PATH),
    "annotation_col": ANNOT_COL,
    "n_total": int(len(out_df)),
    "n_evaluated": int(len(out_df)),  # guaranteed
    "tp": tp,
    "fp": fp,
    "tn": tn,
    "fn": fn,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "specificity": specificity,
    "npv": npv,
    "forced_default_rows": int((out_df["judge_error"] == "FORCED_DEFAULT_AFTER_PARSE_FAIL").sum()),
}])

metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)

print("=== NOTO Evaluation (positive class = is_noto_correct=true) ===")
print(f"Total rows:         {len(out_df)}")
print(f"Evaluated rows:     {len(out_df)}")
print(f"Forced-default rows:{int(metrics_df.loc[0, 'forced_default_rows'])}")
print(f"TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
print(f"Accuracy:    {accuracy:.3f}")
print(f"Precision:   {precision:.3f}")
print(f"Recall:      {recall:.3f}")
print(f"F1:          {f1:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"NPV:         {npv:.3f}")
print(f"\n📄 Metrics CSV saved to: {METRICS_OUTPUT_PATH}\n")