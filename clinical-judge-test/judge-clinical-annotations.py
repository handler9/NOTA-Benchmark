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

# Input / output
INPUT_CSV = "medqa_nato_results_gpt copy.csv"   # change if needed
OUTPUT_PATH = "medqa_nato_judge_prompt5.csv"

# Column to judge
PROMPT_COL = "noto_prompt"

# Clinical annotation column (yes/no)
ANNOT_COL = "Clinical Annotations"

# Metrics output CSV
METRICS_OUTPUT_PATH = "medqa_nato_judge_prompt5_metrics.csv"

# APIM max tokens (as requested)
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
# 3. Judge system prompt
# ------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are an expert physician and exam-key editor.

Your task is to decide whether the NOTO option (e.g., "none of the other answers" / "none of the above")
is the correct exam-key answer for the item.

Definition:
- TRUE-NOTA / TRUE-NOTO means the NOTO option is the correct key (because none of the provided substantive options is an acceptable exam key answer).

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
}
"""

# ------------------------------------------------------
# 4. Robust judge JSON parser (brace-scanning)
# ------------------------------------------------------
def safe_parse_judge_json(text: str):
    """
    Extracts the first valid JSON object containing:
      - is_true_nota (bool)
      - rationale (str)
    Handles extra prose, code fences, trailing text, etc.
    """
    default = {
        "is_true_nota": None,
        "rationale": "Model did not respond in the requested JSON format.",
        "parse_ok": False
    }

    if not isinstance(text, str) or not text.strip():
        return default

    cleaned = text.strip()

    # Strip <think> blocks and markdown code fences
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    def normalize_obj(obj: dict):
        is_true = obj.get("is_true_nota", None)
        rat = obj.get("rationale", "")

        # Normalize boolean-ish strings if needed
        if isinstance(is_true, str):
            if is_true.strip().lower() == "true":
                is_true = True
            elif is_true.strip().lower() == "false":
                is_true = False
            else:
                is_true = None

        if not isinstance(is_true, bool):
            is_true = None

        rat = rat if isinstance(rat, str) else str(rat)
        rat = rat.strip() if rat.strip() else "No rationale provided."

        return {"is_true_nota": is_true, "rationale": rat, "parse_ok": True}

    def try_load(s: str):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "is_true_nota" in obj and "rationale" in obj:
                return normalize_obj(obj)
        except Exception:
            return None
        return None

    # 1) Try full string
    obj = try_load(cleaned)
    if obj:
        return obj

    # 2) Brace-scan for the first complete {...} JSON candidate
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
                    if obj:
                        return obj
                    break  # move to next start_idx

    # 3) Regex last resort
    flat = " ".join(cleaned.split())
    m_bool = re.search(r'["\']is_true_nota["\']\s*:\s*(true|false)', flat, re.IGNORECASE)
    m_rat = re.search(r'["\']rationale["\']\s*:\s*["\'](.*?)["\']', cleaned, re.DOTALL | re.IGNORECASE)

    is_true = None
    if m_bool:
        is_true = (m_bool.group(1).lower() == "true")
    rat = m_rat.group(1).strip() if m_rat else default["rationale"]

    if is_true is None and rat == default["rationale"]:
        return default

    return {"is_true_nota": is_true, "rationale": rat, "parse_ok": True}

# ------------------------------------------------------
# 5. POST helper w/ retries
# ------------------------------------------------------
def post_with_retries(url, headers, json_data, timeout=120, max_retries=3):
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
#    IMPORTANT: Do NOT send temperature=0 (unsupported on this deployment).
# ------------------------------------------------------
def call_gpt5_judge(item_prompt: str) -> str:
    # JSON-sticky user wrapper
    user_msg = f"""
Classify whether the following exam item is TRUE-NOTA.

IMPORTANT OUTPUT RULES:
- Output MUST be valid JSON.
- Output MUST contain ONLY the JSON object (no prose, no markdown, no code fences).
- Use lowercase true/false.
- Rationale must be 1–3 sentences.

Return exactly this JSON shape:
{{
  "is_true_nota": true,
  "rationale": "..."
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
        # Do not include temperature
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
def is_prev_judge_ok(prev_row: dict) -> bool:
    raw = prev_row.get("judge_raw")
    err = prev_row.get("judge_error")
    parse_ok = prev_row.get("judge_parse_ok")

    if isinstance(err, str) and err.strip():
        return False
    if not isinstance(raw, str) or not raw.strip():
        return False
    if raw.startswith("ERROR:"):
        return False
    if parse_ok in [False, "False", 0, "0"]:
        return False

    prev_val = prev_row.get("judge_is_true_nota")
    if pd.isna(prev_val):
        return False

    return True

# ------------------------------------------------------
# 7b. Metrics helpers
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
csv_path = Path(INPUT_CSV).resolve()
print(f"📂 Input CSV: {csv_path}")
if not csv_path.exists():
    print(f"❌ Input CSV not found: {csv_path}")
    raise SystemExit

df = pd.read_csv(csv_path)
if PROMPT_COL not in df.columns:
    raise SystemExit(f"❌ Missing required column: {PROMPT_COL}")

# Prefer an ID column if present
id_col = None
for candidate in ["question_id", "id", "qid", "QID", "row_index"]:
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

    if id_col and id_col in prev_df.columns:
        prev_key_col = id_col
    elif "row_index" in prev_df.columns:
        prev_key_col = "row_index"

    if prev_key_col:
        prev_map = {row[prev_key_col]: row.to_dict() for _, row in prev_df.iterrows()}
        print(f"Loaded {len(prev_map)} previous rows.")

# ------------------------------------------------------
# 10. Run judge (Option A: only noto_prompt)
# ------------------------------------------------------
rows_out = []
total = len(df)
print(f"\n🚀 Starting TRUE-NOTA judge on {total} rows...\n")

for idx, row in df.iterrows():
    key = row[id_col] if id_col else (idx + 1)

    prev_row = prev_map.get(key) if prev_map else None
    reused = False

    if prev_row and is_prev_judge_ok(prev_row):
        judge_raw = prev_row.get("judge_raw", "")
        judge_is_true = prev_row.get("judge_is_true_nota")
        judge_rationale = prev_row.get("judge_rationale", "")
        judge_error = prev_row.get("judge_error", "")
        judge_parse_ok = prev_row.get("judge_parse_ok", True)
        reused = True
    else:
        item_prompt = str(row[PROMPT_COL])
        judge_raw = call_gpt5_judge(item_prompt)

        if isinstance(judge_raw, str) and judge_raw.startswith("ERROR:"):
            judge_is_true = None
            judge_rationale = ""
            judge_error = judge_raw
            judge_parse_ok = False
        else:
            parsed = safe_parse_judge_json(judge_raw)
            judge_is_true = parsed["is_true_nota"]
            judge_rationale = parsed["rationale"]
            judge_parse_ok = parsed["parse_ok"]
            judge_error = "" if judge_parse_ok else "PARSE_FAILED"

    out_row = row.to_dict()
    out_row["row_index"] = idx + 1
    out_row["judge_raw"] = judge_raw
    out_row["judge_is_true_nota"] = judge_is_true
    out_row["judge_rationale"] = judge_rationale
    out_row["judge_parse_ok"] = judge_parse_ok
    out_row["judge_error"] = judge_error

    rows_out.append(out_row)

    note = " [reused]" if reused else ""
    print(f"Processed {idx+1}/{total}{note}")

    if AUTOSAVE_EVERY and (len(rows_out) % AUTOSAVE_EVERY == 0):
        pd.DataFrame(rows_out).to_csv(OUTPUT_PATH, index=False)
        print(f"   💾 Autosaved to {OUTPUT_PATH}")

    time.sleep(0.1)

# ------------------------------------------------------
# 11. Save final
# ------------------------------------------------------
out_df = pd.DataFrame(rows_out)
out_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Done! Saved to {OUTPUT_PATH}\n")

# ------------------------------------------------------
# 12. Compute metrics vs clinical annotations (yes/no) and save to CSV
# ------------------------------------------------------
if ANNOT_COL not in out_df.columns:
    print(f"⚠️ Metrics skipped: annotation column not found: {ANNOT_COL}")
else:
    # Normalize labels
    gold = out_df[ANNOT_COL].apply(to_bool)
    pred = out_df["judge_is_true_nota"].apply(to_bool)

    # Keep only rows with both labels
    keep = gold.notna() & pred.notna()
    eval_df = out_df.loc[keep].copy()
    gold = gold.loc[keep]
    pred = pred.loc[keep]

    if len(eval_df) == 0:
        print("⚠️ Metrics skipped: no rows with both gold and pred labels.")
    else:
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
            "input_csv": INPUT_CSV,
            "output_csv": OUTPUT_PATH,
            "annotation_col": ANNOT_COL,
            "n_evaluated": int(len(eval_df)),
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
        }])

        metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)
        print("=== TRUE-NOTA Evaluation (positive class = TRUE-NOTA) ===")
        print(f"Evaluated rows: {len(eval_df)}")
        print(f"TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
        print(f"Accuracy:    {accuracy:.3f}")
        print(f"Precision:   {precision:.3f}")
        print(f"Recall:      {recall:.3f}")
        print(f"F1:          {f1:.3f}")
        print(f"Specificity: {specificity:.3f}")
        print(f"NPV:         {npv:.3f}")
        print(f"\n📄 Metrics CSV saved to: {METRICS_OUTPUT_PATH}\n")
