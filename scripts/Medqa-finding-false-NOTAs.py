#!/usr/bin/env python3
import os
import re
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import requests
from requests.exceptions import RequestException

# ------------------------------------------------------
# 0. Stanford APIM config (GPT-5)
# ------------------------------------------------------
GPT5_URL = (
    "https://apim.stanfordhealthcare.org/openai-eastus2/"
    "deployments/gpt-5/chat/completions?api-version=2024-12-01-preview"
)

MAX_COMPLETION_TOKENS = 5000  # judge responses are short; keep modest

# ------------------------------------------------------
# 1. Judge system prompt (your spec)
# ------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """You are an expert physician and exam-key editor.

Your task is to decide whether the NOTA option (e.g., "none of the other answers" / "none of the above")
is the correct exam-key choice for this multiple-choice item.

You MUST follow this process:

Step 1 — Generate the best answer (independent of the options)
- Determine the single best exam-key answer you would write in free-response form.

Step 2 — Evaluate each NON-NOTA option against that best answer
- For each answer choice except the NOTO option, assign a match score from 0.0 to 1.0:
  1.0 = fully correct and specific enough to be the official exam key
  0.8–0.99 = essentially equivalent / acceptable synonym
  0.5–0.79 = partially correct, vague, incomplete, or missing key qualifier (NOT acceptable as exam key)
  <0.5 = incorrect or irrelevant

Step 3 — Decide whether NOTA is correct
- Identify which option is the NOTA option in the list.
- Let BEST_MATCH be the highest match score among the NON-NOTA options.
- If BEST_MATCH >= 0.80, then NOTA is NOT correct.
- If BEST_MATCH < 0.80, then NOTA IS correct (because none of the provided substantive options is an acceptable exam key answer).

Important rules:
- Do NOT assume one option must be correct.
- "Least wrong" or partially correct does NOT defeat NOTO.
- Penalize generic or incomplete answers: the exam key should be the most correct and specific answer.

Output requirements (STRICT):
- Return ONLY valid JSON. No markdown. No code fences. No extra text.
- Use lowercase true/false.
- Rationale MUST be 1–3 sentences.

"""

# Required JSON shape (we enforce on parse)
REQUIRED_KEYS = {"label", "plausible_options", "reason"}


# ------------------------------------------------------
# 2. Robust JSON parser (brace scan)
# ------------------------------------------------------
def safe_parse_json_obj(text: str):
    default = {
        "label": None,
        "plausible_options": [],
        "reason": "Model did not respond in the requested JSON format.",
        "parse_ok": False,
    }

    if not isinstance(text, str) or not text.strip():
        return default

    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    def normalize(obj: dict):
        label = obj.get("label", None)
        plausible = obj.get("plausible_options", [])
        reason = obj.get("reason", "")

        if isinstance(label, str):
            label = label.strip().upper()
        if label not in {"TRUE_NOTA", "FALSE_NOTA"}:
            label = None

        if plausible is None:
            plausible = []
        if isinstance(plausible, str):
            # allow "A,C" or "A"
            plausible = [p.strip().upper() for p in plausible.split(",") if p.strip()]
        if not isinstance(plausible, list):
            plausible = []
        plausible = [str(x).strip().upper() for x in plausible if str(x).strip()]
        plausible = [x for x in plausible if x in {"A", "B", "C", "D", "E"}]

        if not isinstance(reason, str):
            reason = str(reason)
        reason = reason.strip() if reason.strip() else "No reason provided."

        ok = (label in {"TRUE_NOTA", "FALSE_NOTA"})
        return {"label": label, "plausible_options": plausible, "reason": reason, "parse_ok": ok}

    def try_load(s: str):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and REQUIRED_KEYS.issubset(set(obj.keys())):
                return normalize(obj)
        except Exception:
            return None
        return None

    # 1) Try full
    obj = try_load(cleaned)
    if obj:
        return obj

    # 2) Brace-scan for first full JSON object
    starts = [m.start() for m in re.finditer(r"\{", cleaned)]
    for start in starts:
        depth = 0
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = cleaned[start:i+1]
                    obj = try_load(cand)
                    if obj:
                        return obj
                    break

    return default


# ------------------------------------------------------
# 3. POST helper w/ retries
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
# 4. Call GPT-5 judge (Stanford APIM)
#    IMPORTANT: Do NOT send temperature=0 (unsupported on this deployment).
# ------------------------------------------------------
def call_gpt5_judge(headers, item_prompt: str) -> str:
    user_msg = f"""
You are auditing whether a NOTA-modified medical MCQ is truly TRUE_NOTA.

OUTPUT RULES (STRICT):
- Output MUST be valid JSON only (no prose, no markdown, no code fences).
- Label must be exactly "TRUE_NOTA" or "FALSE_NOTA".
- plausible_options must be an array of letters from ["A","B","C","D","E"] (empty array allowed).
- reason must be concise.

Return exactly this JSON shape:
{{
  "label": "TRUE_NOTA" or "FALSE_NOTA",
  "plausible_options": ["A","C"] or [],
  "reason": "..."
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
        resp = post_with_retries(GPT5_URL, headers=headers, json_data=data)
    except Exception as e:
        return f"ERROR: {e}"

    if resp.status_code != 200:
        return f"ERROR: {resp.status_code} {resp.text}"

    try:
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: Could not parse response JSON: {e} :: {resp.text[:500]}"


# ------------------------------------------------------
# 5. Column inference + label hiding
# ------------------------------------------------------
def pick_col(df, candidates):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def infer_mcq_columns(df):
    """
    Infer: id, stem, option_A..E.
    """
    return {
        "id": pick_col(df, ["sample_id", "question_id", "id", "qid"]),
        "stem": pick_col(df, ["stem", "question", "question_text", "prompt"]),
        "A": pick_col(df, ["option_A", "A", "choice_A"]),
        "B": pick_col(df, ["option_B", "B", "choice_B"]),
        "C": pick_col(df, ["option_C", "C", "choice_C"]),
        "D": pick_col(df, ["option_D", "D", "choice_D"]),
        "E": pick_col(df, ["option_E", "E", "choice_E"]),
    }


def find_label_like_columns(df):
    """
    Columns to hide from the judge prompt (but keep in output structure).
    """
    hide = []
    for c in df.columns:
        lc = c.lower()
        if lc in {"question_type", "label", "nota_label", "true_nota", "is_true_nota", "clinical annotations"}:
            hide.append(c)
        elif "annotation" in lc or "gold" in lc or ("label" in lc) or ("question_type" in lc):
            hide.append(c)
        elif ("nota" in lc) and (("label" in lc) or ("type" in lc) or ("truth" in lc)):
            hide.append(c)
    return sorted(set(hide))


def build_item_prompt(row, colmap, hidden_cols):
    qid = str(row[colmap["id"]]) if colmap["id"] else ""
    stem = str(row[colmap["stem"]]) if colmap["stem"] else ""

    def opt(letter):
        c = colmap.get(letter)
        if c and c in row and pd.notna(row[c]):
            return str(row[c]).strip()
        return ""

    A = opt("A"); B = opt("B"); C = opt("C"); D = opt("D"); E = opt("E")

    # Include limited extra context if present (and not label-like)
    extra = []
    for c in row.index:
        if c in hidden_cols:
            continue
        lc = c.lower()
        if lc in {"specialty", "category", "topic", "discipline"} and pd.notna(row[c]):
            extra.append(f"{c}: {row[c]}")
    extra_block = ("\n" + "\n".join(extra)) if extra else ""

    return f"""Question ID: {qid}

STEM:
{stem}

OPTIONS:
A) {A}
B) {B}
C) {C}
D) {D}
E) {E}
{extra_block}

Remember: If ANY remaining option could reasonably be correct, label FALSE_NOTA.
""".strip()


# ------------------------------------------------------
# 6. Resume helpers
# ------------------------------------------------------
def is_prev_ok(prev_row: dict) -> bool:
    raw = prev_row.get("judge_raw")
    err = prev_row.get("judge_error")
    parse_ok = prev_row.get("judge_parse_ok")

    if isinstance(err, str) and err.strip():
        return False
    if not isinstance(raw, str) or not raw.strip():
        return False
    if raw.startswith("ERROR:"):
        return False
    if parse_ok in [False, "False", 0, "0", None]:
        return False

    label = prev_row.get("judge_label")
    return isinstance(label, str) and label.strip().upper() in {"TRUE_NOTA", "FALSE_NOTA"}


# ------------------------------------------------------
# 7. Main
# ------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/question_key.csv", help="Input CSV path")
    ap.add_argument("--audit", dest="audit", default="metrics/question_key_TRUE_NOTA_audit2.csv",
                    help="Audit output CSV (input cols + judge cols)")
    ap.add_argument("--out", dest="out", default="metrics/question_key_TRUE_NOTA_only2.csv",
                    help="Filtered output CSV (SAME structure as input; only TRUE_NOTA rows)")
    ap.add_argument("--resume", action="store_true", help="Resume from existing audit file")
    ap.add_argument("--autosave_every", type=int, default=25, help="Autosave audit every N processed rows")
    ap.add_argument("--sleep", type=float, default=0.1, help="Small delay between calls")
    ap.add_argument("--limit", type=int, default=0, help="Optional: limit number of rows (0=all)")
    args = ap.parse_args()

    # Load API key from .env
    load_dotenv()
    KEY = os.getenv("SECUREGPT_API_KEY")
    if not KEY:
        raise SystemExit("❌ SECUREGPT_API_KEY not found in .env")

    headers = {
        "Ocp-Apim-Subscription-Key": KEY,
        "Content-Type": "application/json",
    }

    inp_path = Path(args.inp)
    if not inp_path.exists():
        # helpful fallback if you're running in this chat sandbox
        fb = Path("/mnt/data/questions.csv")
        if fb.exists():
            inp_path = fb
        else:
            raise SystemExit(f"❌ Input CSV not found: {args.inp}")

    audit_path = Path(args.audit)
    out_path = Path(args.out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Input CSV: {inp_path.resolve()}")
    df = pd.read_csv(inp_path)

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    colmap = infer_mcq_columns(df)
    if not colmap["stem"] or not colmap["A"] or not colmap["B"] or not colmap["C"] or not colmap["D"]:
        raise SystemExit(
            "❌ Could not infer required MCQ columns.\n"
            f"Detected mapping: {colmap}\n"
            f"Columns: {list(df.columns)}"
        )

    hidden_cols = find_label_like_columns(df)
    if hidden_cols:
        print(f"🙈 Hiding label-like columns from judge prompt: {hidden_cols}")
    else:
        print("🙈 No label-like columns detected to hide (fine).")

    id_col = colmap["id"]
    print(f"Detected ID column: {id_col if id_col else '(none; will use row index)'}")

    # Resume map
    prev_map = None
    prev_key_col = None

    if args.resume and audit_path.exists():
        prev_df = pd.read_csv(audit_path)
        if id_col and id_col in prev_df.columns:
            prev_key_col = id_col
        elif "row_index" in prev_df.columns:
            prev_key_col = "row_index"

        if prev_key_col:
            prev_map = {str(r[prev_key_col]): r.to_dict() for _, r in prev_df.iterrows()}
            print(f"🔁 Resume enabled: loaded {len(prev_map)} prior audit rows from {audit_path}")

    # Run judge
    rows_out = []
    total = len(df)
    print(f"\n🚀 Starting TRUE_NOTA audit on {total} rows...\n")

    for idx, row in df.iterrows():
        key = str(row[id_col]) if id_col else str(idx + 1)

        prev_row = prev_map.get(key) if prev_map else None
        reused = False

        if prev_row and is_prev_ok(prev_row):
            # reuse prior
            judge_raw = prev_row.get("judge_raw", "")
            judge_label = prev_row.get("judge_label")
            judge_plaus = prev_row.get("judge_plausible_options", "")
            judge_reason = prev_row.get("judge_reason", "")
            judge_parse_ok = prev_row.get("judge_parse_ok", True)
            judge_error = prev_row.get("judge_error", "")
            reused = True
        else:
            item_prompt = build_item_prompt(row, colmap, hidden_cols)
            judge_raw = call_gpt5_judge(headers, item_prompt)

            if isinstance(judge_raw, str) and judge_raw.startswith("ERROR:"):
                judge_label = None
                judge_plaus = ""
                judge_reason = ""
                judge_parse_ok = False
                judge_error = judge_raw
            else:
                parsed = safe_parse_json_obj(judge_raw)
                judge_label = parsed["label"]
                judge_plaus = ",".join(parsed["plausible_options"])
                judge_reason = parsed["reason"]
                judge_parse_ok = parsed["parse_ok"]
                judge_error = "" if judge_parse_ok else "PARSE_FAILED"

        out_row = row.to_dict()
        out_row["row_index"] = idx + 1
        out_row["judge_raw"] = judge_raw
        out_row["judge_label"] = judge_label
        out_row["judge_plausible_options"] = judge_plaus
        out_row["judge_reason"] = judge_reason
        out_row["judge_parse_ok"] = judge_parse_ok
        out_row["judge_error"] = judge_error
        out_row["judge_reused"] = reused

        rows_out.append(out_row)

        note = " [reused]" if reused else ""
        print(f"Processed {idx+1}/{total}{note}")

        if args.autosave_every and (len(rows_out) % args.autosave_every == 0):
            pd.DataFrame(rows_out).to_csv(audit_path, index=False)
            print(f"   💾 Autosaved audit to {audit_path}")

        time.sleep(args.sleep)

    audit_df = pd.DataFrame(rows_out)
    audit_df.to_csv(audit_path, index=False)
    print(f"\n✅ Audit saved to: {audit_path}")

    # Filter TRUE_NOTA based on judge_label
    true_mask = audit_df["judge_label"].astype(str).str.upper().eq("TRUE_NOTA")
    kept_audit = audit_df.loc[true_mask].copy()

    # IMPORTANT: output must be SAME structure as input (no judge cols).
    # So we re-select original input columns only, preserving exact order.
    input_cols = list(df.columns)
    filtered_df = kept_audit[input_cols].copy()
    filtered_df.to_csv(out_path, index=False)

    print(f"✅ TRUE_NOTA-only CSV (same structure) saved to: {out_path}")
    print(f"Kept {len(filtered_df)} / {len(df)} rows as TRUE_NOTA.")


if __name__ == "__main__":
    main()
