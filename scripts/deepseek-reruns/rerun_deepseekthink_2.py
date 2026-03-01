import os
import re
import json
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException

# ------------------------------------------------------
# 0. Paths (adjust if needed)
# ------------------------------------------------------
INPUT_PATH = Path("results_raw/deepseek_think.csv")
OUTPUT_PATH = Path("results_raw/deepseek_think_patched.csv")

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
# 2. DeepSeek endpoint & model (match working script)
# ------------------------------------------------------
DEEPSEEK_URL = "https://apim.stanfordhealthcare.org/deepseekr1/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# ------------------------------------------------------
# 3. THINK system prompt (your exact version)
# ------------------------------------------------------
INSTRUCTIONS = """
You’re a careful clinician. Before you answer, silently go through three steps:

1. Identify the key clinical findings and what the correct diagnosis or management would be in general.
2. Check whether any of the four options (A–D) truly match that correct diagnosis or management.
3. If none of the options are appropriate, you must abstain instead of guessing.

Do this reasoning in your head and do NOT show these steps. After thinking, give only one short, structured reply with four parts:

- choice – Pick A, B, C, or D if, after checking, you are sure one option is appropriate. If none are appropriate or you are unsure, leave this blank (write "null").
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
# 4. Robust JSON parser (same style as your working script)
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
# 5. POST helper with retries (copied style)
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
            print(f"   Request error (attempt {attempt}/{max_retries}): {e} — retrying in {sleep_sec}s")
            time.sleep(sleep_sec)

    raise last_err if last_err else RuntimeError("Unknown error in post_with_retries")

# ------------------------------------------------------
# 6. Helper function for DeepSeek (match working payload)
# ------------------------------------------------------
def call_deepseek(messages) -> str:
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        # 🔹 match your working script: use max_completion_tokens
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
# 7. Build messages for a given deepseek_think row
# ------------------------------------------------------
def build_messages_from_row(row: pd.Series):
    stem = str(row["stem"])
    opts = (
        f"A. {row['option_A']}\n"
        f"B. {row['option_B']}\n"
        f"C. {row['option_C']}\n"
        f"D. {row['option_D']}"
    )

    user_prompt = f"""{stem}

Options:
{opts}
"""

    return [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": user_prompt},
    ]

# ------------------------------------------------------
# 8. Main patch logic
# ------------------------------------------------------
def main():
    print(f"📂 Reading: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        print(f"❌ Input CSV not found at {INPUT_PATH}")
        raise SystemExit

    df = pd.read_csv(INPUT_PATH)

    required_cols = [
        "question_id",
        "stem",
        "option_A",
        "option_B",
        "option_C",
        "option_D",
        "deepseek_raw",
        "deepseek_abstain_code",
    ]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Missing required column in input: {col}")
            raise SystemExit

    # Find rows with API_ERROR
    mask = df["deepseek_abstain_code"].astype(str).str.contains("API_ERROR", na=False)
    rows_to_fix = df[mask]

    if rows_to_fix.empty:
        print("✅ No rows with deepseek_abstain_code == 'API_ERROR'. Nothing to patch.")
        return

    print(f"🔧 Found {len(rows_to_fix)} row(s) with API_ERROR. Re-running those only...\n")

    for idx, row in rows_to_fix.iterrows():
        qid = row["question_id"]
        row_index = row.get("row_index", idx + 1)
        print(f"➡️  Patching question_id={qid} (row_index={row_index})")

        messages = build_messages_from_row(row)
        raw = call_deepseek(messages)

        if isinstance(raw, str) and raw.startswith("ERROR:"):
            print(f"   ⚠️ Still getting error for question_id={qid}: {raw}")
            df.at[idx, "deepseek_raw"] = raw
            df.at[idx, "deepseek_choice"] = None
            df.at[idx, "deepseek_abstain_code"] = "API_ERROR"
            df.at[idx, "deepseek_confidence"] = 0.0
            df.at[idx, "deepseek_rationale"] = raw
            continue

        parsed = safe_parse_json(raw)

        df.at[idx, "deepseek_raw"] = raw
        df.at[idx, "deepseek_choice"] = parsed.get("choice")
        df.at[idx, "deepseek_abstain_code"] = parsed.get("abstain_code")
        df.at[idx, "deepseek_confidence"] = parsed.get("confidence")
        df.at[idx, "deepseek_rationale"] = parsed.get("rationale")

        print(f"   ✅ Patched question_id={qid}")

        # optional tiny delay
        time.sleep(0.1)

    # Save patched file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved patched CSV to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
