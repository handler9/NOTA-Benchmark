"""
Patches all 15 runner scripts to pass full conversation context for multi-turn questions.

Changes applied to each script:
  1. Data loading: store prompt_turns (full conversation) alongside question (last turn)
  2. After loading: detect multi-turn IDs
  3. call_model: accept prompt_turns list, build appropriate API payload
  4. Call site: parse prompt_turns, smart resume (reuse single-turn, re-run multi-turn)
"""

import re
from pathlib import Path

RUNNERS_DIR = Path("scripts/healthbench/runners")

# ── Shared data-loading patch ─────────────────────────────────────────────────
OLD_DATA_LOAD = '''        question = r["prompt"][-1]["content"] if r.get("prompt") else ""
        rows_in.append({
            "prompt_id": r["prompt_id"],
            "question": question,'''

NEW_DATA_LOAD = '''        prompt_turns = r.get("prompt", [])
        question = prompt_turns[-1]["content"] if prompt_turns else ""
        rows_in.append({
            "prompt_id": r["prompt_id"],
            "question": question,
            "prompt_turns": json.dumps(prompt_turns),'''

OLD_LOADED_PRINT = 'print(f"  {len(rows_in)} questions loaded")'
NEW_LOADED_PRINT = '''print(f"  {len(rows_in)} questions loaded")
multi_turn_ids = {row["prompt_id"] for row in rows_in if len(json.loads(row["prompt_turns"])) > 1}
print(f"  {len(multi_turn_ids)} multi-turn questions (will re-run with full context)")'''

# ── Shared call-site patch (loop body) ────────────────────────────────────────
OLD_CALL_SITE = '''    qid = row["prompt_id"]
    question = row["question"]

    prev_row = prev_map.get(qid)
    if prev_row and is_result_ok(prev_row):
        raw = prev_row[f"{PREFIX}_raw"]
        reused = True
    else:
        raw = call_model(question)
        reused = False'''

NEW_CALL_SITE = '''    qid = row["prompt_id"]
    question = row["question"]
    prompt_turns = json.loads(row["prompt_turns"])

    prev_row = prev_map.get(qid)
    if prev_row and is_result_ok(prev_row) and qid not in multi_turn_ids:
        raw = prev_row[f"{PREFIX}_raw"]
        reused = True
    else:
        raw = call_model(prompt_turns)
        reused = False'''

# ── Model-specific call_model patches ─────────────────────────────────────────

# GPT
OLD_CALL_MODEL_GPT = '''def call_model(question: str) -> str:
    data = {
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": question},
        ],
        "max_completion_tokens": 5000,
    }'''

NEW_CALL_MODEL_GPT = '''def call_model(prompt_turns: list) -> str:
    data = {
        "messages": [{"role": "system", "content": INSTRUCTIONS}] + [
            {"role": t["role"], "content": t["content"]} for t in prompt_turns
        ],
        "max_completion_tokens": 5000,
    }'''

# Llama
OLD_CALL_MODEL_LLAMA = '''def call_model(question: str) -> str:
    data = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": question},
        ],
        "max_tokens": 5000,
    }'''

NEW_CALL_MODEL_LLAMA = '''def call_model(prompt_turns: list) -> str:
    data = {
        "model": LLAMA_MODEL,
        "messages": [{"role": "system", "content": INSTRUCTIONS}] + [
            {"role": t["role"], "content": t["content"]} for t in prompt_turns
        ],
        "max_tokens": 5000,
    }'''

# DeepSeek
OLD_CALL_MODEL_DEEPSEEK = '''def call_model(question: str) -> str:
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": question},
        ],
        "max_tokens": 5000,
    }'''

NEW_CALL_MODEL_DEEPSEEK = '''def call_model(prompt_turns: list) -> str:
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "system", "content": INSTRUCTIONS}] + [
            {"role": t["role"], "content": t["content"]} for t in prompt_turns
        ],
        "max_tokens": 5000,
    }'''

# Claude
OLD_CALL_MODEL_CLAUDE = '''def call_model(question: str) -> str:
    full_prompt = f"{INSTRUCTIONS}\\n\\nQuestion:\\n{question}"
    payload = {
        "model_id": CLAUDE_MODEL_ID,
        "prompt_text": full_prompt,
        "max_tokens": 5000,
    }'''

NEW_CALL_MODEL_CLAUDE = '''def call_model(prompt_turns: list) -> str:
    if len(prompt_turns) <= 1:
        question = prompt_turns[0]["content"] if prompt_turns else ""
        full_prompt = f"{INSTRUCTIONS}\\n\\nQuestion:\\n{question}"
    else:
        parts = []
        for turn in prompt_turns:
            label = "Patient" if turn["role"] == "user" else "Assistant"
            parts.append(f"{label}: {turn['content']}")
        full_prompt = f"{INSTRUCTIONS}\\n\\nConversation:\\n" + "\\n\\n".join(parts)
    payload = {
        "model_id": CLAUDE_MODEL_ID,
        "prompt_text": full_prompt,
        "max_tokens": 5000,
    }'''

# Gemini
OLD_CALL_MODEL_GEMINI = '''def call_model(question: str) -> str:
    mode, url = autodetect_gemini()
    full_text = f"{INSTRUCTIONS}\\n\\nQuestion:\\n{question}"
    payload, extract = build_payload(mode, full_text)'''

NEW_CALL_MODEL_GEMINI = '''def call_model(prompt_turns: list) -> str:
    if len(prompt_turns) <= 1:
        question = prompt_turns[0]["content"] if prompt_turns else ""
        full_text = f"{INSTRUCTIONS}\\n\\nQuestion:\\n{question}"
    else:
        parts = []
        for turn in prompt_turns:
            label = "Patient" if turn["role"] == "user" else "Assistant"
            parts.append(f"{label}: {turn['content']}")
        full_text = f"{INSTRUCTIONS}\\n\\nConversation:\\n" + "\\n\\n".join(parts)
    mode, url = autodetect_gemini()
    payload, extract = build_payload(mode, full_text)'''


def patch(content, old, new, label):
    if old in content:
        content = content.replace(old, new, 1)
        print(f"    ✅ {label}")
    else:
        print(f"    ⚠️  NOT FOUND: {label}")
    return content


MODEL_PATCHES = {
    "gpt":      OLD_CALL_MODEL_GPT,
    "llama":    OLD_CALL_MODEL_LLAMA,
    "deepseek": OLD_CALL_MODEL_DEEPSEEK,
    "claude":   OLD_CALL_MODEL_CLAUDE,
    "gemini":   OLD_CALL_MODEL_GEMINI,
}
MODEL_PATCHES_NEW = {
    "gpt":      NEW_CALL_MODEL_GPT,
    "llama":    NEW_CALL_MODEL_LLAMA,
    "deepseek": NEW_CALL_MODEL_DEEPSEEK,
    "claude":   NEW_CALL_MODEL_CLAUDE,
    "gemini":   NEW_CALL_MODEL_GEMINI,
}

for script in sorted(RUNNERS_DIR.glob("hb_*.py")):
    model = None
    for m in MODEL_PATCHES:
        if f"_{m}." in script.name:
            model = m
            break
    if model is None:
        print(f"⚠️  Skipping {script.name} — unknown model")
        continue

    print(f"\n{script.name}")
    content = script.read_text()

    content = patch(content, OLD_DATA_LOAD,   NEW_DATA_LOAD,   "data loading")
    content = patch(content, OLD_LOADED_PRINT, NEW_LOADED_PRINT, "multi-turn detection")
    content = patch(content, MODEL_PATCHES[model], MODEL_PATCHES_NEW[model], "call_model")
    content = patch(content, OLD_CALL_SITE,   NEW_CALL_SITE,   "call site")

    script.write_text(content)

print("\nDone. All runner scripts patched.")
