#!/usr/bin/env python3
import os
import json
import requests
from dotenv import load_dotenv

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MAX_OUTPUT_TOKENS = 5000
THINKING_BUDGET = 128

TEST_PROMPT = """
Return ONLY valid JSON (no markdown).
{
  "test": 1
}
"""

GEMINI_ENDPOINTS = [
    ("native_contents", "https://apim.stanfordhealthcare.org/gemini-25-pro/gemini-25-pro"),
    ("chat_v1", "https://apim.stanfordhealthcare.org/gemini-25-pro/v1/chat/completions"),
    ("chat", "https://apim.stanfordhealthcare.org/gemini-25-pro/chat/completions"),
]

# --------------------------------------------------
# LOAD KEY
# --------------------------------------------------

load_dotenv()
KEY = os.getenv("SECUREGPT_API_KEY")

if not KEY:
    print("❌ SECUREGPT_API_KEY not found.")
    raise SystemExit(1)

HEADERS = {
    "Ocp-Apim-Subscription-Key": KEY,
    "Content-Type": "application/json",
}

session = requests.Session()

# --------------------------------------------------
# DIAGNOSTIC RUN
# --------------------------------------------------

print("\n================ GEMINI DIAGNOSTIC ================\n")
print(f"Requested maxOutputTokens: {MAX_OUTPUT_TOKENS}")
print(f"Requested thinkingBudget:  {THINKING_BUDGET}\n")

for mode, url in GEMINI_ENDPOINTS:
    print("---------------------------------------------------")
    print(f"Testing route: {mode}")
    print(f"URL: {url}")
    print("---------------------------------------------------\n")

    if mode == "native_contents":
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": TEST_PROMPT}]
            }],
            "generationConfig": {
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
                "temperature": 0.0,
                "thinkingConfig": {
                    "thinkingBudget": THINKING_BUDGET
                }
            }
        }
    else:
        payload = {
            "model": "gemini-2.5-pro",
            "messages": [{
                "role": "user",
                "content": TEST_PROMPT
            }],
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.0
        }

    try:
        resp = session.post(url, headers=HEADERS, json=payload, timeout=60)
    except Exception as e:
        print(f"❌ REQUEST FAILED: {e}\n")
        continue

    print("HTTP STATUS:", resp.status_code)
    print("\nHEADERS:")
    for k, v in resp.headers.items():
        print(f"  {k}: {v}")

    print("\nRAW BODY (first 5000 chars):")
    print(resp.text[:5000])

    # Try structured inspection
    try:
        data = resp.json()
        print("\n----- STRUCTURED INSPECTION -----")

        # Sometimes APIM wraps in list
        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        # Try common Gemini structure
        if isinstance(data, dict):
            if "candidates" in data:
                candidate = data["candidates"][0]

                print("finishReason:", candidate.get("finishReason"))

                content = candidate.get("content", {})
                parts = content.get("parts", [])
                print("Returned parts count:", len(parts))

            if "usageMetadata" in data:
                usage = data["usageMetadata"]
                print("promptTokenCount:", usage.get("promptTokenCount"))
                print("thoughtsTokenCount:", usage.get("thoughtsTokenCount"))
                print("candidatesTokenCount:", usage.get("candidatesTokenCount"))
                print("totalTokenCount:", usage.get("totalTokenCount"))

            if "modelVersion" in data:
                print("modelVersion:", data.get("modelVersion"))

        print("---------------------------------\n")

    except Exception:
        print("\nCould not JSON-decode response.\n")

print("\n================ END DIAGNOSTIC ================\n")
