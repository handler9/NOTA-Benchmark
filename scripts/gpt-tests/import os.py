import os
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("AIHUB_API_KEY") or os.getenv("SECUREGPT_API_KEY")

url = "https://aihubapi.stanfordhealthcare.org/azure-openai/deployments/gpt-5-4/chat/completions?api-version=2025-04-01-preview"
headers = {
    "api-key": KEY,
    "Content-Type": "application/json",
}
data = {
    "messages": [
        {"role": "user", "content": "Say hello."}
    ],
    "max_completion_tokens": 50,
}

r = requests.post(url, headers=headers, json=data, timeout=60)
print(r.status_code)
print(r.text)how 