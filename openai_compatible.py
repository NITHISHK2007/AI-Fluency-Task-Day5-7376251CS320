import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")

URL = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "system",
            "content": "You are a strict exam-focused study assistant. Give short, direct answers."
        },
        {
            "role": "user",
            "content": "Explain what a queue is in Python."
        }
    ],
    "stream": False
}

response = requests.post(
    URL,
    headers=headers,
    json=payload
)

response.raise_for_status()

data = response.json()

print("=== OPENAI-COMPATIBLE ENDPOINT ===")
print("MODEL:", MODEL)
print()
print(data["choices"][0]["message"]["content"])