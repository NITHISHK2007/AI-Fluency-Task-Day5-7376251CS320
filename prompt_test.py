import os
import time
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

SYSTEM_PROMPT = """
You are a Student Study Assistant.
Explain academic concepts in simple language.
Use short examples when useful.
"""

prompts = [
    "What is a Python list?",
    "Explain binary search in simple terms.",
    "What is a stack in data structures?",
    "What is a Python list?"
]

for i, prompt in enumerate(prompts, 1):

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    start = time.time()

    response = requests.post(
        URL,
        headers=headers,
        json=payload
    )

    response.raise_for_status()

    total = time.time() - start

    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    usage = data.get("usage", {})
    tokens = usage.get("completion_tokens", 0)

    tokens_per_sec = tokens / total if total > 0 else 0

    print(f"\n=== PROMPT {i} ===")
    print("Prompt:", prompt)
    print("\nAnswer:")
    print(answer)
    print(f"\nCompletion tokens: {tokens}")
    print(f"Total time: {total:.2f} seconds")
    print(f"Approx tokens/sec: {tokens_per_sec:.2f}")