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


def chat(system_prompt, user_prompt, stream=False):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": stream
    }

    start = time.time()

    response = requests.post(
        URL,
        headers=headers,
        json=payload,
        stream=stream
    )

    response.raise_for_status()

    if not stream:
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        total = time.time() - start
        return answer, total

    first_token_time = None
    answer = ""

    for line in response.iter_lines():
        if not line:
            continue

        text = line.decode("utf-8")

        if not text.startswith("data: "):
            continue

        data = text[6:]

        if data == "[DONE]":
            break

        import json
        chunk = json.loads(data)

        delta = chunk["choices"][0]["delta"].get("content", "")

        if delta:
            if first_token_time is None:
                first_token_time = time.time()

            print(delta, end="", flush=True)
            answer += delta

    total = time.time() - start
    ttft = first_token_time - start if first_token_time else total

    return answer, total, ttft


SYSTEM_PROMPT = """
You are a Student Study Assistant.
Explain academic concepts in simple language.
Use short examples when useful.
Do not make the explanation unnecessarily complicated.
"""

print("=== STUDENT STUDY ASSISTANT ===")
print("MODEL:", MODEL)

print("\n--- Non-streaming ---")

answer, total = chat(
    SYSTEM_PROMPT,
    "Explain what an array is in Python.",
    stream=False
)

print(answer)
print(f"\nTotal time: {total:.2f} seconds")

print("\n--- Streaming ---")

answer, total, ttft = chat(
    SYSTEM_PROMPT,
    "Explain what a stack is in simple terms.",
    stream=True
)

print(f"\nTTFT: {ttft:.2f} seconds")
print(f"Total time: {total:.2f} seconds")

words = len(answer.split())

if total > 0:
    print(f"Approx tokens/sec: {words / total:.2f}")