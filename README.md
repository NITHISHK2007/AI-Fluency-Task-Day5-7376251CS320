# Day 5 Task — Serving Models Your Way

## Scenario

**Student Study Assistant**

A simple AI study assistant designed to explain programming and data-structure concepts in easy language.

## Task

This project explores model serving concepts including:

* Ollama and local model serving
* Modelfiles
* REST APIs
* OpenAI-compatible endpoints
* Streaming responses
* TTFT and total response time
* Multi-user serving with vLLM
* PagedAttention and continuous batching

## Implementation

Ollama was not installed for this implementation. Instead, the hands-on API demonstrations were implemented using the **Groq OpenAI-compatible API**.

The project contains:

* `api_demo.py` — non-streaming and streaming API demonstration
* `openai_compatible.py` — OpenAI-compatible endpoint demonstration with a different system prompt
* `prompt_test.py` — tests multiple study prompts and records response metrics
* `analysis.md` — complete conceptual analysis and observations
* `outputs/` — screenshots of the program outputs

## Results

The streaming API test produced:

* TTFT: **0.88 seconds**
* Total time: **1.75 seconds**
* Approx. tokens/sec: **162.72**

The prompt testing also included a repeated prompt to compare response performance.

## Running the Project

Install the required packages:

```bash
py -m pip install requests python-dotenv
```

Run the API demonstration:

```bash
py api_demo.py
```

Run the OpenAI-compatible endpoint demonstration:

```bash
py openai_compatible.py
```

Run the prompt tests:

```bash
py prompt_test.py
```

## Security

The Groq API key is stored in `.env`.

The `.gitignore` file excludes `.env`, virtual environments, Python cache files, and compiled Python files from Git.

## Note

The Day 5 task specifically includes Ollama hands-on requirements. Since Ollama was not installed, no fake Ollama outputs were created. Ollama-specific concepts are discussed conceptually in `analysis.md`, while the working API demonstrations use Groq's OpenAI-compatible endpoint.
