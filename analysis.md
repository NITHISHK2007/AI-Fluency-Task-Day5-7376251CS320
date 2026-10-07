# Day 5 Task — Serving Models Your Way

## Scenario: Student Study Assistant

### 3.1 Explanation of Concepts

My chosen scenario is a **Student Study Assistant**. The assistant is intended to explain programming and data-structure concepts in simple language for a student.

### What is Ollama?

Ollama is a local model-serving platform that provides different ways to interact with a language model. The task describes three main ways of accessing it: the command line, the REST API, and an OpenAI-compatible endpoint.

In a normal Ollama setup, the model files are stored locally on the machine. A request is sent to the Ollama server, the server loads or uses the required model, processes the prompt, and returns the generated response. The REST API can provide both normal and streaming responses.

For this task, Ollama was not installed or executed on my machine. Therefore, the Ollama-specific commands and local model-loading observations were not experimentally verified. Instead, I implemented the REST and OpenAI-compatible API portions using the Groq API, which also provides an OpenAI-compatible interface.

### What is a Modelfile?

A Modelfile is used by Ollama to package a base model together with default configuration and behaviour. The task identifies `FROM`, `PARAMETER`, and `SYSTEM` as important instructions.

For the Student Study Assistant scenario, a suitable Modelfile would define a base model, parameters such as context length or temperature, and a system instruction telling the model to behave as a simple academic assistant.

A custom Ollama model does not necessarily contain a completely new set of model weights. It can package a base model with additional configuration and instructions. Therefore, creating such a custom model does not mean downloading another complete model from scratch.

Because Ollama was not used for this implementation, no `ollama create` or actual Modelfile execution was performed.

### Program system prompt versus Modelfile system prompt

The task is designed to demonstrate that the behaviour packaged in a Modelfile provides a default, while a program can supply its own system prompt when making a request.

In my implementation, the program directly supplied the system prompt:

> You are a Student Study Assistant. Explain academic concepts in simple language. Use short examples when useful.

The OpenAI-compatible test then supplied a different system prompt:

> You are a strict exam-focused study assistant. Give short, direct answers.

The resulting response was short and exam-focused, showing that the program-level system instruction was used for that API request.

This is useful for agent development because a fixed default behaviour can be overridden when a particular application or task needs different behaviour.

### `/api/generate` versus `/api/chat`

The Ollama REST API provides different request styles. `/api/generate` is intended for generating a response from a prompt, while `/api/chat` is designed around a sequence of chat messages containing roles such as system and user.

The OpenAI-compatible endpoint provides an interface similar to the OpenAI API format. This is useful because application code can be moved between compatible model-serving systems with fewer changes.

In my implementation, I used the Groq OpenAI-compatible endpoint:

`https://api.groq.com/openai/v1/chat/completions`

This allowed the same Python application structure to send system and user messages.

### Streaming, TTFT and total time

Streaming means that the model response is returned in smaller pieces as they are generated instead of waiting for the complete answer.

**TTFT (Time To First Token)** measures the time from sending the request until the first generated token is received.

**Total time** measures the time taken for the complete response.

Streaming can make an application feel faster because the user starts seeing the answer earlier, even though the total time needed to generate the complete answer may be similar.

In my streaming test, the Student Study Assistant produced:

* TTFT: **0.88 seconds**
* Total time: **1.75 seconds**
* Approximate tokens/sec: **162.72**

### `num_ctx`, `OLLAMA_KEEP_ALIVE` and `OLLAMA_NUM_PARALLEL`

`num_ctx` controls the context length available to the model. A larger context allows more tokens to be kept in the active context but also increases KV-cache memory usage.

`OLLAMA_KEEP_ALIVE` controls how long a model remains loaded after a request.

`OLLAMA_NUM_PARALLEL` controls how many requests can be processed in parallel by an Ollama server.

The KV cache is important because increasing context length increases the memory required to store the attention information for the active requests.

These settings were not experimentally tested because Ollama was not installed for this task.

### KV cache and PagedAttention

During model serving, the KV cache stores attention-related information for tokens that have already been processed. With multiple requests and long contexts, the KV cache can consume a significant amount of GPU memory.

PagedAttention addresses this problem by managing the KV cache using fixed-size blocks and a block table rather than requiring one large continuous memory allocation for every request.

Prefix caching provides an additional benefit when requests repeatedly use the same initial prompt. For a Student Study Assistant, a common system prompt could be reused across many requests, reducing repeated computation.

These concepts are part of the vLLM analysis required by the task and were not implemented as a hands-on experiment.

### Static batching and continuous batching

Static batching groups a fixed set of requests together before processing them.

Continuous batching allows new requests to join the processing schedule as other requests finish. This is more suitable for a server receiving requests continuously from many users.

Important serving metrics include:

* **Throughput** — amount of work or tokens processed per unit of time.
* **TTFT** — time until the first token is produced.
* **TPOT** — time per output token after generation begins.
* **P95 latency** — the latency below which 95% of requests complete.

Increasing throughput can sometimes increase latency because more requests compete for compute and memory. Therefore, a production serving system needs to balance throughput and responsiveness.

---

## 3.2 Comparison Table

| Basis for comparison                    | Ollama                                                                               | vLLM                                                                               |
| --------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Built for                               | Local model serving and individual developers/small applications                     | High-throughput serving for applications with many concurrent users                |
| Hardware it needs                       | A suitable local machine with enough CPU/GPU/RAM for the selected model              | Usually GPU-oriented infrastructure designed for high-performance serving          |
| How it handles several requests at once | Supports parallel request configuration                                              | Designed around efficient continuous batching                                      |
| How it manages memory                   | Local model serving with model and KV-cache memory                                   | Uses techniques such as PagedAttention for efficient KV-cache management           |
| Setup effort and model format           | Simple local setup and model-oriented workflow                                       | More production-oriented serving setup                                             |
| My choice if only I use the scenario    | Ollama, because the Student Study Assistant is small and local serving is sufficient | Not necessary for this scenario                                                    |
| My choice if 100 people use it at once  | A production serving system such as vLLM would be more suitable                      | vLLM, because concurrent requests require efficient batching and memory management |

For one student using the Student Study Assistant, a local Ollama setup would be sufficient because the workload is small and does not require a large multi-user serving infrastructure.

For approximately 100 simultaneous users, a server such as vLLM would be more appropriate because the system needs efficient scheduling, batching and memory management.

---

## 3.3 Minimal Implementation

The task asks for a Modelfile, an Ollama REST API script, and an OpenAI-compatible endpoint test.

Ollama was not installed on my system, so I did not create fake `ollama list`, `ollama show`, `ollama ps`, or `ollama create` outputs.

Instead, I implemented the API portions using Groq's OpenAI-compatible API.

### REST-style implementation

`api_demo.py` sends a non-streaming request and a streaming request using Python `requests`.

The non-streaming request successfully returned an explanation of a Python array/list.

The streaming request successfully returned an explanation of a stack.

The measured streaming results were:

| Metric             |       Result |
| ------------------ | -----------: |
| TTFT               | 0.88 seconds |
| Total time         | 1.75 seconds |
| Approx. tokens/sec |       162.72 |

### OpenAI-compatible implementation

`openai_compatible.py` sends a chat-completion request using the OpenAI-compatible API format.

The system prompt was changed to:

> You are a strict exam-focused study assistant. Give short, direct answers.

The user prompt asked for an explanation of a queue in Python.

The response followed the short, direct academic style and explained FIFO, `collections.deque`, and `queue.Queue`.

This demonstrates how the program can supply its own system instruction when using an OpenAI-compatible endpoint.

---

## 3.4 Observation

Four prompts were tested using `prompt_test.py`. One prompt was deliberately repeated.

| Prompt                                 | Completion tokens | Total time | Approx. tokens/sec |
| -------------------------------------- | ----------------: | ---------: | -----------------: |
| What is a Python list?                 |               781 |     2.64 s |             296.19 |
| Explain binary search in simple terms. |               816 |     2.46 s |             332.10 |
| What is a stack in data structures?    |               746 |     2.46 s |             303.87 |
| What is a Python list? — repeated      |               758 |     2.45 s |             309.62 |

The repeated prompt was faster in the second run in this particular test: the first run took 2.64 seconds and the repeated run took 2.45 seconds.

The model consistently behaved as a Student Study Assistant and provided simple academic explanations.

The test also showed that response time and token generation speed vary between prompts. The binary-search response had the highest measured speed at approximately 332.10 tokens/sec, while the Python-list response in the first run was approximately 296.19 tokens/sec.

Because the implementation used Groq rather than Ollama, I could not verify whether the Ollama model remained loaded in memory using `ollama ps`, nor could I compare Ollama's first-load and second-load behaviour.

TTFT was measured for the streaming API demonstration, where the first token arrived after 0.88 seconds and the complete response finished after 1.75 seconds.

---

## 3.5 Suitability

For the Student Study Assistant scenario used by one student, local Ollama would be a suitable solution when the computer has enough memory for the selected model. It provides a simple way to run and access a model locally.

A shared deployment serving many users would require a more production-oriented serving system. vLLM or a similar server would be more suitable because many simultaneous requests create GPU-memory and scheduling challenges.

The important advantage of a Modelfile is that it can package default behaviour and model parameters so that an application starts with a consistent configuration. The REST API makes the model accessible programmatically, while an OpenAI-compatible endpoint makes application integration easier.

---

## 3.6 Conclusion

This task helped me understand the difference between local model serving and production-oriented model serving.

For a small personal Student Study Assistant, a local serving solution such as Ollama is sufficient when the required model fits the available hardware. Modelfiles can provide a reusable default behaviour, while REST APIs allow Python programs to communicate with the model.

Streaming improves the user experience by allowing the first generated content to appear before the complete response is finished. TTFT and total time provide different views of the response experience.

When the number of users increases significantly, model serving becomes a memory and scheduling problem. Techniques such as PagedAttention and continuous batching are designed to make multi-user serving more efficient, which is why a system such as vLLM becomes more appropriate for larger deployments.

For this implementation, Ollama-specific hands-on steps were not executed because Ollama was not installed. The working API demonstration and measurements were performed using Groq's OpenAI-compatible API, and the Ollama-specific observations are therefore treated as conceptual rather than experimentally verified.
