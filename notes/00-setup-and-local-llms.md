# Module 0 — Environment and Local Models

> Phase 0 · Week 0 · [Syllabus](../SYLLABUS.md#module-0-environment-and-local-models)

**Goal:** by the end of this module you can run an open-source LLM on your own laptop, call it from Python,
stream its answer, and measure how fast it is — with no API key and no cost.

**Why local first?** Local models are free, private (nothing leaves your machine), work offline and never
rate-limit you. They also force you to understand things hosted APIs hide: model size, memory, quantization and speed.
Everything you learn here transfers directly to OpenAI, Claude or Gemini later, because Ollama speaks the same
OpenAI-style API.

---

## 0.1 Dev environment

### What you need

| Tool | Why | Check it works |
|---|---|---|
| Python 3.11+ | The language of AI engineering | `python --version` |
| `uv` | Fast installer + virtual-env manager (replaces `pip` + `venv`) | `uv --version` |
| Git + GitHub account | Version control; this repo is your portfolio | `git --version` |
| VS Code (+ Python, Jupyter extensions) | Editor and notebooks | open a `.py` file |
| Jupyter | Quick experiments | `uv run jupyter lab` |

### Install `uv`

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### One project, one environment

Every project gets its **own virtual environment** so library versions never clash.

```bash
mkdir -p projects/00-hello-local-llm && cd projects/00-hello-local-llm
uv init                # creates pyproject.toml, .python-version, main.py
uv add ollama openai   # installs into .venv and records them in pyproject.toml
uv run python main.py  # runs inside the project's .venv — no "activate" needed
```

Key idea: `pyproject.toml` lists what you depend on; `uv.lock` pins exact versions so the project runs the same on any machine.
Commit both. Never commit `.venv/`.

### Git basics you will use daily

```bash
git switch -c module-0          # new branch for the work
git add -A && git commit -m "Module 0: hello local LLM"
git push -u origin module-0
```

A minimal `.gitignore` for every Python project:

```gitignore
.venv/
__pycache__/
.env
*.gguf
.ipynb_checkpoints/
```

> **Exercise 0.1** — Install the tools above, create `projects/00-hello-local-llm` with `uv init`, and push it to GitHub.

---

## 0.2 Ollama — run models locally

[Ollama](https://ollama.com) downloads open models, runs them on your CPU/GPU and exposes a local HTTP API on
`http://localhost:11434`.

### Install and first chat

```bash
# macOS / Windows: download the installer from https://ollama.com/download
# Linux:
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2:3b     # download a ~2 GB model
ollama run llama3.2:3b      # interactive chat in the terminal (/bye to exit)
```

### Commands to know

| Command | What it does |
|---|---|
| `ollama pull <model>` | Download a model |
| `ollama run <model>` | Chat in the terminal (pulls first if needed) |
| `ollama list` | Models on disk |
| `ollama ps` | Models currently loaded in memory |
| `ollama show <model>` | Parameters, context length, quantization, licence |
| `ollama rm <model>` | Delete a model |
| `ollama serve` | Start the server manually (the desktop app does this for you) |

### Starter models (check [ollama.com/library](https://ollama.com/library) for the latest tags)

| Model tag | Size on disk (approx.) | Good for |
|---|---|---|
| `llama3.2:3b` | ~2 GB | Fast default for experiments |
| `qwen2.5:7b` | ~4.7 GB | Stronger reasoning, multilingual, JSON output |
| `mistral:7b` | ~4.1 GB | Solid general model |
| `phi4-mini` | ~2.5 GB | Small but good at reasoning |
| `gemma3:4b` | ~3.3 GB | Google's small open model |
| `nomic-embed-text` | ~0.3 GB | Embeddings (not chat) — used in 0.5 and Module 4 |

**Model tags** read as `name:size-variant`, e.g. `qwen2.5:7b-instruct-q4_K_M` = Qwen 2.5, 7 billion
parameters, instruction-tuned, 4-bit quantization.

> **Exercise 0.2** — Pull `llama3.2:3b` and one 7B model. Ask both the same three questions (a fact, a
> reasoning puzzle, "write a Python function that reverses a string"). Note which answers are better and which is faster.

---

## 0.3 Hardware sizing and quantization

### How much memory does a model need?

A model is mostly its **weights**: one number per parameter.

| Precision | Bytes per parameter | 7B model weights |
|---|---|---|
| FP16 (16-bit) | 2 | ~14 GB |
| Q8 (8-bit) | ~1 | ~7.5 GB |
| Q4 (4-bit) | ~0.5–0.6 | ~4–4.5 GB |

Rule of thumb: **RAM needed ≈ parameters (billions) × bytes per parameter + 1–2 GB** for the context (KV cache)
and runtime.

| Your RAM | Comfortable models (Q4) |
|---|---|
| 8 GB | 1B–4B (e.g. `llama3.2:3b`, `phi4-mini`) |
| 16 GB | up to 7B–8B (e.g. `qwen2.5:7b`, `llama3.1:8b`) |
| 32 GB+ | 13B–14B, or 7B at Q8 |

Apple Silicon uses unified memory (GPU shares RAM), so it runs local models well. On Windows/Linux an NVIDIA GPU
with enough VRAM speeds things up greatly; without one, models run on CPU — slower but fine for learning.

### Quantization in one paragraph

Quantization stores each weight with fewer bits (e.g. 4 instead of 16). The model gets ~4× smaller and faster,
with a small loss in quality. `Q4_K_M` is the usual sweet spot; `Q8_0` is near-original quality at twice the size.

### GGUF

**GGUF** is the single-file format used by `llama.cpp` (the engine inside Ollama and LM Studio). A `.gguf` file
holds the quantized weights plus the tokenizer and metadata. You will export your own fine-tuned model to GGUF
in Module 7.

> **Exercise 0.3** — Run `ollama show qwen2.5:7b` and find its parameter count, quantization and context length.
> Using the rule of thumb, calculate whether a 14B model at Q4 fits on your machine.

---

## 0.4 Calling a local model from Python

Ollama gives you **two** ways in. Learn both — the second one is what makes switching to hosted providers trivial later.

### Option A — the `ollama` Python library

```python
# chat_ollama.py        run: uv run python chat_ollama.py
import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": "You are a concise tutor."},
        {"role": "user", "content": "Explain what an LLM is in two sentences."},
    ],
    options={"temperature": 0.2},  # lower = more deterministic
)
print(response.message.content)
```

**Messages** are the core abstraction of every chat API:

- `system` — standing instructions (persona, rules, format).
- `user` — what the person asks.
- `assistant` — what the model replied (you send earlier replies back to give it memory of the conversation).

### Streaming

Without streaming you wait for the whole answer. With streaming you print tokens as they arrive — essential for good UX.

```python
import ollama

stream = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Write a haiku about vectors."}],
    stream=True,
)
for chunk in stream:
    print(chunk.message.content, end="", flush=True)
print()
```

### Option B — the OpenAI-compatible endpoint

Ollama also serves the OpenAI API at `http://localhost:11434/v1`. The **same code** later works with OpenAI,
Groq, OpenRouter and many others by changing only `base_url`, `api_key` and `model`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # key is required but ignored

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Name three uses of embeddings."}],
)
print(response.choices[0].message.content)
print(response.usage)  # prompt_tokens, completion_tokens, total_tokens
```

### Raw HTTP (to see what libraries do for you)

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'
```

### Useful generation options

| Option | Effect |
|---|---|
| `temperature` | Randomness. 0–0.3 for facts/code, 0.7–1.0 for creative text |
| `top_p` | Sample only from the most likely tokens whose probabilities add up to `p` |
| `num_ctx` | Context window size in tokens (Ollama's default is small — raise it for long documents) |
| `num_predict` | Maximum tokens to generate |
| `seed` | Fixed seed for reproducible output |

> **Exercise 0.4** — Build a terminal chat loop that keeps the conversation history (append each user and
> assistant message to the `messages` list) and exits on `quit`.

---

## 0.5 Local embeddings

An **embedding** turns text into a list of numbers (a vector) so that texts with similar meaning have vectors
that point in similar directions. This is the foundation of RAG (Module 4).

```bash
ollama pull nomic-embed-text
```

```python
import math
import ollama

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

texts = [
    "The cat sat on the mat.",
    "A kitten is resting on a rug.",
    "Stock markets fell sharply today.",
]
vectors = ollama.embed(model="nomic-embed-text", input=texts).embeddings
print(len(vectors[0]), "dimensions")                     # 768 for nomic-embed-text
print("cat vs kitten:", round(cosine(vectors[0], vectors[1]), 3))  # high
print("cat vs stocks:", round(cosine(vectors[0], vectors[2]), 3))  # low
```

**Cosine similarity** ranges from −1 to 1: close to 1 means "same meaning", near 0 means unrelated. You will
meet it again in Module 2 (math essentials) and Module 4 (vector databases).

> **Exercise 0.5** — Embed ten sentences from two different topics and, for each sentence, print its most
> similar neighbour. Does the model group them by topic?

---

## 0.6 Working with AI coding assistants responsibly

AI assistants (Claude Code, Copilot, Cursor, ChatGPT) speed you up, but in a learning programme they can also
stop you learning. Ground rules for this course:

1. **Write the first version yourself.** Use the assistant to review, explain errors or suggest improvements afterwards.
2. **Never paste code you cannot explain.** If you cannot say what each line does, ask the assistant to explain it until you can.
3. **Never paste secrets** (API keys, `.env` contents, customer data) into a chat.
4. **Verify claims.** Libraries change fast; check the official docs for any API the assistant suggests.
5. **Keep a "learned" log** in your notes: one line per new concept you picked up from the assistant.

---

## Build — Hello local LLM

Create `projects/00-hello-local-llm/main.py` that:

1. Reads the model name from an environment variable `MODEL` (default `llama3.2:3b`).
2. Sends a prompt given on the command line and **streams** the answer.
3. After the answer, prints **tokens generated** and **tokens per second**.

Hint: the last streamed chunk has `done=True` and carries `eval_count` (tokens generated) and `eval_duration`
(nanoseconds spent generating). Tokens per second = `eval_count / (eval_duration / 1e9)`.

Usage you are aiming for:

```bash
uv run python main.py "Explain quantization to a 10-year-old"
MODEL=qwen2.5:7b uv run python main.py "Explain quantization to a 10-year-old"
```

## Checkpoint

- [ ] `ollama list` shows at least two chat models and `nomic-embed-text`.
- [ ] The script works against two different local models by changing only `MODEL`.
- [ ] You recorded tokens/second for each model in a small table in your project README.
- [ ] You can explain in your own words: model size vs RAM, what Q4 means, and what an embedding is.

## Key terms

| Term | Meaning |
|---|---|
| Parameters | The learned numbers (weights) inside a model; "7B" = 7 billion |
| Quantization | Storing weights with fewer bits to save memory and speed up inference |
| GGUF | Single-file model format used by llama.cpp, Ollama and LM Studio |
| Inference | Running a trained model to get output (as opposed to training it) |
| Token | A chunk of text (roughly ¾ of an English word) the model reads and writes |
| Context window | Maximum tokens (prompt + answer) the model can handle at once |
| Embedding | A vector that represents the meaning of a text |
| Streaming | Receiving the answer token by token as it is generated |

## Further reading

- Ollama docs and API reference — <https://github.com/ollama/ollama/tree/main/docs>
- Ollama Python library — <https://github.com/ollama/ollama-python>
- uv documentation — <https://docs.astral.sh/uv/>
