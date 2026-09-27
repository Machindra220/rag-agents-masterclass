# Module 1 — Python for AI Engineering

> Phase 1 · Weeks 1–3 · Level 1 Understand · [Syllabus](../SYLLABUS.md#module-1-python-for-ai-engineering)

**Goal:** write Python that belongs in a production codebase, not just scripts that work on your laptop.

Every later module sits on these eight skills: typed data, validated LLM output, concurrency, resilient API calls,
clean pipelines, an HTTP service and tests. The examples call the local Ollama model from Module 0.

**Setup for this module**

```bash
mkdir -p projects/01-llm-api-service && cd projects/01-llm-api-service
uv init
uv add ollama pydantic pydantic-settings fastapi "uvicorn[standard]" httpx python-dotenv tenacity
uv add --dev pytest pytest-asyncio
```

All code in these notes was run with Python 3.11, Pydantic 2.13 and FastAPI 0.141.

---

## 1.1 Modern Python refresher

### Data structures you will use constantly

| Type | Use it for | LLM-world example |
|---|---|---|
| `list` | Ordered items | Chat `messages`, retrieved chunks |
| `dict` | Key → value lookup | A JSON reply, model options |
| `set` | Unique items, fast membership | Allowed tool names |
| `tuple` | Fixed, immutable group | `(score, chunk_id)` pairs |

```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi"},
]
user_turns = [m["content"] for m in messages if m["role"] == "user"]   # list comprehension
by_role = {m["role"]: m["content"] for m in messages}                  # dict comprehension
top3 = sorted([(0.91, "c7"), (0.42, "c2"), (0.88, "c1")], reverse=True)[:3]
```

### Functions with type hints

Type hints document intent and let your editor and tools catch bugs before you run the code.

```python
def build_prompt(question: str, context: list[str], max_chars: int = 4000) -> str:
    joined = "\n---\n".join(context)[:max_chars]
    return f"Answer using only this context:\n{joined}\n\nQuestion: {question}"
```

### Classes and dataclasses

Use a **dataclass** for plain data, a **class** when data comes with behaviour.

```python
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Conversation:
    system_prompt: str
    history: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.history.append(Message(role, content))

    def to_api(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}] + [
            {"role": m.role, "content": m.content} for m in self.history
        ]
```

### Modules and packages

Split code by responsibility as soon as a file passes ~200 lines:

```text
projects/01-llm-api-service/
├── pyproject.toml
├── .env                 # never committed
├── src/llm_service/
│   ├── __init__.py
│   ├── config.py        # settings (1.2)
│   ├── schemas.py       # Pydantic models (1.3)
│   ├── llm.py           # model client + retries (1.4, 1.5)
│   └── api.py           # FastAPI app (1.7)
└── tests/
    └── test_api.py      # pytest (1.8)
```

> **Exercise 1.1** — Extend `Conversation` with a `trim(max_messages: int)` method that keeps the system prompt
> and only the most recent `max_messages` turns. Use it in your Module 0 chat loop.

---

## 1.2 Project hygiene: environments, secrets, JSON

### Secrets live in `.env`, never in code

```dotenv
# .env  (add ".env" to .gitignore!)
MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434
# OPENAI_API_KEY=sk-...      # from Module 3 onward
```

Commit a `.env.example` with the same keys and fake values so others know what to set.

### Typed settings with `pydantic-settings`

Instead of scattering `os.getenv` calls, load all configuration once, with types and defaults:

```python
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_host: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    request_timeout_s: float = 60.0
    openai_api_key: SecretStr | None = None  # optional until Module 3


settings = Settings()
print(settings.model, settings.openai_api_key)  # SecretStr prints as '**********'
```

Environment variables override `.env` values, which override defaults — so `MODEL=qwen2.5:7b uv run ...` just works.
`SecretStr` hides the key in logs and error messages; call `.get_secret_value()` only where you pass it to a client.

### Parsing JSON from LLM output

Models often wrap JSON in prose or ``` fences. Extract, then parse:

```python
import json
import re


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply that may contain extra prose or ``` fences."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in model output")
    return json.loads(match.group(0))


reply = 'Sure! Here is the data:\n```json\n{"city": "Pune", "population_millions": 7.4}\n```'
print(extract_json(reply))  # {'city': 'Pune', 'population_millions': 7.4}
```

This is a fallback. In 1.3 you will make the model return valid JSON in the first place.

> **Exercise 1.2** — Move your Module 0 script's model name and host into a `Settings` class and a `.env` file.

---

## 1.3 Data validation with Pydantic

LLM output is **untrusted input**: it can be missing fields, have wrong types or invent values. Pydantic turns
"some JSON" into a typed object or a clear error.

```python
from pydantic import BaseModel, Field, ValidationError, field_validator


class MovieReview(BaseModel):
    title: str
    rating: int = Field(ge=1, le=5, description="Stars from 1 to 5")
    sentiment: str
    key_points: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("sentiment")
    @classmethod
    def normalise_sentiment(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"positive", "negative", "mixed"}:
            raise ValueError("sentiment must be positive, negative or mixed")
        return v


good = '{"title": "Inception", "rating": 5, "sentiment": " Positive ", "key_points": ["visuals"]}'
review = MovieReview.model_validate_json(good)
print(review.sentiment, review.rating)  # positive 5

bad = '{"title": "Inception", "rating": 9, "sentiment": "great"}'
try:
    MovieReview.model_validate_json(bad)
except ValidationError as e:
    for err in e.errors():
        print(err["loc"], err["msg"])
# ('rating',) Input should be less than or equal to 5
# ('sentiment',) Value error, sentiment must be positive, negative or mixed
```

Useful methods:

| Method | Does |
|---|---|
| `Model.model_validate(dict)` | dict → object (validates) |
| `Model.model_validate_json(str)` | JSON string → object |
| `obj.model_dump()` / `obj.model_dump_json()` | object → dict / JSON |
| `Model.model_json_schema()` | JSON Schema — hand this to the LLM |

### Structured output: make the model follow your schema

Ollama (and OpenAI, Anthropic, Gemini) can constrain generation to a JSON Schema. Pass the Pydantic schema,
then validate the reply with the same model:

```python
import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Review the film Inception."}],
    format=MovieReview.model_json_schema(),   # constrain output to the schema
    options={"temperature": 0},
)
review = MovieReview.model_validate_json(response.message.content)
print(review)
```

Schema-constrained output guarantees **shape**, not **truth** — validators still catch values that are the
right type but wrong (a rating of 0, an unknown label).

> **Exercise 1.3** — Define an `Invoice` model (vendor, date, line items with quantity and price, total).
> Ask a local model to extract it from a pasted invoice text, and add a validator that checks the line items
> add up to the total.

---

## 1.4 Async and concurrency

LLM calls spend almost all their time **waiting** on the network or the model. `async` lets one Python process
wait on many calls at once instead of one after another.

```python
import asyncio
import random
import time


async def fake_llm(prompt: str) -> str:
    await asyncio.sleep(random.uniform(0.5, 1.0))  # stands in for a network call
    return f"answer to {prompt!r}"


async def sequential(prompts):
    return [await fake_llm(p) for p in prompts]


async def concurrent(prompts, max_in_flight: int = 3):
    limit = asyncio.Semaphore(max_in_flight)  # simple rate limit

    async def one(p):
        async with limit:
            return await fake_llm(p)

    return await asyncio.gather(*(one(p) for p in prompts))


async def main():
    prompts = [f"q{i}" for i in range(6)]
    t = time.perf_counter(); await sequential(prompts); print(f"sequential: {time.perf_counter() - t:.1f}s")
    t = time.perf_counter(); await concurrent(prompts); print(f"concurrent: {time.perf_counter() - t:.1f}s")


asyncio.run(main())
# sequential: 4.7s
# concurrent: 1.6s
```

Key ideas:

- `async def` defines a coroutine; `await` pauses it until the result is ready, letting others run.
- `asyncio.gather(...)` runs many coroutines concurrently and returns results **in input order**.
- `asyncio.Semaphore(n)` caps how many run at once — your first rate limiter. Providers reject you (HTTP 429) if you send too much.
- Never call blocking code (`time.sleep`, `requests.get`) inside `async def`; use `asyncio.sleep`, `httpx.AsyncClient`, `ollama.AsyncClient`.

With the real model:

```python
import asyncio
import ollama


async def ask_all(questions: list[str]) -> list[str]:
    client = ollama.AsyncClient()
    responses = await asyncio.gather(
        *(client.chat(model="llama3.2:3b", messages=[{"role": "user", "content": q}]) for q in questions)
    )
    return [r.message.content for r in responses]


print(asyncio.run(ask_all(["What is RAG?", "What is an agent?", "What is MCP?"])))
```

A local model on one laptop often processes requests one at a time, so the speed-up is smaller than with hosted
APIs; `OLLAMA_NUM_PARALLEL` controls how many it serves in parallel. The pattern is what matters — it pays off
fully from Module 3 onward.

> **Exercise 1.4** — Time 5 questions sent sequentially vs with `gather` against your local model and against
> the fake function. Explain the difference in your notes.

---

## 1.5 Robust API calls: errors, retries, timeouts, logging

### Classify failures first

| Kind | Examples | Retry? |
|---|---|---|
| Transient | Timeout, HTTP 429 (rate limit), 500/502/503, connection reset | Yes, with backoff |
| Permanent | 400 bad request, 401 bad key, 404 unknown model, validation error in *your* input | No — fix the request |
| Bad output | Invalid JSON, schema violation | Sometimes: retry once with the error in the prompt |

### Retries with exponential backoff and jitter

```python
import asyncio
import functools
import logging
import random

log = logging.getLogger("llm")


class TransientError(Exception):
    """Worth retrying: timeouts, 429 rate limits, 5xx server errors."""


def retry(max_attempts: int = 4, base_delay: float = 0.5, max_delay: float = 8.0):
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except (TransientError, asyncio.TimeoutError) as exc:
                    if attempt == max_attempts:
                        raise
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))
                    delay *= random.uniform(0.5, 1.5)  # jitter so clients don't retry in lockstep
                    log.warning("attempt %d failed (%s); retrying in %.2fs", attempt, exc, delay)
                    await asyncio.sleep(delay)
        return wrapper
    return decorator


calls = 0

@retry(max_attempts=4, base_delay=0.1)
async def flaky_llm(prompt: str) -> str:
    global calls
    calls += 1
    if calls < 3:
        raise TransientError("503 service unavailable")
    return "ok"


async def main():
    result = await asyncio.wait_for(flaky_llm("hi"), timeout=5)  # overall timeout
    print(result, "after", calls, "calls")  # ok after 3 calls

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
asyncio.run(main())
```

Writing it once yourself teaches the idea. In real projects use the `tenacity` library, which does the same with
less code:

```python
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

@retry(stop=stop_after_attempt(4), wait=wait_random_exponential(multiplier=0.5, max=8),
       retry=retry_if_exception_type((TransientError, TimeoutError)))
async def call_model(prompt: str) -> str: ...
```

### Timeouts

Always set one — a hung request otherwise blocks forever. Two levels:

- **Per request:** `ollama.AsyncClient(timeout=60)`, `httpx.AsyncClient(timeout=30)`.
- **Overall:** `await asyncio.wait_for(coro, timeout=90)` around the whole retry loop.

### Structured logging

Plain `print` cannot be searched or charted. Log **events with fields** as JSON — the same data Langfuse will
collect for you in Module 12.

```python
import json
import logging
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))  # extra structured fields
        return json.dumps(entry)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
log = logging.getLogger("llm")
log.addHandler(handler)
log.setLevel(logging.INFO)

start = time.perf_counter()
# ... call the model ...
log.info("llm_call", extra={"fields": {"model": "llama3.2:3b", "prompt_tokens": 42,
                                       "completion_tokens": 128,
                                       "latency_ms": round((time.perf_counter() - start) * 1000)}})
# {"ts": ..., "level": "INFO", "logger": "llm", "msg": "llm_call", "model": "llama3.2:3b", ...}
```

Log: model, token counts, latency, attempt number, outcome. Never log secrets, and be careful logging full prompts
that may contain personal data (Module 4 covers PII redaction).

> **Exercise 1.5** — Stop Ollama (`ollama stop` or quit the app) while your script runs. Make sure it retries,
> logs each attempt as JSON, then fails with a clear error message instead of a stack trace.

---

## 1.6 Pipeline-oriented coding

AI systems are pipelines: load → clean → chunk → embed → store → retrieve → generate. Write each step as a small
function with clear input and output, then compose them. **Generators** (`yield`) stream items one at a time, so a
pipeline can process a 2 GB file without loading it into memory.

```python
from collections.abc import Iterable, Iterator


def read_lines(text: str) -> Iterator[str]:
    for line in text.splitlines():
        yield line


def clean(lines: Iterable[str]) -> Iterator[str]:
    for line in lines:
        line = line.strip()
        if line:
            yield line


def batch(items: Iterable[str], size: int) -> Iterator[list[str]]:
    current: list[str] = []
    for item in items:
        current.append(item)
        if len(current) == size:
            yield current
            current = []
    if current:
        yield current


doc = "first line\n\n  second line  \nthird\nfourth\nfifth"
for group in batch(clean(read_lines(doc)), size=2):
    print(group)
# ['first line', 'second line']
# ['third', 'fourth']
# ['fifth']
```

Principles:

- **One job per function**, pure where possible (same input → same output, no hidden state) — easy to test.
- **Typed boundaries** — Pydantic models or dataclasses between steps, not loose dicts.
- **Batching** — embedding APIs accept many texts per call; `batch()` above is how you feed them.
- **Idempotent steps** — re-running ingestion should not duplicate data (matters in Module 4).

> **Exercise 1.6** — Add an `embed_batches` step that sends each batch to `ollama.embed(model="nomic-embed-text", input=batch)`
> and yields `(text, vector)` pairs. Run it on a text file of your choice.

---

## 1.7 FastAPI basics

FastAPI turns typed Python functions into an HTTP API, validates requests with Pydantic and generates interactive
docs at `/docs`. It is how you will serve every project in this course.

Worked example — a summarisation service (your Build uses the same pattern for `/ask`):

```python
# app.py
from typing import Protocol

import ollama
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field


class LLM(Protocol):
    async def complete(self, prompt: str) -> str: ...


class OllamaLLM:
    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.client = ollama.AsyncClient(host=host)

    async def complete(self, prompt: str) -> str:
        response = await self.client.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response.message.content


class SummariseRequest(BaseModel):
    text: str = Field(min_length=20, max_length=20_000)
    max_words: int = Field(default=50, ge=10, le=300)


class SummariseResponse(BaseModel):
    summary: str
    word_count: int


app = FastAPI(title="Summariser")


def get_llm() -> LLM:
    return OllamaLLM()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/summarise", response_model=SummariseResponse)
async def summarise(req: SummariseRequest, llm: LLM = Depends(get_llm)) -> SummariseResponse:
    prompt = f"Summarise the text below in at most {req.max_words} words.\n\n{req.text}"
    try:
        summary = (await llm.complete(prompt)).strip()
    except Exception as exc:  # the model server is down, timed out, etc.
        raise HTTPException(status_code=503, detail="LLM unavailable") from exc
    return SummariseResponse(summary=summary, word_count=len(summary.split()))
```

Run it and open <http://127.0.0.1:8000/docs>:

```bash
uv run uvicorn app:app --reload
curl -X POST localhost:8000/summarise -H "Content-Type: application/json" \
     -d '{"text": "FastAPI is a modern Python web framework for building APIs quickly.", "max_words": 15}'
```

What to notice:

- **Request/response models** — a bad request gets an automatic `422` with details; the LLM is never called.
- **Dependency injection** — `Depends(get_llm)` means the endpoint never builds its own client. Tests swap in a fake (1.8); later you swap Ollama for OpenAI by changing one function.
- **`Protocol`** — any object with an async `complete(prompt)` method counts as an `LLM`. This is the seam that makes the provider ladder in the syllabus painless.
- **Error mapping** — infrastructure failures become a clean `503`, not a stack trace.

> **Exercise 1.7** — Add a `GET /models` endpoint that returns the names from `ollama.AsyncClient().list()`, and a
> `stream: bool` request option that returns a `StreamingResponse` of tokens.

---

## 1.8 Testing: pytest, fixtures and mocking

### Testing vs evaluation — two different questions

| | Testing (this module) | Evaluation (Module 3) |
|---|---|---|
| Question | Does my **code** behave correctly? | Is the **model's answer** good? |
| LLM | Replaced by a fake — deterministic | Real model |
| Result | Pass / fail | A score (accuracy, groundedness …) |
| Runs | Every commit, in seconds, offline | On a golden dataset, slower, costs tokens |

Unit tests must never depend on a real model: they would be slow, flaky and (with paid APIs) cost money.

### Testing the FastAPI example with a fake LLM

```python
# test_app.py      run: uv run pytest -q
import pytest
from fastapi.testclient import TestClient

from app import app, get_llm


class FakeLLM:
    def __init__(self, reply: str = "A short summary.", fail: bool = False):
        self.reply, self.fail, self.prompts = reply, fail, []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise ConnectionError("model server down")
        return self.reply


@pytest.fixture
def fake_llm():
    fake = FakeLLM()
    app.dependency_overrides[get_llm] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


LONG_TEXT = "FastAPI is a modern Python web framework for building APIs quickly."


def test_summarise_returns_typed_json(client, fake_llm):
    r = client.post("/summarise", json={"text": LONG_TEXT, "max_words": 20})
    assert r.status_code == 200
    assert r.json() == {"summary": "A short summary.", "word_count": 3}
    assert "at most 20 words" in fake_llm.prompts[0]


def test_rejects_too_short_text(client, fake_llm):
    r = client.post("/summarise", json={"text": "too short"})
    assert r.status_code == 422          # Pydantic validation error
    assert fake_llm.prompts == []        # the LLM was never called


def test_llm_failure_returns_503(client, fake_llm):
    fake_llm.fail = True
    r = client.post("/summarise", json={"text": LONG_TEXT})
    assert r.status_code == 503
```

```text
$ uv run pytest -q
...                                                                      [100%]
3 passed in 0.64s
```

### Concepts

- **Fixture** — reusable setup (`@pytest.fixture`); pytest passes it to any test that names it as an argument. Code after `yield` is cleanup.
- **Fake** — a small hand-written stand-in (like `FakeLLM`) that records calls and can simulate failures. Prefer fakes for your own interfaces.
- **Mock** — `unittest.mock` objects created on the fly; handy for patching third-party code:

  ```python
  from unittest.mock import AsyncMock, patch

  from app import OllamaLLM

  async def test_ollama_llm_uses_configured_model():
      with patch("app.ollama.AsyncClient") as client_cls:
          client_cls.return_value.chat = AsyncMock(
              return_value=type("R", (), {"message": type("M", (), {"content": "hi"})()})()
          )
          llm = OllamaLLM(model="qwen2.5:7b")
          assert await llm.complete("hello") == "hi"
          assert client_cls.return_value.chat.call_args.kwargs["model"] == "qwen2.5:7b"
  ```

  (async tests need `pytest-asyncio`: add `asyncio_mode = "auto"` under `[tool.pytest.ini_options]` in `pyproject.toml`.)
- **What to test** — validation (bad input rejected), the prompt you build, error paths (timeouts, retries, 503s), parsing of model output. Not the model's intelligence — that is evaluation.

> **Exercise 1.8** — Add tests for your retry logic: a fake that fails twice then succeeds should be called
> three times; one that always fails should raise after `max_attempts`.

---

## Build — Tested LLM API service

In `projects/01-llm-api-service`, build a FastAPI service with:

1. `POST /ask` taking `{"question": str}` and returning a typed response:
   `{"answer": str, "model": str, "latency_ms": int, "attempts": int}`.
2. Settings (model, host, timeout) loaded from `.env` via `pydantic-settings` (1.2).
3. The model asked for **structured output** with a Pydantic schema, validated on return (1.3).
4. An async Ollama client with a per-request timeout and retries with backoff on transient errors (1.4–1.5).
5. JSON structured logs for every model call: model, tokens, latency, attempts, outcome (1.5).
6. `GET /health`.
7. A pytest suite using a fake LLM: happy path, invalid input (422), model down (503), retry then success (1.8).

## Checkpoint

- [ ] `uv run pytest` passes **with Ollama switched off and no network** — proof that tests do not touch the real model.
- [ ] `curl` against the running service returns typed JSON from the real local model.
- [ ] Changing `MODEL` in `.env` switches the model without code changes.
- [ ] Logs show one JSON line per model call with latency and token counts.
- [ ] You can explain: why async helps, which errors to retry, and the difference between testing and evaluation.

## Key terms

| Term | Meaning |
|---|---|
| Type hint | Annotation of expected types (`def f(x: int) -> str`) checked by tools, not at runtime |
| Pydantic model | Class that validates and converts data into typed fields |
| JSON Schema | Standard description of a JSON shape; used for structured LLM output |
| Coroutine | An `async def` function; runs when awaited |
| Semaphore | Counter that limits how many tasks run at once |
| Exponential backoff | Waiting 0.5 s, 1 s, 2 s, 4 s … between retries |
| Jitter | Random variation added to backoff so clients don't retry simultaneously |
| Dependency injection | Passing collaborators (like the LLM client) in rather than creating them inside |
| Fixture | Reusable test setup in pytest |
| Fake / mock | Test stand-ins for real dependencies |

## Further reading

- Pydantic docs — <https://docs.pydantic.dev/latest/>
- pydantic-settings — <https://docs.pydantic.dev/latest/concepts/pydantic_settings/>
- Python `asyncio` — <https://docs.python.org/3/library/asyncio.html>
- FastAPI tutorial — <https://fastapi.tiangolo.com/tutorial/>
- FastAPI testing and dependency overrides — <https://fastapi.tiangolo.com/advanced/testing-dependencies/>
- pytest — <https://docs.pytest.org/>
- Ollama structured outputs — <https://ollama.com/blog/structured-outputs>
