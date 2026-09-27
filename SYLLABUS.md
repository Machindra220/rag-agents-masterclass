# AI Agents & RAG — Brick-by-Brick Learning Syllabus

A self-study path to building production-grade RAG and Agentic AI systems, modelled on the
[IIT Roorkee PG Certificate in Agentic and RAG Systems for AI Engineering](https://ihub-iitroorkee.emeritus.org/iitr-post-graduate-certificate-in-agentic-and-rag-systems-for-ai-engineering)
(iHUB DivyaSampark with Emeritus, Batch 2).

Living version (editable, with progress tracker): <https://claude.ai/code/artifact/274efa6c-5971-4df1-8624-c27d3571b517>

## How to use this syllabus

We learn in 14 modules (a setup module plus the programme's 13) across 4 levels and 6 phases.
That takes 1 setup week plus 24 weeks at 8–10 hours a week, and every phase ends in a working project in this repo.

- **One brick at a time.** Each module splits into numbered bricks (1.1, 1.2 …). Finish one brick, run its exercise, then move on.
- **Build, then measure.** Every phase ends with a project; we move on only once its checkpoint passes.
- **Everything compounds.** The Python service from Phase 1 becomes the RAG system in Phase 3, the agent in Phase 5 and the deployed capstone in Phase 6.
- **Local models first.** Every brick runs on free local models through Ollama first; paid and hosted providers are added one at a time (see *Model strategy*).
- **Free tools only**, as in the programme; paid tiers are optional.
- **Suggested weekly rhythm:** ~3 h concepts, ~5 h coding, ~1 h notes on what worked and what broke.
- **Hardware:** 4+ cores, 16 GB RAM recommended; fine-tuning (Module 7) uses a free Colab GPU.
- Bricks marked **[Added]** are not in the programme brochure or website; they fill gaps for self-study.

## Model strategy: local first, then one provider at a time

| Stage | When | Provider / runtime | What you learn |
|---|---|---|---|
| 1 | Module 0 onward | **Ollama** (Llama 3.x, Qwen, Mistral, Phi, Gemma; `nomic-embed-text` for embeddings) | Free, private, offline LLMs; OpenAI-compatible local API |
| 2 | Module 2 | **LM Studio / llama.cpp** | GGUF files, quantization levels, CPU vs GPU speed |
| 3 | Module 2 | **Hugging Face** (Transformers, Inference API) | Open-model ecosystem, model cards, licences |
| 4 | Module 2–3 | **Groq / OpenRouter** free tiers | Hosted open models, rate limits, speed |
| 5 | Module 3 | **OpenAI** (GPT) | Proprietary API, structured outputs, function calling |
| 6 | Module 3 | **Anthropic** (Claude) | Tool use, long context, prompt caching |
| 7 | Module 3 | **Google** (Gemini) | Multimodal input, very long context |
| 8 | Module 12 | **LiteLLM** router over all of the above | One interface, model routing, fallbacks, cost tracking |

Each new provider gets the same benchmark task from Module 2 so results are comparable.

## Roadmap at a glance

| Level | Phase | Weeks | Modules | Project (programme week) |
|---|---|---|---|---|
| — | 0 Setup and Local LLMs **[Added]** | 0 | 0 | Hello local LLM |
| 1 Understand | 1 AI Engineering Foundations | 1–3 | 1 | Tested LLM API service |
| 1 Understand | 2 Foundation Models and LLM Engineering | 4–6 | 2–3 | Project 1 Prompt Lab (week 6) |
| 2 Build | 3 Enterprise RAG and Retrieval | 7–12 | 4–5 | Project 2 Mini-RAG (week 9), Project 3 Grounded Q&A (week 12) |
| 3 Adapt and Act | 4 Model Adaptation and Fine-Tuning | 13–15 | 6–7 | Project 4 Specialise a Model (week 14) |
| 3 Adapt and Act | 5 Agentic AI and Autonomous Workflows | 16–20 | 8–10 | Project 5 Tool-Using Agent (week 20) |
| 4 Ship | 6 Infrastructure, Security, Governance, Deployment | 21–24 | 11–13 | Project 6 Connected Assistant (week 21), Capstone (week 24) |

## Modules, brick by brick

### Phase 0 — Setup and Local LLMs (week 0) [Added]

#### Module 0: Environment and Local Models

- **0.1 Dev environment:** Python 3.11+, `uv`, Git and GitHub, VS Code, Jupyter.
- **0.2 Ollama:** install, pull and chat with small models (Llama 3.2 3B, Qwen 2.5 7B, Phi, Mistral).
- **0.3 Hardware sizing:** which model sizes fit 8 GB vs 16 GB RAM; GGUF and 4-bit/8-bit quantization basics.
- **0.4 Calling a local model from Python:** the Ollama API and its OpenAI-compatible endpoint; streaming.
- **0.5 Local embeddings:** `nomic-embed-text` via Ollama.
- **0.6 Working with AI coding assistants** responsibly.

**Build:** a Python script that chats with a local model, streams tokens and prints tokens per second.

**Checkpoint:** the same script works against two different local models by changing one setting.

### Phase 1 — AI Engineering Foundations (weeks 1–3) · Level 1 Understand

#### Module 1: Python for AI Engineering

- **1.1 Modern Python refresher:** syntax, modules and packages, data structures, functions, OOP (classes, dataclasses).
- **1.2 Project hygiene:** virtual environments, `uv`/`pip`, `.env` files, secrets management, JSON parsing.
- **1.3 Data validation with Pydantic:** models, validators, parsing LLM JSON output into typed objects.
- **1.4 Async and concurrency:** `async`/`await`, `asyncio.gather`, rate limits, calling several APIs at once.
- **1.5 Robust API calls:** error handling, retries with backoff, timeouts, structured logging.
- **1.6 Pipeline-oriented coding:** composable steps, generators, small reusable functions.
- **1.7 FastAPI basics:** routes, request/response models, dependency injection, running with Uvicorn.
- **1.8 Testing:** pytest, fixtures, mocking an LLM call; testing (deterministic) vs evaluation (quality).

**Build:** a FastAPI service with one `/ask` endpoint that calls the local Ollama model, validates the reply with Pydantic, retries on failure and has pytest tests with a mocked LLM.

**Checkpoint:** tests pass without network access; the endpoint returns typed JSON.

### Phase 2 — Foundation Models and LLM Engineering (weeks 4–6) · Level 1 Understand

#### Module 2: LLM Architecture and Model Ecosystem

- **2.1 Math essentials [Added]:** vectors, dot product and cosine similarity, softmax, probability; temperature and top-p sampling.
- **2.2 How text becomes numbers:** tokenization, embeddings, positional encoding.
- **2.3 The Transformer:** self-attention, multi-head attention, decoder-only models, next-token prediction.
- **2.4 Context windows:** limits, overflow behaviour, long-context trade-offs.
- **2.5 The model ecosystem:** GPT, Claude, Gemini, Llama, Mistral, Qwen; open-source vs proprietary; licences.
- **2.6 Reasoning models and small language models [Added]:** when "thinking" models help, when a 3B model is enough.
- **2.7 Structured outputs and streaming:** JSON mode and schemas, streaming tokens to a client.
- **2.8 Cost, latency, quality trade-offs:** token pricing, measuring latency, choosing a model per use case.

**Build:** a benchmark script that sends the same task to several local models (then each hosted provider as it is added) and logs tokens, cost, latency and output.

**Checkpoint:** you can explain attention in your own words and defend a model choice with numbers.

#### Module 3: Prompt Engineering, Evaluation and Reliability

- **3.1 Core prompting:** zero-shot, few-shot, system prompts, role and format instructions.
- **3.2 Reasoning prompts:** chain-of-thought, tool-aware prompting, retrieval-aware prompting.
- **3.3 Safety and hallucination:** safety prompts, grounding instructions, "say I don't know".
- **3.4 Prompts as code:** templates, versioning, keeping prompts separate from logic.
- **3.5 Evaluating LLMs:** golden datasets, LLM-as-judge, pairwise comparison, critic–creator loops.
- **3.6 Eval tooling:** RAGAS and DeepEval basics, reliability testing across repeated runs.
- **3.7 Adding hosted providers [Added]:** OpenAI, Anthropic and Gemini keys, prompt caching, comparing against local models.
- **3.8 Programmatic prompt optimisation [Added, optional]:** DSPy basics.

**Build — Project 1, Prompt Lab (programme week 6):** solve one task three ways (zero-shot, few-shot, structured chain-of-thought), score each against a golden dataset with an LLM judge, and write up the winner; compare a local open model with a proprietary one.

**Checkpoint:** a notebook or CLI with scores per strategy and a short written comparison.

### Phase 3 — Enterprise RAG and Retrieval Infrastructure (weeks 7–12) · Level 2 Build

#### Module 4: Retrieval Architecture and Vector Infrastructure

- **4.1 Why RAG:** retrieve-then-generate, where it beats fine-tuning, the basic pipeline.
- **4.2 Embedding models:** OpenAI, BGE, Nomic; semantic similarity; local vs hosted embeddings.
- **4.3 Document parsing [Added]:** PDFs, tables, scanned pages and OCR (PyMuPDF, Docling, Unstructured).
- **4.4 Chunking strategies:** fixed-size, recursive, semantic, overlap; how chunk size changes answers.
- **4.5 Vector databases:** FAISS, Chroma, Qdrant; HNSW and IVF indexing.
- **4.6 Ingestion pipelines:** loaders, metadata enrichment, re-indexing.
- **4.7 Hybrid retrieval basics:** keyword + vector search with metadata filters.
- **4.8 Knowledge lifecycle and privacy:** updating and deleting documents, PII detection and redaction.
- **4.9 Frameworks [Added]:** LangChain and LlamaIndex basics, and when plain Python is clearer.

**Build — Project 2, Mini-RAG (programme week 9):** load, chunk, embed and store a corpus you choose in a vector database, then answer questions grounded in the retrieved context — fully local first.

**Checkpoint:** end-to-end pipeline from raw documents to grounded answers.

#### Module 5: Production RAG and Retrieval Optimisation

- **5.1 Sparse vs dense:** BM25 and dense retrieval, reciprocal rank fusion.
- **5.2 Reranking:** cross-encoder rerankers and when they pay off.
- **5.3 Query transformation:** query rewriting, multi-query retrieval, HyDE.
- **5.4 Efficiency:** semantic caching, context compression.
- **5.5 Measuring RAG:** hit rate, precision@K, groundedness with RAGAS, DeepEval, TruLens.
- **5.6 Debugging RAG:** failure taxonomy — retrieval vs generation vs prompt — and tracing each.
- **5.7 Advanced patterns [Added]:** parent–child chunks, contextual retrieval, agentic RAG, GraphRAG, multimodal RAG; long context vs RAG.

**Build — Project 3, Grounded Q&A (programme week 12):** extend Mini-RAG with hybrid retrieval and a reranker, inline citations, and abstention when the answer is not supported.

**Checkpoint:** a retrieval quality report (hit rate, precision@K) on a small question set, before and after each improvement.

### Phase 4 — Model Adaptation and Fine-Tuning (weeks 13–15) · Level 3 Adapt and Act

#### Module 6: Fine-Tuning Strategy and Adaptation Frameworks

- **6.1 Prompting vs RAG vs fine-tuning:** a decision framework by use case.
- **6.2 Datasets:** curation, instruction-tuning formats, synthetic data generation, domain adaptation.
- **6.3 Baselines:** evaluate the base model first; model selection.
- **6.4 Cost, performance and risk:** when fine-tuning is the wrong choice and what to use instead.

#### Module 7: Parameter-Efficient Model Adaptation

- **7.1 PEFT concepts:** LoRA, QLoRA, adapters, quantization.
- **7.2 Tooling:** Hugging Face Transformers, PEFT, TRL, bitsandbytes, Unsloth or Axolotl.
- **7.3 Training runs:** dataset formatting, hyperparameters, GPU/Colab workflow.
- **7.4 Evaluate and serve:** tuned vs base comparison, catastrophic forgetting, serving as an API.
- **7.5 Back to local [Added]:** export the tuned model to GGUF and run it in Ollama; vLLM and llama.cpp serving.

**Build — Project 4, Specialise a Model (programme week 14):** QLoRA fine-tune a small open model (Llama, Mistral or Phi class) for one narrow task and compare it with the base model.

**Checkpoint:** a results summary plus a decision log on when fine-tuning was and was not the right call.

### Phase 5 — Agentic AI and Autonomous Workflows (weeks 16–20) · Level 3 Adapt and Act

#### Module 8: Agent Architecture, Tooling and Memory Systems

- **8.1 What an agent is:** LLM + tools + loop; ReAct (reason, act, observe).
- **8.2 Tool and function calling:** tool schemas, parsing calls, API orchestration; tool calling with local models.
- **8.3 Planning loops:** plan-then-execute, stopping conditions.
- **8.4 Memory:** short-term conversation, long-term vector memory, retrieval-as-a-tool.
- **8.5 Grounded and data agents:** agents over your RAG system, text-to-SQL agents.
- **8.6 Failure-oriented design:** transient, permanent, silent, cascading and adversarial failures.
- **8.7 Guardrails in the loop:** max-iteration guards, circuit breakers, prompt-injection defence.
- **8.8 Framework tour:** CrewAI and Smolagents (listed on the programme website); OpenAI Agents SDK and Claude Agent SDK **[Added]**.

**Build — Project 5, Tool-Using Agent (programme week 20):** a single agent that plans and uses two or three tools (document search, calculator, SQL lookup) with a visible reasoning trace, iteration guards and a failure fallback.

**Checkpoint:** the agent is deployed and callable, and recovers cleanly when a tool fails.

#### Module 9: Agent Orchestration and Human Oversight

- **9.1 LangGraph fundamentals:** nodes, edges, state, state-machine orchestration, checkpointing.
- **9.2 Planner–executor systems:** splitting planning from doing.
- **9.3 Reflection and self-correction:** critique loops that improve an answer.
- **9.4 Human-in-the-loop:** approval gates, escalation, interrupts, audit trails.
- **9.5 Streaming agent UX:** streaming tool calls and reasoning, interruptibility.
- **9.6 Evaluating agents [Added]:** task success, tool-call accuracy, trajectory evaluation.

**Build:** rebuild the Project 5 agent in LangGraph with an approval gate before any risky tool call.

**Checkpoint:** a run pauses for human approval and resumes from saved state.

#### Module 10: Multi-Agent Coordination Systems

- **10.1 Architectures:** supervisor–worker, delegation, planner–worker–reviewer crews.
- **10.2 Shared state and communication:** handoffs, message passing.
- **10.3 Coordination failures:** loops, conflicting outputs, runaway cost.
- **10.4 Cost and latency:** distributed workflows, when parallel agents help.
- **10.5 When not to use multi-agent:** one well-built agent is often the smarter call.
- **10.6 Agent-to-agent protocols [Added]:** A2A basics and how it relates to MCP.

**Build:** optional extension of Project 5 into a planner–worker–reviewer crew.

**Checkpoint:** a written comparison of single-agent vs multi-agent cost and quality on the same task.

### Phase 6 — AI Infrastructure, Security, Governance and Deployment (weeks 21–24) · Level 4 Ship

#### Module 11: MCP Protocols, AI Security and Governance

- **11.1 Model Context Protocol:** hosts, clients, servers; tools and resources; stdio and SSE transports.
- **11.2 Building an MCP server:** exposing one or two tools; connecting it to an assistant.
- **11.3 Threats:** prompt injection, retrieval poisoning, tool abuse, data exfiltration.
- **11.4 Guardrails:** Guardrails AI, NeMo Guardrails, allow-lists, input validation, permission management.
- **11.5 Governance and compliance:** GDPR and India's DPDP Act, audit trails, enterprise AI governance.
- **11.6 OWASP Top 10 for LLM apps and red-teaming [Added]:** attack your own system (e.g. promptfoo).

**Build — Project 6, Connected Assistant (programme week 21):** connect an assistant to an external data source through a custom or existing MCP server, with one safety guard on the exposed tool.

**Checkpoint:** a live, demonstrable MCP-connected assistant that rejects a disallowed tool call.

#### Module 12: AI Deployment, Observability and Runtime Operations

- **12.1 Observability:** logging, tracing and metrics with Langfuse; distributed tracing; token and cost tracking.
- **12.2 Packaging:** Docker images, FastAPI deployment, environment configuration.
- **12.3 CI/CD for AI:** GitHub Actions, regression tests, eval suites as a gate.
- **12.4 Runtime reliability:** prompt versioning, model routing (LiteLLM), fallbacks, semantic caching for cost.
- **12.5 Demo UI [Added]:** Streamlit, Gradio or Chainlit front end.
- **12.6 Hosting [Added]:** deploying to a free-tier cloud host; secrets in production; load testing and latency budgets.

**Build:** containerise the Grounded Q&A service, add Langfuse tracing and a CI pipeline that runs tests and evals.

**Checkpoint:** a dashboard showing latency, cost and quality per request.

#### Module 13: Capstone Project

- **13.1 Architecture design** of the end-to-end system.
- **13.2 RAG integration** — Milestone 1: RAG system checkpoint.
- **13.3 Agent workflow** — Milestone 2: agent integration checkpoint.
- **13.4 Fine-tuned model integration** and deployment pipeline.
- **13.5 Final build, evaluation and demo** — Milestone 3.
- **13.6 Portfolio and career [Added]:** README and architecture write-up, demo video, resume and LinkedIn updates.

### Electives after the capstone [Added]

- Voice agents (speech-to-text, text-to-speech, real-time pipelines).
- Browser and computer-use agents.
- Distilling a large model into a small one.
- Cloud AI platforms: AWS Bedrock, Azure AI Foundry, Google Vertex AI.

## Capstone, tools and progress

The capstone is one GitHub repository holding a deployed, monitored system that combines RAG, an agent and a fine-tuned model.

### Capstone deliverables

- Production RAG pipeline with hybrid search and reranking.
- LangGraph agent with tool calling and failure handling.
- FastAPI deployment with a live, callable endpoint.
- Langfuse dashboard tracking latency, cost and answer quality.
- A demo you can walk a hiring manager through.

### Tool stack by phase

| Phase | Programme tools | Added for self-study |
|---|---|---|
| 0 Setup | — | Ollama, LM Studio, llama.cpp, uv, Git/GitHub, Jupyter |
| 1 Foundations | Python, Pydantic, asyncio, FastAPI, pytest | httpx |
| 2 LLMs and evaluation | GPT, Llama, Mistral; RAGAS, DeepEval, LLM-as-judge | Hugging Face, Groq, OpenRouter, Anthropic, Gemini, DSPy |
| 3 RAG | OpenAI/BGE/Nomic embeddings, FAISS, Chroma, Qdrant, BM25, cross-encoders, TruLens | Docling, Unstructured, LangChain, LlamaIndex |
| 4 Fine-tuning | Hugging Face PEFT, TRL, bitsandbytes, Unsloth, Axolotl | vLLM, GGUF export |
| 5 Agents | LangGraph, CrewAI, Smolagents | OpenAI Agents SDK, Claude Agent SDK, A2A |
| 6 Production | MCP, Guardrails AI, NeMo Guardrails, Langfuse, Docker, GitHub Actions | LiteLLM, promptfoo, Streamlit/Gradio |

### Progress tracker

| Module | Phase | Project | Status |
|---|---|---|---|
| 0 Environment and Local Models | 0 | Hello local LLM | Not started |
| 1 Python for AI Engineering | 1 | LLM API service | Not started |
| 2 LLM Architecture and Ecosystem | 2 | Model benchmark script | Not started |
| 3 Prompting, Evaluation, Reliability | 2 | Project 1 Prompt Lab | Not started |
| 4 Retrieval and Vector Infrastructure | 3 | Project 2 Mini-RAG | Not started |
| 5 Production RAG Optimisation | 3 | Project 3 Grounded Q&A | Not started |
| 6 Fine-Tuning Strategy | 4 | Decision log | Not started |
| 7 Parameter-Efficient Adaptation | 4 | Project 4 Specialise a Model | Not started |
| 8 Agents, Tools, Memory | 5 | Project 5 Tool-Using Agent | Not started |
| 9 Orchestration and Human Oversight | 5 | LangGraph agent with approval gate | Not started |
| 10 Multi-Agent Coordination | 5 | Agent crew comparison | Not started |
| 11 MCP, Security, Governance | 6 | Project 6 Connected Assistant | Not started |
| 12 Deployment and Observability | 6 | Dockerised, traced service with CI | Not started |
| 13 Capstone | 6 | End-to-end production system | Not started |

## Sources

- Programme brochure: *Post Graduate Certificate in Agentic and RAG Systems for AI Engineering*, iHUB DivyaSampark, IIT Roorkee (Batch 2, B2C edition).
- [Programme website](https://ihub-iitroorkee.emeritus.org/iitr-post-graduate-certificate-in-agentic-and-rag-systems-for-ai-engineering) (saved copy): 4 learning levels, 24-week schedule starting 22 December 2026, project weeks, tool list.
- Brick splits, weekly pacing, the local-first model strategy and all **[Added]** items are suggestions for self-study.
