# What These Notes Don't Cover — and Where to Learn It

These notes teach each topic to a working level with code. The items below are **not covered, only touched, or
need hands-on verification**. Study them from the resources listed. Resource links were accurate at time of
writing; if one moves, search the title.

## 1. Things to verify while implementing

Most model-calling code in `notes/` was not executed (no model in the authoring environment). Fast-moving libraries
most likely to need fixes:

| Library | Why it may break | Check |
|---|---|---|
| RAGAS | Metric/import names change between versions | <https://docs.ragas.io> |
| Langfuse SDK | v2 → v3 API change | <https://langfuse.com/docs> |
| LangGraph | `interrupt`, streaming modes evolve | <https://langchain-ai.github.io/langgraph/> |
| CrewAI | Frequent releases | <https://docs.crewai.com> |
| Unsloth / TRL | `SFTTrainer`/`SFTConfig` arguments change | <https://docs.unsloth.ai>, <https://huggingface.co/docs/trl> |
| MCP Python SDK | Transports and client API evolving | <https://modelcontextprotocol.io>, <https://github.com/modelcontextprotocol/python-sdk> |
| Guardrails AI / NeMo Guardrails | Hub validators, config format | <https://www.guardrailsai.com/docs>, <https://docs.nvidia.com/nemo/guardrails/> |
| Ollama model tags | New models, renamed tags | <https://ollama.com/library> |
| Hosted model names/prices | Change every few months | Each provider's models & pricing page |

## 2. Foundations we only skimmed

| Topic | Why it matters | Resources |
|---|---|---|
| Machine-learning basics (training, loss, gradient descent, overfitting) | Understand fine-tuning and evals deeply | Andrew Ng, *Machine Learning Specialization* (Coursera); fast.ai *Practical Deep Learning* (free) |
| Neural networks & backpropagation | What training actually does | 3Blue1Brown *Neural Networks* (YouTube); Karpathy *Neural Networks: Zero to Hero* (free) |
| Linear algebra & probability | Embeddings, attention, sampling | 3Blue1Brown *Essence of Linear Algebra*; Khan Academy probability |
| Building a GPT from scratch | Real intuition for transformers | Karpathy *Let's build GPT*; Sebastian Raschka, *Build a Large Language Model (From Scratch)* (book) |
| PyTorch | Needed to go beyond Unsloth recipes | pytorch.org tutorials; Hugging Face *LLM Course* (free) |

## 3. Topics not covered (or only mentioned)

### Models and training
- **Pre-training and continued pre-training**, tokenizer training — HF *LLM Course*; Raschka's book.
- **Preference tuning: RLHF, DPO, ORPO, GRPO** — HF TRL docs; *DPO* paper (Rafailov et al., 2023).
- **Distillation** (elective) — Unsloth/TRL docs; "knowledge distillation" in HF blog.
- **Mixture-of-Experts, long-context techniques, KV-cache, speculative decoding** — Chip Huyen, *AI Engineering* (book); vLLM docs.
- **Inference optimisation at scale** (vLLM, TensorRT-LLM, batching, GPU sizing) — <https://docs.vllm.ai>.

### RAG and data
- **GraphRAG** implementation — Microsoft GraphRAG <https://microsoft.github.io/graphrag/>; LightRAG.
- **Multimodal RAG** (images, charts, tables with vision models, ColPali) — LlamaIndex multimodal docs; ColPali paper.
- **Embedding fine-tuning** for your domain — Sentence Transformers docs <https://sbert.net>.
- **Search at scale**: sharding, replication, quantised vectors, Elasticsearch/OpenSearch hybrid — Qdrant & OpenSearch docs.
- **Data engineering for AI**: scheduling ingestion (Airflow/Prefect), CDC, data quality — *Designing Data-Intensive Applications* (Kleppmann).

### Agents
- **OpenAI Agents SDK, Claude Agent SDK, Google ADK** hands-on — each vendor's docs.
- **A2A protocol** implementation — <https://a2a-protocol.org>.
- **Computer-use / browser agents** (elective) — Anthropic computer-use docs; Playwright + browser-use.
- **Voice agents** (elective): speech-to-text (Whisper), TTS, real-time pipelines — LiveKit Agents, Pipecat docs.
- **Long-running/background agents, scheduling, durable execution** — LangGraph Platform docs; Temporal.
- **Agent design patterns** — Anthropic, *Building effective agents* (blog); *Agentic Design Patterns* resources.

### Security, governance, responsible AI
- **Full DPDP/GDPR legal compliance** — notes are simplified, **not legal advice**. Read the DPDP Act 2023 and DPDP Rules 2025 (MeitY site); consult legal counsel.
- **EU AI Act, ISO/IEC 42001, NIST AI RMF** — official texts; NIST AI Risk Management Framework site.
- **Bias, fairness, explainability** — Google *Responsible AI Practices*; Hugging Face ethics resources.
- **Advanced red-teaming** (garak, PyRIT, jailbreak research) — OWASP GenAI project <https://genai.owasp.org>; garak, PyRIT repos.
- **Authentication/authorisation for MCP** (OAuth) and remote MCP hardening — MCP spec authorization section.

### Production engineering
- **Kubernetes, autoscaling, GPU clusters** — Kubernetes docs; KServe.
- **Cloud AI platforms** (elective): AWS Bedrock, Azure AI Foundry, Google Vertex AI — each has free learning paths.
- **Frontend engineering** beyond Streamlit (React/Next.js chat UIs, Vercel AI SDK).
- **Cost/FinOps at scale, A/B testing prompts and models in production, online evaluation** — Langfuse docs; Eugene Yan's blog.
- **Databases for AI apps** (Postgres + pgvector, migrations) — pgvector README.

## 4. Books and courses for the whole journey

| Resource | Type | Covers |
|---|---|---|
| Chip Huyen, *AI Engineering* (O'Reilly, 2025) | Book | The best single overview of this whole syllabus |
| Jay Alammar & Maarten Grootendorst, *Hands-On Large Language Models* | Book | Tokens, embeddings, RAG, fine-tuning with code |
| Hugging Face *LLM Course* and *Agents Course* | Free courses | Transformers, fine-tuning, smolagents, LangGraph |
| DeepLearning.AI short courses | Free courses | LangGraph, RAG evaluation, MCP, agent memory, many more |
| Anthropic docs & *Building effective agents* | Docs/blog | Prompting, tool use, agent patterns, MCP |
| LangChain Academy (Intro to LangGraph) | Free course | LangGraph in depth |
| Hamel Husain & Shreya Shankar — evals writing | Blog/course | Practical LLM evaluation |
| OWASP Top 10 for LLM Applications | Standard | Security risks and mitigations |
| Papers: *Attention Is All You Need*, *RAG* (Lewis 2020), *ReAct* (Yao 2022), *LoRA* (Hu 2021), *QLoRA* (Dettmers 2023) | Papers | Original ideas behind each module |

## 5. Skills to practise outside these notes

- Reading papers and changelogs quickly
- Debugging with traces instead of print statements
- Writing clear design docs and decision logs
- Explaining trade-offs to non-technical stakeholders
- Contributing a small fix to an open-source AI library (great portfolio signal)
