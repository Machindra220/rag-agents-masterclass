# IIT Roorkee (iHUB DivyaSampark) vs Crio.Do — AI Engineering Syllabus Comparison

**Verdict:** for **maximum topics and maximum hands-on learning, Crio's curriculum is broader and more
project-heavy**. It has 17 assessed builds against 6 projects + a capstone. It adds coding agents, voice agents,
hands-on AWS, computer vision, data engineering and a full interview sprint. **DivyaSampark's advantages are the
IIT-backed credential, IIT faculty masterclasses, and more explicit coverage of transformer internals, Indian DPDP
compliance and named guardrail frameworks.** The best self-study plan is the existing [SYLLABUS.md](SYLLABUS.md)
plus the Crio-only topics listed in section 2. See section 5.

Sources: DivyaSampark brochure and website (Batch 2, saved copies); Crio "Detailed Curriculum & Projects" page
(saved copy). Comparisons are based only on what each published curriculum states. Teaching quality,
mentor quality and outcomes cannot be judged from a syllabus.

---

## 1. Side by side

| | IIT Roorkee / iHUB DivyaSampark | Crio.Do Fellowship |
|---|---|---|
| Duration | 24 weeks (6 months) | 24 weeks of sprints + 2-week interview sprint |
| Live teaching | 2 sessions/week × 3 h (~48 sessions) by industry programme leaders + ~12 h IIT faculty masterclasses | 60+ live sessions with industry mentors |
| Weekly effort | 8–10 h | ~10 h |
| Structure | 13 modules in 6 pillars / 4 levels | 6 four-week sprints + interview sprint |
| Hands-on | 15+ exercises, 6 module projects, 1 capstone | 17 builds (10 mini, 6 guided, 1 capstone) + 4 optional builds; ~120 build hours |
| Assessment | Rubric-graded projects; pass = 50% projects, 50% capstone, 50% attendance | Every build deployed and auto-graded against a rubric in CI |
| Cloud | Not cloud-specific (Docker, FastAPI, GitHub Actions) | AWS hands-on (Bedrock, AgentCore, SageMaker); Azure/GCP conceptual |
| Tools/credits | Free tools; paid versions not provided | API credits, AWS sandbox and lab kit included |
| Credential | e-Certificate from iHUB DivyaSampark, IIT Roorkee | Crio programme certificate (no university credential stated) |
| Extras | Optional 2-day IIT Roorkee Noida campus immersion; recorded career sessions | Interview sprint, résumé/LinkedIn review, 5 referral interviews; optional DSA and System Design tracks |
| Fee | ₹1,20,000 + GST (website showed ₹1,08,000 offer) | Not stated on the curriculum page |

### Coverage scorecard

● = taught in depth with a project · ◐ = taught or mentioned · ○ = absent

| Area | DivyaSampark | Crio |
|---|---|---|
| Python for production AI | ● (full module) | ◐ (pre-work + prerequisites) |
| Transformer / LLM internals | ● | ◐ |
| ML/DL fundamentals, computer vision | ○ | ● (refresher + SnapClassify) |
| Prompt engineering, structured outputs | ● | ● |
| Context engineering | ○ | ◐ |
| Evaluation (LLM-judge, golden sets, CI gates) | ● | ● (dedicated EvalLab + evals in every build) |
| RAG: embeddings, chunking, vector DBs | ● | ● |
| Advanced retrieval (hybrid, rerank, HyDE, caching, compression) | ● | ◐ (hybrid, rerank, rewriting; no HyDE/compression named) |
| Knowledge graphs / semantic layer over SQL | ○ | ◐ |
| Data engineering for AI (pipelines, incremental sync) | ○ | ◐ |
| Real data connectors (Notion, Google Drive, helpdesk) | ○ | ● |
| Fine-tuning (LoRA/QLoRA) | ● (2 modules) | ● |
| DPO / RL post-training, distillation, model hub | ○ | ◐ |
| Agents: tool calling, ReAct, memory, failure design | ● | ● |
| LangGraph orchestration, HITL | ● | ● |
| Multi-agent systems | ● | ● |
| A2A / ACP protocols | ○ | ◐ |
| MCP | ● | ● (multi-tenant, RBAC, OAuth) |
| Security, guardrails, prompt injection | ● | ● |
| Indian DPDP / GDPR governance | ● (explicit) | ◐ ("compliance literacy") |
| Observability (Langfuse) | ● | ● |
| Drift detection, incident fire-drills, runbooks, rollbacks | ○ | ● |
| Cost engineering (routing, caching, token budgets) | ◐ | ● (RouteCache) |
| Cloud deployment (AWS Bedrock/SageMaker) | ○ | ● |
| Voice / real-time agents | ○ | ● (VoiceMate) |
| Coding agents (Claude Code, Cursor, AGENTS.md, hooks) | ○ | ● (3 sessions + RepoAgent) |
| Problem → production (specs, metrics, handover) | ◐ (capstone) | ● |
| Interview prep, system design, DSA | ◐ (recorded career sessions) | ● |

**Count across 28 areas: DivyaSampark covers 14 in depth, touches 3, and misses 11. Crio covers 19 in depth, touches 9, and misses none.**

---

## 2. Crio topics not covered in the first (DivyaSampark) syllabus

"Partly" means our [SYLLABUS.md](SYLLABUS.md) adds it only as an **[Added]** brick, an elective, or a line in
[notes/99-gaps-and-further-resources.md](notes/99-gaps-and-further-resources.md), not as a taught topic with a project.

### Foundations
1. **ML/DL refresher**: bias/variance, classical metrics, model families. *Not covered* (gaps doc only).
2. **Computer-vision primer**: CNNs, transfer learning in PyTorch, deploying an image classifier, and **classical model vs multimodal LLM** comparison (SnapClassify). *Not covered.*
3. **Context engineering** as its own discipline. *Not covered.*
4. **Design patterns for AI apps and when *not* to use AI/agents.** *Partly* (Module 10.5 covers multi-agent only).
5. **Error-analysis loop** (build → examine → decide) as the core workflow. *Partly* (Module 3 eval loop).
6. **System-design recap for AI apps**: API design, data models, auth, async. *Not covered.*
7. **Multimodal inputs** (images in prompts). *Partly* (Gemini mention only).

### RAG and data
8. **Storage tiers** for retrieval (when to pick which store). *Not covered.*
9. **Knowledge graphs and a semantic layer over SQL.** *Partly* (GraphRAG one-line in Module 5.7).
10. **Data engineering for AI**: pipelines, data quality and freshness, **incremental sync**. *Partly* (content-hash re-ingest in 4.6).
11. **Real enterprise sources**: Notion / Google Drive connectors, conversation memory in RAG. *Not covered.*
12. **Agentic RAG with multi-hop questions** as a build. *Partly* (overview only).

### Agents, MCP and integration
13. **Workflows vs agent harness** distinction. *Partly.*
14. **Router pattern** for agents. *Partly.*
15. **Multi-tenant MCP servers**: per-tenant scoping, **RBAC at the tool boundary**, audit logs, **OAuth**. *Partly* (single-tenant MCP with basic roles in Module 11).
16. **Vendor agent SDKs compared with LangGraph**. *Partly* ([Added] brick 8.8).
17. **ACP protocol**; typed schemas and state machines for agent messaging. *Not covered* (A2A only).
18. **OAuth/SSO and webhooks** for integrating real systems. *Not covered.*
19. **Text-to-SQL with a verification loop** at scale (200-table warehouse, schema retrieval). *Partly* (Module 8.6 basics).
20. **Agents on live systems** (real helpdesk, GitHub, CRM, Stripe test mode). *Not covered.*

### Evaluation and models
21. **Building eval sets from production traces.** *Not covered.*
22. **Evaluate-your-evals** (judge calibration against human labels) as a deliverable. *Partly* (judge-bias table in Module 3).
23. **DPO / RL-style post-training, reasoning-model training.** *Not covered* (gaps doc only).
24. **Distillation** as a hands-on demo. *Not covered* (elective only).
25. **Publishing adapters to a model hub; model cards with data provenance.** *Not covered.*
26. **Fine-tuned model + RAG vs frontier API, cost per 1,000 queries** (DomainCopilot). *Partly* (Module 7 comparison table).

### Production and operations
27. **Drift detection and statistical regression testing.** *Not covered.*
28. **Incident fire-drills, runbooks, rollbacks, time-to-recover, handover docs.** *Not covered.*
29. **Prompt caching + semantic cache gateway with Redis, token budgets, cost-per-query dashboard.** *Partly* (semantic cache in Module 5, routing in Module 12).
30. **PII redaction with Presidio** in a build. *Partly* (mentioned in Module 4.8).
31. **Hands-on AWS**: Bedrock, AgentCore, SageMaker, cost alarms. *Not covered* (elective only).
32. **Scaling LLM APIs.** *Not covered.*
33. **Real-time voice agents**: LiveKit, cascaded vs speech-to-speech, barge-in, WER, cost per minute. *Not covered* (elective only).

### Coding agents (Crio's biggest unique block)
34. **How a coding-agent harness wraps an LLM**; plan → execute → deploy. *Not covered.*
35. **Autonomy levels and safe permissions** for coding agents. *Not covered.*
36. **AGENTS.md / CLAUDE.md, hooks, MCP in coding agents, parallel agents.** *Not covered.*
37. **Agentic code review; AI security and architecture audits.** *Not covered.*
38. **Shipping a feature into a large open-source codebase with agents** (RepoAgent). *Not covered.*

### Product engineering and career
39. **Problem → production**: framing the problem, one-page spec, success metrics, MVP vs careful build, real constraints (legacy systems, tenancy), stakeholder demos. *Partly* (capstone design doc in Module 13).
40. **Agentic system-design interviews, ambiguous-problem decomposition, architecture defence.** *Not covered.*
41. **Data Structures & Algorithms and LLD/HLD system design** tracks. *Not covered.*
42. **Regulated-vertical builds**: insurance claims, healthcare prior-auth, call-centre speech analytics. *Not covered.*

## 3. DivyaSampark topics not (explicitly) in Crio

1. A full **Python for AI Engineering** module (Crio treats it as a prerequisite and pre-work).
2. **Transformer internals** in depth: attention, positional encoding, the GPT/Llama/Mistral ecosystem.
3. **HyDE, multi-query retrieval, context compression** named explicitly.
4. **RAGAS / DeepEval / TruLens** named as the eval stack.
5. A **fine-tuning strategy and decision framework** module, separate from the hands-on PEFT module.
6. **Catastrophic forgetting**, Unsloth/Axolotl, the GPU/Colab lab path.
7. **Failure taxonomy for agents** (transient, permanent, silent, cascading, adversarial) and **circuit breakers**.
8. **Reflection / self-correction loops and streaming agent UX** (tool-call streaming, interruptibility) as named topics.
9. **When not to use multi-agent systems** as a named topic.
10. **Named guardrail frameworks**: Guardrails AI, NeMo Guardrails.
11. **Retrieval poisoning, tool abuse, data exfiltration** as named threats.
12. **GDPR and India's DPDP Act** explicitly, plus enterprise AI governance.
13. **MCP transports (stdio, SSE)** in detail.
14. **IIT faculty masterclasses** and campus immersion.

## 4. Which is more effective?

| Criterion | Winner | Why |
|---|---|---|
| Breadth of topics | **Crio** | Adds coding agents, voice, AWS, CV, data engineering, interview prep, DSA and system design |
| Amount of hands-on building | **Crio** | 17 assessed builds (~120 h) vs 6 projects + capstone |
| Production / ops realism | **Crio** | Drift, fire-drills, runbooks, rollbacks, cost alarms, real AWS deploy, real data sources |
| Job-market relevance (2026) | **Crio** | Coding agents, voice agents, FDE-style interview prep and multi-tenant MCP are in high demand |
| Theory depth (how LLMs work) | **DivyaSampark** | Dedicated transformer and model-ecosystem module |
| Governance for India | **DivyaSampark** | Explicit DPDP/GDPR, named guardrail tools |
| Credential / brand | **DivyaSampark** | IIT Roorkee (iHUB DivyaSampark) certificate, faculty, campus immersion |
| Beginner friendliness | **DivyaSampark** | Python module included; Crio expects you to already be a working programmer |
| Cost transparency | **DivyaSampark** | Fee published; Crio's fee is not on the curriculum page |

**Maximum topics and maximum learning: Crio.** It covers roughly 42 topics the DivyaSampark syllabus
lacks or only mentions, against about 14 in the other direction, and it makes you ship about three times as many
graded builds.

**Choose DivyaSampark if** the IIT credential matters for your career goals, you want stronger LLM theory and
India-specific compliance, or you need the Python on-ramp.

**Choose Crio if** your goal is to be job-ready as an AI / Forward Deployed Engineer with a large portfolio,
including cloud deployment, coding agents and interview preparation.

## 5. Recommended self-study plan (best of both)

Keep the existing 14-module path in [SYLLABUS.md](SYLLABUS.md) and add the Crio-only topics where they fit:

| Where | Add from Crio |
|---|---|
| Before Module 2 | ML/DL refresher + CV primer; build a small SnapClassify (#1–2) |
| Module 3 | Context engineering, error-analysis loop, eval sets from traces (#3, 5, 21) |
| Modules 4–5 | Incremental sync, Notion/Drive connector, knowledge graph / SQL semantic layer (#9–11) |
| Module 7 | DPO demo, distillation, publish adapter + model card to Hugging Face (#23–25) |
| Modules 8–10 | Router pattern, live-API agent, OAuth/webhooks, text-to-SQL with verification (#14, 18–20) |
| Module 11 | Multi-tenant MCP with RBAC + OAuth (#15) |
| Module 12 | Drift alerts, fire-drill + runbook, Redis semantic cache, deploy one build to AWS Bedrock (#27–31) |
| New Module 12b | Voice agent with LiveKit (#33) |
| New Module 13a | Coding agents: CLAUDE.md, hooks, agentic review; ship a PR to an open-source repo (#34–38) |
| After capstone | Interview sprint: agentic system design, decomposition, portfolio defence; DSA and system design as needed (#40–41) |
