# Module 12C — Kubernetes and SRE Observability for AI [v2]

> Ship · Week 28 · [Syllabus](../SYLLABUS.md#module-12c--kubernetes-and-sre-observability-for-ai-week-28-v2)

**Goal:** run an AI service on Kubernetes the way an SRE would: it scales, it is measured, and it alerts on an SLO.

**Learn after:** 12A Cloud AI on AWS (IAM, cloud basics) and 12 Deployment (Docker, tracing).

## Bricks

| Brick | Topic | Exercise |
|---|---|---|
| 12C.1 | Pod, Deployment, Service, ConfigMap, Secret | Run AskDocs on a local kind/k3d cluster |
| 12C.2 | HPA, Ingress, persistent volumes | Scale on CPU or request rate; persist the vector DB |
| 12C.3 | EKS with eksctl, IRSA, Helm | Same chart on EKS; pull secrets via IAM role |
| 12C.4 | Prometheus and PromQL | Expose `/metrics` from FastAPI (requests, latency, tokens, cost) |
| 12C.5 | Grafana and Alertmanager | Dashboard for p95 latency, error rate, cost per request |
| 12C.6 | Datadog APM and LLM Observability | Compare traces with Langfuse from Module 12 |
| 12C.7 | SLIs, SLOs, error budgets | 99% of answers under 3 s; multi-window burn-rate alert |

## Useful PromQL

```promql
histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{app="askdocs"}[5m])))
sum(rate(http_requests_total{app="askdocs",status=~"5.."}[5m])) / sum(rate(http_requests_total{app="askdocs"}[5m]))
```

## Build — AI on EKS

Deploy AskDocs to EKS with HPA, a Grafana dashboard and an SLO burn-rate alert.

**Checkpoint:** the service scales under a Locust load test, and the alert fires on a planted error spike.
