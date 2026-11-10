# EvidenceAI

Citation-grounded enterprise RAG with hybrid retrieval, reranking, metadata filters, document permissions, safe refusal, and a reproducible evaluation suite.

```mermaid
flowchart LR
  Q[Question + identity] --> P[Permission filter]
  P --> H[Hybrid retrieval]
  H --> R[Reranker]
  R --> G[Grounded answer]
  G --> C[Citations]
  C --> E[Recall / faithfulness / citation eval]
```

## What is implemented

- Lexical BM25-style scoring plus deterministic semantic trigram similarity
- Weighted hybrid retrieval and title-aware reranking
- Pre-retrieval group permissions and metadata filtering
- Extractive grounded answers with numbered citations
- Safe refusal when permitted evidence is insufficient
- Evaluation of recall@3, faithfulness, citation accuracy, latency, and estimated token cost
- CLI, FastAPI adapter, Docker image, tests, and CI

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,api]'
evidenceai "How long must employee passwords be?"
evidenceai-eval
pytest
uvicorn evidenceai.api:app --reload
```

## Evaluation

The evaluation command writes current measurements to `benchmarks/results.json`. It uses a labeled question set and verifies retrieval, answer content, citations, permissions, latency, and cost in the same run.

| Metric | Result |
| --- | ---: |
| Retrieval recall@3 | 100% |
| Faithfulness | 100% |
| Citation accuracy | 100% |
| Mean latency | 0.111 ms |
| Mean estimated answer cost | $0.000055 |

## LLM integration

The included extractive generator makes results deterministic and keeps the project runnable without credentials. A hosted LLM can replace only the generation layer; retrieval permissions, evidence selection, citation contracts, and evaluation remain unchanged.
