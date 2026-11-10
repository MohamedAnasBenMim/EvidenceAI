from __future__ import annotations

import json
import statistics
from pathlib import Path

from .dataset import sample_documents
from .rag import EvidenceRAG
from .retrieval import HybridIndex

QUESTIONS = [
    ("How long must employee passwords be?", "security-1", "fourteen"),
    ("How many remote days are allowed?", "hr-1", "three"),
    ("When do travel expenses need finance approval?", "finance-1", "500"),
]


def evaluate() -> dict[str, float]:
    index = HybridIndex(sample_documents())
    rag = EvidenceRAG(index)
    recalls = []
    faithfulness = []
    citation_accuracy = []
    latencies = []
    costs = []
    for question, expected_document, expected_text in QUESTIONS:
        results = index.search(question, {"employees"})
        recalls.append(any(result.document.id == expected_document for result in results[:3]))
        answer = rag.answer(question, {"employees"})
        cited_ids = {citation.document_id for citation in answer.citations}
        citation_accuracy.append(expected_document in cited_ids)
        faithfulness.append(expected_text.lower() in answer.text.lower())
        latencies.append(answer.latency_ms)
        costs.append(answer.estimated_token_cost_usd)
    return {
        "retrieval_recall_at_3": round(sum(recalls) / len(recalls), 4),
        "faithfulness": round(sum(faithfulness) / len(faithfulness), 4),
        "citation_accuracy": round(sum(citation_accuracy) / len(citation_accuracy), 4),
        "mean_latency_ms": round(statistics.mean(latencies), 4),
        "mean_estimated_cost_usd": round(statistics.mean(costs), 6),
    }


def main() -> None:
    report = evaluate()
    path = Path("benchmarks/results.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
