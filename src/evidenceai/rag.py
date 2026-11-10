from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass

from .retrieval import HybridIndex, SearchResult, tokenize


@dataclass(frozen=True, slots=True)
class Citation:
    document_id: str
    title: str
    quote: str


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: list[Citation]
    latency_ms: float
    estimated_token_cost_usd: float
    grounded: bool

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "citations": [asdict(citation) for citation in self.citations]}


class EvidenceRAG:
    def __init__(self, index: HybridIndex) -> None:
        self.index = index

    def answer(
        self,
        question: str,
        user_groups: set[str],
        metadata: dict[str, str] | None = None,
    ) -> Answer:
        started = time.perf_counter()
        retrieved = self.index.search(question, user_groups, metadata, limit=8)
        reranked = self._rerank(question, retrieved)[:3]
        citations = [self._citation(result, question) for result in reranked if result.score > 0]
        if not citations:
            text = "I could not find sufficient permitted evidence to answer this question."
            grounded = False
        else:
            facts = [citation.quote.rstrip(".") for citation in citations]
            text = " ".join(f"{fact} [{index}]" for index, fact in enumerate(facts, 1)) + "."
            grounded = True
        token_count = len(tokenize(question + " " + text))
        return Answer(
            text,
            citations,
            round((time.perf_counter() - started) * 1000, 3),
            round(token_count * 0.000002, 6),
            grounded,
        )

    @staticmethod
    def _rerank(question: str, results: list[SearchResult]) -> list[SearchResult]:
        question_terms = set(tokenize(question))
        return sorted(
            results,
            key=lambda result: (
                result.score + 0.15 * len(question_terms & set(tokenize(result.document.title)))
            ),
            reverse=True,
        )

    @staticmethod
    def _citation(result: SearchResult, question: str) -> Citation:
        sentences = re.split(r"(?<=[.!?])\s+", result.document.text.strip())
        question_terms = set(tokenize(question))
        quote = max(sentences, key=lambda sentence: len(set(tokenize(sentence)) & question_terms))
        return Citation(result.document.id, result.document.title, quote[:280])
