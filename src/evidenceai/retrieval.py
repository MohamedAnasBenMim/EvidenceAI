from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def trigrams(text: str) -> set[str]:
    normalized = " ".join(tokenize(text))
    return {normalized[index : index + 3] for index in range(max(len(normalized) - 2, 0))}


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    text: str
    allowed_groups: frozenset[str]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    document: Document
    score: float
    lexical_score: float
    semantic_score: float


class HybridIndex:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self._tokens = {
            document.id: tokenize(document.title + " " + document.text) for document in documents
        }
        self._frequencies = {doc_id: Counter(tokens) for doc_id, tokens in self._tokens.items()}
        self._document_frequency = Counter(
            token for tokens in self._tokens.values() for token in set(tokens)
        )

    def search(
        self,
        query: str,
        user_groups: set[str],
        metadata: dict[str, str] | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        query_tokens = tokenize(query)
        query_trigrams = trigrams(query)
        results: list[SearchResult] = []
        for document in self.documents:
            if not document.allowed_groups.intersection(user_groups):
                continue
            if metadata and any(
                document.metadata.get(key) != value for key, value in metadata.items()
            ):
                continue
            lexical = self._bm25(document.id, query_tokens)
            document_trigrams = trigrams(document.title + " " + document.text)
            semantic = len(query_trigrams & document_trigrams) / max(
                len(query_trigrams | document_trigrams), 1
            )
            phrase_bonus = 0.25 if query.lower() in document.text.lower() else 0.0
            score = 0.6 * lexical + 0.4 * semantic + phrase_bonus
            if score > 0 and (lexical > 0 or semantic >= 0.15):
                results.append(SearchResult(document, score, lexical, semantic))
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def _bm25(self, document_id: str, query: list[str]) -> float:
        frequency = self._frequencies[document_id]
        score = 0.0
        for token in query:
            document_frequency = self._document_frequency[token]
            if not document_frequency:
                continue
            inverse_frequency = math.log(
                1 + (len(self.documents) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            term_frequency = frequency[token]
            score += (
                inverse_frequency * term_frequency / (term_frequency + 1.2) if term_frequency else 0
            )
        return score
