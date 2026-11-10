from evidenceai.dataset import sample_documents
from evidenceai.evaluation import evaluate
from evidenceai.rag import EvidenceRAG
from evidenceai.retrieval import HybridIndex


def test_hybrid_retrieval_finds_relevant_document() -> None:
    results = HybridIndex(sample_documents()).search("password length", {"employees"})
    assert results[0].document.id == "security-1"


def test_permissions_prevent_confidential_leakage() -> None:
    index = HybridIndex(sample_documents())
    assert index.search("acquisition", {"employees"}) == []
    assert index.search("acquisition", {"board"})[0].document.id == "board-1"


def test_metadata_filter_is_enforced() -> None:
    results = HybridIndex(sample_documents()).search(
        "approval", {"employees"}, {"department": "finance"}
    )
    assert results and all(
        result.document.metadata["department"] == "finance" for result in results
    )


def test_answer_contains_numbered_citation() -> None:
    answer = EvidenceRAG(HybridIndex(sample_documents())).answer(
        "How long must passwords be?", {"employees"}
    )
    assert answer.grounded is True
    assert "[1]" in answer.text
    assert answer.citations[0].document_id == "security-1"


def test_no_evidence_returns_safe_refusal() -> None:
    answer = EvidenceRAG(HybridIndex(sample_documents())).answer("acquisition date", {"employees"})
    assert answer.grounded is False
    assert answer.citations == []


def test_evaluation_meets_quality_gate() -> None:
    metrics = evaluate()
    assert metrics["retrieval_recall_at_3"] == 1.0
    assert metrics["citation_accuracy"] == 1.0
