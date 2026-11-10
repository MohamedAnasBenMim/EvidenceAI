from fastapi import FastAPI
from pydantic import BaseModel

from .dataset import sample_documents
from .rag import EvidenceRAG
from .retrieval import HybridIndex

app = FastAPI(title="EvidenceAI", version="1.0.0")
rag = EvidenceRAG(HybridIndex(sample_documents()))


class Question(BaseModel):
    text: str
    groups: set[str]
    metadata: dict[str, str] | None = None


@app.post("/v1/answer")
def answer(question: Question) -> dict[str, object]:
    return rag.answer(question.text, question.groups, question.metadata).to_dict()
