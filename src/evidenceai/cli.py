import argparse
import json

from .dataset import sample_documents
from .rag import EvidenceRAG
from .retrieval import HybridIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a citation-grounded question")
    parser.add_argument("question")
    parser.add_argument("--group", action="append", default=["employees"])
    args = parser.parse_args()
    answer = EvidenceRAG(HybridIndex(sample_documents())).answer(args.question, set(args.group))
    print(json.dumps(answer.to_dict(), indent=2))
