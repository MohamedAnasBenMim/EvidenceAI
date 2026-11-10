from .retrieval import Document


def sample_documents() -> list[Document]:
    return [
        Document(
            "security-1",
            "Password Policy",
            "Employees must use passwords with at least fourteen characters. Multi-factor authentication is required for administrator accounts.",
            frozenset({"employees", "security"}),
            {"department": "security"},
        ),
        Document(
            "hr-1",
            "Remote Work",
            "Employees may work remotely up to three days per week with manager approval. Requests are recorded in the HR portal.",
            frozenset({"employees", "hr"}),
            {"department": "hr"},
        ),
        Document(
            "finance-1",
            "Expense Policy",
            "Travel expenses above 500 USD require finance approval. Receipts must be submitted within thirty days.",
            frozenset({"employees", "finance"}),
            {"department": "finance"},
        ),
        Document(
            "board-1",
            "Acquisition Plan",
            "The confidential acquisition review is scheduled for December.",
            frozenset({"board"}),
            {"department": "executive"},
        ),
    ]
