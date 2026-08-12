from src.rag.vector_store import get_vector_store


def get_retriever(
    document_id: str,
    k: int = 4,
):
    """
    Create a retriever that searches ONLY within
    the specified document.
    """

    if not document_id:
        raise ValueError(
            "document_id is required for retrieval."
        )

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {
                "document_id": document_id
            },
        },
    )

    return retriever