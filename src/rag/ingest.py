import hashlib
from pathlib import Path

from src.rag.document_loader import (
    load_pdf,
    split_documents,
)
from src.rag.vector_store import get_vector_store


def generate_document_id(file_path: str) -> str:
    """
    Generate a stable document ID using the file contents.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()[:16]


def ingest_pdf(
    file_path: str,
    document_name: str | None = None,
) -> dict:
    """
    Ingest a PDF into ChromaDB with document-level metadata.

    Returns information about the processed document.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    if document_name is None:
        document_name = path.name

    document_id = generate_document_id(
        str(path)
    )

    print(
        f"Processing document: "
        f"{document_name}"
    )

    print(
        f"Document ID: {document_id}"
    )

    # Load PDF
    documents = load_pdf(str(path))

    print(
        f"Loaded {len(documents)} pages."
    )

    # Split and add metadata
    chunks = split_documents(
        documents=documents,
        document_id=document_id,
        document_name=document_name,
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # Get vector store
    vector_store = get_vector_store()

    # Remove previous version of the same document
    vector_store.delete(
        where={
            "document_id": document_id
        }
    )

    # Add chunks
    vector_store.add_documents(
        documents=chunks,
        ids=[
            chunk.metadata["chunk_id"]
            for chunk in chunks
        ],
    )

    print(
        "Document successfully stored in ChromaDB."
    )

    return {
        "document_id": document_id,
        "document_name": document_name,
        "page_count": len(documents),
        "chunk_count": len(chunks),
    }