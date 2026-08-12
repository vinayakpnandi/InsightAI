from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF and return its pages as LangChain documents.
    """

    path = Path(file_path)

    loader = PyPDFLoader(str(path))

    return loader.load()


def split_documents(
    documents: list[Document],
    document_id: str,
    document_name: str,
) -> list[Document]:
    """
    Split documents into chunks and attach metadata
    required for document-level isolation.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):

        page = chunk.metadata.get("page", 0)

        chunk.metadata.update(
            {
                "document_id": document_id,
                "document_name": document_name,
                "source": document_name,
                "page": page + 1,
                "chunk_id": f"{document_id}_{index}",
            }
        )

    return chunks