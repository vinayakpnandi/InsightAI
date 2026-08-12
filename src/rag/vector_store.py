import streamlit as st

from langchain_chroma import Chroma

from src.rag.embeddings import get_embeddings


# --------------------------------------------------
# ChromaDB Configuration
# --------------------------------------------------

PERSIST_DIRECTORY = "data/vectorstore"

COLLECTION_NAME = "insightai_documents"


# --------------------------------------------------
# Load Vector Store
# --------------------------------------------------

@st.cache_resource
def get_vector_store():
    """
    Load the persistent ChromaDB vector store.

    The vector store is cached so Streamlit does not
    recreate the Chroma connection on every rerun.
    """

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    return vector_store