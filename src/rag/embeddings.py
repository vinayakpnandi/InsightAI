import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------------------------
# Embedding Model Configuration
# --------------------------------------------------

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------

@st.cache_resource
def get_embeddings():
    """
    Load the embedding model once and reuse it.

    Streamlit caches this resource so the model
    does not need to be loaded again every time
    the application reruns.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )