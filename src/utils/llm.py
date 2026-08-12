import os

import streamlit as st

from dotenv import load_dotenv
from langchain_ollama import ChatOllama


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


# ============================================================
# MODELS
# ============================================================

MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b",
)

BUSINESS_MODEL_NAME = os.getenv(
    "OLLAMA_BUSINESS_MODEL",
    "qwen3:4b",
)

SQL_MODEL_NAME = os.getenv(
    "OLLAMA_SQL_MODEL",
    "qwen3:4b",
)


# ============================================================
# GENERIC LLM
# ============================================================

@st.cache_resource
def get_llm():
    """
    General-purpose InsightAI LLM.

    Qwen3:4b is used because it has already been
    verified to return usable content through LangChain.
    """

    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=4096,
        keep_alive="10m",
    )


# ============================================================
# BUSINESS LLM
# ============================================================

@st.cache_resource
def get_business_llm():
    """
    LLM used for business insights.

    Uses Qwen3:4b.
    """

    return ChatOllama(
        model=BUSINESS_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=4096,
        keep_alive="10m",
    )


# ============================================================
# SQL LLM
# ============================================================

@st.cache_resource
def get_sql_llm():
    """
    Dedicated LLM for SQL generation.

    Uses Qwen3:4b because SQL generation requires
    reliable structured output.
    """

    return ChatOllama(
        model=SQL_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=4096,
        keep_alive="10m",
    )