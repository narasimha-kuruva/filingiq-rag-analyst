"""
Embedder — wraps embedding model selection with automatic fallback.

Primary: Google Gemini embedding-001 (requires API key).
Fallback: sentence-transformers/all-MiniLM-L6-v2 (runs locally, no API needed).
"""

import os
import logging

from langchain_core.embeddings import Embeddings

from config import EMBEDDING_MODEL, LOCAL_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def get_embedding_function() -> Embeddings:
    """
    Return a LangChain-compatible embedding function.

    Tries Google Gemini embeddings first; falls back to local
    sentence-transformers if the API key is missing or the call fails.
    """
    api_key = _get_api_key()

    if api_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=api_key,
            )
            # Quick smoke test — embed a single word to verify connectivity
            embeddings.embed_query("test")
            logger.info("Using Google Gemini embeddings (%s)", EMBEDDING_MODEL)
            return embeddings
        except Exception as e:
            logger.warning(
                "Gemini embeddings failed (%s), falling back to local model.", e
            )

    # ── Fallback: local sentence-transformers ──────────────────────────
    return _get_local_embeddings()


def _get_api_key() -> str | None:
    """Retrieve Google API key from Streamlit secrets or env var."""
    # Streamlit Cloud / local secrets.toml
    try:
        import streamlit as st
        key = st.secrets.get("GOOGLE_API_KEY")
        if key and key != "your-gemini-api-key-here":
            return key
    except Exception:
        pass

    # Environment variable
    key = os.environ.get("GOOGLE_API_KEY")
    if key and key != "your-gemini-api-key-here":
        return key

    return None


def _get_local_embeddings() -> Embeddings:
    """Load local sentence-transformers model as fallback."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    logger.info("Using local embeddings (%s)", LOCAL_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
