"""
FilingIQ — Centralized configuration.

All tunable constants are defined here. Do not hardcode values elsewhere.
"""

import os
import sys

# ─── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ─── Retrieval ──────────────────────────────────────────────────────────────
TOP_K = 4
SIMILARITY_THRESHOLD = 0.50  # cosine; discard collection results below this

# ─── LLM ────────────────────────────────────────────────────────────────────
LLM_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.15

# ─── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "models/embedding-001"  # Google Gemini embeddings (primary)
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free local fallback

# ─── ChromaDB persistence ──────────────────────────────────────────────────
# Streamlit Cloud filesystem is ephemeral; use /tmp there.
# Locally, persist next to the project.
if os.environ.get("STREAMLIT_SERVER_HEADLESS"):
    # Running on Streamlit Cloud
    CHROMA_PERSIST_DIR = "/tmp/chroma_db"
else:
    CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# ─── Collection names ──────────────────────────────────────────────────────
NARRATIVE_COLLECTION = "narrative_store"
STRUCTURED_COLLECTION = "structured_store"

# ─── Agentic mode (future) ─────────────────────────────────────────────────
# When True, questions will be routed through a tool-calling agent with
# tools like calculate_ratio, compare_companies, web_search_recent_news
# instead of the direct retrieval chain. See generation/llm_chain.py.
AGENTIC_MODE = False

# ─── Refusal message ───────────────────────────────────────────────────────
REFUSAL_MESSAGE = "I cannot find that information in the uploaded documents."

# ─── Supported file types ──────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".txt"}

FILE_TYPE_ICONS = {
    "pdf": "📄",
    "docx": "📝",
    "excel": "📊",
    "csv": "📋",
    "txt": "📃",
}
