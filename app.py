"""
FilingIQ — SEC Filing Intelligence Platform
============================================
Streamlit entrypoint — UI only, no business logic.

An AI analyst assistant that uses dual-index RAG to answer questions
from uploaded financial documents with grounded, cited responses.
"""

import io
import logging
import streamlit as st

from config import FILE_TYPE_ICONS, REFUSAL_MESSAGE
from ingestion.router import ingest_file
from retrieval.embedder import get_embedding_function
from retrieval.vector_store import DualVectorStore
from retrieval.retriever import retrieve
from generation.llm_chain import generate
from utils.citation_formatter import format_citation, format_sources_list

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FilingIQ — AI Filing Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium look ────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }

    /* File list in sidebar */
    .file-item {
        padding: 8px 12px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.05);
        margin-bottom: 6px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.9rem;
    }

    /* Source expander styling */
    .source-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        background: #f0f4ff;
        color: #4338ca;
        font-size: 0.85rem;
        margin: 2px 4px;
        border: 1px solid #c7d2fe;
    }

    /* Architecture box */
    .arch-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px;
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 20px;
        line-height: 1.5;
    }

    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #faf5ff 100%);
        border: 1px solid #e0e7ff;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 40px 0;
    }
    .welcome-card h2 {
        color: #4338ca;
        margin-bottom: 10px;
    }
    .welcome-card p {
        color: #6b7280;
        font-size: 1rem;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-top: 20px;
    }
    .feature-item {
        background: white;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .feature-item .icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    .feature-item .label {
        color: #374151;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def _init_session_state():
    """Initialize all session state variables on first run."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store" not in st.session_state:
        with st.spinner("🔧 Initializing embedding model..."):
            embedding_fn = get_embedding_function()
            st.session_state.vector_store = DualVectorStore(embedding_fn)

    if "processed_files" not in st.session_state:
        # Populate processed_files from database on startup
        st.session_state.processed_files = st.session_state.vector_store.get_all_sources()

    if "processed_uploads" not in st.session_state:
        st.session_state.processed_uploads = set()


_init_session_state()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 FilingIQ")
    st.markdown("*AI-Powered Filing Analyst*")
    st.markdown("---")

    # ── File uploader ──────────────────────────────────────────────────
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop your financial documents here",
        accept_multiple_files=True,
        type=["pdf", "docx", "xlsx", "csv", "txt"],
        help="Supports: PDF (10-K filings), DOCX (memos), XLSX (financials), CSV (ratios), TXT (notes)",
    )

    # ── Process uploaded files ─────────────────────────────────────────
    if uploaded_files:
        # Keep only processed uploads that are still present in the file uploader
        current_names = {f.name for f in uploaded_files}
        st.session_state.processed_uploads = {
            uid for uid in st.session_state.processed_uploads
            if uid.split("_")[0] in current_names
        }

        for uploaded_file in uploaded_files:
            upload_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if upload_id not in st.session_state.processed_uploads:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Read file bytes into a BytesIO buffer
                        file_bytes = uploaded_file.read()
                        buffer = io.BytesIO(file_bytes)

                        # Ingest
                        documents, collection_target = ingest_file(
                            buffer, uploaded_file.name
                        )

                        if not documents:
                            st.warning(
                                f"⚠️ No content extracted from {uploaded_file.name}"
                            )
                            continue

                        # Perform the replacement operation (idempotent delete-before-insert)
                        st.session_state.vector_store.replace_file(
                            documents, collection_target
                        )

                        # Track processed upload in current session
                        st.session_state.processed_uploads.add(upload_id)

                        # Synchronize session state from ChromaDB (single source of truth)
                        st.session_state.processed_files = st.session_state.vector_store.get_all_sources()

                        st.success(f"✅ {uploaded_file.name} processed successfully!")

                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                        logger.error(
                            "Failed to process %s: %s", uploaded_file.name, e,
                            exc_info=True,
                        )

    # ── Indexed documents list ─────────────────────────────────────────
    if st.session_state.processed_files:
        st.markdown("---")
        st.markdown("### 📋 Indexed Documents")

        for f in st.session_state.processed_files:
            icon = FILE_TYPE_ICONS.get(f["file_type"], "📄")
            st.markdown(
                f'<div class="file-item">{icon} <strong>{f["filename"]}</strong>'
                f'<br><small>{f["doc_count"]} chunks indexed</small></div>',
                unsafe_allow_html=True,
            )

        # ── Clear button ───────────────────────────────────────────────
        st.markdown("")
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            st.session_state.vector_store.clear_all()
            st.session_state.processed_files = []
            st.session_state.processed_uploads = set()
            st.session_state.messages = []
            st.rerun()

    # ── Architecture footer ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="arch-box">'
        "🏗️ <strong>Architecture</strong><br>"
        "Dual-index RAG: narrative + structured data indexed separately "
        "in ChromaDB for higher retrieval accuracy. Gemini 1.5 Flash "
        "generates grounded, cited answers."
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">FilingIQ</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    "AI-powered SEC filing analyst — upload documents, ask questions, get cited answers."
    "</p>",
    unsafe_allow_html=True,
)

# ── Welcome message when no documents are uploaded ─────────────────────────
if not st.session_state.processed_files:
    st.markdown(
        """
        <div class="welcome-card">
            <h2>👋 Welcome to FilingIQ</h2>
            <p>Upload your financial documents in the sidebar to get started.
            I can analyze PDFs, Word documents, Excel files, CSVs, and text files.</p>
            <div class="feature-grid">
                <div class="feature-item">
                    <div class="icon">📄</div>
                    <div class="label">10-K Filings (PDF)</div>
                </div>
                <div class="feature-item">
                    <div class="icon">📝</div>
                    <div class="label">Analyst Memos (DOCX)</div>
                </div>
                <div class="feature-item">
                    <div class="icon">📊</div>
                    <div class="label">Financial Statements (XLSX)</div>
                </div>
                <div class="feature-item">
                    <div class="icon">📋</div>
                    <div class="label">Financial Ratios (CSV)</div>
                </div>
                <div class="feature-item">
                    <div class="icon">📃</div>
                    <div class="label">Notes & Transcripts (TXT)</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render conversation history ────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources expander for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("🔍 View Sources", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    citation = format_citation(src)
                    score = src.get("similarity_score", 0)
                    origin = src.get("origin", "?")

                    st.markdown(f"**Source {i}** — `[{origin}]` {citation} *(score: {score:.4f})*")
                    st.markdown(
                        f'<small style="color:#6b7280;">{src.get("content_preview", "")}</small>',
                        unsafe_allow_html=True,
                    )
                    if i < len(msg["sources"]):
                        st.markdown("---")

# ── Chat input ─────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask a question about your uploaded documents...",
    disabled=False,
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        from retrieval.retriever import is_greeting_query
        
        if not st.session_state.processed_files and not is_greeting_query(prompt):
            answer = "👋 Please upload some financial documents in the sidebar first so I can analyze them and answer your questions."
            sources = []
            st.markdown(answer)
        else:
            with st.spinner("🔍 Searching documents & generating answer..."):
                # Retrieve
                retrieval_result = retrieve(
                    prompt, st.session_state.vector_store
                )

                # Generate
                gen_result = generate(prompt, retrieval_result)
                answer = gen_result.answer
                sources = gen_result.sources

            # Display answer
            st.markdown(answer)

            # Display sources
            if sources:
                with st.expander("🔍 View Sources", expanded=False):
                    for i, src in enumerate(sources, 1):
                        citation = format_citation(src)
                        score = src.get("similarity_score", 0)
                        origin = src.get("origin", "?")

                        st.markdown(
                            f"**Source {i}** — `[{origin}]` {citation} *(score: {score:.4f})*"
                        )
                        st.markdown(
                            f'<small style="color:#6b7280;">{src.get("content_preview", "")}</small>',
                            unsafe_allow_html=True,
                        )
                        if i < len(sources):
                            st.markdown("---")

    # Save to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
