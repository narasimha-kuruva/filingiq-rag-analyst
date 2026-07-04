# 📊 FilingIQ — SEC Filing Intelligence Platform

**FilingIQ** is an AI-powered analyst assistant that lets you upload SEC filings (10-K PDFs, analyst memos, financial spreadsheets, and more) and ask natural-language questions. It uses a **dual-index Retrieval-Augmented Generation (RAG)** architecture to deliver grounded, cited answers — separating narrative prose from structured financial data for higher retrieval accuracy. Powered by Google Gemini 1.5 Flash and ChromaDB.

> 🎯 **Key differentiator**: Unlike single-index RAG systems, FilingIQ maintains separate vector stores for narrative documents (PDF, DOCX, TXT) and structured data (XLSX, CSV), preventing embedding-space dilution and delivering more precise retrieval.

---

## 🏗️ Architecture & Layered Boundaries

FilingIQ is built with a strictly decoupled, layered architecture to maintain a clean separation of concerns:

```
┌────────────────────────────────────────────────────────┐
│                      UI Layer                          │
│                   (app.py UI)                          │
└───────────┬─────────────────────────────────┬──────────┘
            │ (User Upload)                   │ (Natural Language Query)
            ▼                                 ▼
┌───────────────────────┐           ┌────────────────────┐
│     Utility Layer     │           │  Generation Layer  │
│  (utils/file_hash.py) │           │ (generation/llm)   │
└───────────┬───────────┘           └─────────▲──────────┘
            │                                 │
            ▼                                 │ (Grounded Response)
┌───────────────────────┐           ┌─────────┴──────────┐
│    Ingestion Layer    │           │  Retriever Layer   │
│  (ingestion/loaders)  │           │ (retrieval/search) │
└───────────┬───────────┘           └─────────▲──────────┘
            │ (Document Chunks)               │
            ▼                                 │ (Similarity Retrieve)
┌─────────────────────────────────────────────┴──────────┐
│                 Vector Store Layer                     │
│            (retrieval/vector_store.py)                 │
│  • Storage-focused operations (Index / Skip / Replace) │
│  • Idempotency management & ChromaDB metadata query    │
└────────────────────────────────────────────────────────┘
```

### Layer Responsibilities
* **UI Layer (`app.py`)**: Manages the Streamlit interface, chat history state, file upload events, and triggers the ingestion pipeline.
* **Utility Layer (`utils/file_hash.py`)**: Handles standalone file hash generation using SHA256 hashes of entire files.
* **Ingestion Layer (`ingestion/`)**: Dispatches files to type-specific loaders, generates text chunks, and initializes chunk-level metadata.
* **Vector Store Layer (`retrieval/vector_store.py`)**: Manages ChromaDB collections. Handles version detection, deterministic metadata-based chunk IDs, and safe document replacement (`sync_document()`).
* **Retriever Layer (`retrieval/retriever.py`)**: Executes cosine similarity lookups across the dual narrative/structured collections and filters by relevance threshold.
* **Generation Layer (`generation/`)**: Constructs strict grounding prompts and generates assistant responses via Google Gemini 1.5 Flash.

---

## ⚡ Intelligent Versioning & Deduplication

FilingIQ implements an intelligent indexing pipeline that treats ChromaDB as the single source of truth:

1. **Startup Synchronization**: On launch, the Streamlit session state processed file list is synchronized directly from active ChromaDB records, ensuring consistent state across restarts.
2. **File Hashing**: When a file is uploaded, the Utility layer computes a SHA256 hash of its entire content.
3. **Smart Ingest Decisions**:
   * **Skip**: If the file hash matches a document hash already found in ChromaDB, the system skips loader chunking, embedding generation, and vector store writes.
   * **Replace**: If the file is modified (hash mismatch), the vector store executes a **Safe Replacement Workflow** (pre-validates incoming metadata, pre-generates deterministic chunk IDs, deletes outdated vectors, and inserts new chunks).
   * **Index**: New files are indexed directly.
4. **Deterministic Chunk IDs**: Chunk IDs are computed as `SHA256(filename + file_type + index + chunk_text)` to prevent duplicate vector slots and ensure stability across document updates.

---

## 📁 Supported File Types

| Type | Extension | What It Handles | Citation Format |
|------|-----------|-----------------|-----------------|
| 📄 PDF | `.pdf` | 10-K filings, annual reports | `p.12` |
| 📝 DOCX | `.docx` | Analyst memos, research notes | `Section: Risk Factors, ¶3` |
| 📃 TXT | `.txt` | Transcripts, plain notes | `Chunk 5` |
| 📊 XLSX | `.xlsx` | Income statements, balance sheets | `Sheet1, Row 34` |
| 📋 CSV | `.csv` | Financial ratios, peer comparisons | `Row 12` |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- A free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/filingiq-rag-analyst.git
cd filingiq-rag-analyst

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
# Edit .streamlit/secrets.toml and add your Gemini API key

# 5. Generate sample test data (optional)
python generate_sample_data.py

# 6. Run the app
streamlit run app.py
```

### Run Tests (optional)
Generate the sample data files first:
```bash
python generate_sample_data.py
```

To verify the ingestion layer, run the ingestion tests:
```bash
python test_ingestion.py
```

To verify the database and retrieval logic, run the retrieval tests:
```bash
python test_retrieval.py
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repo
4. Set `app.py` as the entrypoint
5. Add your Gemini API key under **Settings → Secrets**:
   ```toml
   GOOGLE_API_KEY = "your-actual-key-here"
   ```
6. Deploy!

---

## ⚙️ Configuration

All tunable parameters are centralized in [`config.py`](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 1000 | Character count per text chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between adjacent chunks |
| `TOP_K` | 4 | Number of results per collection |
| `SIMILARITY_THRESHOLD` | `0.50` | Minimum cosine similarity to keep results |
| `LLM_MODEL` | `gemini-2.5-flash` | Google Gemini model |
| `TEMPERATURE` | 0.15 | LLM temperature (low = less creative drift) |
| `EMBEDDING_MODEL` | `models/embedding-001` | Primary embedding model |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local fallback embedding model |
| `CHROMA_PERSIST_DIR` | `chroma_db` | Persistent vector store directory (or `/tmp/chroma_db` on Streamlit Cloud) |
| `AGENTIC_MODE` | `False` | Future: enable tool-calling agent |

---

## 🔐 How Grounding Works

FilingIQ uses a **strict grounding prompt** and routing logic that enforces:
1. **Document-only answers** — the LLM cannot use its training data
2. **Mandatory citations** — every answer must end with a `Sources:` line
3. **Deterministic refusal** — if nothing relevant is found, the system returns a fixed message *without even calling the LLM*, saving API quota
4. **Safe arithmetic** — calculations are allowed only with numbers present in the retrieved context
5. **Greeting short-circuiting** — simple greeting or capability queries (e.g., 'hello', 'who are you') bypass database retrieval and route to a direct greeting generator, reducing query overhead

---

## ⚠️ Known Limitations & Future Work

### Current Limitations
- **Ephemeral vector store on Streamlit Cloud**: ChromaDB persists to disk, but Streamlit Cloud's filesystem resets on app restart/redeploy. Documents must be re-uploaded after each restart.
  - *Future fix*: Swap in a hosted vector DB (Chroma Cloud, Supabase pgvector, or Pinecone free tier).
- **Single-session**: No user authentication or multi-user support. All uploads are shared within a single app instance.
- **Embedding model fallback**: If the Gemini API key has no quota, the app falls back to local `sentence-transformers/all-MiniLM-L6-v2`, which is slower and produces different embeddings.

### Planned Agentic Upgrade
The codebase is designed for a future **agentic layer** (controlled by `config.AGENTIC_MODE`):
- `retriever.py` and `llm_chain.py` are standalone composable functions, ready to be wrapped as LangChain/LangGraph tools
- Planned tools:
  - `calculate_ratio(numerator, denominator)` — compute financial ratios on-the-fly
  - `compare_companies(company_a, company_b, metric)` — side-by-side comparisons
  - `web_search_recent_news(query)` — augment with real-time context
- A ReAct or function-calling agent will decide which tool(s) to invoke per question

---

## 📄 License

This project is for educational and portfolio purposes. Not affiliated with the SEC.