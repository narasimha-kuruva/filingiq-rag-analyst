"""
Vector Store — dual-collection ChromaDB setup.

Maintains two physically separate collections:
  - narrative_store  → PDF, DOCX, TXT chunks (prose)
  - structured_store → Excel, CSV row-documents (numeric facts)

Keeping them separate prevents embedding-space dilution between
narrative prose and tabular facts.
"""

import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import (
    CHROMA_PERSIST_DIR,
    NARRATIVE_COLLECTION,
    STRUCTURED_COLLECTION,
)

logger = logging.getLogger(__name__)


class DualVectorStore:
    """Manages two ChromaDB collections: narrative and structured."""

    def __init__(self, embedding_fn: Embeddings):
        self._embedding_fn = embedding_fn
        self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        self.narrative = Chroma(
            client=self._client,
            collection_name=NARRATIVE_COLLECTION,
            embedding_function=self._embedding_fn,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.structured = Chroma(
            client=self._client,
            collection_name=STRUCTURED_COLLECTION,
            embedding_function=self._embedding_fn,
            collection_metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialized at %s (collections: %s, %s)",
            CHROMA_PERSIST_DIR,
            NARRATIVE_COLLECTION,
            STRUCTURED_COLLECTION,
        )

    # ── Add documents ──────────────────────────────────────────────────

    def add_documents(
        self, documents: list[Document], collection_target: str
    ) -> None:
        """
        Add documents to the appropriate collection.

        Parameters
        ----------
        documents : list[Document]
            Documents to index.
        collection_target : str
            "narrative" or "structured".
        """
        if not documents:
            logger.warning("No documents to add.")
            return

        store = self._get_store(collection_target)
        store.add_documents(documents)
        logger.info(
            "Added %d documents to '%s' collection.",
            len(documents),
            collection_target,
        )

    # ── Query ──────────────────────────────────────────────────────────

    def similarity_search_with_score(
        self, query: str, collection_target: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        """
        Run similarity search on a specific collection.

        Returns list of (Document, cosine_similarity) tuples.
        Higher score = more similar. Clamped between 0.0 and 1.0.
        """
        store = self._get_store(collection_target)
        results = store.similarity_search_with_score(query, k=k)
        
        mapped_results = []
        for idx, (doc, distance) in enumerate(results, start=1):
            # Cosine similarity = 1.0 - cosine_distance
            similarity = max(0.0, min(1.0, 1.0 - distance))
            mapped_results.append((doc, similarity))

            # Log raw retrieval details
            logger.info(
                f"--------------------------------------------------\n"
                f"Result #{idx}\n"
                f"--------------------------------------------------\n"
                f"Raw Distance:\n{distance}\n\n"
                f"Similarity:\n{similarity}\n\n"
                f"Metadata:\n{doc.metadata}\n\n"
                f"Chunk Length:\n{len(doc.page_content)}\n\n"
                f"Chunk:\n{doc.page_content}"
            )
            
        return mapped_results

    # ── Collection stats ───────────────────────────────────────────────

    def get_document_count(self, collection_target: str) -> int:
        """Return the number of documents in a collection."""
        store = self._get_store(collection_target)
        return store._collection.count()

    def get_all_sources(self) -> list[dict]:
        """Return a list of unique source filenames across both collections."""
        sources: dict[str, str] = {}  # filename → file_type

        for target in ("narrative", "structured"):
            store = self._get_store(target)
            try:
                result = store._collection.get(include=["metadatas"])
                if result and result["metadatas"]:
                    for meta in result["metadatas"]:
                        fname = meta.get("source_filename", "unknown")
                        ftype = meta.get("file_type", "unknown")
                        sources[fname] = ftype
            except Exception:
                pass

        return [
            {"filename": fname, "file_type": ftype}
            for fname, ftype in sources.items()
        ]

    # ── Clear ──────────────────────────────────────────────────────────

    def clear_all(self) -> None:
        """Delete and re-create both collections."""
        for name in (NARRATIVE_COLLECTION, STRUCTURED_COLLECTION):
            try:
                self._client.delete_collection(name)
            except Exception:
                pass

        # Re-initialize
        self.narrative = Chroma(
            client=self._client,
            collection_name=NARRATIVE_COLLECTION,
            embedding_function=self._embedding_fn,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.structured = Chroma(
            client=self._client,
            collection_name=STRUCTURED_COLLECTION,
            embedding_function=self._embedding_fn,
            collection_metadata={"hnsw:space": "cosine"},
        )
        logger.info("Both collections cleared and re-created.")

    # ── Internal ───────────────────────────────────────────────────────

    def _get_store(self, collection_target: str) -> Chroma:
        if collection_target == "narrative":
            return self.narrative
        elif collection_target == "structured":
            return self.structured
        else:
            raise ValueError(
                f"Unknown collection target '{collection_target}'. "
                "Use 'narrative' or 'structured'."
            )
