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

    # ── Deterministic IDs ──────────────────────────────────────────────

    def _generate_content_id(self, filename: str, text: str) -> str:
        """Generate a deterministic content-based SHA256 ID."""
        import hashlib
        hasher = hashlib.sha256()
        hasher.update(filename.encode("utf-8"))
        hasher.update(text.encode("utf-8"))
        return hasher.hexdigest()

    # ── Add documents ──────────────────────────────────────────────────
    
    def add_documents(
        self, documents: list[Document], collection_target: str
    ) -> None:
        """
        Add documents to the appropriate collection using deterministic content-based IDs.

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
        
        # Generate deterministic IDs based on content hash
        ids = [
            self._generate_content_id(
                doc.metadata.get("source_filename", "unknown"),
                doc.page_content
            )
            for doc in documents
        ]
        
        store.add_documents(documents, ids=ids)
        logger.info(
            "Added %d documents to '%s' collection with deterministic content IDs.",
            len(documents),
            collection_target,
        )

    # ── Replace file (idempotent delete-before-insert) ─────────────────

    def replace_file(self, documents: list[Document], collection_target: str) -> dict:
        """
        Perform an idempotent delete-before-insert replacement operation.
        
        Parameters
        ----------
        documents : list[Document]
            Documents to index.
        collection_target : str
            "narrative" or "structured".

        Returns
        -------
        dict
            Indexing statistics: {"exists": bool, "deleted": int, "inserted": int, "count_before": int, "count_after": int}
        """
        if not documents:
            return {
                "exists": False,
                "deleted": 0,
                "inserted": 0,
                "count_before": 0,
                "count_after": 0
            }

        filename = documents[0].metadata.get("source_filename", "unknown")
        
        # Get target store
        store = self._get_store(collection_target)
        
        # Determine count before upload
        count_before = self.get_document_count(collection_target)
        
        # Check if files already exist
        try:
            result = store._collection.get(
                where={"source_filename": filename},
                include=[]
            )
            ids_to_delete = result.get("ids", [])
            exists = len(ids_to_delete) > 0
            deleted_count = len(ids_to_delete)
        except Exception as e:
            logger.warning("Failed to query existing files in replace_file: %s", e)
            exists = False
            deleted_count = 0
            ids_to_delete = []

        # Delete existing documents
        if deleted_count > 0:
            try:
                store.delete(ids=ids_to_delete)
                logger.info(
                    "Deleted %d existing documents for file '%s' from '%s' collection.",
                    deleted_count,
                    filename,
                    collection_target
                )
            except Exception as e:
                logger.error("Failed to delete existing files in replace_file: %s", e, exc_info=True)
                deleted_count = 0

        # Insert new documents with deterministic SHA256 IDs
        self.add_documents(documents, collection_target)
        
        # Get count after upload
        count_after = self.get_document_count(collection_target)

        # Log the exact UPLOAD metrics block requested by the user
        log_msg = (
            "\n--------------------------------------------------\n"
            "UPLOAD\n"
            "--------------------------------------------------\n\n"
            f"File:\n{filename}\n\n"
            f"Already Exists:\n{exists}\n\n"
            f"Collection:\n{collection_target}\n\n"
            f"Documents Before:\n{count_before}\n\n"
            f"Deleted:\n{deleted_count}\n\n"
            f"Inserted:\n{len(documents)}\n\n"
            f"Documents After:\n{count_after}"
        )
        logger.info(log_msg)

        return {
            "exists": exists,
            "deleted": deleted_count,
            "inserted": len(documents),
            "count_before": count_before,
            "count_after": count_after
        }

    # ── Delete file (legacy/utility) ───────────────────────────────────

    def delete_file(self, filename: str, collection_target: str) -> int:
        """
        Delete all documents belonging to a filename in a specific collection.
        
        Returns the number of deleted documents.
        """
        store = self._get_store(collection_target)
        try:
            # Query for matching documents to get their IDs
            result = store._collection.get(
                where={"source_filename": filename},
                include=[]
            )
            if result and result.get("ids"):
                ids_to_delete = result["ids"]
                store.delete(ids=ids_to_delete)
                logger.info(
                    "Deleted %d documents for file '%s' from '%s' collection.",
                    len(ids_to_delete),
                    filename,
                    collection_target,
                )
                return len(ids_to_delete)
        except Exception as e:
            logger.error(
                "Failed to delete file '%s' from '%s' collection: %s",
                filename,
                collection_target,
                e,
                exc_info=True,
            )
        return 0

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
        """Return a list of unique source filenames across both collections with document counts."""
        sources: dict[str, dict] = {}  # filename → {"filename": str, "file_type": str, "doc_count": int}

        for target in ("narrative", "structured"):
            store = self._get_store(target)
            try:
                result = store._collection.get(include=["metadatas"])
                if result and result["metadatas"]:
                    for meta in result["metadatas"]:
                        fname = meta.get("source_filename", "unknown")
                        ftype = meta.get("file_type", "unknown")
                        if fname not in sources:
                            sources[fname] = {
                                "filename": fname,
                                "file_type": ftype,
                                "doc_count": 0,
                            }
                        sources[fname]["doc_count"] += 1
            except Exception as e:
                logger.warning("Failed to get sources from '%s': %s", target, e)

        return list(sources.values())

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
