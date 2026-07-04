"""
Retriever — query routing across dual vector store collections.

1. Embeds the user's question.
2. Searches both narrative and structured collections (top-k each).
3. Applies SIMILARITY_THRESHOLD filtering per collection.
4. Merges results tagged with [NARRATIVE] or [STRUCTURED] origin.
5. Returns refusal message if nothing passes threshold.

This module is kept as a standalone composable function so it can later
be wrapped as a LangChain/LangGraph tool for the agentic layer.
"""

import logging
from dataclasses import dataclass, field

from langchain.schema import Document

from config import TOP_K, SIMILARITY_THRESHOLD, REFUSAL_MESSAGE
from retrieval.vector_store import DualVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval output."""

    context: str
    """Formatted context string to inject into the LLM prompt."""

    sources: list[dict] = field(default_factory=list)
    """List of source metadata dicts with similarity scores."""

    is_relevant: bool = True
    """False if no results passed the similarity threshold."""


def retrieve(query: str, vector_store: DualVectorStore) -> RetrievalResult:
    """
    Search both collections and return merged, filtered context.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    vector_store : DualVectorStore
        Initialized dual-collection vector store.

    Returns
    -------
    RetrievalResult
        Contains formatted context, source metadata, and relevance flag.
    """
    all_results: list[tuple[Document, float, str]] = []  # (doc, score, origin)

    # ── Search both collections ────────────────────────────────────────
    for target, label in [("narrative", "NARRATIVE"), ("structured", "STRUCTURED")]:
        try:
            count = vector_store.get_document_count(target)
            if count == 0:
                logger.debug("Skipping empty collection: %s", target)
                continue

            results = vector_store.similarity_search_with_score(
                query, collection_target=target, k=TOP_K
            )

            if not results:
                continue

            # Check best score against threshold
            best_score = max(score for _, score in results)

            if best_score < SIMILARITY_THRESHOLD:
                logger.debug(
                    "Collection '%s' best score %.3f below threshold %.3f — discarded.",
                    target,
                    best_score,
                    SIMILARITY_THRESHOLD,
                )
                continue

            # Keep results that individually meet threshold
            for doc, score in results:
                if score >= SIMILARITY_THRESHOLD:
                    all_results.append((doc, score, label))

        except Exception as e:
            logger.warning("Error searching '%s' collection: %s", target, e)

    # ── Nothing relevant found ─────────────────────────────────────────
    if not all_results:
        return RetrievalResult(
            context="",
            sources=[],
            is_relevant=False,
        )

    # ── Sort by score (highest first) and build context ────────────────
    all_results.sort(key=lambda x: x[1], reverse=True)

    context_parts: list[str] = []
    sources: list[dict] = []

    for doc, score, label in all_results:
        context_parts.append(f"[{label}] {doc.page_content}")
        sources.append(
            {
                **doc.metadata,
                "similarity_score": round(score, 4),
                "origin": label,
                "content_preview": doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content,
            }
        )

    context = "\n\n---\n\n".join(context_parts)

    return RetrievalResult(
        context=context,
        sources=sources,
        is_relevant=True,
    )
