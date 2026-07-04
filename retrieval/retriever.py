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

from langchain_core.documents import Document

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

    is_greeting: bool = False
    """True if the query is a greeting or introductory question."""


def is_greeting_query(query: str) -> bool:
    """Check if the user's query is a simple greeting or introductory question."""
    cleaned = query.strip().lower().strip("?!.,")
    greetings = {
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
        "good evening", "howdy", "hola", "yo", "hi there", "hello there",
        "who are you", "what are you", "what can you do", "help", "who created you",
        "what is filingiq", "what is this app", "how does this work", "how do I use this",
        "what's up", "sup", "how are you", "how are you doing", "hello!"
    }
    if cleaned in greetings:
        return True

    words = cleaned.split()
    if words and words[0] in {"hi", "hello", "hey", "howdy", "hola"}:
        if len(words) <= 3:
            return True

    return False


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
    import datetime
    is_greeting = is_greeting_query(query)
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. QUERY LOGGING
    logger.info(
        f"==================================================\n"
        f"QUERY\n"
        f"==================================================\n"
        f"Timestamp: {timestamp_str}\n"
        f"Question:\n{query}\n\n"
        f"Greeting:\n{is_greeting}"
    )

    if is_greeting:
        logger.info(
            "==================================================\n"
            "FINAL CONTEXT SENT TO GEMINI\n"
            "==================================================\n"
            "<Greeting query detected - Short-circuiting directly to greeting generation>"
        )
        return RetrievalResult(
            context="",
            sources=[],
            is_relevant=True,
            is_greeting=True,
        )

    all_results: list[tuple[Document, float, str]] = []  # (doc, score, origin)

    # ── Search both collections ────────────────────────────────────────
    for target, label in [("narrative", "NARRATIVE"), ("structured", "STRUCTURED")]:
        try:
            count = vector_store.get_document_count(target)
            
            # 2. COLLECTION STATISTICS
            logger.info(
                f"Collection: {target}\n"
                f"Documents indexed: {count}\n"
                f"Top-K: {TOP_K}"
            )

            if count == 0:
                logger.info(
                    f"Threshold: {SIMILARITY_THRESHOLD}\n"
                    f"Number of retrieved chunks in '{target}': 0\n"
                    f"Number of discarded chunks in '{target}': 0\n"
                    f"Number of retained chunks in '{target}': 0"
                )
                continue

            results = vector_store.similarity_search_with_score(
                query, collection_target=target, k=TOP_K
            )

            if not results:
                logger.info(
                    f"Threshold: {SIMILARITY_THRESHOLD}\n"
                    f"Number of retrieved chunks in '{target}': 0\n"
                    f"Number of discarded chunks in '{target}': 0\n"
                    f"Number of retained chunks in '{target}': 0"
                )
                continue

            retrieved_count = len(results)
            retained_chunks = []
            discarded_chunks = []

            # Check best score against threshold
            best_score = max(score for _, score in results)

            if best_score < SIMILARITY_THRESHOLD:
                # All chunks in this collection are discarded
                for doc, score in results:
                    discarded_chunks.append({
                        "score": score,
                        "reason": f"Best score in collection '{target}' ({best_score:.4f}) is below threshold ({SIMILARITY_THRESHOLD:.4f})"
                    })
            else:
                for doc, score in results:
                    if score >= SIMILARITY_THRESHOLD:
                        retained_chunks.append((doc, score))
                        all_results.append((doc, score, label))
                    else:
                        discarded_chunks.append({
                            "score": score,
                            "reason": f"Individual chunk score ({score:.4f}) is below threshold ({SIMILARITY_THRESHOLD:.4f})"
                        })

            # 4. THRESHOLD FILTERING
            logger.info(
                f"Threshold: {SIMILARITY_THRESHOLD}\n"
                f"Number of retrieved chunks in '{target}': {retrieved_count}\n"
                f"Number of discarded chunks in '{target}': {len(discarded_chunks)}\n"
                f"Number of retained chunks in '{target}': {len(retained_chunks)}"
            )

            for idx, (doc, score) in enumerate(results, start=1):
                is_kept = any(r[0] == doc for r in retained_chunks)
                status = "KEPT" if is_kept else "DISCARDED"
                msg = (
                    f"Chunk #{idx}\n"
                    f"Similarity: {score:.4f}\n"
                    f"Status: {status}"
                )
                if not is_kept:
                    reason = next((d["reason"] for d in discarded_chunks if d["score"] == score), "Below threshold")
                    msg += f"\nReason: {reason}"
                logger.info(msg)

        except Exception as e:
            logger.warning("Error searching '%s' collection: %s", target, e)

    # ── Nothing relevant found ─────────────────────────────────────────
    if not all_results:
        # 5. FINAL CONTEXT SENT TO GEMINI
        logger.info(
            "==================================================\n"
            "FINAL CONTEXT SENT TO GEMINI\n"
            "==================================================\n\n"
            "<No relevant context found - Short-circuiting directly to refusal>"
        )
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

    # 5. FINAL CONTEXT SENT TO GEMINI
    logger.info(
        "==================================================\n"
        "FINAL CONTEXT SENT TO GEMINI\n"
        "==================================================\n\n"
        f"{context}"
    )

    return RetrievalResult(
        context=context,
        sources=sources,
        is_relevant=True,
    )
