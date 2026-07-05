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

from config import TOP_K, NARRATIVE_SIMILARITY_THRESHOLD, STRUCTURED_SIMILARITY_THRESHOLD, REFUSAL_MESSAGE
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


def _rerank_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """
    Placeholder for a future reranking stage.

    Parameters
    ----------
    query : str
        The user's query.
    candidates : list[dict]
        Retrieved candidate documents with similarity scores.

    Returns
    -------
    list[dict]
        The list of candidates, unmodified.
    """
    return candidates


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

    # ── Phase 1: Vector Search (Candidate Retrieval) ───────────────────
    candidates: list[dict] = []

    for target, label in [("narrative", "NARRATIVE"), ("structured", "STRUCTURED")]:
        try:
            count = vector_store.get_document_count(target)
            results = []
            if count > 0:
                results = vector_store.similarity_search_with_score(
                    query, collection_target=target, k=TOP_K
                )

            for doc, score in results:
                candidates.append({
                    "doc": doc,
                    "score": score,
                    "collection": target,
                    "origin": label
                })

        except Exception as e:
            logger.warning("Error searching '%s' collection during search phase: %s", target, e)

    # ── Phase 2: Reranking Step (Optional Placeholder) ─────────────────
    candidates = _rerank_candidates(query, candidates)

    # ── Phase 3: Threshold Filtering & Logging ──────────────────────────
    retained_candidates: list[dict] = []

    for target, label in [("narrative", "NARRATIVE"), ("structured", "STRUCTURED")]:
        # Determine target threshold
        if target == "narrative":
            threshold = NARRATIVE_SIMILARITY_THRESHOLD
        else:
            threshold = STRUCTURED_SIMILARITY_THRESHOLD

        # Filter candidates belonging to this collection
        coll_candidates = [c for c in candidates if c["collection"] == target]

        retained = []
        discarded = []

        scores = [c["score"] for c in coll_candidates]
        highest_score = max(scores) if scores else None
        lowest_score = min(scores) if scores else None

        for item in coll_candidates:
            if item["score"] >= threshold:
                retained.append(item)
                retained_candidates.append(item)
            else:
                discarded.append(item)

        # Log detailed audit diagnostics
        logger.info(
            f"Collection: {target}\n"
            f"Threshold: {threshold:.4f}\n"
            f"Top-K requested: {TOP_K}\n"
            f"Retrieved count: {len(coll_candidates)}\n"
            f"Retained count: {len(retained)}\n"
            f"Discarded count: {len(discarded)}\n"
            f"Highest similarity: {f'{highest_score:.4f}' if highest_score is not None else 'None'}\n"
            f"Lowest similarity: {f'{lowest_score:.4f}' if lowest_score is not None else 'None'}"
        )

        # Log details of each candidate from this collection
        for idx, item in enumerate(coll_candidates, start=1):
            is_kept = item in retained
            status = "KEPT" if is_kept else "DISCARDED"
            msg = (
                f"Chunk #{idx}\n"
                f"Similarity: {item['score']:.4f}\n"
                f"Status: {status}"
            )
            if not is_kept:
                msg += f"\nReason: Individual chunk score ({item['score']:.4f}) is below threshold ({threshold:.4f})"
            logger.info(msg)

    # ── Phase 4: Context Generation ────────────────────────────────────
    if not retained_candidates:
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

    # Sort all retained candidates by score descending
    retained_candidates.sort(key=lambda x: x["score"], reverse=True)

    context_parts: list[str] = []
    sources: list[dict] = []

    for item in retained_candidates:
        doc = item["doc"]
        score = item["score"]
        label = item["origin"]
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
