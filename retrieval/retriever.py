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

from config import RETRIEVAL_POLICIES
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


def rerank_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """
    Placeholder for a future reranking stage.
    Must not modify candidate order or scores. It simply returns the candidates unchanged.
    Its purpose is to establish a stable extension point for future reranking implementations.
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
    indexed_document_counts: dict[str, int] = {}

    for target, label in [("narrative", "NARRATIVE"), ("structured", "STRUCTURED")]:
        try:
            policy = RETRIEVAL_POLICIES.get(target, {"threshold": 0.50, "top_k": 4})
            top_k = policy.get("top_k", 4)
            count = vector_store.get_document_count(target)
            indexed_document_counts[target] = count

            results = []
            if count > 0:
                results = vector_store.similarity_search_with_score(
                    query, collection_target=target, k=top_k
                )

            for doc, score in results:
                candidates.append({
                    "doc": doc,
                    "score": score,
                    "collection": target,
                    "origin": label,
                    "metadata": doc.metadata.copy() if doc.metadata else {}
                })

        except Exception as e:
            logger.warning("Error searching '%s' collection during search phase: %s", target, e)
            indexed_document_counts[target] = 0

    # ── Phase 2: Optional Future Reranking ─────────────────────────────
    candidates = rerank_candidates(query, candidates)

    # ── Phase 3: Filtering & Diagnostics Logging ───────────────────────
    retained_candidates: list[dict] = []

    for target, label in [("narrative", "NARRATIVE"), ("structured", "STRUCTURED")]:
        policy = RETRIEVAL_POLICIES.get(target, {"threshold": 0.50, "top_k": 4})
        threshold = policy.get("threshold", 0.50)
        top_k = policy.get("top_k", 4)
        indexed_count = indexed_document_counts.get(target, 0)

        # Filter candidates belonging to this collection
        coll_candidates = [c for c in candidates if c["collection"] == target]

        retained = []
        discarded = []

        scores = [c["score"] for c in coll_candidates]
        highest_score = max(scores) if scores else 0.0
        lowest_score = min(scores) if scores else 0.0
        average_score = sum(scores) / len(scores) if scores else 0.0

        for item in coll_candidates:
            if item["score"] >= threshold:
                retained.append(item)
                retained_candidates.append(item)
            else:
                discarded.append(item)

        # Determine failure reason if applicable
        failure_reason = "None"
        if indexed_count == 0 or not coll_candidates:
            failure_reason = "RETRIEVAL_FAILURE"
        elif not retained:
            failure_reason = "THRESHOLD_FAILURE"

        # Prepare sorted score lists for logging
        retrieved_scores_desc = sorted([c["score"] for c in coll_candidates], reverse=True)
        accepted_scores_desc = sorted([c["score"] for c in retained], reverse=True)
        rejected_scores_desc = sorted([c["score"] for c in discarded], reverse=True)

        retrieved_scores_str = "\n".join([f"{s:.4f}" for s in retrieved_scores_desc]) if retrieved_scores_desc else "None"
        accepted_scores_str = "\n".join([f"{s:.4f}" for s in accepted_scores_desc]) if accepted_scores_desc else "None"
        rejected_scores_str = "\n".join([f"{s:.4f}" for s in rejected_scores_desc]) if rejected_scores_desc else "None"

        # Log detailed diagnostics as requested
        logger.info(
            f"--- Retrieval Diagnostics for {target.upper()} ---\n"
            f"Query: {query}\n"
            f"Collection Name: {target}\n"
            f"Retrieval Policy: {policy}\n"
            f"Top-K: {top_k}\n"
            f"Number of Indexed Documents: {indexed_count}\n"
            f"Retrieved Candidate Count: {len(coll_candidates)}\n"
            f"Retained Candidate Count: {len(retained)}\n"
            f"Discarded Candidate Count: {len(discarded)}\n"
            f"Highest Similarity: {highest_score:.4f}\n"
            f"Lowest Similarity: {lowest_score:.4f}\n"
            f"Average Similarity: {average_score:.4f}\n"
            f"Failure Reason: {failure_reason}\n\n"
            f"Retrieved:\n{retrieved_scores_str}\n\n"
            f"Threshold:\n{threshold:.4f}\n\n"
            f"Accepted:\n{accepted_scores_str}\n\n"
            f"Rejected:\n{rejected_scores_str}"
        )

        # Log chunk-level status details
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

    # ── Phase 4: Context Construction ──────────────────────────────────
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
                **item["metadata"],
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
