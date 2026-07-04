"""
LLM Chain — builds the retrieval → prompt → Gemini generation pipeline.

# ═══════════════════════════════════════════════════════════════════════════
# TODO: AGENTIC LAYER (future iteration)
# ═══════════════════════════════════════════════════════════════════════════
# When config.AGENTIC_MODE is enabled, this module should:
# 1. Wrap `retrieve()` and the LLM call as LangChain/LangGraph "tools".
# 2. Add additional tools:
#    - calculate_ratio(numerator, denominator) → computed financial ratio
#    - compare_companies(company_a, company_b, metric) → side-by-side comparison
#    - web_search_recent_news(query) → retrieve recent news for context
# 3. Use a ReAct or tool-calling agent that decides which tool(s) to invoke
#    based on the user's question.
# 4. The core retriever.retrieve() and this module's generate() function
#    remain unchanged — they are already composable standalone functions.
"""

import os
import logging
from dataclasses import dataclass

from config import LLM_MODEL, TEMPERATURE, REFUSAL_MESSAGE, AGENTIC_MODE
from generation.prompts import RAG_PROMPT
from retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Container for the LLM response + metadata."""

    answer: str
    sources: list[dict]
    is_relevant: bool


def _get_api_key() -> str | None:
    """Retrieve Google API key from Streamlit secrets or env var."""
    try:
        import streamlit as st
        key = st.secrets.get("GOOGLE_API_KEY")
        if key and key != "your-gemini-api-key-here":
            return key
    except Exception:
        pass

    key = os.environ.get("GOOGLE_API_KEY")
    if key and key != "your-gemini-api-key-here":
        return key

    return None


def generate(
    question: str,
    retrieval_result: RetrievalResult,
) -> GenerationResult:
    """
    Generate a grounded answer using retrieved context + Gemini.

    If AGENTIC_MODE is True, this will eventually route through a
    tool-calling agent. Currently always uses direct chain.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    retrieval_result : RetrievalResult
        Output from retriever.retrieve().

    Returns
    -------
    GenerationResult
        Contains the answer text, source metadata, and relevance flag.
    """
    # ── If it is a greeting, handle with a greeting prompt ──────────────
    if getattr(retrieval_result, "is_greeting", False):
        api_key = _get_api_key()
        if not api_key:
            return GenerationResult(
                answer="👋 Hello! I am FilingIQ, your document-grounded financial analyst assistant. "
                       "Please configure your Gemini API key in the secrets or environment to start asking questions about uploaded documents.",
                sources=[],
                is_relevant=True,
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.prompts import PromptTemplate

            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                google_api_key=api_key,
                temperature=TEMPERATURE,
                convert_system_message_to_human=True,
            )

            greeting_template = """You are FilingIQ, a document-grounded financial analyst assistant.
The user is greeting you or asking about your capabilities.
Respond with a polite, professional, and friendly response. Introduce yourself, state your features (analyzing 10-K PDFs, DOCX memos, XLSX financials, CSV ratios, and TXT notes), and invite the user to upload files in the sidebar and ask questions. Keep it concise (1-2 short paragraphs).

USER QUERY: {question}
RESPONSE:"""
            prompt = PromptTemplate(input_variables=["question"], template=greeting_template)
            prompt_text = prompt.format(question=question)

            # Log before Gemini API call (greeting)
            logger.info(
                f"=== GEMINI CALL DETAILS (GREETING) ===\n"
                f"User Question: {question}\n"
                f"Context Length: 0 (Greeting)\n"
                f"Prompt Length: {len(prompt_text)}\n"
                f"Number of retrieved chunks: 0"
            )

            response = llm.invoke(prompt_text)
            answer = response.content

            # Log after Gemini returns (greeting)
            logger.info(
                f"=== GEMINI CALL RESPONSE (GREETING) ===\n"
                f"Generation completed successfully.\n"
                f"Answer Length: {len(answer)}\n"
                f"COMPLETE generated answer:\n{answer}"
            )

            return GenerationResult(
                answer=answer,
                sources=[],
                is_relevant=True,
            )
        except Exception as e:
            return GenerationResult(
                answer="👋 Hello! I am FilingIQ, your AI financial analyst. How can I help you analyze your financial documents today?",
                sources=[],
                is_relevant=True,
            )

    # ── Short-circuit: nothing relevant found ──────────────────────────
    if not retrieval_result.is_relevant:
        return GenerationResult(
            answer=REFUSAL_MESSAGE,
            sources=[],
            is_relevant=False,
        )

    # ── Future: agentic routing ────────────────────────────────────────
    if AGENTIC_MODE:
        # Placeholder for tool-calling agent
        raise NotImplementedError(
            "Agentic mode is not yet implemented. "
            "Set AGENTIC_MODE = False in config.py."
        )

    # ── Direct chain: retriever → prompt → Gemini ─────────────────────
    api_key = _get_api_key()
    if not api_key:
        return GenerationResult(
            answer="⚠️ Google API key not configured. Please add your Gemini API key "
                   "to `.streamlit/secrets.toml` or set the `GOOGLE_API_KEY` "
                   "environment variable.",
            sources=[],
            is_relevant=False,
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=api_key,
            temperature=TEMPERATURE,
            convert_system_message_to_human=True,
        )

        # Format prompt
        prompt_text = RAG_PROMPT.format(
            context=retrieval_result.context,
            question=question,
        )

        # Log before Gemini API call
        logger.info(
            f"=== GEMINI CALL DETAILS ===\n"
            f"User Question: {question}\n"
            f"Context Length: {len(retrieval_result.context)}\n"
            f"Prompt Length: {len(prompt_text)}\n"
            f"Number of retrieved chunks: {len(retrieval_result.sources)}"
        )

        # Invoke LLM
        response = llm.invoke(prompt_text)
        answer = response.content

        # Log after Gemini returns
        logger.info(
            f"=== GEMINI CALL RESPONSE ===\n"
            f"Generation completed successfully.\n"
            f"Answer Length: {len(answer)}\n"
            f"COMPLETE generated answer:\n{answer}"
        )

        return GenerationResult(
            answer=answer,
            sources=retrieval_result.sources,
            is_relevant=True,
        )

    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        return GenerationResult(
            answer=f"⚠️ An error occurred during generation: {str(e)}",
            sources=retrieval_result.sources,
            is_relevant=False,
        )
