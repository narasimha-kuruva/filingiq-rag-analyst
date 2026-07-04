"""
Citation Formatter — converts raw metadata dicts into human-readable
citation strings for display in chat responses.
"""


def format_citation(metadata: dict) -> str:
    """
    Convert a metadata dict into a human-readable citation string.

    Examples
    --------
    PDF:   "sample_10k.pdf, p.12"
    DOCX:  "analyst_memo.docx, Section: Risk Factors, ¶3"
    TXT:   "notes.txt, Chunk 5"
    Excel: "financials.xlsx, Sheet1, Row 34"
    CSV:   "ratios.csv, Row 12"
    """
    filename = metadata.get("source_filename", "unknown")
    file_type = metadata.get("file_type", "unknown")

    if file_type == "pdf":
        page = metadata.get("page_number", "?")
        return f"{filename}, p.{page}"

    elif file_type == "docx":
        section = metadata.get("section_title", "Untitled")
        para = metadata.get("paragraph_index", "?")
        return f"{filename}, Section: {section}, ¶{para}"

    elif file_type == "txt":
        chunk = metadata.get("chunk_index", "?")
        return f"{filename}, Chunk {chunk}"

    elif file_type == "excel":
        sheet = metadata.get("sheet_name", "?")
        row = metadata.get("row_index", "?")
        return f"{filename}, {sheet}, Row {row}"

    elif file_type == "csv":
        row = metadata.get("row_index", "?")
        return f"{filename}, Row {row}"

    else:
        return filename


def format_sources_list(sources: list[dict]) -> str:
    """
    Format a list of source metadata dicts into a bulleted citation block.

    Returns a string like:
        • sample_10k.pdf, p.12 (score: 0.82)
        • financials.xlsx, Sheet1, Row 4 (score: 0.75)
    """
    if not sources:
        return "No sources available."

    lines: list[str] = []
    seen: set[str] = set()  # deduplicate identical citations

    for src in sources:
        citation = format_citation(src)
        score = src.get("similarity_score", 0)

        key = f"{citation}:{score}"
        if key in seen:
            continue
        seen.add(key)

        lines.append(f"• {citation} (score: {score:.4f})")

    return "\n".join(lines)
