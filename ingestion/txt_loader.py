"""
TXT Loader — reads plain text and chunks it.

No page or section concept is available, so each chunk is tagged with a
sequential chunk_index for citation.
"""

import io
from typing import Union

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_txt(
    source: Union[str, io.BytesIO],
    filename: str = "unknown.txt",
) -> list[Document]:
    """
    Load a TXT file and return chunked Documents with chunk_index metadata.

    Parameters
    ----------
    source : str | BytesIO
        File path or in-memory buffer.
    filename : str
        Original filename for metadata.

    Returns
    -------
    list[Document]
        Each chunk carries metadata:
        {"source_filename", "file_type": "txt", "chunk_index"}.
    """
    # Read the raw text
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
    else:
        raw_text = source.read().decode("utf-8", errors="replace")

    if not raw_text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(raw_text)

    documents: list[Document] = []
    for idx, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source_filename": filename,
                    "file_type": "txt",
                    "chunk_index": idx,
                },
            )
        )

    return documents
