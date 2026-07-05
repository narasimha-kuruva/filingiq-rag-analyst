"""
PDF Loader — extracts text page-by-page and chunks with page-level metadata.

Uses pdfplumber for reliable page-level extraction (handles tables and
multi-column layouts better than PyPDFLoader).
"""

import io
from typing import Union

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(
    source: Union[str, io.BytesIO],
    filename: str = "unknown.pdf",
) -> list[Document]:
    """
    Load a PDF and return chunked Documents with page-number metadata.

    Parameters
    ----------
    source : str | BytesIO
        File path (str) or in-memory buffer (BytesIO from Streamlit uploader).
    filename : str
        Original filename for metadata.

    Returns
    -------
    list[Document]
        Each chunk carries metadata:
        {"source_filename", "file_type": "pdf", "page_number"}.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    chunk_index = 0

    # pdfplumber accepts both file paths and file-like objects
    with pdfplumber.open(source) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            chunks = splitter.split_text(text)
            for chunk in chunks:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source_filename": filename,
                            "file_type": "pdf",
                            "page_number": page_num,
                            "chunk_index": chunk_index,
                        },
                    )
                )
                chunk_index += 1

    return documents
