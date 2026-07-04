"""
Ingestion Router — detects file type from extension and dispatches to the
correct loader function.

Returns a unified list[Document] regardless of source type, each with
standardized metadata including source_filename and file_type.
"""

import io
import os
from typing import Union

from langchain.schema import Document

from config import SUPPORTED_EXTENSIONS
from ingestion.pdf_loader import load_pdf
from ingestion.docx_loader import load_docx
from ingestion.txt_loader import load_txt
from ingestion.excel_loader import load_excel
from ingestion.csv_loader import load_csv


# Map extensions to (loader_function, collection_target)
# "narrative" → narrative_store, "structured" → structured_store
_LOADER_MAP = {
    ".pdf":  (load_pdf,   "narrative"),
    ".docx": (load_docx,  "narrative"),
    ".txt":  (load_txt,   "narrative"),
    ".xlsx": (load_excel, "structured"),
    ".csv":  (load_csv,   "structured"),
}


def detect_file_type(filename: str) -> str:
    """Return the lowercase extension (e.g. '.pdf') from a filename."""
    _, ext = os.path.splitext(filename)
    return ext.lower()


def ingest_file(
    source: Union[str, io.BytesIO],
    filename: str,
) -> tuple[list[Document], str]:
    """
    Route a file to the appropriate loader based on its extension.

    Parameters
    ----------
    source : str | BytesIO
        File path (str) or in-memory buffer (BytesIO, e.g. from Streamlit uploader).
    filename : str
        Original filename — used for extension detection and metadata.

    Returns
    -------
    tuple[list[Document], str]
        (documents, collection_target) where collection_target is
        "narrative" or "structured".

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    ext = detect_file_type(filename)

    if ext not in _LOADER_MAP:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: {supported}"
        )

    loader_fn, collection_target = _LOADER_MAP[ext]
    documents = loader_fn(source, filename)

    return documents, collection_target
