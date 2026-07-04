"""
CSV Loader — converts each row to a natural-language sentence.

Same row-to-sentence approach as Excel, but simpler since there's only one
"sheet" (the CSV itself).
"""

import io
from typing import Union

import pandas as pd
from langchain.schema import Document


def _row_to_sentence(row: pd.Series, row_index: int, filename: str) -> str:
    """Convert a single DataFrame row into a natural-language sentence."""
    parts: list[str] = []
    for col, val in row.items():
        if pd.notna(val):
            parts.append(f"{col} = {val}")
    fields = ", ".join(parts)
    return f"In '{filename}', row {row_index}: {fields}"


def load_csv(
    source: Union[str, io.BytesIO],
    filename: str = "unknown.csv",
) -> list[Document]:
    """
    Load a CSV file and return one Document per row.

    Parameters
    ----------
    source : str | BytesIO
        File path or in-memory buffer.
    filename : str
        Original filename for metadata.

    Returns
    -------
    list[Document]
        Each row is a Document with metadata:
        {"source_filename", "file_type": "csv", "row_index"}.
    """
    if isinstance(source, str):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)

    # Drop completely empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    documents: list[Document] = []
    for idx, row in df.iterrows():
        row_index = int(idx) + 2  # 1-indexed + header row
        sentence = _row_to_sentence(row, row_index, filename)

        documents.append(
            Document(
                page_content=sentence,
                metadata={
                    "source_filename": filename,
                    "file_type": "csv",
                    "row_index": row_index,
                },
            )
        )

    return documents
