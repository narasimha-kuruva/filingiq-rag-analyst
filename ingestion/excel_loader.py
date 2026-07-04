"""
Excel Loader — converts each row to a natural-language sentence.

Loads all sheets with pandas, converts each row into a human-readable
sentence embedding, and stores one Document per row with sheet+row metadata.
"""

import io
from typing import Union

import pandas as pd
from langchain_core.documents import Document


def _row_to_sentence(row: pd.Series, sheet_name: str, row_index: int) -> str:
    """Convert a single DataFrame row into a natural-language sentence."""
    parts: list[str] = []
    for col, val in row.items():
        if pd.notna(val):
            parts.append(f"{col} = {val}")
    fields = ", ".join(parts)
    return f"In sheet '{sheet_name}', row {row_index}: {fields}"


def load_excel(
    source: Union[str, io.BytesIO],
    filename: str = "unknown.xlsx",
) -> list[Document]:
    """
    Load an Excel workbook and return one Document per row.

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
        {"source_filename", "file_type": "excel", "sheet_name", "row_index"}.
    """
    # sheet_name=None → returns dict {sheet_name: DataFrame}
    sheets = pd.read_excel(source, sheet_name=None, engine="openpyxl")

    documents: list[Document] = []
    for sheet_name, df in sheets.items():
        # Drop completely empty rows
        df = df.dropna(how="all").reset_index(drop=True)

        for idx, row in df.iterrows():
            row_index = int(idx) + 2  # Excel rows are 1-indexed + header row
            sentence = _row_to_sentence(row, sheet_name, row_index)

            documents.append(
                Document(
                    page_content=sentence,
                    metadata={
                        "source_filename": filename,
                        "file_type": "excel",
                        "sheet_name": str(sheet_name),
                        "row_index": row_index,
                    },
                )
            )

    return documents
