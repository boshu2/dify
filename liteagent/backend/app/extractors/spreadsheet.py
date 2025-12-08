"""
Spreadsheet document extractor using pandas.

Extracts text content from CSV, Excel files for RAG processing.
"""
import io
from typing import BinaryIO

from app.extractors.base import DocumentExtractor, ExtractedContent


class SpreadsheetExtractor(DocumentExtractor):
    """Extract text from spreadsheet files using pandas."""

    @property
    def supported_extensions(self) -> set[str]:
        return {".csv", ".xlsx", ".xls"}

    async def extract(self, file: BinaryIO, filename: str) -> ExtractedContent:
        """Extract text from spreadsheet file."""
        import pandas as pd

        content = file.read()
        file_io = io.BytesIO(content)
        ext = filename.lower().split(".")[-1]

        metadata = {
            "filename": filename,
            "format": ext,
        }

        # Read based on file type
        if ext == "csv":
            # Try different encodings for CSV
            try:
                df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), encoding="latin-1")
            sheets_data = {"Sheet1": df}
        else:
            # Excel file - read all sheets
            file_io.seek(0)
            excel_file = pd.ExcelFile(file_io, engine="openpyxl")
            sheets_data = {
                sheet: pd.read_excel(excel_file, sheet_name=sheet)
                for sheet in excel_file.sheet_names
            }
            metadata["sheet_count"] = len(sheets_data)
            metadata["sheet_names"] = list(sheets_data.keys())

        # Convert to text format
        text_parts = []
        total_rows = 0
        total_cols = 0

        for sheet_name, df in sheets_data.items():
            total_rows += len(df)
            total_cols = max(total_cols, len(df.columns))

            # Add sheet header for multi-sheet files
            if len(sheets_data) > 1:
                text_parts.append(f"=== {sheet_name} ===")

            # Convert headers
            headers = " | ".join(str(col) for col in df.columns)
            text_parts.append(headers)
            text_parts.append("-" * len(headers))

            # Convert rows (limit to prevent huge outputs)
            max_rows = 1000  # Configurable limit
            for idx, row in df.head(max_rows).iterrows():
                row_text = " | ".join(str(val) for val in row.values)
                text_parts.append(row_text)

            if len(df) > max_rows:
                text_parts.append(f"... ({len(df) - max_rows} more rows)")

            text_parts.append("")  # Blank line between sheets

        metadata["row_count"] = total_rows
        metadata["column_count"] = total_cols

        full_text = "\n".join(text_parts)

        return ExtractedContent(
            text=full_text,
            metadata=metadata,
            pages=len(sheets_data),
        )
