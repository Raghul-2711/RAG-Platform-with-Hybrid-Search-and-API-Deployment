"""
Loader for PDF documents using PyMuPDF.
"""

import os
import re
from typing import List

import fitz

from .base import DocumentLoader, LoadedDocument


class PDFLoader(DocumentLoader):
    """Loads PDF files page-by-page."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    _SECTION_HEADING_PATTERN = re.compile(
        r"^\s*(bibliography|references|reference list)\s*$",
        re.IGNORECASE,
    )

    def can_load(self, file_path: str) -> bool:
        """Return True if the file is a PDF document."""
        extension = os.path.splitext(file_path)[1].lower()
        return extension in self.SUPPORTED_EXTENSIONS

    def load(self, file_path: str) -> List[LoadedDocument]:
        """Extract text from each PDF page as a normalized document."""
        if not self.can_load(file_path):
            raise ValueError(f"Unsupported PDF file: {file_path}")

        filename = os.path.basename(file_path)
        documents = []
        bibliography_started = False

        with fitz.open(file_path) as pdf:
            total_pages = len(pdf)

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if not text:
                    continue

                lines = [line.strip() for line in text.splitlines() if line.strip()]

                if lines and self._SECTION_HEADING_PATTERN.match(lines[0]):
                    bibliography_started = True

                content_type = (
                    "bibliography" if bibliography_started else "content"
                )

                documents.append(
                    LoadedDocument(
                        source=filename,
                        text=text,
                        metadata={
                            "document_name": filename,
                            "file_type": "pdf",
                            "page_number": page_number,
                            "total_pages": total_pages,
                            "content_type": content_type,
                        },
                    )
                )

        return documents