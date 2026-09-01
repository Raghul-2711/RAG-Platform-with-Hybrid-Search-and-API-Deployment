"""
Loader for plain-text and Markdown documents.
"""

import os
from typing import List

from .base import DocumentLoader, LoadedDocument


class TextLoader(DocumentLoader):
    """Loads .txt and .md files as normalized documents."""

    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    def can_load(self, file_path: str) -> bool:
        """Return True if the file is a supported text format."""
        extension = os.path.splitext(file_path)[1].lower()
        return extension in self.SUPPORTED_EXTENSIONS

    def load(self, file_path: str) -> List[LoadedDocument]:
        """Read a text/Markdown file and return one normalized document."""
        if not self.can_load(file_path):
            raise ValueError(f"Unsupported text file: {file_path}")

        filename = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        return [
            LoadedDocument(
                source=filename,
                text=text,
                metadata={
                    "document_name": filename,
                    "file_type": os.path.splitext(filename)[1]
                    .lower()
                    .lstrip("."),
                },
            )
        ]