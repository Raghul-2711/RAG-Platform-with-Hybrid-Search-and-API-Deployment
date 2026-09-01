"""
Document loader package.

Provides loaders for TXT, Markdown, PDF, DOCX, and PPTX files.
"""

from .base import DocumentLoader, LoadedDocument
from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .docx_loader import DocxLoader
from .pptx_loader import PptxLoader


__all__ = [
    "DocumentLoader",
    "LoadedDocument",
    "TextLoader",
    "PDFLoader",
    "DocxLoader",
    "PptxLoader",
]