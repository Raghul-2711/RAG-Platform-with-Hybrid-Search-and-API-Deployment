"""
Base abstractions for document loaders.

Each loader converts a supported file format into a normalized
document representation that the RAG chunking pipeline can consume.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from abc import ABC, abstractmethod


@dataclass
class LoadedDocument:
    """Normalized representation of a loaded document."""

    source: str
    text: str
    metadata: Dict = field(default_factory=dict)


class DocumentLoader(ABC):
    """Abstract interface implemented by all document loaders."""

    @abstractmethod
    def can_load(self, file_path: str) -> bool:
        """Return True if this loader supports the given file."""
        pass

    @abstractmethod
    def load(self, file_path: str) -> List[LoadedDocument]:
        """Load a file and return normalized documents."""
        pass