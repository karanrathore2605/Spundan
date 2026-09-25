"""
Document loader module for the RAG pipeline.
Loads raw text documents from the filesystem and attaches source metadata.
"""

from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a loaded text document with its source origin and metadata."""
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Responsible for loading text files from a directory."""

    def __init__(self, directory_path: Path):
        self.directory_path = Path(directory_path)

    def load(self) -> List[Document]:
        """
        Loads all .txt and .md files from the configured directory.
        
        Returns:
            List[Document]: Extracted documents with content and source paths.
        """
        documents: List[Document] = []

        if not self.directory_path.exists():
            logger.warning("Documents directory %s does not exist.", self.directory_path)
            return documents

        if not self.directory_path.is_dir():
            logger.warning("Path %s is not a directory.", self.directory_path)
            return documents

        supported_extensions = ["*.txt", "*.md"]
        file_paths = []
        for ext in supported_extensions:
            file_paths.extend(self.directory_path.glob(ext))

        for file_path in sorted(file_paths):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if not content:
                    logger.debug("Skipping empty file: %s", file_path.name)
                    continue

                documents.append(
                    Document(
                        content=content,
                        source=file_path.name,
                        metadata={
                            "file_path": str(file_path),
                            "file_size": file_path.stat().st_size,
                        }
                    )
                )
                logger.debug("Loaded document '%s' (%d characters)", file_path.name, len(content))
            except Exception as e:
                logger.error("Failed to read file %s: %s", file_path.name, str(e))

        logger.info("Successfully loaded %d documents from %s", len(documents), self.directory_path)
        return documents
