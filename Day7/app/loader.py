import logging
from pathlib import Path
from app.models import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads and validates text documents for the RAG pipeline."""

    def __init__(self, documents_dir: Path | str):
        self.documents_dir = Path(documents_dir)

    def load_document(self, file_path: Path | str) -> Document | None:
        """
        Load a single document from disk.
        Returns None if file is empty or unreadable.
        """
        path = Path(file_path)
        if not path.is_file():
            print(f"Warning: File not found: {path}")
            return None

        # Try reading with utf-8 first, then fallback to utf-8-sig / latin-1
        content = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                content = path.read_text(encoding=encoding)
                break
            except (UnicodeDecodeError, OSError):
                continue

        if content is None:
            print(f"Warning: Could not decode '{path.name}', skipping.")
            return None

        stripped_content = content.strip()
        if not stripped_content:
            # Edge Case 2: Empty document
            print(f"Warning: document '{path.name}' is empty and was skipped.")
            return None

        return Document(
            content=stripped_content,
            source=path.name,
            metadata={"source": path.name, "path": str(path)},
        )

    def load_documents(self) -> list[Document]:
        """
        Load all supported text files from the configured documents directory.
        Gracefully handles empty directories and files.
        """
        if not self.documents_dir.exists():
            # Edge Case 1: Directory does not exist
            print(f"No documents found.\nPlease add a text document first to: {self.documents_dir}")
            return []

        documents: list[Document] = []
        text_files = sorted(list(self.documents_dir.glob("*.txt")))

        if not text_files:
            # Edge Case 1: No documents found
            print("No documents found.\nPlease add a text document first.")
            return []

        for file_path in text_files:
            doc = self.load_document(file_path)
            if doc:
                documents.append(doc)

        if not documents:
            print("No documents found.\nPlease add a text document first.")

        return documents
