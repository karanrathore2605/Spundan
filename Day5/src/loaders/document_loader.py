from pathlib import Path

from pypdf import PdfReader

from src.models import Document


class DocumentLoader:
    """
    Loads PDF and TXT documents from the documents directory.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(self, documents_dir: Path):
        self.documents_dir = documents_dir

    def load_documents(self) -> list[Document]:
        """
        Load all supported documents.
        """

        if not self.documents_dir.exists():
            raise FileNotFoundError(
                f"Documents directory not found: {self.documents_dir}"
            )

        files = [
            file
            for file in self.documents_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not files:
            raise FileNotFoundError(
                f"No PDF or TXT files found in: {self.documents_dir}"
            )

        documents: list[Document] = []

        for file_path in sorted(files):
            if file_path.suffix.lower() == ".pdf":
                documents.extend(self._load_pdf(file_path))

            elif file_path.suffix.lower() == ".txt":
                documents.append(self._load_txt(file_path))

        return documents

    def _load_pdf(self, file_path: Path) -> list[Document]:
        """
        Load every page of a PDF as a separate Document object.
        """

        reader = PdfReader(str(file_path))

        documents: list[Document] = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            text = self._clean_text(text)

            if not text:
                continue

            documents.append(
                Document(
                    text=text,
                    source=file_path.name,
                    page=page_number,
                )
            )

        return documents

    def _load_txt(self, file_path: Path) -> Document:
        """
        Load a TXT file.
        """

        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        text = self._clean_text(text)

        return Document(
            text=text,
            source=file_path.name,
            page=None,
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Basic text cleanup.
        """

        lines = [
            line.strip()
            for line in text.splitlines()
        ]

        lines = [
            line
            for line in lines
            if line
        ]

        return "\n".join(lines)