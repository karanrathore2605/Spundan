from pathlib import Path

from app.models import Document


class DocumentLoader:

    def __init__(self, documents_dir: Path):
        self.documents_dir = documents_dir

    def load_documents(self) -> list[Document]:

        if not self.documents_dir.exists():
            raise FileNotFoundError(
                f"Documents directory not found: {self.documents_dir}"
            )

        documents = []

        for file_path in self.documents_dir.glob("*.txt"):

            try:
                content = file_path.read_text(
                    encoding="utf-8"
                ).strip()

                if not content:
                    continue

                documents.append(
                    Document(
                        content=content,
                        source=file_path.name
                    )
                )

            except OSError as exc:
                print(
                    f"Warning: Could not read {file_path}: {exc}"
                )

        return documents