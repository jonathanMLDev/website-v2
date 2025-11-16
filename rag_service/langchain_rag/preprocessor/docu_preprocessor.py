"""
Documentation preprocessing pipeline for LangChain RAG
"""

from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    markdown = None
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import hashlib


class DocuPreprocessor:
    """Process Boost library documentation for RAG"""

    def __init__(
        self,
        doc_data_dir: str = "",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.doc_data_dir = Path(doc_data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def load_documents(self) -> List[Document]:
        """Load and process all documents from data directory"""
        docs_path = self.doc_data_dir
        if docs_path.exists():
            return self._process_documentation(docs_path)
        return []

    def _process_documentation(self, docs_path: Path) -> List[Document]:
        """Process Boost documentation files"""
        documents = []

        for file_path in tqdm(docs_path.rglob("*"), desc="Processing documentation"):
            if file_path.is_file() and file_path.suffix in [".txt", ".md", ".html"]:
                try:
                    content = self._read_file(file_path)
                    if content:
                        first_line = content.split("\n")[0]
                        if "Source URL:" in first_line:
                            url = first_line.split("Source URL:")[1].strip()
                            source = "Boost Documentation"
                        else:
                            url = str(file_path.relative_to(docs_path))
                            source = "github.com/boostorg"
                        doc_id = hashlib.md5(url.encode()).hexdigest()
                        doc = Document(
                            page_content=content.replace(first_line, ""),
                            id=doc_id,
                            metadata={
                                "source": source,
                                "type": "documentation",
                                "library": self._extract_library_name(file_path),
                                "file_type": file_path.suffix,
                                "url": url,
                                "version": "1.89.0"
                            },
                        )
                        documents.append(doc)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        return documents

    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read and clean file content"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Clean content based on file type
            if file_path.suffix == ".html":
                soup = BeautifulSoup(content, "html.parser")
                content = soup.get_text()
            elif file_path.suffix == ".md":
                # Convert markdown to plain text
                if MARKDOWN_AVAILABLE and markdown:
                    html_content = markdown.markdown(content)
                    soup = BeautifulSoup(html_content, "html.parser")
                    content = soup.get_text()
                else:
                    # Fallback: just use the markdown text as-is
                    # Remove markdown syntax roughly
                    content = re.sub(r'#{1,6}\s+', '', content)  # Remove headers
                    content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)  # Remove bold
                    content = re.sub(r'\*(.+?)\*', r'\1', content)  # Remove italic

            # Clean up whitespace
            # content = re.sub(r"\s+", " ", content).strip()
            content = re.sub(r"\n\s*\n+", "\n\n", content).strip()
            return content if len(content) > 50 else None

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def _extract_library_name(self, file_path: Path) -> str:
        """Extract Boost library name from file path"""
        parts = file_path.parts
        for i, part in enumerate(parts):
            if part == "en" and i + 1 < len(parts):
                return parts[i + 1]
        return "unknown"

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunked_documents = []
        for doc in documents:
            doc.id = doc.id if doc.id else hashlib.md5(doc.metadata.get("url", "").encode()).hexdigest()
            chunks = self.text_splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                if len(chunk.page_content) < 50:
                    continue
                chunk.id = f"{doc.id}-{i:03d}"
                chunked_documents.append(chunk)

        return chunked_documents

