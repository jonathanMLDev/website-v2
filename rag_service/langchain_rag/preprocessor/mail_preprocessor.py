"""
Mail preprocessing pipeline for LangChain RAG
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from tqdm import tqdm
import hashlib
import structlog

logger = structlog.get_logger(__name__)
from ..llm import LLMHelper


class MailPreprocessor:
    """Process Boost mailing list data for RAG"""

    def __init__(
        self,
        mail_data_dir: str = "",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.mail_data_dir = Path(mail_data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.content_by_url: Dict[str, str] = {}
        self.logger = logger.bind(component="MailPreprocessor")

        from config.rag_config import DEFAULT_CONFIG
        self.llm_helper = LLMHelper(config=DEFAULT_CONFIG)

    def load_emails(self) -> List[Document]:
        """Load and process all emails from data directory"""
        emails_path = self.mail_data_dir
        if emails_path.exists():
            return self._process_mail_data(emails_path)
        return []

    def _process_mail_data(self, mail_path: Path) -> List[Document]:
        """Process Boost mailing list data"""
        documents = []

        for thread_file in tqdm(mail_path.rglob("*.json"), desc="Processing mail threads"):
            try:
                with open(thread_file, "r", encoding="utf-8") as f:
                    mail_data = json.load(f)
                thread_docs = self.process_mail_list(mail_data)
                documents.extend(thread_docs)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in {thread_file}: {e}")
            except Exception as e:
                print(f"Error processing {thread_file}: {e}")
        self.logger.info(f"Processed {len(documents)} documents")

        # documents = self.complete_metadata(documents)
        self.logger.info(f"Completed metadata for {len(documents)} documents")

        return documents

    def process_mail_list(self, mail_data: Any) -> List[Document]:
        """Process individual mail thread"""
        documents = []

        try:
            if isinstance(mail_data, list):
                mail_list = mail_data
            else:
                mail_list = mail_data.get("messages", [])

            # Extract thread information
            thread_url = mail_list[0].get("thread_url", "unknown") if mail_list else "unknown"
            subject = mail_list[0].get("subject", "No Subject") if mail_list else "No Subject"

            # Process each message in the thread
            for message in mail_list:
                content = self._extract_message_content(message)
                msg_id = message.get("message_id", "Unknown")
                if "@@" not in msg_id:
                    msg_id = f"@@MailingList@@{msg_id}"
                if content:
                    message_url = message.get("message_url", message.get("url", "Unknown"))
                    self.content_by_url[message_url] = content
                    doc_id = hashlib.md5(message_url.encode()).hexdigest()
                    doc = Document(
                        page_content=content,
                        id=doc_id,
                        metadata={
                            "source": msg_id,
                            "type": "mail",
                            "thread_id": thread_url,
                            "subject": subject,
                            "author": message.get("sender_address", "Unknown"),
                            "date": message.get("date", "Unknown"),
                            "url": message_url,
                            "parent": message.get("parent", "Unknown"),
                            # "parent_content": self.content_by_url.get(message.get("parent", "Unknown"), ""),
                        },
                    )
                    doc = self.complete_metadata([doc])
                    documents.append(doc)
        except Exception as e:
            print(f"Error processing mail thread: {e}")

        return documents

    def _extract_message_content(self, message: Dict[str, Any]) -> Optional[str]:
        """Extract and clean message content"""
        content = message.get("content", message.get("body", ""))
        if not content:
            return None

        # Clean up whitespace and quotes
        content = re.sub(r"\s+", " ", content).strip()
        content = content.replace('"', '"').replace('"', '"')

        return content if len(content) > 20 else None

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

    def complete_metadata(
        self,
        documents: List[Document],
        llm_helper=None,
        batch_size: int = 10,
        update_existing: bool = False
    ) -> List[Document]:
        """
        Complete metadata for documents using LLM helper

        Args:
            documents: List of documents to enrich with metadata
            llm_helper: LLMHelper instance (will create one if not provided)
            batch_size: Number of documents to process in each batch
            update_existing: If True, update existing metadata fields; if False, only add missing fields

        Returns:
            List of documents with completed metadata (categories, sentiment, libraries)
        """
        if not documents:
            return documents

        # Create LLMHelper if not provided
        if llm_helper is None:
            llm_helper = self.llm_helper

        self.logger.info(f"Completing metadata for {len(documents)} documents...")

        for i, doc in enumerate(documents):

            try:
            # Update documents with extracted metadata
                doc.metadata['categories'] = llm_helper.process_pipeline("classify", doc)
                doc.metadata['sentiment'] = llm_helper.process_pipeline("sentiment", doc)
                doc.metadata['libraries'] = llm_helper.process_pipeline("libraries", doc)
            except Exception as e:
                if update_existing or 'categories' not in doc.metadata:
                    doc.metadata['categories'] = ['Discussion']
                if update_existing or 'sentiment' not in doc.metadata:
                    doc.metadata['sentiment'] = 'Neutral'
                if update_existing or 'libraries' not in doc.metadata:
                    doc.metadata['libraries'] = ['General']

        return documents

