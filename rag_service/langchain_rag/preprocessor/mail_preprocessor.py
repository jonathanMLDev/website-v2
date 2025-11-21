"""
Mail preprocessing pipeline for LangChain RAG
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from dateutil.parser import parse as parse_date
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from ..llm import LLMHelper

logger = structlog.get_logger(__name__)


class MailPreprocessor:
    """Process Boost mailing list data for RAG"""

    def __init__(
        self, mail_data_dir: str = "", chunk_size: int = 512, chunk_overlap: int = 50
    ):
        self.mail_data_dir = Path(mail_data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.logger = logger.bind(component="MailPreprocessor")

        from config.rag_config import DEFAULT_CONFIG

        self.llm_helper = LLMHelper(config=DEFAULT_CONFIG)

    def load_emails(self) -> List[Document]:
        """Load and process all emails from data directory"""
        emails_path = self.mail_data_dir
        if emails_path.exists():
            return self._process_mail_data_from_path(emails_path)
        return []

    def _process_mail_data_from_path(self, mail_path: Path) -> List[Document]:
        """Process Boost mailing list data"""
        documents = []

        for thread_file in tqdm(
            mail_path.rglob("*.json"), desc="Processing mail threads"
        ):
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

        return documents

    def process_mail_list(self, mail_data: Any) -> List[Document]:
        """Process individual mail thread"""
        documents = []

        try:
            if isinstance(mail_data, list):
                mail_list = mail_data
            elif "messages" in mail_data:
                mail_list = mail_data.get("messages", [])
            else:
                mail_list = [mail_data]

            # Process each message in the thread
            for message in mail_list:
                content = self._extract_message_content(message)
                if content:
                    doc_id = self._create_doc_id_from_message(message)
                    metadata = self.complete_metadata(message, content)
                    doc = Document(page_content=content, id=doc_id, metadata=metadata)
                    documents.append(doc)
        except Exception as e:
            print(f"Error processing mail thread: {e}")

        return documents

    def _create_doc_id_from_message(self, message: dict) -> str:
        """Create document ID from message URL."""
        message_url = message.get("message_url", message.get("url", "Unknown"))
        return hashlib.md5(message_url.encode()).hexdigest()

    def _parse_date_to_timestamp(self, date_value: Any) -> Optional[int]:
        """
        Parse date value to timestamp (integer).

        Args:
            date_value: Date value (datetime, int, str, or None)

        Returns:
            Timestamp as integer, or None if parsing fails
        """
        current_time = datetime.now().timestamp()
        if date_value is None:
            return None

        # Already a timestamp (integer)
        if isinstance(date_value, int):
            return date_value

        # Already a datetime object
        if isinstance(date_value, datetime):
            return int(date_value.timestamp())

        # String - try to parse
        if isinstance(date_value, str):
            if date_value == "Unknown" or not date_value.strip():
                return current_time
            try:
                dt = parse_date(date_value)
                return int(dt.timestamp())
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Failed to parse date '{date_value}': {e}")
                return current_time

        return current_time

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
            if not doc.id:
                doc.id = self._create_doc_id_from_message(doc.metadata)
            chunks = self.text_splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                if len(chunk.page_content) < 50:
                    continue
                chunk.id = f"{doc.id}-{i:03d}"
                chunked_documents.append(chunk)

        return chunked_documents

    def _extract_basic_metadata(
        self,
        raw_message: Dict[str, Any],
    ) -> Dict:
        """Extract and add basic metadata from raw message data."""
        metadata = {}
        msg_id = raw_message.get("message_id", "Unknown")
        if "@@" not in msg_id:
            msg_id = f"@@MailingList@@{msg_id}"
        thread_url = raw_message.get("thread_url", raw_message.get("thread", ""))
        subject = raw_message.get("subject", "")
        author_info = raw_message.get(
            "sender_address", raw_message.get("sender_name", "Unknown")
        )
        date_value = raw_message.get("date")
        date_timestamp = self._parse_date_to_timestamp(date_value)
        message_url = raw_message.get("message_url", raw_message.get("url", "Unknown"))

        metadata["source"] = msg_id
        metadata["type"] = "mail"
        metadata["thread_id"] = thread_url or "unknown"
        metadata["subject"] = subject or "No Subject"
        metadata["author"] = author_info
        metadata["date"] = date_timestamp
        metadata["url"] = message_url
        metadata["parent"] = raw_message.get("parent", "Unknown")

        return metadata

    def _extract_llm_metadata(
        self,
        message: Dict,
        content: str = None,
        llm_helper: Any = None,
    ) -> Dict:
        """Extract and add LLM-generated metadata."""
        metadata: Dict[str, Any] = {}
        content = content or message.get("content", "")
        if not content:
            return metadata

        if llm_helper is None:
            llm_helper = self.llm_helper

        metadata.setdefault("categories", ["Discussion"])
        metadata.setdefault("sentiment", "Neutral")
        metadata.setdefault("libraries", ["General"])

        try:
            metadata["categories"] = llm_helper.process_pipeline("classify", content)
            metadata["sentiment"] = llm_helper.process_pipeline("sentiment", content)
            metadata["libraries"] = llm_helper.process_pipeline("libraries", content)
        except Exception:  # pragma: no cover
            pass

        return metadata

    def complete_metadata(
        self,
        message: Dict,
        content: str = None,
        llm_helper=None,
    ) -> Dict:
        """
        Complete metadata for a single mail message.

        All metadata production happens here. Extracts basic metadata from the raw
        message payload and augments it with LLM-enriched fields.

        Args:
            message: Raw message dictionary with original metadata fields.
            content: Optional content string (falls back to message["content"]).
            llm_helper: Optional LLMHelper instance (defaults to self.llm_helper).

        Returns:
            Combined metadata dictionary with both basic and LLM-derived fields.
        """
        if not message:
            return None

        basic_metadata = self._extract_basic_metadata(message)

        complement_metadata = self._extract_llm_metadata(message, content, llm_helper)
        metadata = (basic_metadata or {}) | (complement_metadata or {})

        return metadata
