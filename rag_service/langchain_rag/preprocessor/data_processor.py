"""
Data processing pipeline for LangChain RAG (Backward compatibility wrapper)

This module provides a combined interface that uses MailPreprocessor and DocuPreprocessor
"""

from typing import List, Optional, Any
from langchain_core.documents import Document

from .mail_preprocessor import MailPreprocessor
from .docu_preprocessor import DocuPreprocessor


class BoostDataProcessor:
    """Process Boost library documentation and mail data for RAG (Backward compatibility)"""

    def __init__(
        self,
        config: Optional[Any] = None
    ):
        if config is None:
            from config.rag_config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG

        self.mail_preprocessor = MailPreprocessor(
            mail_data_dir=config.mail_data_dir,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.docu_preprocessor = DocuPreprocessor(
            doc_data_dir=config.doc_data_dir,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

    def load_documents_and_emails(self) -> List[Document]:
        """Load and process all documents and emails from data directory"""
        documents = self.load_documents()
        emails = self.load_emails()
        return documents + emails

    def load_documents(self) -> List[Document]:
        """Load and process all documents from data directory"""
        return self.docu_preprocessor.load_documents()

    def load_emails(self) -> List[Document]:
        """Load and process all emails from data directory"""
        return self.mail_preprocessor.load_emails()

    def process_mail_list(self, mail_data) -> List[Document]:
        """Process individual mail thread"""
        return self.mail_preprocessor.process_mail_list(mail_data)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        # Use mail preprocessor's chunk_documents (both have the same implementation)
        return self.mail_preprocessor.chunk_documents(documents)
