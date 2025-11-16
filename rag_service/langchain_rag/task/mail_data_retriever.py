"""
Mail Data Retriever Module

Provides unified interface for retrieving mail data from multiple sources:
- HyperKitty PostgreSQL database
- ChromaDB vector store

This module abstracts the data source and provides a consistent interface.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Literal, Tuple
from dateutil import parser
from langchain_core.documents import Document
import structlog

logger = structlog.get_logger(__name__)

import psycopg2
from django.conf import settings

class MailDataRetriever:
    """Unified mail data retriever supporting multiple sources"""

    def __init__(self, source: Literal["hyperkitty", "chromadb", "auto"] = "auto"):
        """
        Initialize mail data retriever

        Args:
            source: Data source to use
                - "hyperkitty": Use PostgreSQL via HyperKittyLoader
                - "chromadb": Use ChromaDB via RAGService
                - "auto": Try ChromaDB first, fallback to HyperKitty
        """
        self.logger = logger.bind(component="MailDataRetriever")
        self.source = source
        self.hyperkitty_loader = None
        self.rag_service = None

        # Initialize sources based on preference
        if source in ["hyperkitty", "auto"]:
            # Check if HyperKitty database is configured
            if not settings.HYPERKITTY_DATABASE_NAME:
                self.logger.warning("HYPERKITTY_DATABASE_NAME not configured. HyperKitty source will not be available.")
            else:
                self.logger.info("HyperKitty database configured (will connect on demand)")

        if source in ["chromadb", "auto"]:
            try:
                # Import here to avoid circular import
                from ...services import RAGService
                self.rag_service = RAGService.get_pipeline()
                self.logger.info("RAG service initialized for ChromaDB")
            except Exception as e:
                self.logger.warning(f"Failed to initialize RAG service: {e}")

    def get_mails(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        source: Optional[Literal["hyperkitty", "chromadb", "auto"]] = None,
        embedding_conclusion: Optional[bool] = False
    ) -> List[Dict[str, Any]]:
        """
        Get mail data from available source

        Args:
            start_date: Start date for filtering (inclusive)
            end_date: End date for filtering (inclusive)
            limit: Maximum number of emails to return
            source: Override default source for this call

        Returns:
            List of email dictionaries with keys:
            - message_id
            - subject
            - sender_address
            - sender_name
            - date (datetime object)
            - content
            - url
            - thread_id
            - mailinglist
        """
        source_to_use = source or self.source

        # Try ChromaDB first if auto or explicit
        if source_to_use in ["chromadb", "auto"]:
            try:
                return self._get_mails_from_chromadb(
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit,
                    embedding_conclusion=embedding_conclusion,
                )
            except Exception as e:
                self.logger.warning(f"ChromaDB retrieval failed: {e}")
                if source_to_use == "chromadb":
                    raise
                # Fallback to HyperKitty if auto mode

        # Use HyperKitty
        if source_to_use in ["hyperkitty", "auto"]:
            if not self.hyperkitty_loader:
                raise ValueError("HyperKitty loader not available")
            return self._get_mails_from_hyperkitty(
                start_date=start_date,
                end_date=end_date,
                limit=limit,
            )

        raise ValueError(f"Invalid source: {source_to_use}")

    def _build_hyperkitty_query(self, start_date: Optional[datetime], end_date: Optional[datetime], limit: Optional[int]) -> Tuple[str, Dict[str, Any]]:
        """Build SQL query and parameters for HyperKitty email retrieval."""
        where_clauses = []
        params = {}

        if start_date:
            where_clauses.append("date >= %(start_date)s")
            params["start_date"] = start_date
        if end_date:
            where_clauses.append("date < %(end_date)s")
            params["end_date"] = end_date

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        limit_sql = f"LIMIT {limit}" if limit else ""

        query = f"""
            SELECT
                message_id,
                subject,
                content,
                sender_id as sender_address,
                sender_name,
                date,
                mailinglist_id,
                thread_id,
                in_reply_to
            FROM hyperkitty_email
            WHERE {where_sql}
            ORDER BY date DESC
            {limit_sql}
        """
        return query, params

    def _convert_hyperkitty_row_to_email(self, row: tuple) -> Dict[str, Any]:
        """Convert a database row to email dictionary."""
        message_id, subject, content, sender_address, sender_name, date, mailinglist_id, thread_id, in_reply_to = row

        url = f"https://lists.boost.org/hyperkitty/thread/{thread_id}/" if thread_id else ""

        return {
            "message_id": message_id or "",
            "subject": subject or "",
            "content": content or "",
            "sender_address": sender_address or "",
            "sender_name": sender_name or "",
            "date": date,
            "url": url,
            "thread_id": str(thread_id) if thread_id else "",
            "mailinglist": mailinglist_id or "",
            "to": "",
            "cc": "",
            "reply_to": in_reply_to or "",
        }

    def _normalize_email_dates(self, emails: List[Dict[str, Any]]) -> None:
        """Normalize date format in email dictionaries."""
        for email in emails:
            if isinstance(email.get("date"), str):
                try:
                    email["date"] = parser.parse(email["date"])
                except (ValueError, TypeError):
                    email["date"] = None

    def _get_mails_from_hyperkitty(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        mailing_list: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get mails from HyperKitty PostgreSQL database"""
        if not settings.HYPERKITTY_DATABASE_NAME:
            raise ValueError("HYPERKITTY_DATABASE_NAME setting not configured")

        try:
            conn = psycopg2.connect(settings.HYPERKITTY_DATABASE_URL)
        except Exception as e:
            raise ValueError(f"Failed to connect to HyperKitty database: {e}") from e

        emails = []
        try:
            query, params = self._build_hyperkitty_query(start_date, end_date, limit)
            with conn.cursor(name="fetch_emails") as cursor:
                cursor.execute(query, params)
                for row in cursor:
                    emails.append(self._convert_hyperkitty_row_to_email(row))
        except Exception as e:
            self.logger.exception(f"Error querying HyperKitty database: {e}")
            raise ValueError(f"Failed to query HyperKitty database: {e}") from e
        finally:
            conn.close()

        self._normalize_email_dates(emails)
        return emails

    def _get_mails_from_chromadb(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        embedding_conclusion: Optional[bool] = False,
    ) -> List[Document]:
        """Get mails from ChromaDB vector store"""
        # Try to initialize RAG service if not already initialized (lazy initialization)
        if not self.rag_service:
            try:
                # Import here to avoid circular import
                from ...services import RAGService
                self.rag_service = RAGService.get_pipeline()
                self.logger.info("RAG service initialized for ChromaDB (lazy init)")
            except Exception as e:
                self.logger.error(f"Failed to initialize RAG service: {e}", exc_info=True)
                raise ValueError(f"RAG service not initialized: {e}") from e

        # Get vector store from retriever
        # The RAG pipeline has hybrid_retriever which has vector_store
        if not hasattr(self.rag_service, 'base_retrievers') or 'mail' not in self.rag_service.base_retrievers:
            raise ValueError("Mail retriever not available in RAG service")

        mail_retriever = self.rag_service.base_retrievers['mail']
        vector_store = mail_retriever.vector_store

        # If vector store doesn't exist, ChromaDB hasn't been initialized yet
        if not vector_store:
            self.logger.warning("ChromaDB vector store not initialized. ChromaDB may be empty or not set up yet.")
            # Return empty result instead of raising error
            if embedding_conclusion:
                return {"documents": [], "embeddings": []}
            else:
                return {"documents": []}

        # Build filter for ChromaDB query
        where_filter = self._build_time_filter(start_date, end_date)

        # Get documents from ChromaDB
        if embedding_conclusion:
            include = ["documents", "metadatas", "embeddings"]
        else:
            include = ["documents", "metadatas"]
        results = vector_store._collection.get(
            where=where_filter,
            include=include,
            limit=limit or 10000
        )

        if not results or not results.get("metadatas"):
            self.logger.info("No mail documents found in ChromaDB")
            return []

        # Convert ChromaDB results to email format
        documents = []
        for i, metadata in enumerate(results["metadatas"]):
            document = Document(id = results["ids"][i],page_content=results["documents"][i], metadata=metadata)
            documents.append(document)

        if embedding_conclusion:
            return {"documents": documents, "embeddings": results["embeddings"]}
        else:
            return {"documents": documents}

    def retrieve_relevant_emails(
        self,
        question: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        fetch_k: int = 30
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant emails for a topic"""
        if not self.rag_service:
            # Import here to avoid circular import
            from ...services import RAGService
            self.rag_service = RAGService.get_pipeline()
        filters = self._build_time_filter(start_date, end_date)
        return self.rag_service.retrieve(question, fetch_k, filters)


    def _build_time_filter(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build time filter for ChromaDB query

        ChromaDB requires using $and to combine multiple operators on the same field.

        Args:
            start_date: Start date (datetime object or timestamp)
            end_date: End date (datetime object or timestamp)

        Returns:
            ChromaDB filter dictionary with type and date conditions, or None if no filters
        """
        # Always include type filter for mail documents
        # conditions = [{"type": "mail"}]
        conditions = []
        # Build date conditions
        if start_date and end_date:
            # Both dates provided - use $and to combine $gte and $lte
            start_ts = int(start_date.timestamp()) if isinstance(start_date, datetime) else int(start_date)
            end_ts = int(end_date.timestamp()) if isinstance(end_date, datetime) else int(end_date)
            conditions.append({"date": {"$gte": start_ts}})
            conditions.append({"date": {"$lte": end_ts}})
        elif start_date:
            # Only start date
            start_ts = int(start_date.timestamp()) if isinstance(start_date, datetime) else int(start_date)
            conditions.append({"date": {"$gte": start_ts}})
        elif end_date:
            # Only end date
            end_ts = int(end_date.timestamp()) if isinstance(end_date, datetime) else int(end_date)
            conditions.append({"date": {"$lte": end_ts}})

        # If only type filter, return simple dict; otherwise use $and
        if len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}