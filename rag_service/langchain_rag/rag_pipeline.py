"""
LangChain RAG Pipeline with hybrid retrieval
"""

import os
import time
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
import structlog
from tqdm import tqdm

from config.rag_config import LangChainConfig

from .llm.llm_helper import LLMHelper
from .preprocessor import BoostDataProcessor
from .retrieve import (
    CachedRetriever,
    InstrumentedRetriever,
    LangChainHybridRetriever,
    MultiBaseRetriever,
    PerformanceTelemetry,
    QueryCache,
)

logger = structlog.get_logger(__name__)


class LangChainRAGPipeline:
    """High-performance RAG pipeline using LangChain with hybrid retrieval"""

    def __init__(self, config: LangChainConfig = None, **kwargs):
        """
        Initialize the pipeline with configuration and supporting components.

        Args:
            config: Optional LangChainConfig instance. When omitted a new
                instance is created from ``kwargs`` or DEFAULT_CONFIG.
            **kwargs: Keyword arguments forwarded to LangChainConfig when
                ``config`` is not supplied.
        """
        # Use provided config or create from kwargs
        if config is None:
            config = LangChainConfig(**kwargs)

        self.config = config
        self.logger = logger.bind(component="LangChainRAGPipeline")

        # Validate configuration
        self.config.validate()
        self.logger.info("Configuration validated successfully")

        # Initialize caching and telemetry
        self.llm_helper = None
        self.cache = None
        self.telemetry = None
        if self.config.enable_cache:
            self.cache = QueryCache(
                cache_dir=self.config.cache_dir,
                ttl_seconds=self.config.cache_ttl_seconds,
                max_memory_entries=self.config.cache_max_memory_entries,
                max_disk_size_mb=self.config.cache_max_disk_size_mb,
                auto_cleanup_interval=self.config.cache_auto_cleanup_interval,
            )
            self.logger.info(
                f"Query cache initialized (TTL: {self.config.cache_ttl_seconds}s, "
                f"Max memory: {self.config.cache_max_memory_entries}, "
                f"Max disk: {self.config.cache_max_disk_size_mb}MB)"
            )

        if self.config.enable_telemetry:
            self.telemetry = PerformanceTelemetry(
                log_file=self.config.telemetry_log_file
            )
            self.logger.info("Performance telemetry initialized")

        # Initialize components
        self._setup_data_processor()
        self._setup_retriever()
        self.logger.info("LangChain RAG pipeline initialized successfully")

    def _setup_data_processor(self):
        """Instantiate the data processor used to load and chunk documents."""
        self.data_processor = BoostDataProcessor(config=self.config)
        self.logger.info("Data processor initialized")

    def _setup_retriever(self):
        """
        Configure base retrievers and wrap them with cache/telemetry layers.

        The resulting retriever hierarchy is stored on the instance for later
        queries.
        """
        os.environ["CHROMA_DISABLE_PERSISTENCE_CACHE"] = "1"

        # Create base retrievers
        base_retrievers = self._create_base_retrievers()

        # Create and wrap multi-base retriever
        retriever = self._create_wrapped_retriever(base_retrievers)

        self.hybrid_retriever = retriever
        self.multi_base_retriever = MultiBaseRetriever(base_retrievers=base_retrievers)
        self.base_retrievers = base_retrievers

        # Load or reindex for each base retriever
        self._load_or_reindex_retrievers()

    def _create_base_retrievers(self) -> Dict[str, LangChainHybridRetriever]:
        """Create base retrievers for each configured document type."""
        base_retrievers = {}
        retriever_types = self.config.base_retriever_types
        self.logger.info(f"Creating base retrievers for types: {retriever_types}")

        for retriever_type in self.config.base_retriever_types:
            chroma_persist_dir = os.path.join(
                self.config.chroma_persist_dir, retriever_type
            )

            base_retriever = LangChainHybridRetriever(
                config=self.config,
                chroma_persist_dir=chroma_persist_dir,
            )

            base_retrievers[retriever_type] = base_retriever
            self.logger.info(f"Created base retriever for type: {retriever_type}")

        return base_retrievers

    def _create_wrapped_retriever(
        self, base_retrievers: Dict[str, LangChainHybridRetriever]
    ):
        """
        Build the MultiBaseRetriever and augment it with telemetry and cache.

        Args:
            base_retrievers: Mapping of retriever key to instantiated retriever.

        Returns:
            A retriever instance that first records telemetry metrics and then
            leverages query caching (if enabled).
        """
        retriever = MultiBaseRetriever(base_retrievers=base_retrievers)

        if self.config.enable_telemetry and self.telemetry:
            retriever = InstrumentedRetriever(
                retriever=retriever,
                telemetry=self.telemetry,
                stage_name="hybrid_retrieval",
            )
            self.logger.info("Retriever wrapped with telemetry")

        if self.config.enable_cache and self.cache:
            retriever = CachedRetriever(retriever=retriever, cache=self.cache)
            self.logger.info("Retriever wrapped with cache")

        return retriever

    def _load_or_reindex_retrievers(self):
        """Ensure every base retriever has an up-to-date index on disk."""
        for retriever_type, base_retriever in self.base_retrievers.items():
            try:
                if self._try_load_retriever_index(retriever_type, base_retriever):
                    continue

                self._reindex_retriever(retriever_type, base_retriever)
            except (
                Exception
            ) as e:  # pragma: no cover - keep pipeline usable if one retriever fails
                self.logger.exception(
                    f"Failed to load/reindex {retriever_type} retriever: %s", e
                )
                continue

    def _try_load_retriever_index(
        self, retriever_type: str, base_retriever: LangChainHybridRetriever
    ) -> bool:
        """
        Try to load an existing retriever index.

        Returns:
            True when the persisted index has been loaded, False otherwise.
        """
        if self.config.force_reindex:
            return False

        try:
            status = base_retriever.load_index()
            if all(status.values()):
                self.logger.info(
                    f"{retriever_type} retriever index loaded successfully"
                )
                return True
        except (
            Exception
        ) as e:  # pragma: no cover - fall back to reindexing when loading fails unexpectedly
            self.logger.exception(
                f"Failed to load {retriever_type} retriever index: {e}"
            )
            self.logger.info(f"Force reindexing {retriever_type}...")

        return False

    def _reindex_retriever(
        self, retriever_type: str, base_retriever: LangChainHybridRetriever
    ):
        """Reindex a retriever by loading, chunking, and embedding documents."""
        documents = self._load_documents_for_type(retriever_type)
        if not documents:
            self.logger.warning(f"No documents found for {retriever_type} retriever")
            return

        self.logger.info(f"Loaded {len(documents)} documents for {retriever_type}")
        self.logger.info(f"Chunking {retriever_type} documents...")
        chunked_documents = self.data_processor.chunk_documents(documents)
        chunked_count = len(chunked_documents)
        self.logger.info(f"Chunked {chunked_count} documents for {retriever_type}")

        status = {
            "vector_store": False,
            "sparse_retriever": False,
        }
        prev_status = None if self.config.force_reindex else status
        base_retriever.reindex(chunked_documents, prev_status)
        self.logger.info(f"{retriever_type} retriever reindexed successfully")

    def _load_documents_for_type(self, retriever_type: str) -> List[Document]:
        """
        Load documents for a specific retriever type

        Args:
            retriever_type: Type of retriever
                (e.g., "mail", "documentation", "git", "slack")

        Returns:
            List of documents for this retriever type
        """
        documents = []

        if retriever_type == "mail":
            documents = self.data_processor.load_emails()
        elif retriever_type == "documentation":
            documents = self.data_processor.load_documents()
        elif retriever_type == "git":
            # TODO: Implement git data loading
            self.logger.warning("Git data loading not yet implemented")
            # documents = self.data_processor.load_git_data(self.config.git_data_dir)
        elif retriever_type == "slack":
            # TODO: Implement slack data loading
            self.logger.warning("Slack data loading not yet implemented")
            # documents = self.data_processor.load_slack_data(self.config.slack_data_dir)
        else:
            self.logger.warning(f"Unknown retriever type: {retriever_type}")

        return documents

    def retrieve(
        self,
        question: str,
        fetch_k: int = 10,
        filters: Dict[str, Any] = None,
        str_results: bool = False,
    ):
        """Retrieve relevant documents with optional type filtering

        Args:
            question: Search query
            fetch_k: Number of results to return
            filters: Dictionary of filters to apply to the retrieval
            str_results: If True, return text content instead of Document objects

        Returns:
            List of documents or text strings
        """
        retrieve_list = []
        try:
            # Get relevant documents with optional type filtering
            relevant_docs = self.hybrid_retriever.retrieve(question, fetch_k, filters)

            # Create context from documents
            if str_results:
                retrieve_list = [doc.page_content for doc in relevant_docs]
            else:
                retrieve_list = relevant_docs
        except (
            Exception
        ) as e:  # pragma: no cover - defensive catch keeps retrieval API responsive
            self.logger.exception("Error during query: %s", e)
        return retrieve_list

    def query(
        self, question: str, fetch_k: int = 10, filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Query the RAG pipeline with optional type filtering

        Args:
            question: Search query
            fetch_k: Number of results to return
            filters: Dictionary of filters to apply to the retrieval

        Returns:
            Dictionary with answer, source documents, and metadata
        """
        start_time = time.time()
        self.logger.info(
            "Processing query: %s (fetch_k=%d, filters=%s)", question, fetch_k, filters
        )

        # Start telemetry tracking
        if self.telemetry:
            self.telemetry.start_query(question)

        try:
            # Get relevant documents with optional type filtering
            if self.telemetry:
                self.telemetry.start_stage("document_retrieval")

            retrieve_list = self.retrieve(question, fetch_k, filters)

            if self.telemetry:
                self.telemetry.end_stage(
                    "document_retrieval",
                    {"fetch_k": fetch_k, "results": len(retrieve_list)},
                )

            if not self.llm_helper:
                self.llm_helper = LLMHelper(config=self.config)
            answer = self.llm_helper.process_pipeline(
                "answer_question", retrieve_list, question
            )

            response = {
                "answer": answer,
                "source_documents": retrieve_list,
                "query_time": time.time() - start_time,
                "retrieval_method": "hybrid",
            }

            # End telemetry tracking
            if self.telemetry:
                self.telemetry.end_query(results_count=len(retrieve_list), success=True)

            self.logger.info(
                "Query processed successfully in %.3fs", time.time() - start_time
            )
            return response

        except (
            Exception
        ) as e:  # pragma: no cover - provide friendly error response on unexpected failures
            # End telemetry tracking with failure
            if self.telemetry:
                self.telemetry.end_query(results_count=0, success=False)

            self.logger.exception("Error during query: %s", e)
            return {
                "answer": (
                    "I apologize, but I encountered an error while "
                    "processing your query."
                ),
                "source_documents": [],
                "metadata": {"error": str(e)},
            }

    def convert_mail_to_json(self, mail) -> Dict[str, Any]:
        """Convert mail to JSON"""
        mail_json = {}
        mail_json["message_id"] = mail.message_id
        mail_json["subject"] = mail.subject
        mail_json["content"] = mail.content
        mail_json["thread_url"] = mail.thread_url
        mail_json["parent"] = mail.parent
        mail_json["children"] = mail.children
        mail_json["sender_address"] = mail.sender_address
        mail_json["from_field"] = mail.from_field
        mail_json["date"] = mail.date
        mail_json["to"] = mail.to
        mail_json["cc"] = mail.cc
        mail_json["reply_to"] = mail.reply_to
        mail_json["url"] = mail.url
        return mail_json

    def _normalize_mail_data(self, mail: Any) -> Tuple[Dict[str, Any], str]:
        """Normalize mail data to dictionary format and extract message_id."""
        if isinstance(mail, dict):
            mail_json = mail
            message_id = mail.get("message_id", "Unknown")
        else:
            mail_json = self.convert_mail_to_json(mail)
            message_id = mail.message_id if hasattr(mail, "message_id") else "Unknown"
        return mail_json, message_id

    def _process_new_mail(self, mail_json: Dict[str, Any], mail_retriever: Any) -> bool:
        """Process and add a new mail document."""
        doc = self.data_processor.process_mail_list([mail_json])
        chunked_docs = self.data_processor.chunk_documents(doc)
        if mail_retriever:
            mail_retriever.add_documents(chunked_docs)
            return True
        else:
            self.logger.warning(
                "Mail retriever not available, skipping document addition"
            )
            return False

    def _build_mail_processing_result(
        self,
        added_count: int,
        updated_count: int,
        failed_messages: List[str],
        total_processed: int,
    ) -> Dict[str, Any]:
        """Build result dictionary for mail processing."""
        result = {
            "added_count": added_count,
            "updated_count": updated_count,
            "failed_count": len(failed_messages),
            "failed_messages": failed_messages,
            "total_processed": total_processed,
        }

        self.logger.info(
            f"Mail data processing complete: {added_count} added, "
            f"{updated_count} updated (already exist), "
            f"{len(failed_messages)} failed out of {total_processed} total"
        )

        if failed_messages:
            self.logger.warning(f"Failed messages: {failed_messages}")

        return result

    def add_mail_data(self, mail_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add new documents to the RAG pipeline
        (both vector store and BM25) with duplicate checking

        Args:
            mail_data: List of mail messages to add

        Returns:
            Dictionary with statistics:
                - added: Number of messages added
                - skipped: Number of messages skipped (already exist)
                - failed: Number of messages that failed to process
                - failed_messages: List of message IDs that failed
        """
        self.logger.info("Processing %d mail documents for addition", len(mail_data))

        failed_messages = []
        updated_count = 0
        added_count = 0
        mail_retriever = self.base_retrievers.get("mail")

        for mail in mail_data:
            message_id = None
            try:
                mail_json, message_id = self._normalize_mail_data(mail)

                # Check if document already exists
                if mail_retriever and mail_retriever.document_exists(mail_json["url"]):
                    self.update_mail_data(mail_json)
                    updated_count += 1
                    continue

                # Process new mail
                if self._process_new_mail(mail_json, mail_retriever):
                    added_count += 1
            except (
                Exception
            ) as e:  # pragma: no cover - continue processing remaining mails
                self.logger.exception("Error processing mail data: %s", e)
                if message_id:
                    failed_messages.append(message_id)

        # Clear cache after adding documents
        if self.cache:
            self.cache.clear()
            self.logger.info("Cache cleared after adding documents")

        return self._build_mail_processing_result(
            added_count, updated_count, failed_messages, len(mail_data)
        )

    def update_mail_data(self, message: Any) -> bool:
        """Update an existing document in both vector store and BM25

        Args:
            message: Dictionary containing mail message data or mail object

        Returns:
            True if update successful, False otherwise
        """
        # Get message_id properly based on type
        if isinstance(message, dict):
            message_id = message.get("message_id", "Unknown")
            message_dict = message
        else:
            message_id = getattr(message, "message_id", "Unknown")
            message_dict = self.convert_mail_to_json(message)

        self.logger.info(f"Updating document {message_id}")

        try:
            # Create updated document from dict
            updated_doc = self.data_processor.process_mail_list([message_dict])
            # Chunk the updated document
            chunked_docs = self.data_processor.chunk_documents(updated_doc)
            # Update in mail base retriever
            mail_retriever = self.base_retrievers.get("mail")
            if mail_retriever:
                mail_retriever.update_documents(chunked_docs)
            else:
                self.logger.warning(
                    "Mail retriever not available, skipping document update"
                )
                return False
            return True
        except (
            Exception
        ) as e:  # pragma: no cover - updating one mail must not crash pipeline
            self.logger.exception("Error updating mail data: %s", e)
            return False

    def delete_document(self, doc_url: str):
        """Delete a document from both vector store and BM25"""
        self.logger.info(f"Deleting document {doc_url}")

        try:
            # Try to find and delete from all base retrievers
            deleted = False
            for retriever_type, base_retriever in self.base_retrievers.items():
                if base_retriever.document_exists(doc_url):
                    doc_id = base_retriever.get_document_id(doc_url)
                    base_retriever.delete_documents([doc_id])
                    deleted = True
                    self.logger.info(
                        f"Deleted document {doc_url} from {retriever_type} retriever"
                    )

            if deleted:
                if self.cache:
                    self.cache.clear()
                    self.logger.info("Cache cleared after deleting document")
                return True
            else:
                self.logger.info(f"Document {doc_url} does not exist in any retriever")
                return False

        except (
            Exception
        ) as e:  # pragma: no cover - propagate unexpected deletion failures
            self.logger.exception(f"Error deleting document {doc_url}: %s", e)
            raise

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        # Aggregate stats from all base retrievers
        total_documents = 0
        retriever_stats = {}
        for retriever_type, base_retriever in self.base_retrievers.items():
            doc_count = len(base_retriever.documents)
            total_documents += doc_count
            retriever_stats[retriever_type] = {
                "document_count": doc_count,
                "dense_top_k": base_retriever.dense_top_k,
                "sparse_top_k": base_retriever.sparse_top_k,
            }

        stats = {
            "total_documents": total_documents,
            "embedding_model": self.config.embedding_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "base_retrievers": retriever_stats,
            "retriever_types": list(self.base_retrievers.keys()),
        }

        # Add cache stats if caching is enabled
        if self.cache:
            stats["cache"] = self.cache.get_stats()

        # Add telemetry stats if telemetry is enabled
        if self.telemetry:
            stats["telemetry"] = self.telemetry.get_aggregated_metrics()

        return stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_stats()
        return {"error": "Caching is not enabled"}

    def get_telemetry_report(self) -> Dict[str, Any]:
        """Get telemetry report"""
        if self.telemetry:
            return self.telemetry.get_aggregated_metrics()
        return {"error": "Telemetry is not enabled"}

    def print_performance_report(self):
        """Print formatted performance report"""
        if self.telemetry:
            self.telemetry.print_report()
        else:
            self.logger.info("Telemetry is not enabled")

    def clear_cache(self):
        """Clear query cache"""
        if self.cache:
            self.cache.clear()
            self.logger.info("Cache cleared manually")
        else:
            self.logger.info("Caching is not enabled")

    def cleanup_cache(self) -> Dict[str, int]:
        """Manually trigger cache cleanup. Returns cleanup statistics."""
        if self.cache:
            expired = self.cache.cleanup_expired_files()
            size_limit = self.cache.enforce_disk_size_limit()
            self.logger.info(
                f"Cache cleanup: {expired} expired files, {size_limit} for size limit"
            )
            return {"expired": expired, "size_limit": size_limit}
        else:
            self.logger.info("Caching is not enabled")
            return {"expired": 0, "size_limit": 0}

    def get_cache_size_info(self) -> Dict[str, Any]:
        """Get cache size information"""
        if self.cache:
            return self.cache.get_cache_size_info()
        return {"error": "Caching is not enabled"}

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        responses = []

        for question in tqdm(questions, desc="Processing queries"):
            response = self.query(question)
            responses.append(response)

        return responses


# Example usage and testing
def create_langchain_rag_pipeline(config: LangChainConfig = None, **kwargs):
    """Create and return a configured LangChain RAG pipeline"""
    if config is None:
        if kwargs:
            config = LangChainConfig(**kwargs)
        else:
            from config.rag_config import DEFAULT_CONFIG

            config = DEFAULT_CONFIG

    return LangChainRAGPipeline(config=config)


if __name__ == "__main__":
    # Example usage
    pipeline = create_langchain_rag_pipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        force_reindex=False,
    )

    # Test retrieval
    test_question = "How does Boost.Asio handle asynchronous operations?"
    docs = pipeline.retrieve(test_question, fetch_k=5)

    print(f"Question: {test_question}")
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n{i}. Score: {doc.metadata.get('final_score', 0):.3f}")
        print(f"   Type: {doc.metadata.get('type', 'unknown')}")
        print(f"   Content: {doc.page_content[:200]}...")
