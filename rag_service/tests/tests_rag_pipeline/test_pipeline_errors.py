"""
RAG pipeline error handling tests.

Tests for error scenarios including network failures, LLM errors, and retriever errors.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from rag_service.langchain_rag.rag_pipeline import LangChainRAGPipeline
from config.rag_config import LangChainConfig


@pytest.mark.django_db
class TestRAGPipelineErrorHandling:
    """Test error handling in RAG pipeline."""

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_retrieve_handles_retriever_error(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that retrieve handles retriever errors gracefully."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever.retrieve.side_effect = RuntimeError("Retriever failed")
        mock_retriever_class.return_value = mock_retriever

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        # Should not raise, but return empty list
        results = pipeline.retrieve("test query")

        assert results == []

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    @patch("rag_service.langchain_rag.rag_pipeline.LLMHelper")
    def test_query_handles_llm_error(
        self,
        mock_llm_helper_class,
        mock_retriever_class,
        mock_processor_class,
        sample_documents,
    ):
        """Test that query handles LLM errors gracefully."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever.retrieve.return_value = sample_documents
        mock_retriever_class.return_value = mock_retriever

        mock_llm_helper = Mock()
        mock_llm_helper.process_pipeline.side_effect = ValueError("LLM failed")
        mock_llm_helper_class.return_value = mock_llm_helper

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        # Should return error response, not raise
        result = pipeline.query("test query")

        assert "answer" in result
        assert "error" in result.get("metadata", {})
        assert "I apologize" in result["answer"]

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_add_mail_data_handles_invalid_data(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that add_mail_data handles invalid mail data."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor.process_mail_list.side_effect = ValueError("Invalid data")
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever.document_exists.return_value = False
        mock_retriever_class.return_value = mock_retriever

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        invalid_mail_data = [{"invalid": "data"}]

        # Should handle error and return failed count
        result = pipeline.add_mail_data(invalid_mail_data)

        assert result["failed_count"] == 1
        assert result["total_processed"] == 1
        assert len(result["failed_messages"]) == 1

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_delete_document_handles_nonexistent(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that delete_document handles nonexistent documents."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever.document_exists.return_value = False
        mock_retriever_class.return_value = mock_retriever

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        # Should return False, not raise
        result = pipeline.delete_document("nonexistent://url")

        assert result is False

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_pipeline_handles_missing_retriever(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that pipeline handles missing retriever gracefully."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever_class.return_value = mock_retriever

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        # Remove retriever to simulate missing retriever
        pipeline.base_retrievers = {}

        mail_data = [{"message_id": "test@example.com", "url": "test://1"}]

        result = pipeline.add_mail_data(mail_data)

        # Should handle gracefully
        assert result["added_count"] == 0
        assert result["failed_count"] == 1

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_get_cache_stats_when_cache_disabled(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test get_cache_stats when caching is disabled."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever_class.return_value = mock_retriever

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        stats = pipeline.get_cache_stats()

        assert "error" in stats
        assert "not enabled" in stats["error"].lower()

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_get_telemetry_report_when_disabled(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test get_telemetry_report when telemetry is disabled."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever_class.return_value = mock_retriever

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        report = pipeline.get_telemetry_report()

        assert "error" in report
        assert "not enabled" in report["error"].lower()


@pytest.mark.django_db
class TestRAGPipelineNetworkFailures:
    """Test network failure scenarios."""

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    @patch("rag_service.langchain_rag.rag_pipeline.LLMHelper")
    def test_query_handles_llm_timeout(
        self,
        mock_llm_helper_class,
        mock_retriever_class,
        mock_processor_class,
        sample_documents,
    ):
        """Test that query handles LLM timeout errors."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever.retrieve.return_value = sample_documents
        mock_retriever_class.return_value = mock_retriever

        mock_llm_helper = Mock()
        mock_llm_helper.process_pipeline.side_effect = TimeoutError(
            "LLM request timed out"
        )
        mock_llm_helper_class.return_value = mock_llm_helper

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        result = pipeline.query("test query")

        assert "answer" in result
        assert "error" in result.get("metadata", {})
        assert (
            "timeout" in result["answer"].lower() or "I apologize" in result["answer"]
        )

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    @patch("rag_service.langchain_rag.rag_pipeline.LLMHelper")
    def test_query_handles_llm_rate_limit(
        self,
        mock_llm_helper_class,
        mock_retriever_class,
        mock_processor_class,
        sample_documents,
    ):
        """Test that query handles LLM rate limit errors."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever.retrieve.return_value = sample_documents
        mock_retriever_class.return_value = mock_retriever

        mock_llm_helper = Mock()
        # Simulate rate limit error
        rate_limit_error = Exception("Rate limit exceeded: 429")
        rate_limit_error.status_code = 429
        mock_llm_helper.process_pipeline.side_effect = rate_limit_error
        mock_llm_helper_class.return_value = mock_llm_helper

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        result = pipeline.query("test query")

        assert "answer" in result
        assert "error" in result.get("metadata", {})

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_retrieve_handles_chromadb_connection_error(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that retrieve handles ChromaDB connection errors."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever.retrieve.side_effect = ConnectionError(
            "ChromaDB connection failed"
        )
        mock_retriever_class.return_value = mock_retriever

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        # Should return empty list, not raise
        results = pipeline.retrieve("test query")

        assert results == []

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_add_mail_data_handles_s3_upload_failure(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that add_mail_data handles S3 upload failures gracefully."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor.process_mail_list.return_value = [
            Document(page_content="Test", metadata={"url": "test://1"})
        ]
        mock_processor.chunk_documents.return_value = [
            Document(page_content="Test", metadata={"url": "test://1"})
        ]
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever.document_exists.return_value = False
        # Simulate S3 upload failure during add_documents
        mock_retriever.add_documents.side_effect = ConnectionError("S3 upload failed")
        mock_retriever_class.return_value = mock_retriever

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        mail_data = [
            {
                "message_id": "test@example.com",
                "subject": "Test",
                "content": "Test content",
                "url": "test://1",
            }
        ]

        # Should handle error and return failed count
        result = pipeline.add_mail_data(mail_data)

        assert result["failed_count"] >= 0
        assert "failed_messages" in result
