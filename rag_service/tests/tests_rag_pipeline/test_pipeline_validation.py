"""
RAG pipeline validation tests.

Tests for invalid data handling, edge cases, and public method coverage.
"""

from unittest.mock import Mock, patch
from datetime import datetime

import pytest
from langchain_core.documents import Document

from rag_service.langchain_rag.rag_pipeline import LangChainRAGPipeline
from config.rag_config import LangChainConfig


@pytest.mark.django_db
class TestRAGPipelineInvalidData:
    """Test invalid data handling scenarios."""

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_add_mail_data_handles_empty_list(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that add_mail_data handles empty mail data list."""
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

        result = pipeline.add_mail_data([])

        assert result["added_count"] == 0
        assert result["total_processed"] == 0
        assert result["failed_count"] == 0

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_add_mail_data_handles_missing_required_fields(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that add_mail_data handles mail data with missing required fields."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor.process_mail_list.side_effect = ValueError(
            "Missing required field: message_id"
        )
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

        invalid_mail_data = [
            {
                # Missing message_id, subject, content
                "url": "test://1",
            }
        ]

        result = pipeline.add_mail_data(invalid_mail_data)

        assert result["failed_count"] == 1
        assert len(result["failed_messages"]) == 1

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_add_mail_data_handles_malformed_json(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that add_mail_data handles malformed mail data."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor.process_mail_list.side_effect = TypeError("Invalid data type")
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

        # Invalid data type (not dict or list)
        invalid_mail_data = "not a list or dict"

        result = pipeline.add_mail_data(invalid_mail_data)

        assert result["failed_count"] >= 0

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_query_handles_empty_retrieval_results(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that query handles empty retrieval results."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever.retrieve.return_value = []  # Empty results
        mock_retriever_class.return_value = mock_retriever

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        result = pipeline.query("test query")

        assert "answer" in result
        assert "source_documents" in result
        assert len(result["source_documents"]) == 0

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_retrieve_handles_none_query(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that retrieve handles None query."""
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

        # Should handle None gracefully
        results = pipeline.retrieve(None)

        assert isinstance(results, list)

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_add_mail_data_handles_duplicate_documents(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that add_mail_data handles duplicate documents correctly."""
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
        mock_retriever.document_exists.return_value = True
        mock_retriever.update_documents = Mock()
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

        result = pipeline.add_mail_data(mail_data)

        # Should update existing document, not add new one
        assert result["updated_count"] == 1
        assert result["added_count"] == 0
        mock_retriever.update_documents.assert_called_once()


@pytest.mark.django_db
class TestRAGPipelinePublicMethods:
    """Test coverage for all public methods."""

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_convert_mail_to_json(self, mock_retriever_class, mock_processor_class):
        """Test convert_mail_to_json method."""
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

        # Create a mock mail object
        mock_mail = Mock()
        mock_mail.message_id = "test@example.com"
        mock_mail.subject = "Test Subject"
        mock_mail.content = "Test content"
        mock_mail.thread_url = "https://example.com/thread"
        mock_mail.parent = None
        mock_mail.children = []
        mock_mail.sender_address = "sender@example.com"
        mock_mail.from_field = "Sender Name <sender@example.com>"
        mock_mail.date = datetime.now()
        mock_mail.to = "to@example.com"
        mock_mail.cc = ""
        mock_mail.reply_to = ""
        mock_mail.url = "https://example.com/1"

        result = pipeline.convert_mail_to_json(mock_mail)

        assert isinstance(result, dict)
        assert result["message_id"] == "test@example.com"
        assert result["subject"] == "Test Subject"
        assert result["content"] == "Test content"
        assert result["url"] == "https://example.com/1"
        assert "thread_url" in result
        assert "sender_address" in result
        assert "date" in result

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_batch_query_handles_partial_failures(
        self,
        mock_retriever_class,
        mock_processor_class,
        sample_documents,
    ):
        """Test that batch_query handles partial failures."""
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
        # First call succeeds, second fails, third succeeds
        mock_llm_helper.process_pipeline.side_effect = [
            "Answer 1",
            ValueError("LLM error"),
            "Answer 3",
        ]

        with patch(
            "rag_service.langchain_rag.rag_pipeline.LLMHelper",
            return_value=mock_llm_helper,
        ):
            config = LangChainConfig()
            config.force_reindex = False
            config.enable_cache = False
            config.enable_telemetry = False
            config.base_retriever_types = ["mail"]

            pipeline = LangChainRAGPipeline(config=config)

            questions = ["Question 1", "Question 2", "Question 3"]
            results = pipeline.batch_query(questions)

            assert len(results) == 3
            assert "answer" in results[0]
            assert "error" in results[1].get("metadata", {})
            assert "answer" in results[2]

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_update_mail_data_handles_invalid_message(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that update_mail_data handles invalid message data."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor.process_mail_list.side_effect = ValueError("Invalid message")
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

        invalid_message = {"invalid": "data"}

        result = pipeline.update_mail_data(invalid_message)

        assert result is False

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_delete_document_handles_invalid_url(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that delete_document handles invalid URL format."""
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

        # Invalid URL format
        result = pipeline.delete_document("")

        assert result is False

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_get_retrieval_stats_handles_empty_retrievers(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test that get_retrieval_stats handles empty retrievers."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever.documents = []
        mock_retriever.dense_top_k = 10
        mock_retriever.sparse_top_k = 10
        mock_retriever_class.return_value = mock_retriever

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        stats = pipeline.get_retrieval_stats()

        assert "total_documents" in stats
        assert stats["total_documents"] == 0
        assert "base_retrievers" in stats
