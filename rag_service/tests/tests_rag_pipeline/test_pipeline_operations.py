"""
RAG pipeline operation tests.

Tests for CRUD operations: add, update, delete, and statistics.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from rag_service.langchain_rag.rag_pipeline import LangChainRAGPipeline
from config.rag_config import LangChainConfig


@pytest.mark.django_db
class TestRAGPipelineOperations:
    """Test CRUD operations for the RAG pipeline."""

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_pipeline_add_mail_data(self, mock_retriever_class, mock_processor_class):
        """Test adding mail data to pipeline."""
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
        mock_retriever.add_documents = Mock()
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

        assert result["added_count"] == 1
        assert result["total_processed"] == 1
        mock_retriever.add_documents.assert_called_once()

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_pipeline_update_mail_data(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test updating existing mail data in pipeline."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor.process_mail_list.return_value = [
            Document(page_content="Updated", metadata={"url": "test://1"})
        ]
        mock_processor.chunk_documents.return_value = [
            Document(page_content="Updated", metadata={"url": "test://1"})
        ]
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever.update_documents = Mock()
        mock_retriever_class.return_value = mock_retriever

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        mail_data = {
            "message_id": "test@example.com",
            "subject": "Updated Subject",
            "content": "Updated content",
            "url": "test://1",
        }

        result = pipeline.update_mail_data(mail_data)

        assert result is True
        mock_retriever.update_documents.assert_called_once()

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_pipeline_delete_document(self, mock_retriever_class, mock_processor_class):
        """Test deleting a document from pipeline."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever.document_exists.return_value = True
        mock_retriever.get_document_id.return_value = "test-doc-id"
        mock_retriever.delete_documents = Mock()
        mock_retriever_class.return_value = mock_retriever

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        result = pipeline.delete_document("test://1")

        assert result is True
        mock_retriever.delete_documents.assert_called_once()

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_pipeline_get_retrieval_stats(
        self, mock_retriever_class, mock_processor_class
    ):
        """Test getting retrieval statistics."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_retriever = Mock()
        mock_retriever.load_index.return_value = {
            "vector_store": True,
            "sparse_retriever": True,
        }
        mock_retriever.documents = [Document(page_content="Test", metadata={})]
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
        assert "base_retrievers" in stats
        assert "mail" in stats["base_retrievers"]
        assert stats["total_documents"] == 1

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    @patch("rag_service.langchain_rag.rag_pipeline.LLMHelper")
    @patch("rag_service.langchain_rag.rag_pipeline.tqdm")
    def test_batch_query_processing(
        self,
        mock_tqdm,
        mock_llm_helper_class,
        mock_retriever_class,
        mock_processor_class,
        sample_documents,
    ):
        """Test batch query processing."""
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
        mock_llm_helper.process_pipeline.return_value = "Test answer"
        mock_llm_helper_class.return_value = mock_llm_helper

        # Mock tqdm to return the list directly
        mock_tqdm.side_effect = lambda x, **kwargs: x

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        questions = ["Question 1", "Question 2", "Question 3"]
        results = pipeline.batch_query(questions)

        assert len(results) == 3
        assert all("answer" in result for result in results)

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_clear_cache(self, mock_retriever_class, mock_processor_class):
        """Test clear_cache method."""
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
        config.enable_cache = True
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)
        mock_cache = Mock()
        pipeline.cache = mock_cache

        pipeline.clear_cache()

        mock_cache.clear.assert_called_once()

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_cleanup_cache(self, mock_retriever_class, mock_processor_class):
        """Test cleanup_cache method."""
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
        config.enable_cache = True
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)
        mock_cache = Mock()
        mock_cache.cleanup_expired_files.return_value = 5
        mock_cache.enforce_disk_size_limit.return_value = 3
        pipeline.cache = mock_cache

        result = pipeline.cleanup_cache()

        assert result["expired"] == 5
        assert result["size_limit"] == 3
        mock_cache.cleanup_expired_files.assert_called_once()
        mock_cache.enforce_disk_size_limit.assert_called_once()

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_get_cache_size_info(self, mock_retriever_class, mock_processor_class):
        """Test get_cache_size_info method."""
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
        config.enable_cache = True
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)
        mock_cache = Mock()
        mock_cache.get_cache_size_info.return_value = {
            "total_size": 1024,
            "file_count": 10,
        }
        pipeline.cache = mock_cache

        result = pipeline.get_cache_size_info()

        assert result["total_size"] == 1024
        assert result["file_count"] == 10
        mock_cache.get_cache_size_info.assert_called_once()

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_print_performance_report(self, mock_retriever_class, mock_processor_class):
        """Test print_performance_report method."""
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
        config.enable_telemetry = True
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)
        mock_telemetry = Mock()
        pipeline.telemetry = mock_telemetry

        pipeline.print_performance_report()

        mock_telemetry.print_report.assert_called_once()
