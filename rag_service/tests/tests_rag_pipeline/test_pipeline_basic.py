"""
Basic RAG pipeline tests.

Tests for initialization, singleton pattern, and basic operations.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from rag_service.services import RAGService
from rag_service.langchain_rag.rag_pipeline import LangChainRAGPipeline
from config.rag_config import LangChainConfig


@pytest.mark.django_db
class TestRAGPipelineBasic:
    """Basic integration tests for the RAG pipeline."""

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_pipeline_initialization(self, mock_retriever_class, mock_processor_class):
        """Test that pipeline initializes correctly with all components."""
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

        assert pipeline.config == config
        assert pipeline.data_processor is not None
        assert "mail" in pipeline.base_retrievers

    @patch("rag_service.services.LangChainRAGPipeline")
    def test_rag_service_singleton(self, mock_pipeline_class):
        """Test that RAGService maintains singleton pattern."""
        RAGService.reset_pipeline()

        mock_pipeline_first = Mock()
        mock_pipeline_second = Mock()
        mock_pipeline_class.side_effect = [
            mock_pipeline_first,
            mock_pipeline_second,
        ]

        # First call should create pipeline
        pipeline1 = RAGService.get_pipeline()
        assert pipeline1 == mock_pipeline_first

        # Second call should return same instance
        pipeline2 = RAGService.get_pipeline()
        assert pipeline2 == pipeline1
        assert mock_pipeline_class.call_count == 1

        # Reset and verify new instance
        RAGService.reset_pipeline()
        pipeline3 = RAGService.get_pipeline()
        assert pipeline3 == mock_pipeline_second
        assert pipeline3 != pipeline1
        assert mock_pipeline_class.call_count == 2

    def test_rag_service_retrieve(self):
        """Test RAGService.retrieve method."""
        RAGService.reset_pipeline()

        mock_pipeline = Mock()
        mock_pipeline.retrieve.return_value = [
            Document(page_content="Test content", metadata={})
        ]

        with patch.object(RAGService, "get_pipeline", return_value=mock_pipeline):
            results = RAGService.retrieve("test query", fetch_k=5)

        assert len(results) == 1
        mock_pipeline.retrieve.assert_called_once_with("test query", 5, None, False)

    def test_rag_service_query(self):
        """Test RAGService.query method."""
        RAGService.reset_pipeline()

        mock_pipeline = Mock()
        mock_pipeline.query.return_value = {
            "answer": "Test answer",
            "source_documents": [],
            "query_time": 0.5,
        }

        with patch.object(RAGService, "get_pipeline", return_value=mock_pipeline):
            result = RAGService.query("test query", fetch_k=5)

        assert result["answer"] == "Test answer"
        mock_pipeline.query.assert_called_once_with("test query", 5, None)

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    def test_pipeline_retrieve_with_documents(
        self, mock_retriever_class, mock_processor_class, sample_documents
    ):
        """Test pipeline retrieval with actual documents."""
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

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        results = pipeline.retrieve("Boost.Asio", fetch_k=2)

        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)
        mock_retriever.retrieve.assert_called_once()

    @patch("rag_service.langchain_rag.rag_pipeline.BoostDataProcessor")
    @patch("rag_service.langchain_rag.rag_pipeline.LangChainHybridRetriever")
    @patch("rag_service.langchain_rag.rag_pipeline.LLMHelper")
    def test_pipeline_query_end_to_end(
        self,
        mock_llm_helper_class,
        mock_retriever_class,
        mock_processor_class,
        sample_documents,
    ):
        """Test full query pipeline from question to answer."""
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
        mock_llm_helper.process_pipeline.return_value = (
            "Boost.Asio is a networking library."
        )
        mock_llm_helper_class.return_value = mock_llm_helper

        config = LangChainConfig()
        config.force_reindex = False
        config.enable_cache = False
        config.enable_telemetry = False
        config.base_retriever_types = ["mail"]

        pipeline = LangChainRAGPipeline(config=config)

        result = pipeline.query("What is Boost.Asio?", fetch_k=2)

        assert "answer" in result
        assert "source_documents" in result
        assert "query_time" in result
        assert result["answer"] == "Boost.Asio is a networking library."
        assert len(result["source_documents"]) == 2
