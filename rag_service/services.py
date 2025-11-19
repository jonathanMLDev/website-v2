"""
RAG Service Wrapper.

Provides a singleton RAG service that other Django apps can import and use.
"""

from typing import Optional, List, Dict, Any

from .langchain_rag.rag_pipeline import LangChainRAGPipeline

# Global pipeline instance
_pipeline: Optional[LangChainRAGPipeline] = None


class RAGService:
    """Singleton RAG service wrapper."""

    @staticmethod
    def get_pipeline(**kwargs) -> LangChainRAGPipeline:
        """
        Get or create the RAG pipeline instance.

        Args:
            **kwargs: Optional configuration overrides

        Returns:
            LangChainRAGPipeline instance
        """
        global _pipeline

        if _pipeline is None:
            _pipeline = LangChainRAGPipeline()

        return _pipeline

    @staticmethod
    def reset_pipeline():
        """Reset the global pipeline instance (useful for testing)."""
        global _pipeline
        _pipeline = None

    @staticmethod
    def retrieve(
        question: str,
        fetch_k: int = 10,
        filter_types: List[str] = None,
        str_results: bool = False,
    ) -> List[str]:
        """
        Retrieve relevant documents.

        Args:
            question: Search query
            fetch_k: Number of results to return
            filter_types: List of document types to filter by
            str_results: If True, return text content instead of Document objects

        Returns:
            List of documents or text strings
        """
        pipeline = RAGService.get_pipeline()
        return pipeline.retrieve(question, fetch_k, filter_types, str_results)

    @staticmethod
    def query(
        question: str,
        fetch_k: int = 10,
        filter_types: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline.

        Args:
            question: Search query
            fetch_k: Number of results to return
            filter_types: List of document types to filter by

        Returns:
            Dictionary with answer, source documents, and metadata
        """
        pipeline = RAGService.get_pipeline()
        return pipeline.query(question, fetch_k, filter_types)
