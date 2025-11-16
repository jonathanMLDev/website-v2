"""
Multi-base retriever that manages multiple base retrievers and combines results
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import structlog

logger = structlog.get_logger(__name__)

from .hybrid_retriever import LangChainHybridRetriever


class MultiBaseRetriever:
    """
    Manages multiple base retrievers (one per data source type) and combines results
    based on filter_types.
    """

    def __init__(
        self,
        base_retrievers: Dict[str, LangChainHybridRetriever],
    ):
        """
        Initialize multi-base retriever with a dictionary of base retrievers

        Args:
            base_retrievers: Dictionary mapping retriever type (e.g., "mail", "documentation")
                           to LangChainHybridRetriever instances
        """
        self.base_retrievers = base_retrievers
        self.logger = logger.bind(component="MultiBaseRetriever")
        self.logger.info(f"Initialized with {len(base_retrievers)} base retrievers: {list(base_retrievers.keys())}")

    def retrieve(
        self,
        query: str,
        fetch_k: int = 10,
        filter_types: Optional[List[str]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve documents from one or more base retrievers based on filter_types

        Args:
            query: Search query
            fetch_k: Number of results to return
            filter_types: List of document types to filter by (e.g., ["mail", "documentation"])
                        If None or empty, retrieves from all base retrievers and combines results
            **kwargs: Additional arguments passed to base retrievers

        Returns:
            List of Document objects from selected base retrievers
        """
        # If no filter_types specified, use all base retrievers
        if not filter_types or len(filter_types) == 0:
            selected_types = list(self.base_retrievers.keys())
            self.logger.debug(f"No filter_types specified, using all retrievers: {selected_types}")
        else:
            # Normalize filter types to lowercase
            filter_types_normalized = [ft.lower() for ft in filter_types]
            # Select only the base retrievers that match filter_types
            selected_types = [
                ret_type for ret_type in self.base_retrievers.keys()
                if ret_type.lower() in filter_types_normalized
            ]

            if not selected_types:
                selected_types = list(self.base_retrievers.keys())

            self.logger.debug(f"Selected base retrievers: {selected_types} for filter_types: {filter_types}")

        # Retrieve from selected base retrievers
        all_results = []
        for retriever_type in selected_types:
            retriever = self.base_retrievers[retriever_type]
            try:
                # Retrieve from this base retriever
                # Note: We don't pass filter_types to individual retrievers since
                # each retriever is already filtered by type (mail retriever only has mail docs, etc.)
                results = retriever.retrieve(query, fetch_k=fetch_k, filters=filter_types, **kwargs)
                self.logger.debug(
                    f"Retrieved {len(results)} documents from {retriever_type} retriever"
                )
                all_results.extend(results)
            except Exception as e:
                self.logger.exception(f"Error retrieving from {retriever_type} retriever: {e}")
                continue

        # Combine and deduplicate results
        combined_results = self._combine_and_deduplicate(all_results, fetch_k)

        self.logger.info(
            f"Retrieved {len(combined_results)} documents from {len(selected_types)} base retrievers"
        )

        return combined_results

    def _combine_and_deduplicate(
        self,
        documents: List[Document],
        fetch_k: int
    ) -> List[Document]:
        """
        Combine results from multiple retrievers and deduplicate by document ID/URL

        Args:
            documents: List of documents from multiple retrievers
            fetch_k: Maximum number of documents to return

        Returns:
            Deduplicated and sorted list of documents
        """
        if not documents:
            return []

        # Create a dictionary to deduplicate by document URL
        seen_docs = {}
        for doc in documents:
            # Use URL as unique identifier
            doc_url = doc.metadata.get("url", "")
            doc_id = doc.id or doc_url

            # If we haven't seen this document, or if this one has a higher score
            if doc_id not in seen_docs:
                seen_docs[doc_id] = doc
            else:
                # Keep the document with the higher final_score
                existing_score = seen_docs[doc_id].metadata.get("final_score", 0.0)
                new_score = doc.metadata.get("final_score", 0.0)
                if new_score > existing_score:
                    seen_docs[doc_id] = doc

        # Convert back to list and sort by final_score
        deduplicated = list(seen_docs.values())
        deduplicated.sort(
            key=lambda x: x.metadata.get("final_score", 0.0),
            reverse=True
        )

        # Return top fetch_k
        return deduplicated[:fetch_k]

    def get_retriever(self, retriever_type: str) -> Optional[LangChainHybridRetriever]:
        """
        Get a specific base retriever by type

        Args:
            retriever_type: Type of retriever to get (e.g., "mail", "documentation")

        Returns:
            Base retriever instance or None if not found
        """
        return self.base_retrievers.get(retriever_type.lower())

    def get_available_types(self) -> List[str]:
        """
        Get list of available retriever types

        Returns:
            List of available retriever type names
        """
        return list(self.base_retrievers.keys())

    def add_documents(self, documents: List[Document], retriever_type: str):
        """
        Add documents to a specific base retriever

        Args:
            documents: Documents to add
            retriever_type: Type of retriever to add documents to
        """
        retriever = self.get_retriever(retriever_type.lower())
        if retriever:
            retriever.add_documents(documents)
        else:
            self.logger.warning(f"No retriever found for type: {retriever_type}")

    def update_documents(self, documents: List[Document], retriever_type: str):
        """
        Update documents in a specific base retriever

        Args:
            documents: Documents to update
            retriever_type: Type of retriever to update documents in
        """
        retriever = self.get_retriever(retriever_type.lower())
        if retriever:
            retriever.update_documents(documents)
        else:
            self.logger.warning(f"No retriever found for type: {retriever_type}")

    def delete_documents(self, document_ids: List[str], retriever_type: str):
        """
        Delete documents from a specific base retriever

        Args:
            document_ids: Document IDs to delete
            retriever_type: Type of retriever to delete documents from
        """
        retriever = self.get_retriever(retriever_type.lower())
        if retriever:
            retriever.delete_documents(document_ids)
        else:
            self.logger.warning(f"No retriever found for type: {retriever_type}")

