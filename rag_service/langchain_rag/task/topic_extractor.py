"""
Topic Extraction Module

Extracts main topics from recent discussions using LLM agents.
"""

from typing import List
import structlog

logger = structlog.get_logger(__name__)
from typing import Dict, Any
import numpy as np

from ..llm import LLMHelper
from langchain_core.documents import Document

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None

class TopicExtractor:
    """Extract topics from discussions using LLM"""

    def __init__(self):
        self.logger = logger.bind(component="TopicExtractor")
        self.llm_helper = self._initialize_llm_helper()

    def _initialize_llm_helper(self):
        """Initialize LLM helper"""
        try:
            from config.rag_config import DEFAULT_CONFIG
            llm_helper = LLMHelper(config=DEFAULT_CONFIG)
            self.logger.info("Using LLM helper for topic extraction")
            return llm_helper
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM helper: {e}")
            return None

    def extract_topics_by_llm(self, emails: Dict[str, Any], max_topics: int = 10) -> List[str]:
        """
        Extract main topics from recent discussions

        Args:
            recent_discussions: List of discussion dictionaries with 'subject', 'content', 'date'
            max_topics: Maximum number of topics to extract

        Returns:
            List of topic strings
        """
        documents = emails['documents']
        if not documents:
            self.logger.warning("No documents provided for topic extraction")
            return []

        try:
            topics = self.llm_helper.process_pipeline("extract_topics", documents)
            if topics:
                topics = topics[:max_topics]
                self.logger.info(f"Extracted {len(topics)} topics: {topics}")
            else:
                self.logger.warning("No topics extracted from discussions")
            return topics
        except Exception as e:
            self.logger.error(f"Failed to extract topics: {e}")
            return []

    def extract_topics_by_thread(self, emails: Dict[str, Any], max_topics: int = 10) -> List[str]:
        """
        Extract main topics from recent discussions by thread

        Args:
            emails: Dictionary with 'documents' and 'embeddings' keys
            max_topics: Maximum number of topics to extract

        Returns:
            List of topic strings
        """
        documents = emails.get('documents', [])
        thread_ids = [doc.metadata['thread_id'] for doc in documents]
        if not thread_ids:
            self.logger.warning("No thread_ids provided for topic extraction")
            return []
        try:
            sorted_threads = self._group_documents_by_thread(thread_ids, max_topics)
            topics = self._summarize_clusters(sorted_threads, documents)

            return topics

        except Exception as e:
            self.logger.error(f"Failed to extract topics by clustering: {e}")
            return []


    def _group_documents_by_thread(self, thread_ids: List[str], max_topics: int = 10) -> List[List[Document]]:
        """
        Cluster documents by thread

        Args:
            emails: Dictionary with 'documents' and 'embeddings' keys
            max_topics: Maximum number of topics to extract

        Returns:
            List of clusters
        """
        unique_thread_ids, counts = np.unique(thread_ids, return_counts=True)
        cluster_result_info = []
        for i, unique_thread_id in enumerate(unique_thread_ids):
            member_list = [i for i, thread_id in enumerate(thread_ids) if thread_id == unique_thread_id]
            if len(member_list) < 2:
                continue
            cluster_result_info.append(member_list)
        return sorted(cluster_result_info, key=lambda x: len(x), reverse=True)[:max_topics]

    def extract_topics_by_clustering(self, emails: Dict[str, Any], max_topics: int = 10) -> List[str]:
        """
        Extract main topics from recent discussions by clustering embeddings

        Args:
            emails: Dictionary with 'documents' and 'embeddings' keys
            max_topics: Maximum number of topics to extract

        if not documents:
            self.logger.warning("No documents provided for topic extraction")
            return []

        try:
            topics = self.llm_helper.process_pipeline("extract_topics", documents)
            if topics:
                topics = topics[:max_topics]
        Returns:
            List of topic strings (summaries for top n clusters)
        """
        documents = emails.get('documents', [])
        embeddings = emails.get('embeddings', [])

        # Check if documents or embeddings are empty (use len() to avoid NumPy array truthiness issues)
        if not documents or (embeddings is None or len(embeddings) == 0):
            self.logger.warning("No documents or embeddings provided for clustering")
            return []

        if len(documents) != len(embeddings):
            self.logger.error(f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings")
            return []

        if not SKLEARN_AVAILABLE:
            self.logger.error("sklearn not available. Install with: pip install scikit-learn")
            return []

        try:
            sorted_clusters = self._cluster_documents_by_kmeans(embeddings, max_topics)
            topics = self._summarize_clusters(sorted_clusters, documents)

            return topics

        except Exception as e:
            self.logger.error(f"Failed to extract topics by clustering: {e}")
            return []

    def _cluster_documents_by_kmeans(self, embeddings: List[List[float]], n_clusters: int = 10) -> List[List[Document]]:
        """
        Cluster documents into clusters

        Args:
            documents: List of documents to cluster
            n_clusters: Number of clusters to create

        Returns:
            List of clusters
        """
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)

        # Determine number of clusters (use min of max_topics and number of documents)
        n_clusters_to_create = min(round(n_clusters*1.5), len(embeddings))
        if n_clusters_to_create < 2:
            self.logger.warning(f"Not enough documents for clustering. Need at least 2, got {len(embeddings)}")
            return []

        # Perform KMeans clustering
        self.logger.info(f"Clustering {len(embeddings)} documents into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters_to_create, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)

        # Get cluster sizes and sort by size (descending)
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))
        sorted_cluster_info = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:n_clusters]
        cluster_result_info = []
        for cluster_id, cluster_size in sorted_cluster_info:
            indics_of_cluster = [i for i, cluster_label in enumerate(cluster_labels) if cluster_label == cluster_id]
            if len(indics_of_cluster) < 2:
                continue
            cluster_result_info.append(indics_of_cluster)

        return cluster_result_info

    def _summarize_clusters(self, sorted_clusters, documents: List[Document]) -> List[str]:
        """
        Summarize a cluster of documents into a topic

        Args:
            cluster_docs: List of documents in the cluster

        Returns:
            Topic summary string
        """
        n_topics = len(sorted_clusters)
        if not self.llm_helper:
            # Fallback: use first document's subject
            return "Unknown"

        # Extract topics for top n clusters
        topics = []
        for cluster_indicies in sorted_clusters:
            # Get documents in this cluster
            cluster_docs = [documents[i] for i in cluster_indicies]

            if not cluster_docs:
                continue

            # Generate summary for this cluster using LLM
            try:
                topic_summary = self.llm_helper.process_pipeline("total_summarize", cluster_docs)
                if topic_summary:
                    topics.append(topic_summary)
                    # self.logger.info(f"Cluster {cluster_id} (size {cluster_size}): {topic_summary[:100]}...")
            except Exception as e:
                self.logger.error(f"Failed to summarize cluster {cluster_indicies}: {e}")
                # Fallback: use first document's subject as topic
                if cluster_docs:
                    first_doc = cluster_docs[0]
                    if isinstance(first_doc, Document):
                        subject = first_doc.metadata.get('subject', 'Unknown')
                        topics.append(subject)

        return topics
