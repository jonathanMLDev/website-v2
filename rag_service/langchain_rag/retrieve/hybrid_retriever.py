"""
Hybrid retriever for LangChain RAG pipeline (Dense + Sparse only)
"""

import gc
import hashlib
import math
import os
import pickle
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List

import structlog
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = structlog.get_logger(__name__)

# Global cache for embedding models to avoid reloading
_embedding_model_cache: Dict[str, HuggingFaceEmbeddings] = {}
_cache_lock = threading.Lock()


def get_cached_embedding_model(
    model_name: str, device: str = None
) -> HuggingFaceEmbeddings:
    """
    Get or create a cached embedding model instance.

    This function implements a singleton pattern to ensure the same embedding model
    is reused across multiple retriever instances, avoiding expensive reloads.

    Args:
        model_name: Name of the embedding model
        device: Device to use ('cpu', 'cuda', etc.). Auto-detect if None

    Returns:
        Cached HuggingFaceEmbeddings instance
    """
    # Auto-detect device if not provided
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # Create cache key from model name and device
    cache_key = f"{model_name}:{device}"

    with _cache_lock:
        if cache_key not in _embedding_model_cache:
            logger.info(
                f"Loading embedding model: {model_name} on {device} (first time)"
            )
            _embedding_model_cache[cache_key] = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
            )
            logger.info(f"Embedding model cached: {cache_key}")
        else:
            logger.debug(f"Reusing cached embedding model: {cache_key}")

    return _embedding_model_cache[cache_key]


class LangChainHybridRetriever:
    """Hybrid retriever combining dense (vector) and sparse (BM25) retrieval for LangChain"""

    def __init__(
        self,
        config: Any,
        chroma_persist_dir: str = None,
    ):
        """
        Create a new hybrid retriever instance.

        Args:
            config: Configuration object providing embedding model, top-k values,
                persistence paths, and other knobs from ``LangChainConfig``.
            chroma_persist_dir: Optional override directory for storing ChromaDB
                artifacts; defaults to ``config.chroma_persist_dir``.
        """
        if config is None:
            from config.rag_config import DEFAULT_CONFIG

            config = DEFAULT_CONFIG

        self.documents = []
        self.embedding_model_name = config.embedding_model
        self.dense_top_k = config.dense_top_k
        self.sparse_top_k = config.sparse_top_k
        self.final_top_k = config.final_top_k
        self.force_reindex = config.force_reindex
        self.chroma_persist_dir = chroma_persist_dir or config.chroma_persist_dir
        self.logger = logger.bind(component="LangChainHybridRetriever")
        self.vector_store = None
        self.dense_retriever = None
        self.half_life = config.half_life
        # Initialize components
        self._setup_embedding_model()

    def reindex(self, documents: List[Document], status: dict = None):
        """Reindex the hybrid retriever - create new indices from documents"""
        if status is None:
            status = {
                "vector_store": False,
                "sparse_retriever": False,
            }
        if all(status.values()):
            return

        try:
            self.logger.info("Starting reindex with %d documents", len(documents))
            self.documents = documents

            # Force reindex by clearing existing data
            self._clear_existing_indices(status)

            # Create new indices
            if not status["vector_store"]:
                status["vector_store"] = self._create_vector_store()
            if not status["sparse_retriever"]:
                status["sparse_retriever"] = self._create_sparse_retriever()

            self.logger.info("Reindex completed successfully")
        except (
            Exception
        ) as e:  # pragma: no cover - unexpected failures should surface to callers
            self.logger.exception("Failed to reindex: %s", e)
            raise

    def load_index(self) -> dict:
        """Load the hybrid retriever index - load from saved data"""
        status = {
            "vector_store": False,
            "sparse_retriever": False,
        }

        self.logger.info(f"Loading existing indices from {self.chroma_persist_dir}")
        status["vector_store"] = self._load_vector_store()
        status["sparse_retriever"] = self._load_sparse_retriever()

        return status

    def _clear_existing_indices(self, status: dict):
        """Clear existing indices for reindexing"""
        try:
            self.logger.info("Clearing existing indices")
            if not status["vector_store"]:
                self._clear_vector_store()
            if not status["sparse_retriever"]:
                self._clear_sparse_retriever()
        except Exception as e:
            self.logger.exception("Failed to clear indices: %s", e)
            raise

    def _clear_vector_store(self):
        """Clear ChromaDB vector store"""
        try:
            self.logger.info("Clearing vector store")
            # ChromaDB clears on recreate
            self.logger.info("Vector store ready for reindex")
        except Exception as e:
            self.logger.info("No existing vector store to clear: %s", e)

    def _clear_sparse_retriever(self):
        """Clear TF-IDF vectorizer files for reindexing"""
        try:
            self.logger.info("Clearing TF-IDF vectorizer")
            # Only TF-IDF vectorizer is persisted (BM25 created dynamically)
            tfidf_file = os.path.join(self.chroma_persist_dir, "tfidf_vectorizer.pkl")
            if os.path.exists(tfidf_file):
                os.remove(tfidf_file)
                self.logger.info("Deleted tfidf_vectorizer.pkl")

            self.logger.info("TF-IDF vectorizer cleared")
        except Exception as e:
            self.logger.warning("Error clearing TF-IDF vectorizer: %s", e)

    def _setup_embedding_model(self):
        """Setup embedding model using cached instance"""
        try:
            device = "cuda" if self._is_cuda_available() else "cpu"
            self.embedding_model = get_cached_embedding_model(
                model_name=self.embedding_model_name, device=device
            )
            self.logger.info(
                "Embedding model initialized (cached): %s on %s",
                self.embedding_model_name,
                device,
            )
        except Exception as e:
            self.logger.exception("Failed to initialize embedding model: %s", e)
            raise

    def _create_vector_store(self, batch_size: int = 100) -> bool:
        """Create new vector store with documents"""
        try:
            # Create embedded ChromaDB vector store
            self.logger.info("Creating embedded Chroma vector store")

            # Initialize empty ChromaDB vector store
            self.vector_store = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.chroma_persist_dir,
            )

            # Add documents in batches with progress tracking
            total_docs = len(self.documents)

            with tqdm(
                total=total_docs, desc="Vectorizing documents", unit="doc"
            ) as pbar:
                for i in range(0, total_docs, batch_size):
                    batch = self.documents[i : i + batch_size]
                    self.vector_store.add_documents(batch)
                    pbar.update(len(batch))

            self.logger.info("Embedded vector store created successfully")

            # Create dense retriever
            self.dense_retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": self.dense_top_k, "score_threshold": 0.3},
            )

            self.logger.info(
                "Vector store and dense retriever initialized successfully"
            )
            return True
        except Exception as e:
            self.logger.exception("Failed to create vector store: %s", e)
            return False

    def _load_vector_store(self) -> bool:
        """Load existing vector store"""
        try:
            self.logger.info("Loading embedded vector store")

            # Check if ChromaDB directory exists and has data
            if not os.path.exists(self.chroma_persist_dir):
                self.logger.info(
                    "ChromaDB directory does not exist: %s", self.chroma_persist_dir
                )
                return False

            # Check for chroma.sqlite3 or collection data
            chroma_db_file = os.path.join(self.chroma_persist_dir, "chroma.sqlite3")
            if not os.path.exists(chroma_db_file):
                self.logger.info("ChromaDB data file not found: %s", chroma_db_file)
                return False

            # Load existing ChromaDB
            self.vector_store = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.chroma_persist_dir,
            )
            # Verify collection has documents
            try:
                # Try to get collection count
                collection = self.vector_store._collection
                count = collection.count()
                if count == 0:
                    self.logger.info("Vector store exists but is empty")
                    return False
                self.logger.info("Found %d documents in vector store", count)
            except Exception as e:
                self.logger.warning("Could not verify collection count: %s", e)

            # docs = self.vector_store._collection.get(include=["documents", "metadatas", "embeddings"])
            # self.vector_store.delete_collection()
            # self.vector_store = Chroma.from_documents(
            #     documents=docs["documents"],
            #     metadatas=docs["metadatas"],
            #     embeddings=docs["embeddings"],
            #     persist_directory=self.chroma_persist_dir,
            # )
            # Get documents from vector store
            self.documents = []  # Will be populated during retrieval

            # Create dense retriever
            self.dense_retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": self.dense_top_k, "score_threshold": 0.3},
            )

            self.logger.info("Vector store loaded successfully")
            gc.collect()  # Clean up
            return True
        except Exception as e:
            self.logger.exception("Failed to load vector store: %s", e)
            return False

    def _create_sparse_retriever(self) -> bool:
        """Create and save TF-IDF vectorizer for reranking (BM25 created dynamically)"""
        try:
            self.logger.info("Creating TF-IDF vectorizer for reranking")

            # Prepare documents for TF-IDF
            doc_texts = [doc.page_content for doc in self.documents]

            # Setup TF-IDF for reranking
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000, stop_words="english", ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_texts)

            # Save TF-IDF vectorizer
            os.makedirs(self.chroma_persist_dir, exist_ok=True)
            tfidf_path = os.path.join(self.chroma_persist_dir, "tfidf_vectorizer.pkl")
            with open(tfidf_path, "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)

            self.logger.info("TF-IDF vectorizer created and saved successfully")
            return True
        except Exception as e:
            self.logger.exception("Failed to create TF-IDF vectorizer: %s", e)
            return False

    def _load_sparse_retriever(self) -> bool:
        """Load TF-IDF vectorizer for reranking
        (BM25 created dynamically during retrieval)"""
        try:
            self.logger.info("Loading TF-IDF vectorizer")

            tfidf_path = os.path.join(self.chroma_persist_dir, "tfidf_vectorizer.pkl")
            if not os.path.exists(tfidf_path):
                self.logger.info(
                    "TF-IDF vectorizer file not found: %s "
                    "(will be created during reindexing)",
                    tfidf_path,
                )
                return False

            # Load TF-IDF vectorizer for reranking
            with open(tfidf_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)

            self.logger.info("TF-IDF vectorizer loaded successfully")
            return True
        except Exception as e:
            self.logger.warning(
                "Failed to load TF-IDF vectorizer: %s "
                "(will be created during reindexing)",
                e,
            )
            return False

    def retrieve(
        self, query: str, fetch_k: int = 10, filters: Dict[str, Any] = None
    ) -> List[Document]:
        """Perform hybrid retrieval (Dense + Sparse) with optional type filtering

        Args:
            query: Search query
            fetch_k: Number of results to return
            filter_types: List of document types to filter by
                (e.g., ["documentation", "mail", "slack"])
                         If None or empty, no filtering is applied

        Returns:
            List of Document objects matching the query and filters
        """
        # Dense retrieval
        if filters is None:
            dense_docs = self.dense_retriever.invoke(query)
        else:
            dense_docs = self.dense_retriever.invoke(query, filters)

        if len(dense_docs) == 0:
            return []
        # sparse_docs = self.sparse_retriever.invoke(query)
        sparse_retriever = BM25Retriever.from_documents(
            documents=dense_docs, k=self.sparse_top_k
        )

        # Sparse retrieval
        sparse_docs = sparse_retriever.invoke(query)

        # Combine and rerank results
        combined_docs = self._combine_results(dense_docs, sparse_docs)

        # Rerank using TF-IDF similarity
        reranked_docs = self._rerank_results(query, combined_docs)
        fetch_k = fetch_k if fetch_k is not None else self.final_top_k
        fetch_k = min(fetch_k, len(reranked_docs))

        gc.collect()
        return reranked_docs[:fetch_k]

    def _combine_results(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
    ) -> List[Document]:
        """Combine results from dense and sparse retrievers"""
        # Create a dictionary to store combined scores
        combined_scores = {}

        # Add dense retrieval scores
        for i, doc in enumerate(dense_docs):
            doc_id = doc.id if doc.id else id(doc.metadata["url"])
            # Higher score for earlier results
            combined_scores[doc_id] = {
                "document": doc,
                "dense_score": 1.0 / (i + 1),
                "sparse_score": 0.0,
            }

        # Add sparse retrieval scores
        for i, doc in enumerate(sparse_docs):
            doc_id = doc.id if doc.id else id(doc.metadata["url"])

            if doc_id in combined_scores:
                combined_scores[doc_id]["sparse_score"] = 1.0 / (i + 1)
            else:
                combined_scores[doc_id] = {
                    "document": doc,
                    "dense_score": 0.0,
                    "sparse_score": 1.0 / (i + 1),
                }

        # Create combined documents with ensemble scoring
        combined_docs = []
        for doc_id, scores in combined_scores.items():
            # Weighted ensemble scoring (60% dense, 40% sparse)
            ensemble_score = 0.6 * scores["dense_score"] + 0.4 * scores["sparse_score"]

            # Update document metadata with ensemble score
            doc = scores["document"]
            doc.metadata["ensemble_score"] = ensemble_score

            combined_docs.append(doc)

        return combined_docs

    def _rerank_results(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank results using query-document similarity"""
        # Use TF-IDF similarity for reranking
        query_vector = self.tfidf_vectorizer.transform([query])

        reranked_docs = []
        for doc in docs:
            doc_text = doc.page_content
            doc_vector = self.tfidf_vectorizer.transform([doc_text])
            similarity = cosine_similarity(query_vector, doc_vector)[0][0]
            # similarity = 0.5

            # Combine original score with TF-IDF similarity
            original_score = doc.metadata.get("ensemble_score", 0.0)
            final_score = 0.9 * original_score + 0.1 * similarity

            # Add time exponential bonus for mail documents 0~1.0
            if doc.metadata.get("type") == "mail" and "date" in doc.metadata:
                # Clean content without modifying original unnecessarily
                cleaned_content = doc_text.replace(">>", "")
                if cleaned_content != doc_text:
                    doc.page_content = cleaned_content

                time_bonus = self._calculate_time_bonus(doc.metadata["date"])
                final_score = final_score * (0.5 + time_bonus)

            # Update document metadata
            doc.metadata["final_score"] = final_score

            # Remove temporary metadata
            # (don't delete "source" - it's useful for tracking)
            if "ensemble_score" in doc.metadata:
                del doc.metadata["ensemble_score"]

            reranked_docs.append(doc)

        # Sort by final score
        reranked_docs.sort(
            key=lambda x: x.metadata.get("final_score", 0.0), reverse=True
        )
        return reranked_docs

    def _calculate_time_bonus(self, doc_dt: Any) -> float:
        """Calculate exponential time bonus for mail documents based on recency"""
        try:
            # Parse the date string (assuming ISO format or common formats)
            # Try multiple date formats
            if isinstance(doc_dt, datetime):
                doc_date = doc_dt
            elif isinstance(doc_dt, int):
                doc_date = datetime.fromtimestamp(doc_dt)
            elif isinstance(doc_dt, str):
                date_formats = [
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d",
                    "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822 format
                    "%a, %d %b %Y %H:%M:%S",
                ]

                doc_date = None
                for fmt in date_formats:
                    try:
                        doc_date = datetime.strptime(doc_dt.strip(), fmt)
                        break
                    except ValueError:
                        continue
            else:
                self.logger.warning("Invalid date type: %s", type(doc_dt))
                return 0.5

            if doc_date is None:
                # If parsing fails, return no bonus
                self.logger.warning("Failed to parse date: %s", doc_dt)
                return 0.0

            # Remove timezone info if present for comparison
            if doc_date.tzinfo is not None:
                doc_date = doc_date.replace(tzinfo=None)

            # Calculate days since the email
            current_date = datetime.now()
            days_ago = (current_date - doc_date).days

            # Exponential decay: more recent = higher bonus
            # Bonus decays with half-life of ~365 days (1 year)
            # Recent emails (< 1 year) get significant bonus
            # Older emails get diminishing bonus

            # decay_rate = 0.002  # Adjust this to control decay speed
            # time_bonus = math.exp(-decay_rate * days_ago)

            time_bonus = math.exp(-days_ago / self.half_life)

            time_bonus = max(0.1, min(1.0, time_bonus))

            return time_bonus

        except Exception as e:
            self.logger.warning("Error calculating time bonus: %s", e)
            return 0.0

    def document_exists(self, doc_url: str) -> bool:
        """Check if a document with the given ID exists in the vector store

        Args:
            doc_url: Document URL to check

        Returns:
            True if document exists, False otherwise
        """
        try:
            doc_id = self.get_document_id(doc_url)

            # Query ChromaDB for the document ID
            results = self.vector_store.get_by_ids([doc_id])
            return len(results) > 0
        except Exception as e:
            self.logger.warning(
                "Error checking document existence for URL %s: %s", doc_url, e
            )
            return False

    def get_document_id(self, doc_url: str) -> str:
        """Get the document ID for a given URL"""
        return hashlib.md5(doc_url.encode()).hexdigest() + "-000"

    def check_existing_documents(
        self, documents: List[Document]
    ) -> tuple[List[Document], List[Document]]:
        """Check which documents already exist in the database

        Args:
            documents: List of documents to check

        Returns:
            Tuple of (new_documents, existing_documents)
        """
        new_documents = []
        existing_documents = []

        for doc in documents:
            doc_url = doc.metadata.get("url", "N/A")
            if self.document_exists(doc_url):
                existing_documents.append(doc)
                msg_id = doc.metadata.get("message_id", "N/A")
                self.logger.debug(
                    "Document %s already exists (URL: %s)", msg_id, doc_url
                )
            else:
                new_documents.append(doc)

        self.logger.info(
            "Checked %d documents: %d new, %d already exist",
            len(documents),
            len(new_documents),
            len(existing_documents),
        )

        return new_documents, existing_documents

    def add_documents(self, documents: List[Document]):
        """Add new documents to vector store (BM25 created dynamically during retrieval)"""
        try:
            self.logger.info("Adding %d documents to hybrid retriever", len(documents))

            # Add to vector store
            self.vector_store.add_documents(documents)
            self.logger.info("Documents added to vector store")

            # Note: BM25 is created dynamically during retrieval for memory efficiency
            # TF-IDF vectorizer will need to be retrained on next full reindex

            # Update documents list
            self.documents.extend(documents)
            self.logger.info("Total documents: %d", len(self.documents))

        except Exception as e:
            self.logger.exception("Failed to add documents: %s", e)
            raise

    def update_documents(self, updated_documents: List[Document]):
        """Update existing documents in both retrievers"""
        try:
            self.logger.info("Updating %d documents", len(updated_documents))
            document_ids = [doc.id for doc in updated_documents]
            # Delete old documents
            self.delete_documents(document_ids)

            # Add updated documents
            self.add_documents(updated_documents)

            self.logger.info("Documents updated successfully")
        except Exception as e:
            self.logger.exception("Failed to update documents: %s", e)
            raise

    def delete_documents(self, document_ids: List[str]):
        """Delete documents from vector store (BM25 created dynamically during retrieval)"""
        try:
            self.logger.info("Deleting %d documents", len(document_ids))

            # Delete from vector store (ChromaDB)
            try:
                self.vector_store.delete(ids=document_ids)
                self.logger.info("Documents deleted from vector store")
            except Exception as e:
                self.logger.warning("Could not delete from vector store: %s", e)

            # Note: BM25 is created dynamically during retrieval for memory efficiency
            # TF-IDF vectorizer will need to be retrained on next full reindex

            self.logger.info("Documents deleted successfully")
        except Exception as e:
            self.logger.exception("Failed to delete documents: %s", e)
            raise

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False
