"""
LangChain RAG Pipeline Configuration
"""

import os
from typing import List
from dataclasses import dataclass, field


@dataclass
class LangChainConfig:
    """Configuration class for LangChain RAG pipeline"""

    # Data settings
    mail_data_dir: str = "data/message_by_thread"
    doc_data_dir: str = "data/documentation"
    # chunk_size: int = 5000  # for embeddinggemma-300m
    # chunk_overlap: int = 300  # for embeddinggemma-300m
    chunk_size: int = 1024   # for sentence-transformers/all-MiniLM-L6-v2
    chunk_overlap: int = 100  # for sentence-transformers/all-MiniLM-L6-v2

    # Model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_model: str = "google/gemini-2.5-flash-lite"  # OpenAI model name
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    # Vector store settings
    chroma_persist_dir: str = "rag_service/langchain_rag/chroma_db"
    
    # Base retriever types - list of data source types to create retrievers for
    # Options: "mail", "documentation", "git", "slack", etc.
    base_retriever_types: List[str] = field(default_factory=lambda: ["mail"])

    # Retrieval settings
    dense_top_k: int = 100
    sparse_top_k: int = 100
    final_top_k: int = 10
    half_life: int = 1500

    # Persistence settings
    force_reindex: bool = False

    # Cache settings
    enable_cache: bool = True
    cache_dir: str = "rag_service/langchain_rag/query_cache"
    cache_ttl_seconds: int = 3600  # 1 hour default TTL
    cache_max_memory_entries: int = 1000  # Max entries in memory cache
    cache_max_disk_size_mb: int = 500  # Max disk cache size in MB
    cache_auto_cleanup_interval: int = 100  # Cleanup every N operations

    # Telemetry settings
    enable_telemetry: bool = True
    telemetry_log_file: str = "logs/monitoring_metrics.json"

    def validate(self) -> bool:
        """Validate configuration"""
        import warnings
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        # Only validate data directories for enabled retriever types
        # Note: Directories are optional - if they don't exist, the loaders will return empty lists
        # This allows using ChromaDB as the source without requiring file-based data directories
        if "mail" in self.base_retriever_types and not os.path.exists(self.mail_data_dir):
            warnings.warn(f"mail_data_dir does not exist: {self.mail_data_dir}. "
                         f"Email loading from files will be skipped. Using ChromaDB only.")
        if "documentation" in self.base_retriever_types and not os.path.exists(self.doc_data_dir):
            warnings.warn(f"doc_data_dir does not exist: {self.doc_data_dir}. "
                         f"Documentation loading from files will be skipped. Using ChromaDB only.")
        if self.base_retriever_types is None or len(self.base_retriever_types) == 0:
            raise ValueError("base_retriever_types must be a non-empty list")
        valid_types = ["mail", "documentation", "git", "slack"]
        for retriever_type in self.base_retriever_types:
            if retriever_type not in valid_types:
                raise ValueError(f"Invalid base_retriever_type: {retriever_type}. Must be one of {valid_types}")
        return True


# Default configuration instance
DEFAULT_CONFIG = LangChainConfig()

