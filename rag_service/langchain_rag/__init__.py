"""
LangChain RAG Pipeline

High-performance RAG pipeline using LangChain with hybrid retrieval.
"""

from .rag_pipeline import LangChainRAGPipeline
from .retrieve import (
    LangChainHybridRetriever,
    MultiBaseRetriever,
    QueryCache,
    PerformanceTelemetry,
    CachedRetriever,
    InstrumentedRetriever,
)
from .preprocessor import (
    BoostDataProcessor,
    MailPreprocessor,
    DocuPreprocessor,
)
from .llm import BaseAgent, OpenAIAgent, HuggingFaceAgent, LLMHelper
from .task import (
    MailDataRetriever,
    TopicExtractor,
    WeeklyCommunitySummaryGenerator,
)

__all__ = [
    # Main pipeline
    "LangChainRAGPipeline",
    # Retrieve module
    "LangChainHybridRetriever",
    "MultiBaseRetriever",
    "QueryCache",
    "PerformanceTelemetry",
    "CachedRetriever",
    "InstrumentedRetriever",
    # Preprocessor module
    "BoostDataProcessor",
    "MailPreprocessor",
    "DocuPreprocessor",
    # LLM module
    "BaseAgent",
    "OpenAIAgent",
    "HuggingFaceAgent",
    "LLMHelper",
    # Task module
    "MailDataRetriever",
    "TopicExtractor",
    "WeeklyCommunitySummaryGenerator",
]
