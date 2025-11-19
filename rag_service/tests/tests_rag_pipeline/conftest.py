"""
Shared fixtures for RAG pipeline tests.
"""

from datetime import datetime

import pytest
from langchain_core.documents import Document

from config.rag_config import LangChainConfig


@pytest.fixture
def mock_pipeline_config():
    """Create a minimal test configuration."""
    config = LangChainConfig()
    config.force_reindex = False
    config.enable_cache = False
    config.enable_telemetry = False
    config.base_retriever_types = ["mail"]
    return config


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Boost.Asio is a cross-platform C++ library for network and low-level I/O programming.",
            metadata={
                "source": "@@MailingList@@test1@example.com",
                "type": "mail",
                "subject": "Boost.Asio Introduction",
                "url": "https://example.com/1",
                "date": int(datetime.now().timestamp()),
            },
        ),
        Document(
            page_content="Boost.Beast is a C++ header-only library serving as a foundation for writing interoperable networking libraries.",
            metadata={
                "source": "@@MailingList@@test2@example.com",
                "type": "mail",
                "subject": "Boost.Beast Overview",
                "url": "https://example.com/2",
                "date": int(datetime.now().timestamp()),
            },
        ),
    ]
