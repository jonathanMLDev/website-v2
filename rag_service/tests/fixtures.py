"""
Pytest fixtures for RAG Service tests.
"""

from datetime import datetime, timedelta

import pytest
from model_bakery import baker

from rag_service.models import CommunitySummary, LibraryFAQ, LibrarySummary


@pytest.fixture
def community_summary(db):
    """Create a test community summary."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    summary_data = {
        "summary_by_topic": [
            {
                "subject": "Test Topic: Boost Library Updates",
                "assertions": [
                    {
                        "content": "needs for this library",
                        "reference url": ["https://example.com/url1", "https://example.com/url2"],
                    },
                    {
                        "content": "relation with boost.asio",
                        "reference url": ["https://example.com/url3"],
                    },
                ],
                "chronological_summary": [
                    {
                        "Date": "2018-09-10",
                        "summary": "importance of this library",
                        "reference url": ["https://example.com/url4"],
                    },
                    {
                        "Date": "2021-02-20",
                        "summary": "advanced properties of this library",
                        "reference url": ["https://example.com/url5"],
                    },
                ],
            },
        ],
        "overall_stats": {
            "topics_count": 1,
            "recent_emails": 15,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
        },
        "ai_generated": True,
    }

    return baker.make(
        "rag_service.CommunitySummary",
        start_date=start_date,
        end_date=end_date,
        summary_data=summary_data,
        topics_count=1,
        recent_emails_count=15,
        is_active=True,
        need_review=False,
    )


@pytest.fixture
def community_summary_needs_review(db):
    """Create a test community summary that needs review."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    summary_data = {
        "summary_by_topic": [],
        "overall_stats": {
            "topics_count": 0,
            "recent_emails": 0,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
        },
        "ai_generated": True,
    }

    return baker.make(
        "rag_service.CommunitySummary",
        start_date=start_date,
        end_date=end_date,
        summary_data=summary_data,
        topics_count=0,
        recent_emails_count=0,
        is_active=True,
        need_review=True,
    )


@pytest.fixture
def library_summary(db, library, library_version):
    """Create a test library summary."""
    return baker.make(
        "rag_service.LibrarySummary",
        library=library,
        version=library_version,
        summary_text="This is a test library summary.",
        key_features=["Feature 1", "Feature 2"],
        use_cases=["Use case 1", "Use case 2"],
    )


@pytest.fixture
def library_faq(db, library):
    """Create a test library FAQ."""
    return baker.make(
        "rag_service.LibraryFAQ",
        library=library,
        question="What is this library?",
        answer="This is a test library for Boost.",
        is_active=True,
    )

