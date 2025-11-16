"""
Tests for RAG Service models.
"""

import pytest
from model_bakery import baker

from datetime import datetime, timedelta

from rag_service.models import CommunitySummary, LibraryFAQ, LibrarySummary


@pytest.mark.django_db
def test_community_summary_str(community_summary):
    """Test CommunitySummary string representation."""
    expected = f"Community Summary {community_summary.start_date.date()} to {community_summary.end_date.date()}"
    assert str(community_summary) == expected


@pytest.mark.django_db
def test_community_summary_defaults():
    """Test CommunitySummary default values."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    summary = CommunitySummary.objects.create(
        start_date=start_date,
        end_date=end_date,
        summary_data={},
    )

    assert summary.is_active is True
    assert summary.need_review is True
    assert summary.topics_count == 0
    assert summary.recent_emails_count == 0
    assert summary.generated_at is not None


@pytest.mark.django_db
def test_community_summary_ordering():
    """Test that CommunitySummary objects are ordered by generated_at descending."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    summary1 = baker.make(
        "rag_service.CommunitySummary",
        start_date=start_date,
        end_date=end_date,
        summary_data={},
    )

    # Create second summary after a short delay to ensure different generated_at
    import time
    time.sleep(0.01)

    summary2 = baker.make(
        "rag_service.CommunitySummary",
        start_date=start_date,
        end_date=end_date,
        summary_data={},
    )

    summaries = list(CommunitySummary.objects.all())
    assert summaries[0] == summary2  # Most recent first
    assert summaries[1] == summary1


@pytest.mark.django_db
def test_library_summary_str(library_summary):
    """Test LibrarySummary string representation."""
    expected = f"{library_summary.library.name} - {library_summary.version.name} Summary"
    assert str(library_summary) == expected


@pytest.mark.django_db
def test_library_summary_unique_together(db, library, library_version):
    """Test that LibrarySummary enforces unique_together constraint."""
    baker.make(
        "rag_service.LibrarySummary",
        library=library,
        version=library_version,
        summary_text="First summary",
    )

    # Creating a second summary with same library and version should fail
    with pytest.raises(Exception):  # IntegrityError or similar
        baker.make(
            "rag_service.LibrarySummary",
            library=library,
            version=library_version,
            summary_text="Second summary",
        )


@pytest.mark.django_db
def test_library_summary_defaults(db, library, library_version):
    """Test LibrarySummary default values."""
    summary = LibrarySummary.objects.create(
        library=library,
        version=library_version,
        summary_text="Test summary",
    )

    assert summary.key_features == []
    assert summary.use_cases == []
    assert summary.generated_at is not None
    assert summary.updated_at is not None


@pytest.mark.django_db
def test_library_faq_str(library_faq):
    """Test LibraryFAQ string representation."""
    expected = f"{library_faq.library.name}: {library_faq.question[:50]}..."
    assert str(library_faq) == expected


@pytest.mark.django_db
def test_library_faq_defaults(db, library):
    """Test LibraryFAQ default values."""
    faq = LibraryFAQ.objects.create(
        library=library,
        question="Test question?",
        answer="Test answer.",
    )

    assert faq.is_active is True
    assert faq.generated_at is not None
    assert faq.updated_at is not None


@pytest.mark.django_db
def test_library_faq_ordering(db, library):
    """Test that LibraryFAQ objects are ordered by library and question."""
    faq1 = baker.make(
        "rag_service.LibraryFAQ",
        library=library,
        question="B question?",
        answer="Answer 1",
    )

    faq2 = baker.make(
        "rag_service.LibraryFAQ",
        library=library,
        question="A question?",
        answer="Answer 2",
    )

    faqs = list(LibraryFAQ.objects.filter(library=library))
    # Should be ordered by question, so A comes before B
    assert faqs[0].question == "A question?"
    assert faqs[1].question == "B question?"

