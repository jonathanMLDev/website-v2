"""
Tests for RAG Service views.
"""

import pytest
from model_bakery import baker

from datetime import datetime, timedelta


@pytest.mark.django_db
def test_community_summary_view_with_summary(tp, community_summary):
    """Test CommunitySummaryView with an active summary."""
    response = tp.assertGoodView("community")

    assert response.status_code == 200
    assert "community_summary" in response.context
    assert response.context["community_summary"] == community_summary
    assert "summary_by_topic" in response.context
    assert "overall_stats" in response.context

    # Check that reference URLs are normalized
    summary_by_topic = response.context["summary_by_topic"]
    if summary_by_topic:
        topic = summary_by_topic[0]
        if "assertions" in topic:
            for assertion in topic["assertions"]:
                # Should have "reference_url" not "reference url"
                assert "reference_url" in assertion or "reference url" not in assertion
        if "chronological_summary" in topic:
            for entry in topic["chronological_summary"]:
                # Should have "reference_url" not "reference url"
                assert "reference_url" in entry or "reference url" not in entry


@pytest.mark.django_db
def test_community_summary_view_no_summary(tp):
    """Test CommunitySummaryView when no summary exists."""
    response = tp.assertGoodView("community")

    assert response.status_code == 200
    assert response.context["community_summary"] is None
    assert response.context["summary_by_topic"] == []
    assert response.context["overall_stats"] == {}


@pytest.mark.django_db
def test_community_summary_view_excludes_needs_review(
    tp, community_summary_needs_review
):
    """Test that CommunitySummaryView excludes summaries that need review."""
    response = tp.assertGoodView("community")

    assert response.status_code == 200
    # Should not show the summary that needs review
    assert response.context["community_summary"] is None


@pytest.mark.django_db
def test_community_summary_view_uses_latest_reviewed(tp):
    """Test that CommunitySummaryView uses the latest reviewed summary."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    # Create older reviewed summary
    _older_summary = baker.make(
        "rag_service.CommunitySummary",
        start_date=start_date - timedelta(days=7),
        end_date=end_date - timedelta(days=7),
        summary_data={"summary_by_topic": []},
        need_review=False,
    )

    # Create newer reviewed summary
    newer_summary = baker.make(
        "rag_service.CommunitySummary",
        start_date=start_date,
        end_date=end_date,
        summary_data={"summary_by_topic": []},
        need_review=False,
    )

    response = tp.assertGoodView("community")

    assert response.status_code == 200
    # Should use the newer summary
    assert response.context["community_summary"] == newer_summary


@pytest.mark.django_db
def test_community_summary_view_date_parsing(tp):
    """Test that date strings in overall_stats are parsed correctly."""
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
    }

    baker.make(
        "rag_service.CommunitySummary",
        start_date=start_date,
        end_date=end_date,
        summary_data=summary_data,
        need_review=False,
    )

    response = tp.assertGoodView("community")

    assert response.status_code == 200
    date_range = response.context["overall_stats"]["date_range"]
    # Dates should be parsed to datetime objects
    assert isinstance(date_range["start"], datetime)
    assert isinstance(date_range["end"], datetime)
