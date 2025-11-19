"""
Tests for RAG Service management commands.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from io import StringIO

from rag_service.models import CommunitySummary


@pytest.mark.django_db
@patch(
    "rag_service.management.commands.generate_community_summary.generate_weekly_community_summary"
)
def test_generate_community_summary_command_test_mode(mock_generate_summary):
    """Test generate_community_summary command with --test flag."""
    out = StringIO()

    call_command("generate_community_summary", "--test", stdout=out)

    # Should not call the actual generation function
    mock_generate_summary.assert_not_called()

    # Should create a test summary
    assert CommunitySummary.objects.count() == 1
    summary = CommunitySummary.objects.first()
    assert summary.topics_count == 2
    assert "test" in summary.summary_data.get("warning", "").lower()


@pytest.mark.django_db
@patch(
    "rag_service.management.commands.generate_community_summary.generate_weekly_community_summary"
)
def test_generate_community_summary_command_normal_mode(mock_generate_summary):
    """Test generate_community_summary command in normal mode."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    mock_summary_data = {
        "summary_by_topic": [],
        "overall_stats": {
            "topics_count": 3,
            "recent_emails": 25,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
        },
        "ai_generated": True,
    }
    mock_generate_summary.return_value = mock_summary_data

    out = StringIO()

    call_command("generate_community_summary", stdout=out)

    # Should call the generation function
    mock_generate_summary.assert_called_once()

    # Should create a summary
    assert CommunitySummary.objects.count() == 1
    summary = CommunitySummary.objects.first()
    assert summary.topics_count == 3
    assert summary.recent_emails_count == 25
    assert summary.need_review is True  # Default for new summaries


@pytest.mark.django_db
@patch(
    "rag_service.management.commands.generate_community_summary.generate_weekly_community_summary"
)
def test_generate_community_summary_command_error_handling(mock_generate_summary):
    """Test generate_community_summary command error handling."""
    mock_generate_summary.return_value = None

    out = StringIO()
    err = StringIO()

    with pytest.raises(CommandError, match="Failed to generate summary data"):
        call_command("generate_community_summary", stdout=out, stderr=err)


@pytest.mark.django_db
@patch(
    "rag_service.management.commands.generate_community_summary.generate_weekly_community_summary"
)
def test_generate_community_summary_command_with_error_in_result(mock_generate_summary):
    """Test generate_community_summary command when result contains error."""
    mock_generate_summary.return_value = {"error": "Generation failed"}

    out = StringIO()
    err = StringIO()

    with pytest.raises(CommandError, match="Summary generation failed"):
        call_command("generate_community_summary", stdout=out, stderr=err)
