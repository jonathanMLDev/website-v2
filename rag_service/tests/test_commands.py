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
@patch("rag_service.tasks.generate_weekly_community_summary")
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
    assert summary.need_review is False  # Test summaries don't need review
    # Check that summary_data contains summary_by_topic
    assert "summary_by_topic" in summary.summary_data
    assert len(summary.summary_data["summary_by_topic"]) == 2
    # Check that original_summary_data is also set
    assert "summary_by_topic" in summary.original_summary_data
    assert len(summary.original_summary_data["summary_by_topic"]) == 2


@pytest.mark.django_db
@patch("rag_service.tasks.generate_weekly_community_summary")
def test_generate_community_summary_command_normal_mode(mock_generate_summary):
    """Test generate_community_summary command in normal mode."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    mock_summary_data = {
        "summary_by_topic": [
            {
                "subject": "Test Topic 1",
                "assertions": [],
                "chronological_summary": [],
            }
        ],
        "overall_stats": {
            "recent_emails": 25,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
        },
        "ai_model_info": {
            "model_type": "test-type",
            "model_name": "test-model",
            "temperature": 0.2,
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
    assert summary.topics_count == 1
    assert summary.recent_emails_count == 25
    assert summary.need_review is True  # Default for new summaries
    # Check that summary_data and original_summary_data are set correctly
    assert "summary_by_topic" in summary.summary_data
    assert len(summary.summary_data["summary_by_topic"]) == 1
    assert "summary_by_topic" in summary.original_summary_data
    assert len(summary.original_summary_data["summary_by_topic"]) == 1
    # Check model_info is stored
    assert summary.model_info == mock_summary_data["ai_model_info"]


@pytest.mark.django_db
@patch("rag_service.tasks.generate_weekly_community_summary")
def test_generate_community_summary_command_error_handling(mock_generate_summary):
    """Test generate_community_summary command error handling."""
    mock_generate_summary.return_value = None

    out = StringIO()
    err = StringIO()

    with pytest.raises(CommandError, match="Invalid summary data format"):
        call_command("generate_community_summary", stdout=out, stderr=err)


@pytest.mark.django_db
@patch("rag_service.tasks.generate_weekly_community_summary")
def test_generate_community_summary_command_with_error_in_result(mock_generate_summary):
    """Test generate_community_summary command when result contains error."""
    mock_generate_summary.return_value = {"error": "Generation failed"}

    out = StringIO()
    err = StringIO()

    with pytest.raises(CommandError, match="Summary generation failed"):
        call_command("generate_community_summary", stdout=out, stderr=err)


@pytest.mark.django_db
@patch("rag_service.tasks.generate_weekly_community_summary")
def test_generate_community_summary_command_missing_summary_by_topic(
    mock_generate_summary,
):
    """Test generate_community_summary command when summary_by_topic is missing (uses default empty list)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    mock_generate_summary.return_value = {
        "overall_stats": {
            "recent_emails": 25,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
        }
    }

    out = StringIO()

    # Command should handle missing summary_by_topic by using empty list
    call_command("generate_community_summary", stdout=out)

    assert CommunitySummary.objects.count() == 1
    summary = CommunitySummary.objects.first()
    assert summary.topics_count == 0  # Empty list means 0 topics
    assert summary.recent_emails_count == 25
    assert "summary_by_topic" in summary.summary_data
    assert summary.summary_data["summary_by_topic"] == []


@pytest.mark.django_db
@patch("rag_service.tasks.generate_weekly_community_summary")
def test_generate_community_summary_command_missing_overall_stats(
    mock_generate_summary,
):
    """Test generate_community_summary command when overall_stats is missing (uses default empty dict)."""
    mock_generate_summary.return_value = {
        "summary_by_topic": [
            {
                "subject": "Test Topic",
                "assertions": [],
                "chronological_summary": [],
            }
        ]
    }

    out = StringIO()

    # Command should handle missing overall_stats by using empty dict and default dates
    call_command("generate_community_summary", stdout=out)

    assert CommunitySummary.objects.count() == 1
    summary = CommunitySummary.objects.first()
    assert summary.topics_count == 1
    assert summary.recent_emails_count == 0  # Default when overall_stats is missing
