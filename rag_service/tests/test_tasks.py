"""
Tests for RAG Service Celery tasks.
"""

from unittest.mock import Mock, patch
from datetime import datetime, timedelta

import pytest
from model_bakery import baker

from rag_service.models import CommunitySummary
from rag_service.tasks import (
    generate_weekly_community_summary,
    sync_new_mails_to_vector_db,
    update_summary_data,
)


@pytest.mark.django_db
@patch("rag_service.tasks.WeeklyCommunitySummaryGenerator")
def test_generate_weekly_community_summary(mock_generator_class):
    """Test generate_weekly_community_summary task."""
    # Mock the generator
    mock_generator = Mock()
    mock_generator.generate.return_value = {
        "summary_by_topic": [],
        "overall_stats": {
            "topics_count": 0,
            "recent_emails": 0,
            "date_range": {
                "start": (datetime.now() - timedelta(days=7)).isoformat(),
                "end": datetime.now().isoformat(),
            },
        },
        "ai_generated": True,
    }
    mock_generator_class.return_value = mock_generator

    result = generate_weekly_community_summary()

    # Verify generator was called
    mock_generator_class.assert_called_once()
    mock_generator.generate.assert_called_once()

    # Verify result structure
    assert "overall_stats" in result
    assert "summary_by_topic" in result


@pytest.mark.django_db
@patch("rag_service.tasks.MailDataRetriever")
@patch("rag_service.tasks.RAGService")
def test_sync_new_mails_to_vector_db_no_new_emails(
    mock_rag_service, mock_retriever_class
):
    """Test sync_new_mails_to_vector_db when there are no new emails."""
    # Mock the pipeline
    mock_pipeline = Mock()
    mock_rag_service.get_pipeline.return_value = mock_pipeline

    # Mock mail retriever for ChromaDB
    mock_chromadb_retriever = Mock()
    mock_chromadb_retriever.get_mails.return_value = []

    # Mock mail retriever for HyperKitty
    mock_hyperkitty_retriever = Mock()
    mock_hyperkitty_retriever.get_mails.return_value = []

    # Configure retriever class to return different instances
    mock_retriever_class.side_effect = [
        mock_chromadb_retriever,
        mock_hyperkitty_retriever,
    ]

    result = sync_new_mails_to_vector_db()

    assert result["status"] == "success"
    assert result["added_count"] == 0
    assert result["message"] == "No new emails found"


@pytest.mark.django_db
@patch("rag_service.tasks.MailDataRetriever")
@patch("rag_service.tasks.RAGService")
def test_sync_new_mails_to_vector_db_with_new_emails(
    mock_rag_service, mock_retriever_class
):
    """Test sync_new_mails_to_vector_db when there are new emails."""
    # Mock the pipeline
    mock_pipeline = Mock()
    mock_pipeline.add_mail_data.return_value = {
        "added_count": 2,
        "updated_count": 0,
        "failed_count": 0,
    }
    mock_rag_service.get_pipeline.return_value = mock_pipeline

    # Mock mail retriever for ChromaDB (to get latest date)
    mock_chromadb_retriever = Mock()
    mock_chromadb_retriever.get_mails.return_value = []

    # Mock mail retriever for HyperKitty (to get new emails)
    mock_hyperkitty_retriever = Mock()
    new_emails = [
        {
            "message_id": "test1@example.com",
            "subject": "Test Subject 1",
            "content": "Test content 1",
            "url": "https://example.com/1",
            "sender_address": "sender1@example.com",
            "sender_name": "Sender 1",
            "date": datetime.now(),
            "to": "to@example.com",
            "cc": "",
            "reply_to": "",
        },
        {
            "message_id": "test2@example.com",
            "subject": "Test Subject 2",
            "content": "Test content 2",
            "url": "https://example.com/2",
            "sender_address": "sender2@example.com",
            "sender_name": "Sender 2",
            "date": datetime.now(),
            "to": "to@example.com",
            "cc": "",
            "reply_to": "",
        },
    ]
    mock_hyperkitty_retriever.get_mails.return_value = new_emails

    # Configure retriever class to return different instances
    mock_retriever_class.side_effect = [
        mock_chromadb_retriever,
        mock_hyperkitty_retriever,
    ]

    # Mock _get_latest_email_date_from_chromadb
    with patch(
        "rag_service.tasks._get_latest_email_date_from_chromadb", return_value=None
    ):
        result = sync_new_mails_to_vector_db()

    assert result["status"] == "success"
    assert result["added_count"] == 2
    mock_pipeline.add_mail_data.assert_called_once()


@pytest.mark.django_db
@patch("rag_service.tasks.generate_weekly_community_summary")
def test_update_summary_data(mock_generate_summary):
    """Test update_summary_data task."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    mock_summary_data = {
        "summary_by_topic": [],
        "overall_stats": {
            "topics_count": 2,
            "recent_emails": 20,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
        },
        "ai_generated": True,
    }
    mock_generate_summary.return_value = mock_summary_data

    # Create an existing reviewed summary
    existing_summary = baker.make(
        "rag_service.CommunitySummary",
        start_date=start_date - timedelta(days=14),
        end_date=start_date - timedelta(days=7),
        summary_data={},
        need_review=False,
    )

    result = update_summary_data()

    assert result["status"] == "success"
    assert result["community_summary"]["topics_count"] == 2
    assert result["community_summary"]["recent_emails"] == 20

    # Verify new summary was created (need_review=True by default)
    new_summary = (
        CommunitySummary.objects.filter(need_review=True)
        .order_by("-generated_at")
        .first()
    )
    assert new_summary is not None
    assert new_summary.topics_count == 2
    assert new_summary.recent_emails_count == 20
    # Old summary should still be reviewed (need_review=False)
    existing_summary.refresh_from_db()
    assert existing_summary.need_review is False


@pytest.mark.django_db
@patch("rag_service.tasks.generate_weekly_community_summary")
def test_update_summary_data_error_handling(mock_generate_summary):
    """Test update_summary_data error handling."""
    mock_generate_summary.side_effect = Exception("Test error")

    result = update_summary_data()

    assert result["status"] == "error"
    assert "error" in result
    assert "Test error" in result["error"]


@pytest.mark.django_db
@patch("rag_service.tasks.WeeklyCommunitySummaryGenerator")
def test_generate_weekly_community_summary_network_error(mock_generator_class):
    """Test generate_weekly_community_summary handles network errors."""
    mock_generator = Mock()
    mock_generator.generate.side_effect = ConnectionError("Network error")
    mock_generator_class.return_value = mock_generator

    with pytest.raises(ConnectionError):
        generate_weekly_community_summary()


@pytest.mark.django_db
@patch("rag_service.tasks.MailDataRetriever")
@patch("rag_service.tasks.RAGService")
def test_sync_new_mails_to_vector_db_hyperkitty_connection_error(
    mock_rag_service, mock_retriever_class
):
    """Test sync_new_mails_to_vector_db handles HyperKitty connection errors."""
    # Mock the pipeline
    mock_pipeline = Mock()
    mock_rag_service.get_pipeline.return_value = mock_pipeline

    # Mock mail retriever for ChromaDB
    mock_chromadb_retriever = Mock()
    mock_chromadb_retriever.get_mails.return_value = []

    # Mock mail retriever for HyperKitty - connection error
    mock_hyperkitty_retriever = Mock()
    mock_hyperkitty_retriever.get_mails.side_effect = ConnectionError(
        "Database connection failed"
    )

    # Configure retriever class to return different instances
    mock_retriever_class.side_effect = [
        mock_chromadb_retriever,
        mock_hyperkitty_retriever,
    ]

    result = sync_new_mails_to_vector_db()

    assert result["status"] == "error"
    assert "error" in result
    assert "Database connection failed" in result["error"]


@pytest.mark.django_db
@patch("rag_service.tasks.MailDataRetriever")
@patch("rag_service.tasks.RAGService")
def test_sync_new_mails_to_vector_db_invalid_email_data(
    mock_rag_service, mock_retriever_class
):
    """Test sync_new_mails_to_vector_db handles invalid email data."""
    # Mock the pipeline
    mock_pipeline = Mock()
    mock_pipeline.add_mail_data.return_value = {
        "added_count": 0,
        "updated_count": 0,
        "failed_count": 1,
        "failed_messages": ["Invalid email format"],
    }
    mock_rag_service.get_pipeline.return_value = mock_pipeline

    # Mock mail retriever for ChromaDB
    mock_chromadb_retriever = Mock()
    mock_chromadb_retriever.get_mails.return_value = []

    # Mock mail retriever for HyperKitty - returns invalid data
    mock_hyperkitty_retriever = Mock()
    invalid_emails = [
        {
            # Missing required fields
            "message_id": None,
            "subject": None,
        }
    ]
    mock_hyperkitty_retriever.get_mails.return_value = invalid_emails

    # Configure retriever class to return different instances
    mock_retriever_class.side_effect = [
        mock_chromadb_retriever,
        mock_hyperkitty_retriever,
    ]

    # Mock _get_latest_email_date_from_chromadb
    with patch(
        "rag_service.tasks._get_latest_email_date_from_chromadb", return_value=None
    ):
        result = sync_new_mails_to_vector_db()

    assert result["status"] == "success"
    assert result["added_count"] == 0
    assert result["failed_count"] == 1


@pytest.mark.django_db
@patch("rag_service.tasks.MailDataRetriever")
@patch("rag_service.tasks.RAGService")
def test_sync_new_mails_to_vector_db_pipeline_error(
    mock_rag_service, mock_retriever_class
):
    """Test sync_new_mails_to_vector_db handles pipeline errors."""
    # Mock the pipeline - raises error
    mock_pipeline = Mock()
    mock_pipeline.add_mail_data.side_effect = RuntimeError("Pipeline error")
    mock_rag_service.get_pipeline.return_value = mock_pipeline

    # Mock mail retriever for ChromaDB
    mock_chromadb_retriever = Mock()
    mock_chromadb_retriever.get_mails.return_value = []

    # Mock mail retriever for HyperKitty
    mock_hyperkitty_retriever = Mock()
    mock_hyperkitty_retriever.get_mails.return_value = [
        {
            "message_id": "test@example.com",
            "subject": "Test",
            "content": "Test content",
            "url": "https://example.com/1",
            "sender_address": "sender@example.com",
            "sender_name": "Sender",
            "date": datetime.now(),
            "to": "to@example.com",
            "cc": "",
            "reply_to": "",
        }
    ]

    # Configure retriever class to return different instances
    mock_retriever_class.side_effect = [
        mock_chromadb_retriever,
        mock_hyperkitty_retriever,
    ]

    # Mock _get_latest_email_date_from_chromadb
    with patch(
        "rag_service.tasks._get_latest_email_date_from_chromadb", return_value=None
    ):
        result = sync_new_mails_to_vector_db()

    assert result["status"] == "error"
    assert "error" in result
    assert "Pipeline error" in result["error"]


@pytest.mark.django_db
@patch("rag_service.tasks.MailDataRetriever")
@patch("rag_service.tasks.RAGService")
def test_sync_new_mails_to_vector_db_chromadb_error(
    mock_rag_service, mock_retriever_class
):
    """Test sync_new_mails_to_vector_db handles ChromaDB errors."""
    # Mock the pipeline
    mock_pipeline = Mock()
    mock_rag_service.get_pipeline.return_value = mock_pipeline

    # Mock mail retriever for ChromaDB - raises error
    mock_chromadb_retriever = Mock()
    mock_chromadb_retriever.get_mails.side_effect = RuntimeError("ChromaDB error")

    # Configure retriever class
    mock_retriever_class.return_value = mock_chromadb_retriever

    # Mock _get_latest_email_date_from_chromadb to raise error
    with patch(
        "rag_service.tasks._get_latest_email_date_from_chromadb",
        side_effect=RuntimeError("ChromaDB connection failed"),
    ):
        result = sync_new_mails_to_vector_db()

    # Should still proceed with default start date
    assert result["status"] in ["success", "error"]


@pytest.mark.django_db
@patch("rag_service.tasks.generate_weekly_community_summary")
def test_update_summary_data_invalid_summary_data(mock_generate_summary):
    """Test update_summary_data handles invalid summary data structure."""
    # Return invalid summary data (missing required fields)
    mock_generate_summary.return_value = {
        "invalid": "data",
        # Missing "summary_by_topic" and "overall_stats"
    }

    result = update_summary_data()

    # Should handle gracefully
    assert result["status"] in ["success", "error"]


@pytest.mark.django_db
@patch("rag_service.tasks.generate_weekly_community_summary")
def test_update_summary_data_timeout_error(mock_generate_summary):
    """Test update_summary_data handles timeout errors."""
    mock_generate_summary.side_effect = TimeoutError("Operation timed out")

    result = update_summary_data()

    assert result["status"] == "error"
    assert "error" in result
    assert (
        "timed out" in result["error"].lower() or "timeout" in result["error"].lower()
    )
