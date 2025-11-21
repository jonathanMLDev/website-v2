"""
Celery tasks for RAG service.

These tasks handle:
- Weekly community summary generation
- Syncing new mails to vector database
- Updating summary data
- Uploading vector data to S3
"""

import copy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from celery import shared_task
from dateutil.parser import parse

from .langchain_rag.task.community_task import WeeklyCommunitySummaryGenerator
from .langchain_rag.task.mail_data_retriever import MailDataRetriever
from .models import CommunitySummary
from .s3_utils import upload_vector_data_to_s3 as _upload_vector_data_to_s3
from .services import RAGService

logger = structlog.get_logger(__name__)


@shared_task
def generate_weekly_community_summary():
    """Generate weekly community summary."""
    start_date = datetime.now() - timedelta(days=54)
    generator = WeeklyCommunitySummaryGenerator(
        source="chromadb",
        limit=200,
        max_topics=5,
        fetch_k=30,
        start_date=start_date,
        topic_extraction_method="thread",
    )
    return generator.generate()


def _calculate_sync_start_date(pipeline) -> datetime:
    """Calculate the start date for syncing emails."""
    latest_date = _get_latest_email_date_from_chromadb(pipeline)
    if latest_date:
        logger.info("Found latest email in ChromaDB", latest_date=latest_date)
        return latest_date
    else:
        # If no emails in ChromaDB, get emails from last 30 minutes
        start_date = datetime.now() - timedelta(minutes=30)
        logger.info("No emails found in ChromaDB, syncing from last 30 minutes")
        return start_date


def _convert_emails_to_mail_data(emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert email dictionaries to format expected by add_mail_data."""
    mail_data = []
    for email in emails:
        # Ensure date is datetime object
        date_value = email.get("date")
        if isinstance(date_value, str):
            try:
                date_value = parse(date_value)
            except (ValueError, TypeError):
                date_value = datetime.now()

        mail_data.append(
            {
                "message_id": email.get("message_id", ""),
                "subject": email.get("subject", ""),
                "content": email.get("content", ""),
                "thread_url": email.get("url", ""),
                "sender_address": email.get("sender_address", ""),
                "from_field": email.get("sender_name", ""),
                "date": date_value or datetime.now(),
                "to": email.get("to", ""),
                "cc": email.get("cc", ""),
                "reply_to": email.get("reply_to", ""),
                "url": email.get("url", ""),
            }
        )
    return mail_data


@shared_task
def sync_new_mails_to_vector_db():
    """
    Sync new mails from mailing list (HyperKitty) to vector database (ChromaDB).

    This task should run every 30 minutes (configurable).
    """
    logger.info("Starting sync of new mails to vector database")

    try:
        pipeline = RAGService.get_pipeline()
        start_date = _calculate_sync_start_date(pipeline)

        # Get new emails from HyperKitty
        hyperkitty_retriever = MailDataRetriever(source="hyperkitty")
        new_emails = hyperkitty_retriever.get_mails(
            start_date=start_date,
            end_date=datetime.now(),
            limit=1000,  # Reasonable limit per sync
        )

        if not new_emails:
            logger.info("No new emails to sync")
            return {
                "status": "success",
                "added_count": 0,
                "updated_count": 0,
                "failed_count": 0,
                "message": "No new emails found",
            }

        logger.info("Found new emails to sync", count=len(new_emails))
        mail_data = _convert_emails_to_mail_data(new_emails)
        result = pipeline.add_mail_data(mail_data)

        logger.info(
            "Mail sync complete",
            added=result["added_count"],
            updated=result["updated_count"],
            failed=result["failed_count"],
        )

        return {
            "status": "success",
            **result,
        }

    except Exception as e:
        logger.exception("Error syncing new mails to vector database", error=str(e))
        return {
            "status": "error",
            "error": str(e),
            "added_count": 0,
            "updated_count": 0,
            "failed_count": 0,
        }


def _parse_date_from_metadata(date_value: Any) -> Optional[int]:
    """Parse date value from metadata to timestamp."""
    if isinstance(date_value, int):
        return date_value
    elif isinstance(date_value, str):
        try:
            dt = parse(date_value)
            return int(dt.timestamp())
        except (ValueError, TypeError):
            return None
    return None


def _get_latest_email_date_from_chromadb(pipeline):
    """
    Get the latest email date from ChromaDB.

    Args:
        pipeline: RAG pipeline instance

    Returns:
        datetime object of latest email, or None if no emails found
    """
    try:
        mail_retriever = pipeline.base_retrievers.get("mail")
        if not mail_retriever or not mail_retriever.vector_store:
            return None

        # Get all mail documents with date metadata
        results = mail_retriever.vector_store._collection.get(
            where={"type": "mail"},
            include=["metadatas"],
            limit=10000,  # Get enough to find latest
        )

        if not results or not results.get("metadatas"):
            return None

        # Find the latest date
        latest_timestamp = 0
        for metadata in results["metadatas"]:
            date_value = metadata.get("date")
            if date_value:
                timestamp = _parse_date_from_metadata(date_value)
                if timestamp:
                    latest_timestamp = max(latest_timestamp, timestamp)

        if latest_timestamp > 0:
            return datetime.fromtimestamp(latest_timestamp)

        return None

    except Exception as e:
        logger.warning("Error getting latest email date from ChromaDB", error=str(e))
        return None


def _parse_date_from_summary(date_value: Any, default: datetime) -> datetime:
    """Parse date from summary data."""
    if not date_value:
        return default
    if isinstance(date_value, str):
        return parse(date_value)
    return date_value


@shared_task
def update_summary_data():
    """
    Update summary data (community summaries, library summaries, etc.).

    This task should run once per day.
    """
    logger.info("Starting summary data update")

    try:
        summary_data = generate_weekly_community_summary()

        if not isinstance(summary_data, dict):
            logger.error(
                "Invalid summary data type received", data_type=type(summary_data)
            )
            return {
                "status": "error",
                "error": "Invalid summary data format",
            }

        if (
            "summary_by_topic" not in summary_data
            or "overall_stats" not in summary_data
        ):
            logger.error("Summary data missing required fields", data=summary_data)
            return {
                "status": "error",
                "error": "Summary data missing required fields",
            }

        summary_by_topic = {
            "summary_by_topic": summary_data.get("summary_by_topic", []),
        }
        topic_count = len(summary_by_topic["summary_by_topic"])
        overall_stats = summary_data.get("overall_stats", {})
        date_range = overall_stats.get("date_range", {})
        model_info = summary_data.get("ai_model_info", {})

        start_date = _parse_date_from_summary(
            date_range.get("start"), datetime.now() - timedelta(days=7)
        )
        end_date = _parse_date_from_summary(date_range.get("end"), datetime.now())

        original_data = copy.deepcopy(summary_by_topic)
        published_data = copy.deepcopy(summary_by_topic)

        # Create new summary (need_review=True by default, will be set to False by Wagtail after review)
        community_summary = CommunitySummary.objects.create(
            start_date=start_date,
            end_date=end_date,
            original_summary_data=original_data,
            summary_data=published_data,
            topics_count=topic_count,
            recent_emails_count=overall_stats.get("recent_emails", 0),
            model_info=model_info,
        )

        logger.info(
            "Community summary saved",
            summary_id=community_summary.id,
            topics_count=community_summary.topics_count,
            recent_emails_count=community_summary.recent_emails_count,
        )

        logger.info("Summary data update complete")

        return {
            "status": "success",
            "community_summary": {
                "id": community_summary.id,
                "topics_count": community_summary.topics_count,
                "recent_emails": community_summary.recent_emails_count,
            },
            "updated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception("Error updating summary data", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


def generate_library_summary(library_name: str):
    """
    Generate AI summary for a specific library.

    Args:
        library_name: Name of the library to summarize

    Returns:
        Dictionary with summary data
    """
    # TODO: Implement library summary generation
    # See: toAddItems/2_LibrarySummaries.md
    pass


def generate_library_faqs(library_name: str):
    """
    Generate FAQs for a specific library.

    Args:
        library_name: Name of the library to generate FAQs for

    Returns:
        Dictionary with FAQ data
    """
    # TODO: Implement FAQ generation
    # See: toAddItems/3_FAQs.md
    pass


@shared_task
def upload_vector_data_to_s3():
    """
    Upload vector data (ChromaDB) to S3.

    This task should run once per day.
    """
    logger.info("Starting upload of vector data to S3")

    try:
        # Get ChromaDB path from pipeline config
        pipeline = RAGService.get_pipeline()
        chroma_db_path = pipeline.config.chroma_persist_dir

        # Upload to S3
        result = _upload_vector_data_to_s3(
            chroma_db_path=chroma_db_path,
            s3_key_prefix="rag/vector_data",
        )

        logger.info(
            "Successfully uploaded vector data to S3", s3_key=result.get("s3_key")
        )

        return {
            "status": "success",
            **result,
        }

    except Exception as e:
        logger.exception("Error uploading vector data to S3", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


def validate_and_modify_metadata():
    """Validate and modify metadata for emails in ChromaDB."""
    try:
        from .langchain_rag.task.metadata_validate_modify import (
            MetadataValidateModify,
        )

        validator = MetadataValidateModify(
            update_properties=["parent"],
            need_source_data=True,
            st_time=int(datetime(2025, 9, 20, 0, 0, 0).timestamp()),
        )
        validator.run()
        return True
    except Exception as e:
        logger.exception("Error validating and modifying metadata", error=str(e))
        return False
