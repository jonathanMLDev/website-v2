"""
Community Task Module

Handles community-related tasks like weekly community summary generation.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from .mail_data_retriever import MailDataRetriever
from .topic_extractor import TopicExtractor

logger = structlog.get_logger(__name__)


class WeeklyCommunitySummaryGenerator:
    """
    Generate weekly community summary using RAG by topic

    Strategy:
    1. Query recent discussions from mail data source (last 7 days)
    2. Extract main topics from recent discussions using one of:
       - Thread-based grouping (preserves conversation context)
       - Clustering (semantic similarity using embeddings)
       - LLM direct extraction
    3. For each topic, use RAG to retrieve relevant past discussions
    4. Summarize each topic chronologically with retrieved context
    """

    def __init__(
        self,
        source: str = "auto",
        limit: int = 200,
        max_topics: int = 10,
        fetch_k: int = 30,
        topic_extraction_method: str = "thread",
        start_date: Optional[datetime] = None,
    ):
        """
        Initialize weekly community summary generator

        Args:
            source: Data source to use ("auto", "hyperkitty", "chromadb")
            limit: Maximum number of recent emails to retrieve
            max_topics: Maximum number of topics to extract
            fetch_k: Number of relevant discussions to fetch per topic
            topic_extraction_method: Method for topic extraction. Options:
                - "thread": Group by email threads (preserves conversation context) [default]
                - "clustering": Use KMeans clustering on embeddings (semantic similarity)
                - "llm": Use LLM to extract topics directly
            start_date: Start date for email retrieval (defaults to 2025-09-20)
        """
        self.logger = logger.bind(component="WeeklyCommunitySummaryGenerator")
        self.source = source
        self.limit = limit
        self.max_topics = max_topics
        self.fetch_k = fetch_k

        # Validate topic extraction method
        valid_methods = ["thread", "clustering", "llm"]
        if topic_extraction_method not in valid_methods:
            raise ValueError(
                f"Invalid topic_extraction_method: {topic_extraction_method}. "
                f"Must be one of {valid_methods}"
            )
        self.topic_extraction_method = topic_extraction_method
        self.start_date = start_date or datetime(2025, 9, 20, 0, 0, 0)

        # Initialize components
        self.mail_retriever = MailDataRetriever(source=source)
        self.topic_extractor = TopicExtractor()

        self.logger.info("WeeklyCommunitySummaryGenerator initialized")

    def _get_recent_emails(self, date_start: datetime, date_end: datetime) -> Any:
        """Retrieve recent emails for topic identification."""
        recent_emails = self.mail_retriever.get_mails(
            start_date=date_start,
            end_date=date_end,
            limit=self.limit,
            embedding_conclusion=True,
        )
        if not recent_emails:
            self.logger.warning("No recent emails found for summary generation")
        return recent_emails

    def _extract_topics(self, recent_emails: Any) -> List[Any]:
        """Extract main topics from recent discussions."""
        self.logger.info(
            f"Using topic extraction method: {self.topic_extraction_method}"
        )

        if self.topic_extraction_method == "thread":
            topics = self.topic_extractor.extract_topics_by_thread(
                recent_emails, max_topics=self.max_topics
            )
        elif self.topic_extraction_method == "clustering":
            topics = self.topic_extractor.extract_topics_by_clustering(
                recent_emails, max_topics=self.max_topics
            )
        elif self.topic_extraction_method == "llm":
            topics = self.topic_extractor.extract_topics_by_llm(
                recent_emails, max_topics=self.max_topics
            )
        else:
            raise ValueError(
                f"Unknown topic extraction method: {self.topic_extraction_method}"
            )

        if topics:
            self.logger.info(f"Extracted {len(topics)} topics")
        else:
            self.logger.warning("No topics extracted from recent discussions")
        return topics

    def _process_topic(self, topic: Any, end_date: datetime) -> None:
        """Process a single topic and generate its chronological summary."""
        topic_str = topic if isinstance(topic, str) else topic.get("subject", "")
        if topic_str is None:
            return

        self.logger.info(f"Processing topic: {topic_str}")

        relevant_emails = self.mail_retriever.retrieve_relevant_emails(
            question=topic_str,
            fetch_k=self.fetch_k,
            end_date=end_date,
        )

        if not relevant_emails:
            self.logger.warning(f"No discussions found for topic: {topic_str}")
            return

        topic_summary = self.topic_extractor.llm_helper.process_pipeline(
            "chronological_summary", relevant_emails, topic_str
        )
        if isinstance(topic_summary, list):
            topic["chronological_summary"] = topic_summary
        elif isinstance(topic_summary, Dict):
            topic["chronological_summary"] = topic_summary.values()
        else:
            topic["chronological_summary"] = []

    def _build_result(
        self,
        topics: List[Any],
        recent_emails: Any,
        date_start: datetime,
        date_end: datetime,
    ) -> Dict[str, Any]:
        """Build the final result dictionary."""
        email_count = len(recent_emails.get("documents", recent_emails.get("ids", [])))
        return {
            "summary_by_topic": topics,
            "overall_stats": {
                "recent_emails": email_count,
                "date_range": {
                    "start": (
                        date_start.isoformat()
                        if isinstance(date_start, datetime)
                        else str(date_start)
                    ),
                    "end": (
                        date_end.isoformat()
                        if isinstance(date_end, datetime)
                        else str(date_end)
                    ),
                },
            },
            "ai_model_info": self._get_ai_model_info(),
            "message": "This summary is AI-generated from recent mailing list discussions",
        }

    def generate(self) -> Dict[str, Any]:
        """
        Generate weekly community summary

        Returns:
            Dictionary with summary data by topic
        """
        self.logger.info("Starting weekly community summary generation")

        try:
            date_end = datetime.now()
            date_start = self.start_date

            # 1. Get recent emails for topic identification
            recent_emails = self._get_recent_emails(date_start, date_end)
            if not recent_emails:
                return self._build_empty_result(
                    date_start, date_end, "No recent emails found"
                )

            # 2. Extract main topics from recent discussions
            topics = self._extract_topics(recent_emails)
            email_count = len(
                recent_emails.get("documents", recent_emails.get("ids", []))
            )
            if len(topics) == 0:
                return self._build_empty_result(
                    date_start, date_end, "No topics extracted", email_count
                )

            # 3. Retrieve all mails related to each topic and generate chronological summary
            for topic in topics:
                self._process_topic(topic, date_start + timedelta(days=-1))

            # 4. Build final result
            result = self._build_result(topics, recent_emails, date_start, date_end)
            self.logger.info(f"Summary generation completed: {len(topics)} topics")
            return result

        except Exception as e:
            self.logger.exception(f"Error generating weekly community summary: {e}")
            return {
                "summary_by_topic": [],
                "overall_stats": {},
                "ai_model_info": self._get_ai_model_info(),
                "message": f"Summary generation failed{e}",
            }

    def _build_empty_result(
        self, date_start, date_end, message: str, recent_emails_count: int = 0
    ) -> Dict[str, Any]:
        """Build empty result structure"""
        # Handle date_start and date_end - they might be datetime objects or timestamps
        if isinstance(date_start, datetime):
            start_str = date_start.isoformat()
        else:
            start_str = str(date_start)

        if isinstance(date_end, datetime):
            end_str = date_end.isoformat()
        else:
            end_str = str(date_end)

        return {
            "summary_by_topic": [],
            "overall_stats": {
                "recent_emails": recent_emails_count,
                "date_range": {
                    "start": start_str,
                    "end": end_str,
                },
            },
            "ai_model_info": self._get_ai_model_info(),
            "message": message,
        }

    def _get_ai_model_info(self) -> Dict[str, str]:
        """Return information about the LLM that generated the summaries."""
        helper = getattr(self.topic_extractor, "llm_helper", None)
        if helper:
            try:
                return helper.get_model_info()
            except Exception:  # pragma: no cover - best-effort logging only
                self.logger.warning("Unable to fetch LLM model info")
        return {"model_type": "unknown", "model_name": "unknown"}


# Backward compatibility function
def generate_weekly_community_summary() -> Dict[str, Any]:
    """
    Generate weekly community summary using RAG by topic

    This is a convenience function that creates a WeeklyCommunitySummaryGenerator
    and calls its generate() method.

    Returns:
        Dictionary with summary data by topic
    """
    generator = WeeklyCommunitySummaryGenerator()
    return generator.generate()
