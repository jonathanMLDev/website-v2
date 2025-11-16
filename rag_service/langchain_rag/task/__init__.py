"""
Task module - Task-related components
"""

from .mail_data_retriever import MailDataRetriever
from .topic_extractor import TopicExtractor
from .community_task import WeeklyCommunitySummaryGenerator

__all__ = [
    "MailDataRetriever",
    "TopicExtractor",
    "WeeklyCommunitySummaryGenerator",
]

