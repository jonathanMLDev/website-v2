"""
Django models for RAG Service

Models for storing RAG-generated summaries and metadata.
"""

from django.db import models
from django.utils import timezone


class CommunitySummary(models.Model):
    """Stores AI-generated weekly community summaries"""

    start_date = models.DateTimeField(help_text="Start date of the summary period")
    end_date = models.DateTimeField(help_text="End date of the summary period")
    summary_data = models.JSONField(help_text="Summary data by topic")
    topics_count = models.IntegerField(default=0)
    recent_emails_count = models.IntegerField(default=0)
    generated_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True, help_text="Whether this summary is currently active")
    need_review = models.BooleanField(default=True, help_text="Whether this summary needs review before being displayed")

    class Meta:
        verbose_name = "Community Summary"
        verbose_name_plural = "Community Summaries"
        ordering = ["-generated_at"]
        indexes = [
            models.Index(fields=["-generated_at", "is_active"]),
            models.Index(fields=["-generated_at", "need_review"]),
        ]

    def __str__(self):
        return f"Community Summary {self.start_date.date()} to {self.end_date.date()}"


class LibrarySummary(models.Model):
    """Stores AI-generated summaries for Boost libraries"""

    library = models.ForeignKey(
        "libraries.Library",
        on_delete=models.CASCADE,
        related_name="rag_summaries"
    )
    version = models.ForeignKey(
        "versions.Version",
        on_delete=models.CASCADE,
        related_name="library_rag_summaries"
    )
    summary_text = models.TextField(help_text="AI-generated summary of the library")
    key_features = models.JSONField(default=list, help_text="List of key features")
    use_cases = models.JSONField(default=list, help_text="List of use cases")
    generated_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Library Summary"
        verbose_name_plural = "Library Summaries"
        unique_together = [["library", "version"]]
        ordering = ["-generated_at"]

    def __str__(self):
        return f"{self.library.name} - {self.version.name} Summary"


class LibraryFAQ(models.Model):
    """Stores AI-generated FAQs for Boost libraries"""

    library = models.ForeignKey(
        "libraries.Library",
        on_delete=models.CASCADE,
        related_name="rag_faqs"
    )
    question = models.TextField()
    answer = models.TextField()
    generated_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = "Library FAQ"
        verbose_name_plural = "Library FAQs"
        ordering = ["library", "question"]
        indexes = [
            models.Index(fields=["library", "is_active"]),
        ]

    def __str__(self):
        return f"{self.library.name}: {self.question[:50]}..."

