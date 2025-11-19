"""
Django admin interface for RAG Service
"""

from django.contrib import admin
from .models import CommunitySummary, LibrarySummary, LibraryFAQ


@admin.register(CommunitySummary)
class CommunitySummaryAdmin(admin.ModelAdmin):
    list_display = [
        "start_date",
        "end_date",
        "topics_count",
        "recent_emails_count",
        "need_review",
        "generated_at",
    ]
    list_filter = ["need_review", "generated_at"]
    search_fields = ["summary_data"]
    readonly_fields = ["generated_at"]
    date_hierarchy = "generated_at"

    @admin.action(description="Mark selected summaries as reviewed")
    def mark_as_reviewed(self, request, queryset):
        """Admin action to mark summaries as reviewed."""
        updated = queryset.update(need_review=False)
        self.message_user(request, f"{updated} summary(ies) marked as reviewed.")

    @admin.action(description="Mark selected summaries as needing review")
    def mark_as_needs_review(self, request, queryset):
        """Admin action to mark summaries as needing review."""
        updated = queryset.update(need_review=True)
        self.message_user(request, f"{updated} summary(ies) marked as needing review.")

    actions = [mark_as_reviewed, mark_as_needs_review]


@admin.register(LibrarySummary)
class LibrarySummaryAdmin(admin.ModelAdmin):
    list_display = ["library", "version", "generated_at", "updated_at"]
    list_filter = ["version", "generated_at"]
    search_fields = ["library__name", "summary_text"]
    readonly_fields = ["generated_at", "updated_at"]
    raw_id_fields = ["library", "version"]


@admin.register(LibraryFAQ)
class LibraryFAQAdmin(admin.ModelAdmin):
    list_display = ["library", "question_short", "is_active", "generated_at"]
    list_filter = ["is_active", "library", "generated_at"]
    search_fields = ["library__name", "question", "answer"]
    readonly_fields = ["generated_at", "updated_at"]
    raw_id_fields = ["library"]

    @admin.display(description="Question")
    def question_short(self, obj):
        return obj.question[:60] + "..." if len(obj.question) > 60 else obj.question
