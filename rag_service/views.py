"""
Django views for RAG Service.
"""

from dateutil.parser import parse as parse_date
from django.views.generic import TemplateView

from .models import CommunitySummary


class CommunitySummaryView(TemplateView):
    """Display community summary on community page."""

    template_name = "community.html"

    def get_context_data(self, **kwargs):
        """
        Get context data for the community summary template.

        Retrieves the latest reviewed community summary and processes it for display:
        - Normalizes reference URLs (converts "reference url" to "reference_url")
        - Assigns continuous reference numbers across all topics
        - Converts archive API URLs to message URLs
        - Adds count fields for assertions and chronological summaries
        - Parses date ranges for display

        Args:
            **kwargs: Additional keyword arguments passed to the view

        Returns:
            dict: Context dictionary containing:
                - community_summary: The latest reviewed CommunitySummary instance
                - summary_by_topic: Normalized topic data with reference numbers
                - overall_stats: Statistics including date range and counts
                - url_reference_map: Mapping of URLs to reference numbers
        """
        context = super().get_context_data(**kwargs)

        # Get the latest community summary that has been reviewed (need_review=False)
        summary = (
            CommunitySummary.objects.filter(need_review=False)
            .order_by('-generated_at')
            .first()
        )

        if summary:
            context["community_summary"] = summary
            summary_by_topic = summary.summary_data.get("summary_by_topic", [])

            # Collect all unique URLs and create reference number mapping
            all_urls = self._collect_all_urls(summary_by_topic)
            url_to_number = self._create_url_to_number_mapping(all_urls)

            # Normalize topics with reference numbers
            normalized_topics = self._normalize_topics(summary_by_topic, url_to_number)

            context["summary_by_topic"] = normalized_topics
            # For debugging/backward compatibility
            context["url_reference_map"] = url_to_number

            # Parse and add overall stats
            overall_stats_data = summary.summary_data.get("overall_stats", {})
            overall_stats = self._parse_overall_stats(overall_stats_data)
            context["overall_stats"] = overall_stats
        else:
            context["community_summary"] = None
            context["summary_by_topic"] = []
            context["overall_stats"] = {}

        return context

    def _collect_all_urls(self, summary_by_topic):
        """
        Collect all unique URLs from assertions and chronological summaries.

        Args:
            summary_by_topic: List of topic dictionaries

        Returns:
            list: List of unique URL strings
        """
        all_urls = []

        for topic in summary_by_topic:
            # Collect URLs from assertions
            if "assertions" in topic:
                for assertion in topic["assertions"]:
                    urls = (
                        assertion.get("reference url") or
                        assertion.get("reference_url", [])
                    )
                    if isinstance(urls, list):
                        for url in urls:
                            if url and url not in all_urls:
                                all_urls.append(url)

            # Collect URLs from chronological_summary
            if "chronological_summary" in topic:
                for entry in topic["chronological_summary"]:
                    urls = entry.get("reference url") or entry.get("reference_url", [])
                    if isinstance(urls, list):
                        for url in urls:
                            if url and url not in all_urls:
                                all_urls.append(url)

        return all_urls

    def _create_url_to_number_mapping(self, all_urls):
        """
        Create a mapping of URLs to continuous reference numbers (1-indexed).

        Args:
            all_urls: List of unique URL strings

        Returns:
            dict: Dictionary mapping URL strings to integer reference numbers
        """
        return {url: idx for idx, url in enumerate(all_urls, start=1)}

    def _normalize_assertions(self, assertions_list, url_to_number):
        """
        Normalize assertions by converting reference URLs and adding reference numbers.

        Args:
            assertions_list: List of assertion dictionaries
            url_to_number: Dictionary mapping URLs to reference numbers

        Returns:
            tuple: (normalized_assertions list, count integer)
        """
        if not isinstance(assertions_list, list):
            assertions_list = []

        normalized_assertions = []
        for assertion in assertions_list:
            normalized_assertion = assertion.copy()

            # Normalize key from "reference url" to "reference_url"
            if "reference url" in normalized_assertion:
                normalized_assertion["reference_url"] = normalized_assertion.pop("reference url")

            # Add reference numbers paired with URLs
            if "reference_url" in normalized_assertion:
                ref_urls = normalized_assertion["reference_url"]
                if isinstance(ref_urls, list):
                    normalized_assertion["reference_urls_with_numbers"] = [
                        {
                            "url": self.email_url_to_message_url(url),
                            "number": url_to_number.get(url, 0)
                        }
                        for url in ref_urls if url
                    ]

            normalized_assertions.append(normalized_assertion)

        return normalized_assertions, len(normalized_assertions)

    def _normalize_chronological_summary(self, chronological_list, url_to_number):
        """
        Normalize chronological summary entries by converting reference URLs and adding reference numbers.

        Args:
            chronological_list: List of chronological entry dictionaries
            url_to_number: Dictionary mapping URLs to reference numbers

        Returns:
            tuple: (normalized_chronological list, count integer)
        """
        if not isinstance(chronological_list, list):
            chronological_list = []

        normalized_chronological = []
        for entry in chronological_list:
            normalized_entry = entry.copy()

            # Normalize key from "reference url" to "reference_url"
            if "reference url" in normalized_entry:
                normalized_entry["reference_url"] = normalized_entry.pop("reference url")

            # Add reference numbers paired with URLs
            if "reference_url" in normalized_entry:
                ref_urls = normalized_entry["reference_url"]
                if isinstance(ref_urls, list):
                    normalized_entry["reference_urls_with_numbers"] = [
                        {
                            "url": self.email_url_to_message_url(url),
                            "number": url_to_number.get(url, 0)
                        }
                        for url in ref_urls if url
                    ]

            normalized_chronological.append(normalized_entry)

        return normalized_chronological, len(normalized_chronological)

    def _normalize_topics(self, summary_by_topic, url_to_number):
        """
        Normalize all topics by processing assertions and chronological summaries.

        Args:
            summary_by_topic: List of topic dictionaries
            url_to_number: Dictionary mapping URLs to reference numbers

        Returns:
            list: List of normalized topic dictionaries with counts
        """
        normalized_topics = []

        for topic in summary_by_topic:
            normalized_topic = topic.copy()

            # Normalize assertions
            if "assertions" in normalized_topic:
                normalized_assertions, count = self._normalize_assertions(
                    normalized_topic["assertions"], url_to_number
                )
                normalized_topic["assertions"] = normalized_assertions
                normalized_topic["assertions_count"] = count
            else:
                normalized_topic["assertions"] = []
                normalized_topic["assertions_count"] = 0

            # Normalize chronological_summary
            if "chronological_summary" in normalized_topic:
                normalized_chronological, count = self._normalize_chronological_summary(
                    normalized_topic["chronological_summary"], url_to_number
                )
                normalized_topic["chronological_summary"] = normalized_chronological
                normalized_topic["chronological_summary_count"] = count
            else:
                normalized_topic["chronological_summary"] = []
                normalized_topic["chronological_summary_count"] = 0

            normalized_topics.append(normalized_topic)

        return normalized_topics

    def _parse_overall_stats(self, overall_stats):
        """
        Parse date strings in overall_stats for template display.

        Args:
            overall_stats: Dictionary containing overall statistics

        Returns:
            dict: Dictionary with parsed date ranges
        """
        overall_stats = overall_stats.copy()

        if "date_range" in overall_stats:
            date_range = overall_stats["date_range"].copy()

            if date_range.get("start"):
                try:
                    date_range["start"] = parse_date(date_range["start"])
                except (ValueError, TypeError):
                    pass

            if date_range.get("end"):
                try:
                    date_range["end"] = parse_date(date_range["end"])
                except (ValueError, TypeError):
                    pass

            overall_stats["date_range"] = date_range

        return overall_stats

    def email_url_to_message_url(self, old_url: str) -> str:
        """
        Convert archive API URLs to message URLs.

        Replaces:
        - "archives/api/list" with "archives/list"
        - "/email/" with "/message/"

        Args:
            old_url: The original URL string

        Returns:
            The converted URL string
        """
        new_url = old_url
        new_url = new_url.replace("archives/api/list", "archives/list")
        new_url = new_url.replace("/email/", "/message/")
        return new_url
