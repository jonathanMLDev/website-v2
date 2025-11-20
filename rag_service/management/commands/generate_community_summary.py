"""
Django management command to generate and save a community summary.

This command generates a weekly community summary and saves it to the database.

Usage:
    python manage.py generate_community_summary
    python manage.py generate_community_summary --test
"""

from copy import deepcopy
from datetime import datetime, timedelta

from dateutil.parser import parse
from django.core.management.base import BaseCommand, CommandError

from rag_service.models import CommunitySummary


class Command(BaseCommand):
    help = "Generate and save a weekly community summary to the database"

    def add_arguments(self, parser):
        parser.add_argument(
            "--test",
            action="store_true",
            default=False,
            help="Create a test/dummy summary for development purposes",
        )

    def handle(self, *args, **options):
        # Ensure test option defaults to False if not explicitly set
        options.setdefault("test", False)

        self.stdout.write(
            self.style.SUCCESS("Starting community summary generation...")
        )

        try:
            # Generate weekly community summary
            if options["test"]:
                self.stdout.write("Creating test summary data...")
                summary_data = self._create_test_summary()
            else:
                self.stdout.write("Generating summary data...")
                # Import here to avoid dependency issues when using --test
                from rag_service.tasks import generate_weekly_community_summary

                summary_data = generate_weekly_community_summary()

            if not isinstance(summary_data, dict):
                raise CommandError("Invalid summary data format (expected dict)")

            if summary_data.get("error"):
                raise CommandError(
                    f"Summary generation failed: {summary_data.get('error')}"
                )

            summary_by_topic = summary_data.get("summary_by_topic", [])
            if not isinstance(summary_by_topic, list):
                raise CommandError("Summary data missing 'summary_by_topic' list")

            overall_stats = summary_data.get("overall_stats", {})
            if not isinstance(overall_stats, dict):
                raise CommandError("Summary data missing 'overall_stats'")

            date_range = overall_stats.get("date_range", {})

            # Parse date strings to datetime objects
            start_date_str = date_range.get("start")
            end_date_str = date_range.get("end")

            if start_date_str:
                if isinstance(start_date_str, str):
                    start_date = parse(start_date_str)
                else:
                    start_date = start_date_str
            else:
                start_date = datetime.now() - timedelta(days=7)

            if end_date_str:
                if isinstance(end_date_str, str):
                    end_date = parse(end_date_str)
                else:
                    end_date = end_date_str
            else:
                end_date = datetime.now()

            original_summary_data = {"summary_by_topic": deepcopy(summary_by_topic)}
            published_summary_data = deepcopy(original_summary_data)
            topics_count = len(summary_by_topic)
            recent_emails = overall_stats.get("recent_emails", 0)
            model_info = summary_data.get("ai_model_info", {})

            # Create new summary
            # For test summaries, set need_review=False so they display immediately
            need_review = not options.get("test", False)
            community_summary = CommunitySummary.objects.create(
                start_date=start_date,
                end_date=end_date,
                original_summary_data=original_summary_data,
                summary_data=published_summary_data,
                topics_count=topics_count,
                recent_emails_count=recent_emails,
                model_info=model_info,
                need_review=need_review,
            )

            self.stdout.write(
                self.style.SUCCESS(
                    f"\nSuccessfully created community summary!\n"
                    f"  ID: {community_summary.id}\n"
                    f"  Date Range: {start_date.date()} to {end_date.date()}\n"
                    f"  Topics: {community_summary.topics_count}\n"
                    f"  Recent Emails: {community_summary.recent_emails_count}\n"
                    f"  Need Review: {community_summary.need_review}"
                )
            )

        except Exception as e:
            raise CommandError(f"Error generating community summary: {e}")

    def _create_test_summary(self):
        """Create a test/dummy summary for development purposes."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        summary_by_topic = [
            {
                "subject": "Test Topic: Boost Library Updates",
                "assertions": [
                    {
                        "content": "needs for this library",
                        "reference url": [
                            "https://example.com/url1",
                            "https://example.com/url2",
                        ],
                    },
                    {
                        "content": "relation with boost.asio",
                        "reference url": [
                            "https://example.com/url3",
                            "https://example.com/url4",
                        ],
                    },
                    {
                        "content": "Pros and cons of this library",
                        "reference url": [
                            "https://example.com/url5",
                            "https://example.com/url6",
                        ],
                    },
                ],
                "chronological_summary": [
                    {
                        "Date": "2018-09-10",
                        "summary": "importance of this library",
                        "reference url": [
                            "https://example.com/url7",
                            "https://example.com/url8",
                        ],
                    },
                    {
                        "Date": "2021-02-20",
                        "summary": "advanced properties of this library",
                        "reference url": [
                            "https://example.com/url9",
                            "https://example.com/url10",
                        ],
                    },
                ],
            },
            {
                "subject": "Test Topic: Community Discussions",
                "assertions": [
                    {
                        "content": "needs for this Community",
                        "reference url": [
                            "https://example.com/url11",
                            "https://example.com/url12",
                        ],
                    },
                    {
                        "content": "Pros and cons of this Community",
                        "reference url": [
                            "https://example.com/url13",
                            "https://example.com/url14",
                        ],
                    },
                ],
                "chronological_summary": [
                    {
                        "Date": "2018-09-10",
                        "summary": "Should update community page",
                        "reference url": [
                            "https://example.com/url15",
                            "https://example.com/url16",
                        ],
                    },
                    {
                        "Date": "2021-02-20",
                        "summary": "advanced properties of community page",
                        "reference url": [
                            "https://example.com/url17",
                            "https://example.com/url18",
                        ],
                    },
                ],
            },
        ]

        return {
            "summary_by_topic": summary_by_topic,
            "overall_stats": {
                "topics_count": len(summary_by_topic),
                "recent_emails": 15,
                "date_range": {
                    "start": start_date,
                    "end": end_date,
                },
            },
            "ai_model_info": {
                "model_type": "test-type",
                "model_name": "test-model",
                "temperature": 0.2,
            },
            "ai_generated": True,
            "message": "This is a test summary for development purposes.",
        }
