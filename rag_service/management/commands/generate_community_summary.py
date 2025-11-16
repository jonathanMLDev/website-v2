"""
Django management command to generate and save a community summary.

This command generates a weekly community summary and saves it to the database.

Usage:
    python manage.py generate_community_summary
    python manage.py generate_community_summary --deactivate-existing
"""

from datetime import datetime, timedelta

from django.core.management.base import BaseCommand, CommandError
from dateutil.parser import parse

from rag_service.models import CommunitySummary


class Command(BaseCommand):
    help = "Generate and save a weekly community summary to the database"

    def add_arguments(self, parser):
        parser.add_argument(
            "--deactivate-existing",
            action="store_true",
            help="Deactivate all existing summaries before creating a new one",
        )
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

            if not summary_data:
                raise CommandError("Failed to generate summary data")

            # Check if summary generation failed
            if summary_data.get("error"):
                raise CommandError(
                    f"Summary generation failed: {summary_data.get('error')}"
                )

            # Extract data for saving
            overall_stats = summary_data.get("overall_stats", {})
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

            # Deactivate existing summaries if requested
            if options["deactivate_existing"]:
                deactivated_count = CommunitySummary.objects.filter(
                    is_active=True
                ).update(is_active=False)
                if deactivated_count > 0:
                    self.stdout.write(
                        self.style.WARNING(
                            f"Deactivated {deactivated_count} existing summary(ies)"
                        )
                    )

            # Create new summary
            # For test summaries, set need_review=False so they display immediately
            need_review = not options.get("test", False)
            community_summary = CommunitySummary.objects.create(
                start_date=start_date,
                end_date=end_date,
                summary_data=summary_data,
                topics_count=overall_stats.get("topics_count", 0),
                recent_emails_count=overall_stats.get("recent_emails", 0),
                is_active=True,
                need_review=need_review,
            )

            self.stdout.write(
                self.style.SUCCESS(
                    f"\nSuccessfully created community summary!\n"
                    f"  ID: {community_summary.id}\n"
                    f"  Date Range: {start_date.date()} to {end_date.date()}\n"
                    f"  Topics: {community_summary.topics_count}\n"
                    f"  Recent Emails: {community_summary.recent_emails_count}\n"
                    f"  Active: {community_summary.is_active}"
                )
            )

        except Exception as e:
            raise CommandError(f"Error generating community summary: {e}")

    def _create_test_summary(self):
        """Create a test/dummy summary for development purposes."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)


        return {
            "summary_by_topic": [
                {
                    "subject": "Test Topic: Boost Library Updates",
                    "assertions":[
                        {
                            "content": "needs for this library",
                            "reference url": ["https://example.com/url1", "https://example.com/url2"]
                        },
                        {
                            "content": "relation with boost.asio",
                            "reference url": ["https://example.com/url3", "https://example.com/url4"]
                        },
                        {
                            "content": "Pros and cons of this library",
                            "reference url": ["https://example.com/url5", "https://example.com/url6"]
                        },
                    ],
                    "chronological_summary":[
                        {
                            "Date": "2018-09-10",
                            "summary": "importance of this library",
                            "reference url": ["https://example.com/url7", "https://example.com/url8"]
                        },
                        {
                            "Date": "2021-02-20",
                            "summary": "advanced properties of this library",
                            "reference url": ["https://example.com/url9", "https://example.com/url10"]
                        },
                    ],

                },
                {
                    "subject": "Test Topic: Community Discussions",
                    "assertions":[
                        {
                            "content": "needs for this Community",
                            "reference url": ["https://example.com/url11", "https://example.com/url12"]
                        },
                        {
                            "content": "Pros and cons of this Community",
                            "reference url": ["https://example.com/url13", "https://example.com/url14"]
                        },
                    ],
                    "chronological_summary":[
                        {
                            "Date": "2018-09-10",
                            "summary": "Should update community page",
                            "reference url": ["https://example.com/url15", "https://example.com/url16"]
                        },
                        {
                            "Date": "2021-02-20",
                            "summary": "advanced properties of community page",
                            "reference url": ["https://example.com/url17", "https://example.com/url18"]
                        },
                    ],
                },
            ],
            "overall_stats": {
                "topics_count": 2,
                "recent_emails": 15,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
            },
            "ai_generated": True,
            "warning": "This is a test summary for development purposes.",
        }

