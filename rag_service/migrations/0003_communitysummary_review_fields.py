from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("rag_service", "0002_communitysummary_need_review_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="communitysummary",
            name="last_modified_at",
            field=models.DateTimeField(
                auto_now=True, help_text="Timestamp of the latest modification"
            ),
        ),
        migrations.AddField(
            model_name="communitysummary",
            name="original_summary_data",
            field=models.JSONField(
                default=dict,
                editable=False,
                help_text="Raw summary data before reviewer revisions",
            ),
        ),
        migrations.AddField(
            model_name="communitysummary",
            name="main_reviewer",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="main_reviewed_summaries",
                help_text="Reviewer responsible for the published summary",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AddField(
            model_name="communitysummary",
            name="model_info",
            field=models.JSONField(
                default=dict,
                help_text="Metadata about the AI model used to generate the summary",
                blank=True,
            ),
        ),
        migrations.RemoveIndex(
            model_name="communitysummary",
            name="rag_service_generat_3144df_idx",
        ),
        migrations.RemoveField(
            model_name="communitysummary",
            name="is_active",
        ),
    ]
