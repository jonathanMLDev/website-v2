from django.apps import AppConfig


class RagServiceConfig(AppConfig):
    """Configuration for RAG Service Django app"""

    default_auto_field = "django.db.models.BigAutoField"
    name = "rag_service"
    verbose_name = "RAG Service"

    def ready(self):
        """Called when Django starts"""
        # Import signals if needed
        # import rag_service.signals  # noqa
        pass

