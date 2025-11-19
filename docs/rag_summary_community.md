# RAG Service: Community Summary Guide

## Overview

Generates AI-powered summaries of Boost mailing list discussions and displays them on `/community/`.

**Quick Start:**
```bash
python manage.py generate_community_summary --test
# Visit http://localhost:8000/community/
```

## Architecture

```
HyperKitty DB → ChromaDB → Summary Generation → PostgreSQL → Community Page
```

## Data Flow

### 1. Email Collection (Every 30 min)
- **Task:** `sync_new_mails_to_vector_db()` (Celery)
- Syncs emails from HyperKitty to ChromaDB for semantic search
- **Location:** `rag_service/tasks.py`

### 2. Summary Generation (Daily at 1:00 AM)
1. Get recent emails (last 7 days, ~100 emails)
2. Extract topics (thread-based, clustering, or LLM)
3. For each topic: retrieve past emails via RAG → generate summary via LLM
4. Save as `CommunitySummary` to database

**Components:** `WeeklyCommunitySummaryGenerator`, `TopicExtractor`, `MailDataRetriever`, `LLMHelper`
**Location:** `rag_service/langchain_rag/task/community_task.py`

### 3. Display
`CommunitySummaryView` retrieves latest reviewed summary (`need_review=False`) → prepares data → renders in `community.html`
**Location:** `rag_service/views.py`, `templates/community.html`

### 4. S3 Backup (Daily)
- **Task:** `upload_vector_data_to_s3()` (Celery)
- Backs up ChromaDB vector data to S3 for disaster recovery
- **Location:** `rag_service/tasks.py`

## Project Structure

```
rag_service/
├── models.py                    # CommunitySummary model
├── views.py                     # CommunitySummaryView
├── tasks.py                     # Celery tasks
├── services.py                  # RAG service initialization
├── s3_utils.py                  # S3 backup utilities
├── management/commands/
│   └── generate_community_summary.py
├── langchain_rag/
│   ├── rag_pipeline.py          # Main RAG pipeline
│   ├── task/
│   │   ├── community_task.py    # Summary generator
│   │   ├── mail_data_retriever.py
│   │   └── topic_extractor.py
│   ├── retrieve/                # Retrieval components
│   ├── preprocessor/            # Data preprocessing
│   └── llm/                     # LLM integration
└── tests/
    ├── test_models.py           # Model tests
    ├── test_views.py            # View tests
    ├── test_tasks.py            # Task tests
    ├── test_commands.py         # Command tests
    └── tests_rag_pipeline/      # RAG pipeline integration tests
        ├── test_pipeline_basic.py      # Initialization & basic ops
        ├── test_pipeline_operations.py # CRUD operations
        ├── test_pipeline_errors.py     # Error handling
        └── test_pipeline_validation.py # Data validation
```

## Testing

```bash
# Run all tests
pytest rag_service/tests/

# RAG pipeline tests
pytest rag_service/tests/tests_rag_pipeline/

# Specific test file
pytest rag_service/tests/tests_rag_pipeline/test_pipeline_basic.py
```

## Usage

**Test Summary (dummy data, displays immediately):**
```bash
python manage.py generate_community_summary --test
```

**Real Summary (requires ChromaDB + LLM API keys, needs Wagtail review):**
```bash
python manage.py generate_community_summary
```

**Automated Tasks:**
- Email sync: Every 30 min via `sync_new_mails_to_vector_db()`
- Summary generation: Daily at 1:00 AM via `update_summary_data()`
- S3 backup: Daily via `upload_vector_data_to_s3()`

## Data Structure

**Summary Data (stored in `summary_data` field):**
```json
{
  "summary_by_topic": [
    {
      "subject": "Topic Name",
      "assertions": [
        {
          "content": "Key point...",
          "reference_url": ["url1", "url2"]
        }
      ],
      "chronological_summary": [
        {
          "date": "2025-11-06",
          "summary": "What happened...",
          "reference_url": ["url1"]
        }
      ]
    }
  ]
}
```

**Note:** `summary_data` contains only `summary_by_topic`. Statistics (`topics_count`, `recent_emails_count`, `start_date`, `end_date`, `model_info`) are stored as separate model fields.

### Database Model

```python
CommunitySummary(
    start_date=datetime(...),           # Summary period start
    end_date=datetime(...),              # Summary period end
    original_summary_data={...},         # Raw AI-generated data (read-only)
    summary_data={...},                  # Only contains summary_by_topic
    topics_count=5,                      # Calculated from summary_by_topic
    recent_emails_count=200,             # Stored separately
    generated_at=datetime(...),          # Auto-generated timestamp
    last_modified_at=datetime(...),      # Auto-updated on changes
    main_reviewer=User(...),             # Reviewer (set by Wagtail)
    need_review=False,                   # False = reviewed and ready to display
    model_info={"model_type": "...", "model_name": "..."}  # AI model metadata
)
```

## Key Components

**WeeklyCommunitySummaryGenerator** (`rag_service/langchain_rag/task/community_task.py`):
- Main orchestrator for summary generation
- Config: `source="chromadb"`, `limit=200`, `max_topics=5`, `fetch_k=30`, `topic_extraction_method="thread"|"clustering"|"llm"`

**CommunitySummaryView** (`rag_service/views.py`):
- Retrieves latest reviewed summary (`need_review=False`)
- Extracts `summary_by_topic` from `summary_data`
- Normalizes URLs, assigns reference numbers
- Builds `overall_stats` from model fields

## Troubleshooting

**No Summary Displayed:**
```bash
# Check if summary exists
python manage.py shell -c "from rag_service.models import CommunitySummary; print(CommunitySummary.objects.filter(need_review=False).count())"

# Generate test summary
python manage.py generate_community_summary --test
```

**Generation Fails:**
- ChromaDB: Ensure emails synced via `sync_new_mails_to_vector_db`
- LLM: Verify API keys and model availability
- Dependencies: Check RAG packages installed

## Summary

1. **Collect** emails: HyperKitty → ChromaDB (every 30 min via `sync_new_mails_to_vector_db`)
2. **Generate** summaries: recent emails → topics → RAG → LLM (daily at 1 AM via `update_summary_data`)
3. **Store** in PostgreSQL with `need_review=True` (Wagtail sets to `False` after review)
4. **Backup** ChromaDB to S3 (daily via `upload_vector_data_to_s3`)
5. **Display** on `/community/` page (shows latest summary with `need_review=False`)

Runs automatically via Celery or manually via management commands.
