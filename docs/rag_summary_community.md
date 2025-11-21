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
4. Save as `CommunitySummary` to database with:
   - `original_summary_data`: Raw AI output (read-only)
   - `summary_data`: Copy of original (editable by reviewers)
   - `model_info`: AI model metadata
   - `need_review=True`: Requires Wagtail review before display

**Components:** `WeeklyCommunitySummaryGenerator`, `TopicExtractor`, `MailDataRetriever`, `LLMHelper`
**Location:** `rag_service/langchain_rag/task/community_task.py`
**Task:** `update_summary_data()` in `rag_service/tasks.py`

### 3. Review Workflow (Wagtail)
- Summaries are created with `need_review=True`
- Wagtail reviewers can:
  - Edit `summary_data` (published version)
  - Set `main_reviewer` to track who approved
  - Set `need_review=False` when ready to publish
  - `original_summary_data` remains read-only (preserves AI output)

### 4. Display
`CommunitySummaryView` retrieves latest reviewed summary (`need_review=False`) → normalizes data → renders in `community.html`
**Location:** `rag_service/views.py`, `templates/community.html`

### 5. S3 Backup (Daily)
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

Comprehensive test suite with **69 tests** covering all functionality:

```bash
# Run all tests (69 tests)
pytest rag_service/tests/

# RAG pipeline tests (38 tests)
pytest rag_service/tests/tests_rag_pipeline/

# Specific test categories
pytest rag_service/tests/test_models.py        # Model tests (8 tests)
pytest rag_service/tests/test_tasks.py         # Task tests (12 tests)
pytest rag_service/tests/test_views.py         # View tests (5 tests)
pytest rag_service/tests/test_commands.py      # Command tests (6 tests)

# Pipeline test modules
pytest rag_service/tests/tests_rag_pipeline/test_pipeline_basic.py      # Basic ops (6 tests)
pytest rag_service/tests/tests_rag_pipeline/test_pipeline_operations.py # CRUD (9 tests)
pytest rag_service/tests/tests_rag_pipeline/test_pipeline_errors.py     # Errors (13 tests)
pytest rag_service/tests/tests_rag_pipeline/test_pipeline_validation.py # Validation (10 tests)
```

All tests use updated fixtures that match the current model structure (including `original_summary_data`, `model_info`, etc.).

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

**Summary Data (stored in `summary_data` and `original_summary_data` fields):**
```json
{
  "summary_by_topic": [
    {
      "subject": "Topic Name",
      "assertions": [
        {
          "content": "Key point...",
          "reference url": ["url1", "url2"]  // Note: space, not underscore
        }
      ],
      "chronological_summary": [
        {
          "Date": "2025-11-06",  // Note: capital D
          "summary": "What happened...",
          "reference url": ["url1"]  // Note: space, not underscore
        }
      ]
    }
  ]
}
```

**Important Notes:**
- `summary_data` contains only `summary_by_topic` (the published, reviewer-edited version)
- `original_summary_data` contains the raw AI-generated data (read-only, preserved for audit)
- Raw data uses `"reference url"` (with space) and `"Date"` (capital D)
- The view normalizes these to `"reference_url"` (with underscore) and `"date"` (lowercase) for display
- Statistics (`topics_count`, `recent_emails_count`, `start_date`, `end_date`, `model_info`) are stored as separate model fields

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
- Default config: `source="chromadb"`, `limit=200`, `max_topics=5`, `fetch_k=30`, `start_date=54 days ago`, `topic_extraction_method="thread"|"clustering"|"llm"`
- Returns data structure with `summary_by_topic`, `overall_stats`, and `ai_model_info`

**CommunitySummaryView** (`rag_service/views.py`):
- Retrieves latest reviewed summary (`need_review=False`)
- Extracts `summary_by_topic` from `summary_data`
- Normalizes data structure:
  - Converts `"reference url"` → `"reference_url"` (with underscore)
  - Converts `"Date"` → `"date"` (lowercase)
  - Assigns continuous reference numbers to all URLs
  - Converts archive API URLs to message URLs
- Builds `overall_stats` from model fields (`topics_count`, `recent_emails_count`, `start_date`, `end_date`)
- Includes `model_info` in context for display

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
2. **Generate** summaries: recent emails (54 days, up to 200) → topics → RAG → LLM (daily at 1 AM via `update_summary_data`)
3. **Store** in PostgreSQL:
   - `original_summary_data`: Raw AI output (read-only, preserved)
   - `summary_data`: Published version (editable by reviewers)
   - `model_info`: AI model metadata
   - `need_review=True`: Requires Wagtail review before display
4. **Review** in Wagtail: Reviewers edit `summary_data`, set `main_reviewer`, set `need_review=False` when ready
5. **Backup** ChromaDB to S3 (daily via `upload_vector_data_to_s3`)
6. **Display** on `/community/` page (shows latest summary with `need_review=False`)

Runs automatically via Celery or manually via management commands.

## Key Features

- **Dual Data Storage**: Preserves original AI output while allowing reviewer edits
- **Review Workflow**: Wagtail integration for content review and approval
- **Model Tracking**: Stores AI model metadata for reproducibility
- **Audit Trail**: `generated_at`, `last_modified_at`, and `main_reviewer` track changes
- **Data Normalization**: View automatically normalizes field names for consistent display
