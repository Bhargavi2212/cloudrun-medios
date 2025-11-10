# Backend Tests

This directory contains comprehensive tests for the Medi OS backend.

## Test Structure

- `conftest.py` - Shared pytest fixtures for database sessions, test clients, and test data
- `test_auth_service.py` - Tests for authentication service (password reset, etc.)
- `test_auth_api.py` - Tests for authentication API endpoints
- `test_crud.py` - Tests for CRUD operations (notes, patients, consultations)
- `test_note_approval_api.py` - Tests for note approval workflow API endpoints
- `test_document_processing.py` - Tests for document processing service
- `test_ai_models.py` - Tests for AI models service
- `test_storage.py` - Tests for storage service
- `test_queue_service.py` - Tests for queue service
- `test_triage_service.py` - Tests for triage service
- `test_summarizer_service.py` - Tests for summarizer service
- `test_job_queue.py` - Tests for job queue
- `test_pipeline.py` - Tests for AI pipeline
- `test_config.py` - Tests for configuration
- `test_error_handlers.py` - Tests for error handlers

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_auth_service.py
```

### Run with coverage
```bash
pytest --cov=backend --cov-report=html
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test
```bash
pytest tests/test_auth_service.py::test_authenticate_user_success
```

## Test Fixtures

### Database Session
- `db_session` - In-memory SQLite database session for testing

### Test Client
- `client` - FastAPI test client with database session override

### Test Data
- `test_user` - Test user with DOCTOR role
- `test_patient` - Test patient
- `test_consultation` - Test consultation
- `test_queue_state` - Test queue state
- `test_note` - Test note with version
- `auth_headers` - Authentication headers for test user

## Test Coverage Goals

- **Target Coverage**: ≥80%
- **Current Coverage**: See coverage report after running tests

## Writing New Tests

1. Use the existing fixtures from `conftest.py`
2. Follow the naming convention: `test_<function_name>_<scenario>`
3. Use descriptive test names
4. Test both success and failure cases
5. Use `pytest.mark.asyncio` for async tests
6. Use `pytest.raises` for exception testing

## Example Test

```python
def test_create_note_with_version(db_session, test_consultation, test_user):
    """Test creating a note with a version."""
    note_id, version_id = crud.create_note_with_version(
        db_session,
        consultation_id=test_consultation.id,
        content="Test note content",
        author_id=test_user.id,
        is_ai_generated=True,
    )
    
    assert note_id is not None
    assert version_id is not None
```

