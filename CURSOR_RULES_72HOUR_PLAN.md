# Cursor Rules for Medi OS 72-Hour Production Plan

## Overview
This document provides rules and guidelines for Cursor AI to follow when implementing the Medi OS 72-Hour Production Plan. These rules ensure consistency, quality, and alignment with the production plan phases.

## General Principles

### 1. Plan-First Development
- **Always check `PRODUCTION_PLAN_STATUS.md`** before starting any work to understand current status
- **Reference the 72-hour plan phases** to understand priorities and dependencies
- **Update status document** after completing any phase or task
- **Do not skip phases** - follow the plan sequence unless explicitly instructed

### 2. Code Quality Standards
- **Follow existing code patterns** - Match the style and structure of existing code
- **Use type hints** - All Python functions must have type hints
- **Write docstrings** - Google-style docstrings for Python, JSDoc for TypeScript
- **Handle errors gracefully** - Use proper error handling and logging
- **Log appropriately** - Use structured logging with appropriate levels

### 3. Database & Models
- **Use Alembic migrations** - Never modify database schema directly
- **Follow existing model patterns** - Use UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin
- **Validate data** - Use Pydantic models for API request/response validation
- **Handle migrations carefully** - Test migrations on development database first

### 4. API Development
- **Follow RESTful conventions** - Use proper HTTP methods and status codes
- **Use Pydantic models** - All request/response models must be Pydantic
- **Handle errors consistently** - Use `backend/api/error_handlers.py` patterns
- **Document endpoints** - Add OpenAPI documentation
- **Use async/await** - All I/O-bound operations must be async

### 5. Frontend Development
- **Follow existing component patterns** - Match the structure of existing components
- **Use TypeScript strictly** - No `any` types, use proper types
- **Use React Query** - For all API calls and data fetching
- **Use Zustand** - For global state management (auth, UI state)
- **Handle loading states** - Show loading indicators for all async operations
- **Handle errors** - Display user-friendly error messages

## Phase-Specific Rules

### Phase 1 – Foundation

#### db-init Rules
- **Use Alembic** - All schema changes must go through Alembic migrations
- **Follow naming conventions** - Use snake_case for table and column names
- **Add indexes** - Add indexes for frequently queried columns
- **Use enums** - Use SQLAlchemy enums for status fields
- **Add constraints** - Add check constraints for data validation
- **Seed data** - Use `backend/database/seeds.py` for seed data

#### auth-core Rules
- **Use JWT** - Access tokens and refresh tokens
- **Use bcrypt** - For password hashing
- **Use role-based access** - Implement role guards in `backend/security/permissions.py`
- **Audit logging** - Log all authentication events
- **Handle token expiration** - Implement token refresh logic
- **Secure endpoints** - Use `Depends(get_current_user)` for protected endpoints

#### storage-config Rules
- **Use abstraction layer** - Always use `backend/services/storage.py` abstraction
- **Support both backends** - Local storage for dev, GCS for production
- **Generate signed URLs** - For secure file access
- **Handle file uploads** - Validate file types and sizes
- **Store metadata** - Store file metadata in database

#### settings-secrets Rules
- **Use environment variables** - Never hardcode secrets
- **Use `backend/services/config.py`** - For all configuration
- **Provide `.env.example`** - Document all required environment variables
- **Use feature flags** - For feature toggles
- **Support multiple environments** - Dev, staging, production

### Phase 2 – Backend Services

#### scribe-upgrade Rules
- **Use LangGraph pipeline** - Follow `backend/services/make_agent_pipeline.py` pattern
- **Handle errors gracefully** - Fallback to template notes if Gemini fails
- **Log LLM usage** - Track all LLM API calls in `llm_usage` table
- **Version notes** - Use `NoteVersion` model for note versioning
- **Handle async processing** - Use background jobs for long-running tasks
- **Stream responses** - Use streaming for real-time transcription

#### triage-wrapper Rules
- **Load models at startup** - Use FastAPI startup event
- **Cache model predictions** - Cache predictions for performance
- **Log predictions** - Track all triage predictions
- **Handle model errors** - Fallback to default triage level
- **Provide explainability** - Return confidence scores and reasoning

#### summarizer-wrapper Rules
- **Use Gemini API** - For document summarization
- **Cache summaries** - Use `PatientSummary` model for caching
- **Handle large documents** - Chunk large documents if needed
- **Track costs** - Log token usage and costs
- **Handle errors** - Fallback to basic extraction if summarization fails

#### queue-engine Rules
- **Use state machine** - Follow `backend/services/manage_agent_state_machine.py` pattern
- **Handle state transitions** - Validate state transitions
- **Calculate wait times** - Use `backend/services/wait_time_estimator.py`
- **Real-time updates** - Use WebSocket or SSE for real-time updates
- **Handle assignments** - Implement doctor assignment logic

#### api-orchestration Rules
- **Use router pattern** - Organize endpoints by feature
- **Handle errors consistently** - Use `backend/api/error_handlers.py`
- **Add correlation IDs** - For request tracking
- **Log requests** - Use `backend/services/middleware.py` for logging
- **Document APIs** - Add OpenAPI documentation

### Phase 3 – Frontend Completion

#### auth-ui Rules
- **Use Zustand** - For auth state management
- **Handle token refresh** - Automatically refresh tokens
- **Redirect by role** - Redirect users to appropriate dashboards
- **Handle session timeout** - Show session timeout warnings
- **Secure routes** - Use `ProtectedRoute` component

#### dashboards Rules
- **Role-based dashboards** - Separate dashboards for each role
- **Real-time updates** - Use WebSocket or polling for real-time data
- **Show loading states** - Display loading indicators
- **Handle errors** - Show user-friendly error messages
- **Responsive design** - Support mobile and tablet devices

#### ai-interfaces Rules
- **Show status indicators** - Display processing status
- **Handle errors gracefully** - Show error messages and retry options
- **Provide history views** - Show previous AI generations
- **Allow downloads** - Enable download/export functionality
- **Show confidence scores** - Display AI confidence scores

#### advanced-ui Rules
- **Implement search** - Patient search with autocomplete
- **Show history** - Consultation and note history
- **Approval workflows** - Note approval and editing workflows
- **Accessibility** - WCAG 2.1 AA compliance
- **Dark mode** - Support dark mode (if time permits)

### Phase 4 – Testing & QA

#### backend-tests Rules
- **Use pytest** - For all backend tests
- **Test coverage** - Aim for ≥80% coverage
- **Test workflows** - Test end-to-end workflows
- **Test concurrency** - Test concurrent access
- **Mock external services** - Mock Gemini API, storage, etc.

#### frontend-tests Rules
- **Use Vitest** - For unit tests
- **Use Playwright/Cypress** - For E2E tests
- **Test role flows** - Test all role-specific workflows
- **Test error handling** - Test error scenarios
- **Test accessibility** - Test with screen readers

#### load-tests Rules
- **Use k6 or locust** - For load testing
- **Test triage endpoint** - Test triage API under load
- **Test scribe endpoint** - Test scribe API under load
- **Test summarizer endpoint** - Test summarizer API under load
- **Provide recommendations** - Document performance tuning recommendations

### Phase 5 – Deployment & Ops

#### containerization Rules
- **Use multi-stage builds** - Optimize Docker images
- **Use non-root user** - Run containers as non-root user
- **Add health checks** - Implement health check endpoints
- **Use docker-compose** - For local development
- **Optimize images** - Minimize image size

#### gcp-infra Rules
- **Use Cloud Run** - For backend and frontend
- **Use Cloud SQL** - For PostgreSQL database
- **Use Cloud Storage** - For file storage
- **Use Secret Manager** - For secrets management
- **Use Terraform** - For infrastructure as code (if time permits)

#### cicd Rules
- **Use GitHub Actions** - For CI/CD pipeline
- **Run tests** - Run all tests before deployment
- **Run linters** - Run linters before deployment
- **Build images** - Build Docker images
- **Deploy to staging** - Deploy to staging first
- **Run smoke tests** - Run smoke tests after deployment

#### monitoring Rules
- **Use Prometheus** - For metrics collection
- **Use Cloud Monitoring** - For GCP monitoring
- **Log structured data** - Use structured logging
- **Set up alerts** - Configure alerting policies
- **Monitor LLM usage** - Track LLM API usage and costs

### Phase 6 – Documentation & Runbooks

#### dev-docs Rules
- **Document architecture** - Document system architecture
- **Document setup** - Document setup instructions
- **Document API** - Document API endpoints
- **Document deployment** - Document deployment process
- **Document troubleshooting** - Document common issues and solutions

#### user-docs Rules
- **Document workflows** - Document role-specific workflows
- **Document AI features** - Document AI feature usage
- **Add screenshots** - Add screenshots for visual guides
- **Create FAQs** - Answer common questions

#### ops-runbooks Rules
- **Document model updates** - Document how to update models
- **Document database maintenance** - Document database maintenance procedures
- **Document incident response** - Document incident response procedures
- **Document rollback procedures** - Document how to rollback deployments

## File Organization Rules

### Backend Structure
```
backend/
├── api/v1/          # API endpoints (organized by feature)
├── database/        # Database models, migrations, CRUD
├── services/        # Business logic services
├── security/        # Authentication and authorization
├── tests/           # Test files
└── main.py          # FastAPI application
```

### Frontend Structure
```
frontend/src/
├── components/      # Reusable components
├── pages/           # Page components (organized by role)
├── services/        # API service layer
├── stores/          # Zustand stores
├── hooks/           # Custom React hooks
├── types/           # TypeScript type definitions
└── utils/           # Utility functions
```

## Common Patterns

### Error Handling
```python
# Backend
try:
    result = await service.method()
except SpecificError as e:
    logger.error(f"Error: {e}")
    raise HTTPException(status_code=400, detail=str(e))
```

### Logging
```python
# Backend
import logging
logger = logging.getLogger(__name__)
logger.info(f"Processing request: {request_id}")
logger.error(f"Error: {error}", exc_info=True)
```

### API Endpoints
```python
# Backend
@router.post("/endpoint", response_model=ResponseModel)
async def endpoint(
    request: RequestModel,
    current_user: User = Depends(get_current_user),
) -> ResponseModel:
    """Endpoint description."""
    result = await service.process(request)
    return ResponseModel(**result)
```

### React Components
```typescript
// Frontend
export const Component: React.FC<Props> = ({ prop }) => {
  const { data, isLoading, error } = useQuery(['key'], fetchData);
  
  if (isLoading) return <Loading />;
  if (error) return <Error message={error.message} />;
  
  return <div>{data}</div>;
};
```

## Priority Order

When working on multiple tasks, follow this priority order:

1. **Critical bugs** - Fix critical bugs first
2. **Phase 1 tasks** - Complete foundation tasks
3. **Phase 2 tasks** - Complete backend services
4. **Phase 3 tasks** - Complete frontend
5. **Phase 4 tasks** - Add tests
6. **Phase 5 tasks** - Set up deployment
7. **Phase 6 tasks** - Write documentation

## Questions to Ask

Before starting any task, ask:
1. **Is this in the plan?** - Check if the task is in the 72-hour plan
2. **What's the status?** - Check `PRODUCTION_PLAN_STATUS.md` for current status
3. **Are dependencies met?** - Check if dependencies are completed
4. **What's the priority?** - Check priority order
5. **Are there existing patterns?** - Check for existing code patterns to follow

## Updates Required

After completing any task:
1. **Update status** - Update `PRODUCTION_PLAN_STATUS.md`
2. **Update documentation** - Update relevant documentation
3. **Write tests** - Write tests for new functionality
4. **Update API docs** - Update OpenAPI documentation if needed
5. **Commit changes** - Commit changes with descriptive messages

## Notes

- **Do not skip phases** - Complete phases in order
- **Follow existing patterns** - Match existing code style and patterns
- **Test thoroughly** - Write tests for all new functionality
- **Document changes** - Update documentation as you go
- **Ask for clarification** - Ask if unsure about requirements

