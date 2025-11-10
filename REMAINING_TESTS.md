# Remaining Tests to Complete

**Last Updated**: 2025-01-XX  
**Current Status**: Backend ~40% coverage, Frontend ~25% coverage

---

## Backend Tests Status

### ✅ Existing Tests (16 test files)

1. `test_ai_models.py` - AI model tests
2. `test_auth_api.py` - Authentication API tests
3. `test_auth_service.py` - Auth service tests
4. `test_config.py` - Configuration tests
5. `test_crud.py` - Database CRUD tests
6. `test_document_processing.py` - Document processing tests
7. `test_error_handlers.py` - Error handler tests
8. `test_job_queue.py` - Job queue tests
9. `test_model_manager.py` - Model manager tests
10. `test_note_approval_api.py` - Note approval API tests
11. `test_pipeline.py` - Pipeline tests
12. `test_queue_service.py` - Queue service tests
13. `test_storage.py` - Storage service tests
14. `test_summarizer_service.py` - Summarizer service tests
15. `test_triage_service.py` - Triage service tests

### ❌ Missing Integration Tests

#### 1. Manage Agent API Integration Tests
**File**: `backend/tests/test_manage_agent_integration.py`

**Test Cases Needed**:
- [ ] `test_check_in_flow` - Complete check-in → queue entry creation
- [ ] `test_vitals_submission_flow` - Vitals → triage calculation → queue update
- [ ] `test_consultation_lifecycle` - Start → process → complete consultation
- [ ] `test_queue_state_transitions` - All queue stage transitions
- [ ] `test_doctor_assignment` - Automatic and manual doctor assignment
- [ ] `test_wait_time_prediction` - Wait time calculation
- [ ] `test_timeline_generation` - Patient timeline creation
- [ ] `test_document_processing_workflow` - Document upload → processing → review

#### 2. Make Agent (Scribe) Integration Tests
**File**: `backend/tests/test_make_agent_integration.py`

**Test Cases Needed**:
- [ ] `test_full_pipeline_flow` - Upload → transcribe → extract → generate note
- [ ] `test_audio_upload_and_storage` - File upload to storage
- [ ] `test_transcription_accuracy` - Transcription quality checks
- [ ] `test_entity_extraction` - Clinical entity extraction
- [ ] `test_note_generation` - SOAP note generation
- [ ] `test_note_approval_workflow` - Note approval/rejection flow
- [ ] `test_note_versioning` - Note version management
- [ ] `test_job_queue_integration` - Async job processing

#### 3. Queue Service Integration Tests
**File**: `backend/tests/test_queue_integration.py`

**Test Cases Needed**:
- [ ] `test_queue_creation_and_lifecycle` - Complete queue entry lifecycle
- [ ] `test_queue_filtering` - Filter by stage, doctor, priority
- [ ] `test_queue_websocket_updates` - Real-time WebSocket notifications
- [ ] `test_queue_sse_updates` - Server-Sent Events updates
- [ ] `test_queue_concurrent_updates` - Concurrent queue modifications
- [ ] `test_queue_priority_sorting` - Priority-based sorting

#### 4. Authentication & Authorization Integration Tests
**File**: `backend/tests/test_auth_integration.py`

**Test Cases Needed**:
- [ ] `test_jwt_token_flow` - Login → access token → refresh token
- [ ] `test_role_based_access` - RBAC for each role
- [ ] `test_protected_endpoints` - All protected endpoints
- [ ] `test_session_management` - Session creation and expiration
- [ ] `test_password_reset_flow` - Password reset workflow

#### 5. Patient Management Integration Tests
**File**: `backend/tests/test_patients_integration.py`

**Test Cases Needed**:
- [ ] `test_patient_creation_flow` - Patient registration
- [ ] `test_patient_search` - Search functionality
- [ ] `test_patient_history_retrieval` - Medical history access
- [ ] `test_patient_vitals_history` - Vitals tracking

#### 6. Summarizer Integration Tests
**File**: `backend/tests/test_summarizer_integration.py`

**Test Cases Needed**:
- [ ] `test_summarization_workflow` - Patient → summary generation
- [ ] `test_summary_caching` - Cache hit/miss scenarios
- [ ] `test_timeline_generation` - Timeline event creation
- [ ] `test_force_refresh` - Force refresh functionality

#### 7. End-to-End Workflow Tests
**File**: `backend/tests/test_e2e_workflows.py`

**Test Cases Needed**:
- [ ] `test_receptionist_to_doctor_flow` - Check-in → triage → consultation
- [ ] `test_note_generation_workflow` - Consultation → recording → note → approval
- [ ] `test_patient_discharge_flow` - Complete patient lifecycle
- [ ] `test_multi_patient_queue_management` - Multiple patients in queue

#### 8. Error Handling & Edge Cases
**File**: `backend/tests/test_error_cases.py`

**Test Cases Needed**:
- [ ] `test_invalid_audio_format` - Unsupported audio formats
- [ ] `test_missing_required_fields` - Validation errors
- [ ] `test_concurrent_queue_updates` - Race conditions
- [ ] `test_storage_failures` - Storage service errors
- [ ] `test_ai_service_failures` - AI service error handling
- [ ] `test_database_connection_failures` - DB error handling

---

## Frontend Tests Status

### ✅ Existing Tests (2 test files)

1. `StatusIndicator.test.tsx` - Status indicator component
2. `ProtectedRoute.test.tsx` - Protected route component

### ❌ Missing Component Tests

#### 1. Dashboard Components
**Files Needed**:
- [ ] `ReceptionistDashboard.test.tsx`
- [ ] `NurseDashboard.test.tsx`
- [ ] `DoctorDashboard.test.tsx`
- [ ] `AdminDashboard.test.tsx`

**Test Cases**:
- [ ] Dashboard rendering
- [ ] Queue data display
- [ ] Filter functionality
- [ ] Metrics calculation

#### 2. Doctor Workflow Components
**Files Needed**:
- [ ] `DoctorWorkflow.test.tsx`
- [ ] `ConsultationView.test.tsx`

**Test Cases**:
- [ ] Patient selection
- [ ] Audio recording
- [ ] Note generation
- [ ] Note editing
- [ ] Note approval

#### 3. Queue Components
**Files Needed**:
- [ ] `QueueFilters.test.tsx`
- [ ] `MetricCard.test.tsx`
- [ ] `MetricsChart.test.tsx`

**Test Cases**:
- [ ] Queue filtering
- [ ] Metrics display
- [ ] Chart rendering

#### 4. AI Components
**Files Needed**:
- [ ] `NoteApprovalWorkflow.test.tsx`
- [ ] `HistoryView.test.tsx`
- [ ] `ConsultationHistory.test.tsx`

**Test Cases**:
- [ ] Note approval flow
- [ ] History display
- [ ] Timeline rendering

#### 5. Form Components
**Files Needed**:
- [ ] `CheckInView.test.tsx`
- [ ] `VitalsForm.test.tsx`

**Test Cases**:
- [ ] Form validation
- [ ] Form submission
- [ ] Error handling

#### 6. Hooks Tests
**Files Needed**:
- [ ] `useQueueData.test.ts`
- [ ] `usePatientSearch.test.ts`
- [ ] `usePatientCreate.test.ts`

**Test Cases**:
- [ ] Data fetching
- [ ] Error handling
- [ ] Loading states
- [ ] Cache management

#### 7. API Service Tests
**Files Needed**:
- [ ] `api.test.ts`

**Test Cases**:
- [ ] API calls
- [ ] Error handling
- [ ] Request/response transformation

---

## E2E Tests Status

### ✅ Existing E2E Tests (2 test files)

1. `auth.spec.ts` - Authentication flow
2. `receptionist-flow.spec.ts` - Receptionist workflow

### ❌ Missing E2E Tests

#### 1. Nurse Workflow
**File**: `tests/e2e/nurse-flow.spec.ts`
- [ ] Login as nurse
- [ ] View triage queue
- [ ] Enter vitals
- [ ] Verify triage calculation
- [ ] Verify queue update

#### 2. Doctor Workflow
**File**: `tests/e2e/doctor-flow.spec.ts`
- [ ] Login as doctor
- [ ] View patient queue
- [ ] Start consultation
- [ ] Record audio
- [ ] Generate note
- [ ] Approve note
- [ ] Complete consultation

#### 3. Admin Workflow
**File**: `tests/e2e/admin-flow.spec.ts`
- [ ] Login as admin
- [ ] View system overview
- [ ] Manage queue
- [ ] View analytics

#### 4. Complete Patient Journey
**File**: `tests/e2e/patient-journey.spec.ts`
- [ ] Check-in (receptionist)
- [ ] Triage (nurse)
- [ ] Consultation (doctor)
- [ ] Note approval
- [ ] Discharge

---

## Test Coverage Goals

### Backend
- **Current**: ~40%
- **Target**: ≥80%
- **Priority**: Integration tests for workflows

### Frontend
- **Current**: ~25% (2 components)
- **Target**: ≥70%
- **Priority**: Dashboard components, workflow components

### E2E
- **Current**: 2 workflows
- **Target**: All 4 role workflows + patient journey
- **Priority**: Doctor workflow, complete patient journey

---

## Estimated Time

### Backend Integration Tests
- Manage Agent: ~4 hours
- Make Agent: ~3 hours
- Queue Service: ~2 hours
- Auth Integration: ~2 hours
- Patients: ~1 hour
- Summarizer: ~1 hour
- E2E Workflows: ~3 hours
- Error Cases: ~2 hours
**Total**: ~18 hours

### Frontend Component Tests
- Dashboard Components: ~4 hours
- Workflow Components: ~4 hours
- Queue Components: ~2 hours
- AI Components: ~2 hours
- Form Components: ~2 hours
- Hooks: ~2 hours
- API Services: ~1 hour
**Total**: ~17 hours

### E2E Tests
- Nurse Workflow: ~2 hours
- Doctor Workflow: ~3 hours
- Admin Workflow: ~2 hours
- Patient Journey: ~3 hours
**Total**: ~10 hours

### Grand Total: ~45 hours

---

## Priority Order

### High Priority (Critical for Production)
1. Backend integration tests for workflows
2. Frontend dashboard component tests
3. Doctor workflow E2E test

### Medium Priority (Quality Assurance)
4. Frontend workflow component tests
5. Complete patient journey E2E test
6. Error handling tests

### Low Priority (Nice to Have)
7. Admin workflow tests
8. Hooks and API service tests
9. Edge case tests

---

## Next Steps

1. **Start with Backend Integration Tests**
   - Focus on Manage Agent workflow
   - Then Make Agent workflow
   - Then Queue Service

2. **Add Frontend Component Tests**
   - Start with DoctorWorkflow (most complex)
   - Then dashboard components
   - Then supporting components

3. **Complete E2E Tests**
   - Doctor workflow first
   - Then complete patient journey
   - Then other role workflows

