# MediOS Frontend Development Plan

## 1. Vision & Core Architecture

### Mission
To build a clean, intuitive, and role-specific user interface for MediOS that is fast, reliable, and works seamlessly on standard hospital hardware. The UI must directly support and enhance the specific workflows of each user, from receptionists to doctors, to reduce chaos and save time.

### Core Tech Stack
- **Framework**: React 18 (using Vite for build tool)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Server State**: React Query (@tanstack/react-query) for all API fetching, caching, and mutations
- **Global UI State**: Zustand (primarily for authentication and user state)
- **Routing**: React Router DOM v7
- **UI Components**: Radix UI primitives with custom styling
- **Icons**: Lucide React

### Core Principles
1. **Role-First Design**: Every component and view designed with a specific user role in mind
2. **Offline-First Mentality**: UI remains responsive with clear feedback during connection issues
3. **Minimal Clicks**: Workflows optimized to require minimum interactions
4. **Performance First**: Fast loading, efficient rendering, optimized for hospital hardware
5. **Accessibility**: WCAG 2.1 AA compliance for all users

## 2. User Journey & Role-Based Workflows

### The Receptionist's Workflow
**Primary Route**: `/` (root) and `/receptionist`

1. **Login**: Redirected to MainQueueDashboard
2. **Dashboard View**: Real-time list of all patients with status, assigned doctor, wait times
3. **Check-In Process**:
   - Click "Check-In Patient" button
   - Search existing patient OR register new patient
   - Enter chief complaint
   - Patient added to queue with `AWAITING_VITALS` status

### The Nurse's Workflow  
**Primary Route**: `/nurse` and `/triage`

1. **Login**: Redirected to TriageQueueView
2. **Triage Queue**: Filtered list showing only `AWAITING_VITALS` patients
3. **Triage Process**:
   - Select patient from queue
   - Enter complete vital signs using VitalsForm
   - AI calculates triage level automatically
   - Patient status updated to `AWAITING_DOCTOR_ASSIGNMENT`
   - Return to queue (patient removed from nurse's view)

### The Doctor's Workflow
**Primary Routes**: `/doctor/dashboard`, `/doctor/consultation/:id`

1. **Login**: Redirected to DoctorDashboardView
2. **Personal Queue**: Only patients assigned specifically to this doctor
3. **Consultation Process**:
   - Select highest-priority patient
   - Pre-consultation review (chief complaint + vitals)
   - Click "Start Consultation"
   - Activate MakeAgentScribe for AI-powered note-taking
   - Complete consultation and finalize notes

### Admin Workflow
**Primary Route**: `/admin`

1. **System Overview**: Complete hospital floor view
2. **Queue Management**: All patients across all statuses
3. **Staff Management**: Doctor assignments and availability
4. **Analytics**: Wait times, triage distribution, system performance

## 3. Technical Implementation Plan

### Authentication & Authorization

#### Zustand Auth Store Structure
```typescript
interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  clearError: () => void;
}

interface User {
  id: number;
  username: string;
  email: string;
  full_name: string;
  role: 'RECEPTIONIST' | 'NURSE' | 'DOCTOR' | 'ADMIN';
  is_active: boolean;
}
```

#### Role-Based Access Control (RBAC)
- **ProtectedRoute Component**: Wraps all authenticated routes
- **Role Validation**: Checks user role from Zustand store
- **Automatic Redirects**: Routes users to appropriate dashboards
- **Permission Guards**: Component-level permission checking

### API Integration Strategy

#### React Query Configuration
```typescript
// Query Keys Factory
export const queryKeys = {
  patients: {
    all: ['patients'] as const,
    lists: () => [...queryKeys.patients.all, 'list'] as const,
    list: (filters: string) => [...queryKeys.patients.lists(), { filters }] as const,
    details: () => [...queryKeys.patients.all, 'detail'] as const,
    detail: (id: number) => [...queryKeys.patients.details(), id] as const,
  },
  consultations: {
    all: ['consultations'] as const,
    lists: () => [...queryKeys.consultations.all, 'list'] as const,
    queue: () => [...queryKeys.consultations.lists(), 'queue'] as const,
    doctor: (doctorId: number) => [...queryKeys.consultations.lists(), 'doctor', doctorId] as const,
  },
  queue: {
    all: ['queue'] as const,
    main: () => [...queryKeys.queue.all, 'main'] as const,
    triage: () => [...queryKeys.queue.all, 'triage'] as const,
  }
};
```

#### API Service Layer
```typescript
// services/api.ts
class ApiService {
  private baseURL = 'http://localhost:8000/api/v1';
  
  // Authentication
  async login(credentials: LoginRequest): Promise<AuthResponse>;
  async getCurrentUser(): Promise<User>;
  
  // Patient Management
  async getPatients(params?: PaginationParams): Promise<Patient[]>;
  async createPatient(patient: PatientCreate): Promise<Patient>;
  async updatePatient(id: number, patient: PatientUpdate): Promise<Patient>;
  
  // Queue Management
  async getQueue(): Promise<QueueResponse>;
  async checkInPatient(request: CheckInRequest): Promise<CheckInResponse>;
  async submitVitals(consultationId: number, vitals: VitalsSubmission): Promise<TriageResult>;
  
  // AI Scribe
  async processAudio(file: File, metadata: AudioMetadata): Promise<ProcessingResult>;
  async generateNote(text: string, entities: any): Promise<NoteResult>;
}
```

### Component Architecture

#### Global Components
```
src/
├── components/
│   ├── layout/
│   │   ├── AppLayout.tsx              # Main application shell
│   │   ├── Sidebar.tsx                # Role-based navigation
│   │   ├── Header.tsx                 # User info, notifications
│   │   └── LoadingScreen.tsx          # Global loading state
│   ├── auth/
│   │   ├── ProtectedRoute.tsx         # Route protection
│   │   └── RoleGuard.tsx              # Component-level permissions
│   └── ui/                            # Reusable UI components
```

#### Role-Specific Components

##### Receptionist Components (`/receptionist`)
```
src/pages/receptionist/
├── ReceptionistDashboard.tsx          # Main dashboard
├── components/
│   ├── MainQueueDashboard.tsx         # Real-time queue display
│   ├── CheckInView.tsx                # Check-in process container
│   ├── PatientSearch.tsx              # Search existing patients
│   ├── NewPatientForm.tsx             # Register new patient
│   ├── CheckInForm.tsx                # Chief complaint entry
│   └── QueueStatusCard.tsx            # Patient status display
```

##### Nurse Components (`/nurse`)
```
src/pages/nurse/
├── NurseDashboard.tsx                 # Nurse main view
├── components/
│   ├── TriageQueueView.tsx            # Filtered patient queue
│   ├── VitalsForm.tsx                 # Comprehensive vitals entry
│   ├── PatientVitalsCard.tsx          # Individual patient card
│   ├── TriageResultDisplay.tsx        # AI triage results
│   └── VitalsValidation.tsx           # Real-time validation
```

##### Doctor Components (`/doctor`)
```
src/pages/doctor/
├── DoctorDashboard.tsx                # Personal patient queue
├── ConsultationView.tsx               # Main consultation workspace
├── components/
│   ├── PatientQueueCard.tsx           # Priority-sorted patient cards
│   ├── PreConsultationSummary.tsx     # Patient info review
│   ├── MakeAgentScribe.tsx            # AI scribe interface
│   ├── AudioRecorder.tsx              # Voice recording component
│   ├── TranscriptionDisplay.tsx       # Real-time transcription
│   ├── EntityExtraction.tsx           # Medical entity display
│   ├── NoteEditor.tsx                 # Rich text note editing
│   └── ConsultationHistory.tsx        # Previous consultations
```

##### Admin Components (`/admin`)
```
src/pages/admin/
├── AdminDashboard.tsx                 # System overview
├── components/
│   ├── SystemMetrics.tsx              # Performance metrics
│   ├── StaffManagement.tsx            # Doctor assignments
│   ├── QueueAnalytics.tsx             # Wait time analytics
│   ├── TriageDistribution.tsx         # Triage level charts
│   └── SystemHealth.tsx               # Backend status
```

### State Management Strategy

#### Zustand Stores
```typescript
// stores/authStore.ts - Authentication state
// stores/queueStore.ts - Real-time queue updates
// stores/consultationStore.ts - Active consultation state
// stores/uiStore.ts - UI preferences, sidebar state
```

#### React Query Integration
- **Automatic Caching**: 5-minute stale time for patient data
- **Background Refetching**: Keep data fresh without user interaction
- **Optimistic Updates**: Immediate UI updates with rollback on failure
- **Real-time Sync**: WebSocket integration for queue updates

### Real-Time Features

#### WebSocket Integration
```typescript
// hooks/useWebSocket.ts
export const useQueueUpdates = () => {
  const queryClient = useQueryClient();
  
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/queue');
    
    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);
      queryClient.setQueryData(queryKeys.queue.main(), update);
    };
    
    return () => ws.close();
  }, []);
};
```

#### Live Updates Strategy
- **Queue Changes**: Real-time patient status updates
- **Doctor Assignments**: Immediate notification of new patients
- **Triage Results**: Live triage level calculations
- **System Alerts**: Connection status, errors, maintenance

## 4. Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal**: Establish core architecture and authentication

#### Tasks:
- [ ] Set up enhanced Zustand auth store with role management
- [ ] Implement ProtectedRoute with role-based access control
- [ ] Create AppLayout with dynamic sidebar based on user role
- [ ] Build comprehensive API service layer with error handling
- [ ] Set up React Query with proper caching strategies
- [ ] Implement LoginPage with role-based redirects

#### Success Criteria:
- [ ] Users can log in and are redirected to appropriate dashboards
- [ ] Role-based access control prevents unauthorized route access
- [ ] API integration works with proper error handling
- [ ] Loading states and error boundaries function correctly

### Phase 2: Receptionist Flow (Week 2)
**Goal**: Complete receptionist workflow from check-in to queue management

#### Tasks:
- [ ] Build MainQueueDashboard with real-time updates
- [ ] Create CheckInView with patient search and registration
- [ ] Implement PatientSearch with autocomplete and filtering
- [ ] Build NewPatientForm with comprehensive validation
- [ ] Create CheckInForm for chief complaint entry
- [ ] Integrate with backend check-in API
- [ ] Add real-time queue status updates

#### Success Criteria:
- [ ] Receptionist can search and select existing patients
- [ ] New patient registration works with full validation
- [ ] Check-in process adds patients to queue with correct status
- [ ] Real-time dashboard shows all patients and their current status
- [ ] Error handling for duplicate check-ins and validation failures

### Phase 3: Nurse Flow (Week 3)
**Goal**: Complete nurse triage workflow with AI integration

#### Tasks:
- [ ] Build TriageQueueView showing only AWAITING_VITALS patients
- [ ] Create comprehensive VitalsForm with real-time validation
- [ ] Implement vital signs input with proper ranges and warnings
- [ ] Integrate with ManageAgent triage API
- [ ] Display AI-calculated triage results with confidence scores
- [ ] Handle automatic patient status transitions
- [ ] Add vitals history and trending

#### Success Criteria:
- [ ] Nurse sees only patients requiring vitals
- [ ] Vitals form validates input ranges and shows warnings
- [ ] AI triage calculation works with proper error handling
- [ ] Patient automatically moves to doctor assignment after vitals
- [ ] Triage results display with clear reasoning and confidence

### Phase 4: Doctor Flow (Week 4)
**Goal**: Complete doctor consultation workflow with AI scribe

#### Tasks:
- [ ] Build DoctorDashboardView with personal patient queue
- [ ] Create ConsultationView with pre-consultation summary
- [ ] Implement PreConsultationSummary showing patient history
- [ ] Build MakeAgentScribe with full AI pipeline integration
- [ ] Create AudioRecorder with real-time feedback
- [ ] Implement TranscriptionDisplay with live updates
- [ ] Build EntityExtraction display with medical terminology
- [ ] Create NoteEditor with rich text editing capabilities
- [ ] Add consultation completion and note finalization

#### Success Criteria:
- [ ] Doctor sees only their assigned patients in priority order
- [ ] Pre-consultation summary shows complete patient context
- [ ] Audio recording works with proper file handling
- [ ] AI transcription, entity extraction, and note generation function
- [ ] Doctor can edit and finalize AI-generated notes
- [ ] Consultation completion updates patient status correctly

### Phase 5: Admin Dashboard (Week 5)
**Goal**: Complete administrative oversight and system monitoring

#### Tasks:
- [ ] Build AdminDashboard with comprehensive system overview
- [ ] Create SystemMetrics showing performance indicators
- [ ] Implement StaffManagement for doctor assignments
- [ ] Build QueueAnalytics with wait time analysis
- [ ] Create TriageDistribution charts and statistics
- [ ] Add SystemHealth monitoring for backend services
- [ ] Implement real-time alerts and notifications

#### Success Criteria:
- [ ] Admin can view complete hospital floor status
- [ ] System metrics show accurate performance data
- [ ] Staff management allows manual doctor assignments
- [ ] Queue analytics provide actionable insights
- [ ] System health monitoring alerts to issues

### Phase 6: Integration & Polish (Week 6)
**Goal**: End-to-end testing, performance optimization, and production readiness

#### Tasks:
- [ ] Comprehensive end-to-end testing of all user workflows
- [ ] Performance optimization and bundle size reduction
- [ ] Accessibility audit and WCAG 2.1 AA compliance
- [ ] Error boundary implementation and error handling
- [ ] Loading state optimization and skeleton screens
- [ ] Mobile responsiveness testing and optimization
- [ ] Production build optimization and deployment preparation

#### Success Criteria:
- [ ] All user workflows function correctly end-to-end
- [ ] Application loads quickly on standard hospital hardware
- [ ] Accessibility requirements met for all user types
- [ ] Error handling provides clear user feedback
- [ ] Mobile and tablet interfaces work properly
- [ ] Production build is optimized and deployment-ready

## 5. Technical Specifications

### Performance Requirements
- **Initial Load**: < 3 seconds on hospital hardware
- **Route Transitions**: < 500ms between pages
- **API Responses**: < 2 seconds for standard operations
- **Real-time Updates**: < 1 second latency for queue changes
- **Bundle Size**: < 1MB gzipped for initial load

### Browser Support
- **Primary**: Chrome 90+, Firefox 88+, Safari 14+
- **Secondary**: Edge 90+
- **Mobile**: iOS Safari 14+, Chrome Mobile 90+

### Accessibility Requirements
- **WCAG 2.1 AA**: Full compliance for all components
- **Keyboard Navigation**: Complete keyboard accessibility
- **Screen Readers**: ARIA labels and semantic HTML
- **Color Contrast**: Minimum 4.5:1 ratio for all text
- **Focus Management**: Clear focus indicators and logical tab order

### Security Considerations
- **JWT Token Management**: Secure storage and automatic refresh
- **Input Validation**: Client-side validation with server verification
- **XSS Protection**: Sanitized user inputs and CSP headers
- **HTTPS Only**: All API communications over HTTPS
- **Role-Based Access**: Strict component and route protection

## 6. File Structure

```
src/
├── components/
│   ├── layout/
│   │   ├── AppLayout.tsx
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   └── LoadingScreen.tsx
│   ├── auth/
│   │   ├── ProtectedRoute.tsx
│   │   └── RoleGuard.tsx
│   ├── ui/
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── input.tsx
│   │   ├── toast.tsx
│   │   └── [other-ui-components].tsx
│   └── shared/
│       ├── PatientCard.tsx
│       ├── StatusBadge.tsx
│       ├── PriorityIndicator.tsx
│       └── LoadingSpinner.tsx
├── pages/
│   ├── LoginPage.tsx
│   ├── NotFoundPage.tsx
│   ├── receptionist/
│   │   ├── ReceptionistDashboard.tsx
│   │   └── components/
│   ├── nurse/
│   │   ├── NurseDashboard.tsx
│   │   └── components/
│   ├── doctor/
│   │   ├── DoctorDashboard.tsx
│   │   ├── ConsultationView.tsx
│   │   └── components/
│   └── admin/
│       ├── AdminDashboard.tsx
│       └── components/
├── hooks/
│   ├── useAuth.ts
│   ├── useQueue.ts
│   ├── useWebSocket.ts
│   ├── useAudioRecorder.ts
│   └── useLocalStorage.ts
├── services/
│   ├── api.ts
│   ├── websocket.ts
│   ├── audio.ts
│   └── storage.ts
├── stores/
│   ├── authStore.ts
│   ├── queueStore.ts
│   ├── consultationStore.ts
│   └── uiStore.ts
├── types/
│   ├── auth.ts
│   ├── patient.ts
│   ├── consultation.ts
│   ├── queue.ts
│   └── api.ts
├── utils/
│   ├── constants.ts
│   ├── formatters.ts
│   ├── validators.ts
│   └── helpers.ts
└── lib/
    ├── queryClient.ts
    ├── utils.ts
    └── cn.ts
```

## 7. Success Criteria & Testing

### Functional Requirements
- [ ] **Authentication**: Users can log in and are routed based on role
- [ ] **Receptionist Flow**: Can check in patients who appear in nurse queue
- [ ] **Nurse Flow**: Can only see awaiting vitals, submit vitals, patient moves to doctor
- [ ] **Doctor Flow**: Can only see assigned patients, use AI scribe successfully
- [ ] **Admin Flow**: Can view complete system status and manage assignments
- [ ] **Authorization**: Users cannot access unauthorized routes or components

### Performance Requirements
- [ ] **Load Time**: Application loads in < 3 seconds
- [ ] **Responsiveness**: UI responds to interactions in < 500ms
- [ ] **Real-time**: Queue updates appear in < 1 second
- [ ] **Offline**: Application shows appropriate offline states
- [ ] **Memory**: No memory leaks during extended use

### User Experience Requirements
- [ ] **Intuitive**: New users can complete workflows without training
- [ ] **Accessible**: All functionality available via keyboard and screen reader
- [ ] **Responsive**: Works properly on tablets and mobile devices
- [ ] **Error Handling**: Clear error messages with recovery suggestions
- [ ] **Loading States**: Appropriate loading indicators for all operations

### Technical Requirements
- [ ] **Type Safety**: Full TypeScript coverage with no `any` types
- [ ] **Code Quality**: ESLint and Prettier configured and passing
- [ ] **Bundle Size**: Optimized bundle size < 1MB gzipped
- [ ] **Browser Support**: Works in all specified browsers
- [ ] **Security**: No security vulnerabilities in dependencies

## 8. Deployment & Maintenance

### Build Configuration
```json
{
  "scripts": {
    "dev": "vite --host 0.0.0.0 --port 5173",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "lint:fix": "eslint . --ext ts,tsx --fix",
    "type-check": "tsc --noEmit",
    "test": "vitest",
    "test:ui": "vitest --ui"
  }
}
```

### Environment Configuration
```typescript
// .env.development
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws
VITE_ENVIRONMENT=development

// .env.production
VITE_API_BASE_URL=https://api.medios.hospital/api/v1
VITE_WS_URL=wss://api.medios.hospital/ws
VITE_ENVIRONMENT=production
```

### Production Optimizations
- **Code Splitting**: Route-based code splitting for optimal loading
- **Tree Shaking**: Remove unused code from final bundle
- **Asset Optimization**: Image compression and lazy loading
- **Caching Strategy**: Proper cache headers for static assets
- **CDN Integration**: Static asset delivery via CDN

This comprehensive plan provides a roadmap for building a production-ready, role-based healthcare management frontend that integrates seamlessly with the MediOS backend while providing an exceptional user experience for all hospital staff roles.