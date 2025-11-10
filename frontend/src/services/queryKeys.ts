// Query Keys Factory for React Query
export const queryKeys = {
  // Authentication
  auth: {
    all: ['auth'] as const,
    user: () => [...queryKeys.auth.all, 'user'] as const,
  },

  // Patients
  patients: {
    all: ['patients'] as const,
    lists: () => [...queryKeys.patients.all, 'list'] as const,
    list: (filters?: Record<string, unknown>) => [...queryKeys.patients.lists(), { filters }] as const,
    details: () => [...queryKeys.patients.all, 'detail'] as const,
    detail: (id: number) => [...queryKeys.patients.details(), id] as const,
    search: (query: string) => [...queryKeys.patients.all, 'search', query] as const,
  },

  // Consultations
  consultations: {
    all: ['consultations'] as const,
    lists: () => [...queryKeys.consultations.all, 'list'] as const,
    list: (filters?: Record<string, unknown>) => [...queryKeys.consultations.lists(), { filters }] as const,
    details: () => [...queryKeys.consultations.all, 'detail'] as const,
    detail: (id: number) => [...queryKeys.consultations.details(), id] as const,
    patient: (patientId: number) => [...queryKeys.consultations.all, 'patient', patientId] as const,
    doctor: (doctorId: number) => [...queryKeys.consultations.all, 'doctor', doctorId] as const,
    notes: (id: number) => [...queryKeys.consultations.detail(id), 'notes'] as const,
  },

  // Queue Management
  queue: {
    all: ['queue'] as const,
    main: () => [...queryKeys.queue.all, 'main'] as const,
    triage: () => [...queryKeys.queue.all, 'triage'] as const,
    summary: () => [...queryKeys.queue.all, 'summary'] as const,
  },

  // Vitals and Triage
  vitals: {
    all: ['vitals'] as const,
    consultation: (consultationId: number) => [...queryKeys.vitals.all, 'consultation', consultationId] as const,
    history: (patientId: number) => [...queryKeys.vitals.all, 'history', patientId] as const,
  },

  // AI Processing
  ai: {
    all: ['ai'] as const,
    transcription: (consultationId: number) => [...queryKeys.ai.all, 'transcription', consultationId] as const,
    entities: (consultationId: number) => [...queryKeys.ai.all, 'entities', consultationId] as const,
    notes: (consultationId: number) => [...queryKeys.ai.all, 'notes', consultationId] as const,
    status: () => [...queryKeys.ai.all, 'status'] as const,
  },

  // Notes
  notes: {
    all: ['notes'] as const,
    patient: (patientId: number) => [...queryKeys.notes.all, 'patient', patientId] as const,
    detail: (noteId: number) => [...queryKeys.notes.all, 'detail', noteId] as const,
  },
} as const;