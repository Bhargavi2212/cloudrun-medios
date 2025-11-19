// User types
export interface User {
  id: string
  email: string
  first_name?: string
  last_name?: string
  full_name: string
  role: UserRole
  roles?: string[]
  phone?: string
}

export type UserRole = 'ADMIN' | 'DOCTOR' | 'NURSE' | 'RECEPTIONIST'

export interface StandardResponse<T = unknown> {
  success: boolean
  data?: T
  error?: string
  warning?: string
  is_stub?: boolean
}

// Queue / manage-agent types
export type QueueStage =
  | 'waiting'
  | 'triage'
  | 'scribe'
  | 'discharge'

export interface QueuePatient {
  queue_state_id: string
  consultation_id: string
  patient_id: string
  patient_name: string
  age: number | null
  chief_complaint: string | null
  triage_level: number | null
  status: QueueStage
  wait_time_minutes: number
  estimated_wait_minutes: number | null
  queue_position?: number | null
  confidence_level?: 'high' | 'medium' | 'low' | null
  assigned_doctor?: string | null
  assigned_doctor_id?: string | null
  priority_score?: number | null
  prediction_method?: string | null
  check_in_time?: string
  vitals?: Record<string, unknown> | null
}

export interface ManageQueueResponse {
  patients: QueuePatient[]
  total_count: number
  average_wait_time: number
  triage_distribution: Record<number, number>
}

export interface PatientSummary {
  subject_id: number
  summary_markdown: string
  timeline: Record<string, unknown>
  metrics: Record<string, unknown>
  cached: boolean
}

export interface SummarizerHealth {
  enabled: boolean
  is_stub: boolean
}

export interface UploadedAudio {
  audio_id: string
  storage_path: string
  mime_type?: string | null
  size_bytes?: number | null
  signed_url?: string | null
}

// Patient demographics (from patients table)
export interface Patient {
  id: string
  mrn: string
  first_name?: string
  last_name?: string
  date_of_birth?: string
  sex?: 'M' | 'F' | 'Other'
  contact_phone?: string | null
  contact_email?: string | null
  created_at?: string
  updated_at?: string
}

export type PatientSearchResult = Patient

export interface CreatePatientRequest {
  first_name?: string
  last_name?: string
  date_of_birth?: string
  sex?: 'M' | 'F' | 'Other'
  contact_phone?: string | null
  contact_email?: string | null
  mrn?: string
}

// Queue patient (legacy usage)
export interface QueueData {
  patients: QueuePatient[]
  total_count: number
  high_priority_count: number
  average_wait_time: number
  assigned_count: number
}

export interface CheckInRequest {
  patient_id: string
  chief_complaint: string
  priority_level?: number
}

// Vitals types
export interface VitalsSubmission {
  heart_rate: number
  blood_pressure_systolic: number
  blood_pressure_diastolic: number
  respiratory_rate: number
  temperature_celsius: number
  oxygen_saturation: number
  weight_kg?: number
  pain_level?: number
}

export interface TriageResult {
  triage_level: number
  confidence: number
  reasoning: string
  priority_score: number
}

export interface ConsultationRecord {
  id: string
  original_filename?: string | null
  content_type?: string | null
  size_bytes?: number | null
  description?: string | null
  uploaded_at: string
  uploaded_by?: string | null
  signed_url?: string | null
  download_url: string
  status: string
  document_type?: string | null
  confidence?: number | null
  needs_review: boolean
  processed_at?: string | null
  processing_notes?: string | null
  processing_metadata?: Record<string, unknown> | null
  timeline_event_ids?: string[]
}

export interface ExtractEntitiesResponse {
  entities: Record<string, unknown>
  confidence?: number
}

export interface GeneratedNoteResponse {
  note: string
  confidence: number
  note_version_id?: string | null
}

export interface ProcessedAudioResult {
  note_saved?: boolean
  note_save_error?: string
  transcription: string
  entities: Record<string, unknown>
  generated_note: string
  confidence_scores?: Record<string, number>
  warnings?: string[]
  errors?: string[]
  stage_completed?: string
  is_stub?: boolean
}

// Consultation types
export interface Consultation {
  id: string
  patient_id: string
  chief_complaint?: string | null
  triage_level?: number | null
  status?: QueueStage
  assigned_doctor_id?: string | null
  created_at?: string
  notes?: string
  updated_at?: string
}

export interface DocumentStatus {
  file_id: string
  status: string
  processed_at?: string | null
  confidence?: number | null
  needs_review: boolean
  processing_notes?: string | null
  timeline_event_ids: string[]
  metadata: Record<string, unknown>
}

export interface TimelineEventEntry {
  id: string
  patient_id: string
  consultation_id?: string | null
  source_type?: string | null
  source_file_asset_id?: string | null
  event_type: string
  status: string
  title: string
  summary: string
  confidence?: number | null
  extraction_confidence?: number | null
  extraction_metadata?: Record<string, unknown>
  doctor_verified?: boolean
  verified_at?: string | null
  verified_by?: string | null
  event_date?: string | null
  data?: Record<string, unknown>
  notes?: string | null
}

export interface TimelineSummary {
  summary_id: string
  summary: string
  timeline: {
    patient_id: string
    generated_at: string
    events: TimelineEventEntry[]
    highlights?: string[]
    confidence?: number | null
  }
  highlights: string[]
  confidence?: number | null
  cached: boolean
  generated_at: string
  model?: string | null
  token_usage: Record<string, number>
}

export interface ScribeSession {
  id: string
  consultation_id?: string
  patient_id?: string
  status: string
  language?: string
  started_at?: string | null
  ended_at?: string | null
  transcript_snapshot?: string | null
  created_at: string
  updated_at: string
}

export interface ScribeSegment {
  id: string
  session_id: string
  speaker_label?: string | null
  text: string
  start_ms?: number | null
  end_ms?: number | null
  confidence?: number | null
  created_at: string
}

export interface ScribeVital {
  id: number
  session_id: string
  recorded_at: string
  source: string
  heart_rate?: number | null
  respiratory_rate?: number | null
  systolic_bp?: number | null
  diastolic_bp?: number | null
  temperature_c?: number | null
  oxygen_saturation?: number | null
  pain_score?: number | null
}

export interface SoapNoteContent {
  subjective?: Record<string, unknown>
  objective?: Record<string, unknown>
  assessment?: Record<string, unknown>
  plan?: Record<string, unknown>
  entities?: Record<string, unknown>
}

export interface SoapNote {
  id: string
  session_id: string
  consultation_id?: string | null
  status: string
  model_name?: string | null
  content?: SoapNoteContent | null
  raw_markdown?: string | null
  confidence?: Record<string, unknown> | null
  created_at: string
  updated_at: string
}

export interface TriagePredictionSnapshot {
  id: number
  esi_level: number
  probability: number
  probabilities?: Record<string, number>
  flagged: boolean
  created_at: string
}

export interface ScribeSessionDetails {
  session: ScribeSession
  segments: ScribeSegment[]
  vitals: ScribeVital[]
  notes: SoapNote[]
  triage_predictions: TriagePredictionSnapshot[]
}