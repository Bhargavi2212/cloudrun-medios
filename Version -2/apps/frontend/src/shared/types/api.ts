export interface Patient {
  id: string;
  mrn: string;
  first_name: string;
  last_name: string;
  dob?: string | null;
  sex?: string | null;
  contact_info?: Record<string, unknown> | null;
  created_at?: string;
  updated_at?: string;
}

export interface TriageRequestPayload {
  hr: number;
  rr: number;
  sbp: number;
  dbp: number;
  temp_c: number;
  spo2: number;
  pain: number;
}

export interface TriageResponsePayload {
  acuity_level: number;
  model_version: string;
  explanation: string;
}

export interface TranscriptPayload {
  encounter_id: string;
  transcript: string;
  speaker_segments?: Array<{ speaker: string; content: string }>;
  source?: string;
}

export interface SoapNoteResponse {
  id: string;
  encounter_id: string;
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
  model_version: string;
  confidence_score?: number;
  created_at: string;
  updated_at: string;
}

export interface PortableTimelineEvent {
  event_type: string;
  encounter_id: string;
  timestamp: string;
  content: Record<string, unknown>;
  source?: "local" | "federated" | null;
  source_hospital_id?: string | null;
}

export interface PortableSummary {
  id: string;
  encounter_ids: string[];
  summary_text: string;
  structured_data?: StructuredTimelineData | null;
  model_version?: string | null;
  confidence_score?: number | null;
  created_at: string;
}

export interface StructuredTimelineData {
  patient: {
    id: string;
    name: string;
    age: number | null;
    patient_id: string;
  };
  alerts: {
    allergies: string[];
    chronic_conditions: string[];
    recent_events: string[];
    warnings: string[];
  };
  timeline: TimelineEntry[];
  total_entries: number;
  years_of_history: number;
  last_updated: string;
}

export interface TimelineEntry {
  id: string;
  date: string;
  type: "visit" | "lab" | "procedure" | "document" | "discharge";
  title: string;
  source: {
    type: "ai_scribe" | "uploaded_pdf" | "uploaded_image" | "manual_entry";
    original_file?: string | null;
    confidence: number;
    reviewed_by?: string | null;
    reviewed_at?: string | null;
  };
  data: {
    chief_complaint?: string;
    vitals?: {
      hr?: number;
      bp?: string;
      temp?: number;
      rr?: number;
      o2?: number;
    };
    rfv?: string;
    tests?: string[];
    diagnosis?: string;
    medications?: string[];
    plan?: string;
    disposition?: string;
    subjective?: string;
    objective?: string;
    assessment?: string;
    document_type?: string;
    file_name?: string;
  };
  expanded: boolean;
  can_view_original: boolean;
}

export interface PortableProfile {
  patient: Patient;
  timeline: PortableTimelineEvent[];
  summaries: PortableSummary[];
  sources: string[];
}

export interface GlobalModelResponse {
  model_name: string;
  round_id: number;
  weights: Record<string, number[]>;
  contributor_count: number;
}

