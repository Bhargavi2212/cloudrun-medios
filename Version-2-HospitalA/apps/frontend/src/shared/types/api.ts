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
}

export interface PortableSummary {
  id: string;
  encounter_ids: string[];
  summary_text: string;
  model_version?: string | null;
  confidence_score?: number | null;
  created_at: string;
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

