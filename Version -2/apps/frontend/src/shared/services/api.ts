import axios, { AxiosError, AxiosResponse, AxiosInstance } from "axios";
import type { User as AuthUser } from "../types/auth";
import type { PortableProfile } from "../types/api";
import { getCurrentHospital } from "../config/hospitalConfig";

/**
 * Create an axios client instance with the given base URL.
 */
const createClient = (baseURL: string): AxiosInstance => {
  return axios.create({
    baseURL,
    headers: {
      "Content-Type": "application/json",
    },
  });
};

/**
 * Get API clients based on the currently selected hospital.
 * These instances are recreated when the hospital changes.
 */
export const getApiClients = () => {
  const hospital = getCurrentHospital();
  return {
    manageApi: createClient(hospital.apiUrls.manage),
    scribeApi: createClient(hospital.apiUrls.scribe),
    summarizerApi: createClient(hospital.apiUrls.summarizer),
    dolApi: createClient(hospital.apiUrls.dol),
    federationApi: createClient(hospital.apiUrls.federation),
  };
};

// Legacy exports for backward compatibility - these will use current hospital
// Note: These are recreated on each access, so they always use the current hospital
export const getManageApi = () => getApiClients().manageApi;
export const getScribeApi = () => getApiClients().scribeApi;
export const getSummarizerApi = () => getApiClients().summarizerApi;
export const getDolApi = () => getApiClients().dolApi;
export const getFederationApi = () => getApiClients().federationApi;

// For backward compatibility, export functions that return the current instances
// These will be used by existing code until we update all call sites
export const manageApi = new Proxy({} as AxiosInstance, {
  get: (_, prop) => {
    const api = getApiClients().manageApi;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const value = (api as any)[prop];
    return typeof value === "function" ? value.bind(api) : value;
  },
});

export const scribeApi = new Proxy({} as AxiosInstance, {
  get: (_, prop) => {
    const api = getApiClients().scribeApi;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const value = (api as any)[prop];
    return typeof value === "function" ? value.bind(api) : value;
  },
});

export const summarizerApi = new Proxy({} as AxiosInstance, {
  get: (_, prop) => {
    const api = getApiClients().summarizerApi;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const value = (api as any)[prop];
    return typeof value === "function" ? value.bind(api) : value;
  },
});

export const dolApi = new Proxy({} as AxiosInstance, {
  get: (_, prop) => {
    const api = getApiClients().dolApi;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const value = (api as any)[prop];
    return typeof value === "function" ? value.bind(api) : value;
  },
});

export const federationApi = new Proxy({} as AxiosInstance, {
  get: (_, prop) => {
    const api = getApiClients().federationApi;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const value = (api as any)[prop];
    return typeof value === "function" ? value.bind(api) : value;
  },
});

// Standard response wrapper
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export interface StandardResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  warning?: string;
}

// Queue types
export interface QueuePatient {
  queue_state_id: string;
  consultation_id: string;
  patient_id: string;
  patient_name: string;
  age: number | null;
  chief_complaint: string | null;
  triage_level: number | null;
  status: string;
  wait_time_minutes: number;
  estimated_wait_minutes: number | null;
  queue_position: number | null;
  confidence_level: string | null;
  assigned_doctor: string | null;
  assigned_doctor_id: string | null;
  priority_score: number | null;
  prediction_method: string | null;
  check_in_time: string | null;
  vitals: Record<string, unknown> | null;
}

export interface ManageQueueResponse {
  patients: QueuePatient[];
  total_count: number;
  average_wait_time: number;
  triage_distribution: Record<number, number>;
}

// Patient types
export interface Patient {
  id: string;
  mrn: string;
  first_name: string;
  last_name: string;
  dob: string | null;
  sex: string | null;
  contact_info: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface CheckInRequest {
  patient_id: string;
  chief_complaint: string;
  injury?: boolean;
  ambulance_arrival?: boolean;
  seen_72h?: boolean;
}

export interface NurseVitalsPayload {
  hr: number;
  rr: number;
  sbp: number;
  dbp: number;
  temp_c: number;
  spo2: number;
  pain: number;
  notes?: string;
}

export interface NurseVitalsResponse {
  encounter_id: string;
  patient_id: string;
  triage_level: number;
  model_version: string;
  explanation: string;
  vitals: Record<string, number>;
}

export interface CreatePatientRequest {
  mrn?: string;
  first_name: string;
  last_name: string;
  dob?: string;
  sex?: string;
  contact_info?: Record<string, unknown>;
}

// Auth types
export interface LoginRequest {
  email: string;
  password: string;
}

export type User = AuthUser;

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  user: User;
}

// Helper to unwrap responses that may or may not use the StandardResponse envelope
type ApiResponse<T> = StandardResponse<T> | T;

const unwrap = <T>(response: AxiosResponse<ApiResponse<T>>): T => {
  const payload = response.data as ApiResponse<T>;

  if (payload && typeof payload === "object" && "success" in payload) {
    const standardPayload = payload as StandardResponse<T>;
    if (!standardPayload.success) {
      throw new Error(standardPayload.error || "Request failed");
    }
    return (standardPayload.data ?? null) as T;
  }

  return payload as T;
};

// Auth API
export const authAPI = {
  login: (email: string, password: string) =>
    manageApi
      .post<StandardResponse<TokenResponse>>("/auth/login", { email, password })
      .then(unwrap),
  getCurrentUser: () =>
    manageApi
      .get<StandardResponse<User>>("/auth/me")
      .then(unwrap),
};

// Manage API
export const manageAPI = {
  getQueue: () =>
    manageApi
      .get<StandardResponse<ManageQueueResponse>>("/manage/queue")
      .then(unwrap)
      .catch((error: AxiosError) => {
        if (error.response?.status === 404) {
          return {
            patients: [],
            total_count: 0,
            average_wait_time: 0,
            triage_distribution: {},
          } as ManageQueueResponse;
        }
        throw error;
      }),
  checkInPatient: (payload: CheckInRequest) =>
    manageApi
      .post<StandardResponse<{ encounter_id: string; triage_level: number; profile?: PortableProfile; dol_profile_found?: boolean }>>("/manage/check-in", payload)
      .then(unwrap),
  searchPatients: (query: string) =>
    manageApi
      .get<StandardResponse<Patient[]>>("/manage/patients/search", {
        params: { q: query },
      })
      .then(unwrap),
  createPatient: (payload: CreatePatientRequest) =>
    manageApi
      .post<StandardResponse<Patient>>("/manage/patients", payload)
      .then(unwrap),
  recordVitals: (encounterId: string, payload: NurseVitalsPayload) =>
    manageApi
      .post<StandardResponse<NurseVitalsResponse>>(`/manage/encounters/${encounterId}/vitals`, payload)
      .then(unwrap),
};
