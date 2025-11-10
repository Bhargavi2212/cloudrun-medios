import axios, { AxiosError, AxiosResponse } from 'axios'
import { useAuthStore } from '@/store/authStore'
import type {
  CheckInRequest,
  ManageQueueResponse,
  VitalsSubmission,
  TriageResult,
  PatientSearchResult,
  CreatePatientRequest,
  Patient,
  StandardResponse,
  PatientSummary,
  SummarizerHealth,
  ExtractEntitiesResponse,
  GeneratedNoteResponse,
  UploadedAudio,
  ProcessedAudioResult,
  QueueStage,
  ConsultationRecord,
  DocumentStatus,
  TimelineSummary,
} from '@/types'

const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
})

type RequestConfig = Parameters<typeof api.interceptors.request.use>[0]
type ResponseErrorHandler = (error: AxiosError) => Promise<never>

const requestHandler: RequestConfig = (config) => {
  const token = useAuthStore.getState().token
  if (token) {
    config.headers = config.headers ?? {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
}

const requestErrorHandler: ResponseErrorHandler = (error) => Promise.reject(error)

const responseHandler = (response: AxiosResponse) => response

interface RetryableAxiosRequestConfig {
  _retry?: boolean
  headers?: Record<string, string>
  [key: string]: unknown
}

const responseErrorHandler: ResponseErrorHandler = async (error) => {
  const originalRequest = error.config as RetryableAxiosRequestConfig | undefined
  if (error.response?.status === 401 && originalRequest && !originalRequest._retry) {
    const store = useAuthStore.getState()
    const refreshed = await store.tryRefresh()
    if (refreshed) {
      originalRequest._retry = true
      originalRequest.headers = originalRequest.headers ?? {}
      originalRequest.headers.Authorization = `Bearer ${useAuthStore.getState().token}`
      return api(originalRequest as Parameters<typeof api>[0])
    }
    store.logout()
  }
  return Promise.reject(error)
}

api.interceptors.request.use(requestHandler, requestErrorHandler)
api.interceptors.response.use(responseHandler, responseErrorHandler)

const unwrap = <T>(response: AxiosResponse<StandardResponse<T>>): T => {
  const payload = response.data
  if (!payload.success) {
    throw new Error(payload.error || 'Request failed')
  }
  return (payload.data ?? null) as T
}

interface RegisterPayload {
  email: string
  password: string
  first_name?: string
  last_name?: string
  roles: string[]
}

export const authAPI = {
  login: (email: string, password: string) =>
    api.post('/auth/login', { email, password }),
  getCurrentUser: () => api.get('/auth/me'),
  register: (payload: RegisterPayload) => api.post('/auth/register', payload),
  refresh: (refreshToken: string) =>
    api.post('/auth/refresh', { refresh_token: refreshToken }),
  logout: (refreshToken: string) =>
    api.post('/auth/logout', { refresh_token: refreshToken }),
  updateProfile: (payload: { first_name?: string; last_name?: string; phone?: string }) =>
    api.put('/auth/me', payload),
  changePassword: (payload: { current_password: string; new_password: string }) =>
    api.post('/auth/change-password', payload),
  forgotPassword: (email: string) =>
    api.post('/auth/forgot-password', { email }),
  resetPassword: (resetToken: string, newPassword: string) =>
    api.post('/auth/reset-password', { reset_token: resetToken, new_password: newPassword }),
}

export const manageAPI = {
  getQueue: () =>
    api
      .get<ManageQueueResponse>('/manage/queue')
      .then((res) => res.data)
      .catch((error: AxiosError) => {
        if (error.response?.status === 404) {
          return {
            patients: [],
            total_count: 0,
            average_wait_time: 0,
            triage_distribution: {},
          } satisfies ManageQueueResponse
        }
        throw error
      }),
  downloadDocument: async (fileId: string): Promise<Blob> => {
    const response = await api.get(`/manage/records/${fileId}/download`, {
      responseType: 'blob',
    })
    return response.data as Blob
  },
  getQueueSummary: () =>
    api.get('/manage/queue/summary').then((res) => res.data),
  checkInPatient: (payload: CheckInRequest) =>
    api
      .post<StandardResponse<unknown>>('/manage/check-in', {
        patient_id: payload.patient_id,
        chief_complaint: payload.chief_complaint,
      })
      .then(unwrap),
  submitVitals: (consultationId: string, payload: VitalsSubmission) =>
    api
      .post<TriageResult>(`/manage/consultations/${consultationId}/vitals`, payload)
      .then((res) => res.data),
  uploadConsultationRecords: (consultationId: string, files: File[], notes?: string) => {
    const formData = new FormData()
    files.forEach((file) => formData.append('files', file))
    if (notes) {
      formData.append('notes', notes)
    }
    return api
      .post<StandardResponse<{ records: ConsultationRecord[] }>>(
        `/manage/consultations/${consultationId}/records`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } },
      )
      .then(unwrap)
  },
  getConsultationRecords: (consultationId: string) =>
    api
      .get<StandardResponse<{ records: ConsultationRecord[] }>>(
        `/manage/consultations/${consultationId}/records`,
      )
      .then(unwrap),
  getRecordStatus: (fileId: string) =>
    api
      .get<StandardResponse<DocumentStatus>>(`/manage/records/${fileId}/status`)
      .then(unwrap),
  reviewConsultationRecord: (
    fileId: string,
    payload: { resolution: 'approved' | 'needs_review' | 'failed'; notes?: string; update_timeline?: boolean },
  ) =>
    api
      .post<StandardResponse<{ record: ConsultationRecord }>>(
        `/manage/records/${fileId}/review`,
        payload,
      )
      .then((res) => unwrap(res).record),
  getTimelineSummary: (
    patientId: string,
    options?: { forceRefresh?: boolean; visitLimit?: number },
  ) => {
    const params: Record<string, unknown> = {}
    if (options?.forceRefresh) {
      params.force_refresh = options.forceRefresh
    }
    if (options?.visitLimit !== undefined) {
      params.visit_limit = options.visitLimit
    }
    return api
      .get<StandardResponse<TimelineSummary>>(`/manage/patients/${patientId}/timeline`, {
        params,
      })
      .then(unwrap)
  },
}

export const queueAPI = {
  assignQueueState: (queueStateId: string, assignedTo: string) =>
    api
      .post<StandardResponse>(
        `/queue/${queueStateId}/assign`,
        { assigned_to: assignedTo },
      )
      .then(unwrap),
  advanceQueueState: (
    queueStateId: string,
    payload: { next_stage: QueueStage; priority_level?: number; notes?: string },
  ) =>
    api
      .post<StandardResponse>(
        `/queue/${queueStateId}/advance`,
        {
          next_stage: payload.next_stage,
          priority_level: payload.priority_level,
          notes: payload.notes,
        },
      )
      .then(unwrap),
}

const callMakeAgentEndpoint = async <T>(
  path: string,
  payload: unknown,
  method: 'post' | 'put' = 'post',
) => {
  const candidates = ['/make-agent', '/make', '/make_agent', '']
  let lastError: unknown = null

  for (const base of candidates) {
    const url = `${base}/${path}`.replace(/\/+/g, '/')
    try {
      const response =
        method === 'post'
          ? await api.post<StandardResponse<T>>(url, payload)
          : await api.put<StandardResponse<T>>(url, payload)
      return unwrap(response)
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 404) {
        lastError = error
        continue
      }
      throw error
    }
  }

  throw lastError ?? new Error(`make-agent endpoint '${path}' is unavailable`)
}

export const summarizerAPI = {
  getHealth: () =>
    api.get<StandardResponse<SummarizerHealth>>('/summarizer/health').then(unwrap),
  getSummary: (
    subjectId: number,
    options?: { visitLimit?: number; forceRefresh?: boolean },
  ) => {
    const params: Record<string, unknown> = {}
    if (options?.visitLimit !== undefined) {
      params.visit_limit = options.visitLimit
    }
    if (options?.forceRefresh) {
      params.force_refresh = true
    }

    return api
      .get<StandardResponse<PatientSummary>>(`/summarizer/${subjectId}`, {
        params,
      })
      .then(unwrap)
  },
}

export const scribeAPI = {
  extractEntities: (transcript: string) =>
    callMakeAgentEndpoint<ExtractEntitiesResponse>('extract_entities', { transcript }),
  generateNote: (
    transcript: string,
    entities: Record<string, unknown>,
    consultationId?: string,
  ) =>
    callMakeAgentEndpoint<GeneratedNoteResponse>('generate_note', {
      transcript,
      entities,
      consultation_id: consultationId,
    }),
  updateNote: (noteId: string, content: string) =>
    callMakeAgentEndpoint<{ note_id: string; content: string }>(
      'update_note',
      { note_id: noteId, content },
      'put',
    ),
  uploadAudio: (file: File, options?: { consultationId?: string }) => {
    const formData = new FormData()
    formData.append('audio_file', file)
    if (options?.consultationId) {
      formData.append('consultation_id', options.consultationId)
    }
    return api
      .post<StandardResponse<UploadedAudio>>('/make-agent/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then(unwrap)
  },
  processAudio: (audioId: string, options?: { variant?: 'default' | 'medi-os' }) => {
    const path = options?.variant === 'medi-os' ? 'medi-os/process' : 'process'
    return callMakeAgentEndpoint<ProcessedAudioResult>(path, { audio_id: audioId })
  },
  // Get note for a consultation
  getConsultationNote: (consultationId: string) =>
    api
      .get<StandardResponse<{
        note_id: string
        consultation_id: string
        status: string
        content: string
        entities: Record<string, unknown>
        is_ai_generated: boolean
        created_at: string | null
        version_id: string
      }>>(`/make-agent/consultations/${consultationId}/note`)
      .then(unwrap),
  // Update note for a consultation
  updateConsultationNote: (consultationId: string, content: string) =>
    api
      .put<StandardResponse<{
        note_id: string
        version_id: string
        content: string
        updated_at: string | null
      }>>(`/make-agent/consultations/${consultationId}/note`, { content })
      .then(unwrap),
  // Submit note for approval
  submitNoteForApproval: (consultationId: string) =>
    api
      .post<StandardResponse<{
        note_id: string
        status: string
        message: string
      }>>(`/make-agent/consultations/${consultationId}/note/submit`)
      .then(unwrap),
  // Approve note
  approveNote: (consultationId: string) =>
    api
      .post<StandardResponse<{
        note_id: string
        status: string
        message: string
      }>>(`/make-agent/consultations/${consultationId}/note/approve`)
      .then(unwrap),
  // Reject note
  rejectNote: (consultationId: string, rejectionReason?: string) =>
    api
      .post<StandardResponse<{
        note_id: string
        status: string
        message: string
      }>>(`/make-agent/consultations/${consultationId}/note/reject`, {
        rejection_reason: rejectionReason,
      })
      .then(unwrap),
}

export const patientsAPI = {
  search: (query: string) =>
    api
      .get<StandardResponse<PatientSearchResult[]>>('/patients/search', {
        params: { q: query, limit: 25 },
      })
      .then(unwrap),
  create: (payload: CreatePatientRequest) =>
    api.post<StandardResponse<Patient>>('/patients', payload).then(unwrap),
}

export default api
