import { manageApi, summarizerApi } from "./api";

export interface FileAsset {
  id: string;
  patient_id?: string | null;
  encounter_id?: string | null;
  original_filename?: string | null;
  storage_path: string;
  content_type?: string | null;
  size_bytes?: number | null;
  document_type?: string | null;
  upload_method?: string | null;
  status: string;
  confidence?: number | null;
  extraction_status?: string | null;
  extraction_confidence?: number | null;
  extraction_data?: Record<string, unknown> | null;
  confidence_tier?: string | null;
  review_status?: string | null;
  needs_manual_review: boolean;
  processing_notes?: string | null;
  created_at: string;
  updated_at: string;
}

export interface FileUploadResponse {
  file_id: string;
  original_filename?: string | null;
  size_bytes?: number | null;
  content_type?: string | null;
  status: string;
  message: string;
}

export interface DocumentProcessingResult {
  file_id: string;
  success: boolean;
  overall_confidence: number;
  confidence_tier?: string | null;
  needs_review: boolean;
  timeline_event_id?: string | null;
  errors: string[];
  warnings: string[];
}

export interface UploadDocumentPayload {
  file: File;
  patient_id?: string;
  encounter_id?: string;
  upload_method?: string;
}

export const uploadDocument = async (payload: UploadDocumentPayload): Promise<FileUploadResponse> => {
  const formData = new FormData();
  formData.append("file", payload.file);
  if (payload.patient_id) formData.append("patient_id", payload.patient_id);
  if (payload.encounter_id) formData.append("encounter_id", payload.encounter_id);
  if (payload.upload_method) formData.append("upload_method", payload.upload_method);

  const response = await manageApi.post<FileUploadResponse>("/manage/documents/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const getDocument = async (fileId: string): Promise<FileAsset> => {
  const response = await manageApi.get<FileAsset>(`/manage/documents/${fileId}`);
  return response.data;
};

export const listDocuments = async (params?: {
  patient_id?: string;
  encounter_id?: string;
  status?: string;
}): Promise<FileAsset[]> => {
  const response = await manageApi.get<FileAsset[]>("/manage/documents", { params });
  return response.data;
};

export const listPendingDocuments = async (params?: {
  patient_id?: string;
  encounter_id?: string;
}): Promise<FileAsset[]> => {
  const response = await manageApi.get<FileAsset[]>("/manage/documents/pending-review", { params });
  return response.data;
};

export const processDocument = async (fileId: string): Promise<DocumentProcessingResult> => {
  const response = await summarizerApi.post<DocumentProcessingResult>(`/summarizer/documents/${fileId}/process`);
  return response.data;
};

export const confirmDocument = async (fileId: string, notes?: string): Promise<FileAsset> => {
  const formData = new FormData();
  if (notes) formData.append("notes", notes);
  const response = await manageApi.post<FileAsset>(`/manage/documents/${fileId}/confirm`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

export const rejectDocument = async (fileId: string, reason?: string): Promise<FileAsset> => {
  const formData = new FormData();
  if (reason) formData.append("reason", reason);
  const response = await manageApi.post<FileAsset>(`/manage/documents/${fileId}/reject`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
};

