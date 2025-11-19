import axios from "axios";
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
  console.log("[API] uploadDocument called with:", {
    filename: payload.file.name,
    patient_id: payload.patient_id,
    encounter_id: payload.encounter_id,
  });
  
  const formData = new FormData();
  formData.append("file", payload.file);
  if (payload.patient_id) formData.append("patient_id", payload.patient_id);
  if (payload.encounter_id) formData.append("encounter_id", payload.encounter_id);
  if (payload.upload_method) formData.append("upload_method", payload.upload_method);

  // Axios should automatically detect FormData and set Content-Type correctly
  // But we need to remove the default "application/json" header from the axios instance
  try {
    // Create a new axios instance without default headers for this request
    const uploadClient = axios.create({
      baseURL: manageApi.defaults.baseURL,
    });
    
    const response = await uploadClient.post<FileUploadResponse>("/manage/documents/upload", formData);
    console.log("[API] uploadDocument response:", response.status, response.data);
    return response.data;
  } catch (error: unknown) {
    console.error("[API] uploadDocument error:", error);
    const errorObj = error as { response?: { status?: number; data?: unknown; headers?: unknown } };
    if (errorObj?.response) {
      console.error("[API] Error response status:", errorObj.response.status);
      console.error("[API] Error response data:", errorObj.response.data);
      console.error("[API] Error response headers:", errorObj.response.headers);
    }
    throw error;
  }
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

export const deleteDocument = async (fileId: string): Promise<void> => {
  await manageApi.delete(`/manage/documents/${fileId}`);
};

