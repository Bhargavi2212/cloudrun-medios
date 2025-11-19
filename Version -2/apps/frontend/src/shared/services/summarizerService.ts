import { summarizerApi } from "./api";

interface GenerateSummaryPayload {
  patient_id: string;
  encounter_ids: string[];
  highlights?: string[];
}

export const generateSummary = async (payload: GenerateSummaryPayload) => {
  const response = await summarizerApi.post("/summarizer/generate-summary", payload);
  return response.data;
};

export const fetchSummaryHistory = async (patientId: string) => {
  const response = await summarizerApi.get(`/summarizer/history/${patientId}`);
  return response.data;
};

export const getSummary = async (summaryId: string) => {
  const response = await summarizerApi.get(`/summarizer/summary/${summaryId}`);
  return response.data;
};

export interface UpdateSummaryPayload {
  summary_text?: string;
  encounter_ids?: string[];
}

export const updateSummary = async (summaryId: string, payload: UpdateSummaryPayload) => {
  const response = await summarizerApi.put(`/summarizer/summary/${summaryId}`, payload);
  return response.data;
};

export const deleteSummary = async (summaryId: string) => {
  await summarizerApi.delete(`/summarizer/summary/${summaryId}`);
};

