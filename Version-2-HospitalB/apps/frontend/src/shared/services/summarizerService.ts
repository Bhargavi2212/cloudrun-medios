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

