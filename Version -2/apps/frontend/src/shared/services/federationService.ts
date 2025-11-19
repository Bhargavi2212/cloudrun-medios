import { federationApi } from "./api";
import { GlobalModelResponse } from "../types/api";

interface SubmitUpdatePayload {
  model_name: string;
  round_id: number;
  hospital_id: string;
  weights: Record<string, number[]>;
}

export const submitModelUpdate = async (payload: SubmitUpdatePayload) => {
  const response = await federationApi.post("/federation/submit", payload);
  return response.data;
};

export const fetchGlobalModel = async (modelName: string): Promise<GlobalModelResponse> => {
  const response = await federationApi.get<GlobalModelResponse>(`/federation/global-model/${modelName}`);
  return response.data;
};

