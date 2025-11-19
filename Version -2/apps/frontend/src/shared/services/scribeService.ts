import { scribeApi } from "./api";
import { SoapNoteResponse, TranscriptPayload } from "../types/api";

export const createTranscript = async (payload: TranscriptPayload) => {
  const response = await scribeApi.post("/scribe/transcript", payload);
  return response.data;
};

export const generateSoap = async (
  encounterId: string,
  transcript: string,
): Promise<SoapNoteResponse> => {
  const response = await scribeApi.post<SoapNoteResponse>("/scribe/generate-soap", {
    encounter_id: encounterId,
    transcript,
  });
  return response.data;
};

export const getSoapNote = async (noteId: string): Promise<SoapNoteResponse> => {
  const response = await scribeApi.get<SoapNoteResponse>(`/scribe/soap/${noteId}`);
  return response.data;
};

export const listSoapNotes = async (encounterId: string): Promise<SoapNoteResponse[]> => {
  const response = await scribeApi.get<SoapNoteResponse[]>(`/scribe/soap/encounter/${encounterId}`);
  return response.data;
};

export interface UpdateSoapNotePayload {
  subjective?: string;
  objective?: string;
  assessment?: string;
  plan?: string;
}

export const updateSoapNote = async (noteId: string, payload: UpdateSoapNotePayload): Promise<SoapNoteResponse> => {
  const response = await scribeApi.put<SoapNoteResponse>(`/scribe/soap/${noteId}`, payload);
  return response.data;
};

export const deleteSoapNote = async (noteId: string): Promise<void> => {
  await scribeApi.delete(`/scribe/soap/${noteId}`);
};

