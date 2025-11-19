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

