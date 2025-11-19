import { manageApi } from "./api";
import { Patient, PortableProfile, TriageRequestPayload, TriageResponsePayload } from "../types/api";

export interface CreatePatientPayload {
  mrn: string;
  first_name: string;
  last_name: string;
  dob?: string | null;
  sex?: string | null;
  contact_info?: Record<string, unknown> | null;
}

export const fetchPatients = async (): Promise<Patient[]> => {
  const response = await manageApi.get<Patient[]>("/manage/patients");
  return response.data;
};

export const createPatient = async (payload: CreatePatientPayload): Promise<Patient> => {
  const response = await manageApi.post<Patient>("/manage/patients", payload);
  return response.data;
};

export const classifyTriage = async (payload: TriageRequestPayload): Promise<TriageResponsePayload> => {
  const response = await manageApi.post<TriageResponsePayload>("/manage/classify", payload);
  return response.data;
};

export const checkInPatient = async (patientId: string): Promise<PortableProfile> => {
  const response = await manageApi.post<PortableProfile>(`/manage/patients/${patientId}/check-in`);
  return response.data;
};

