import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createPatient, fetchPatients, CreatePatientPayload } from "../../../shared/services/manageService";
import { Patient } from "../../../shared/types/api";

const PATIENTS_KEY = ["patients"];

export const usePatients = () => {
  return useQuery<Patient[]>({
    queryKey: PATIENTS_KEY,
    queryFn: fetchPatients,
  });
};

export const useCreatePatient = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: CreatePatientPayload) => createPatient(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PATIENTS_KEY });
    },
  });
};

