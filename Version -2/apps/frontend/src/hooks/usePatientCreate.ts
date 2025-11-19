import { useMutation } from "@tanstack/react-query";
import { manageAPI, type Patient, type CreatePatientRequest } from "../shared/services/api";

export const usePatientCreate = () => {
  return useMutation<Patient, unknown, CreatePatientRequest>({
    mutationFn: (payload) => manageAPI.createPatient(payload),
  });
};

