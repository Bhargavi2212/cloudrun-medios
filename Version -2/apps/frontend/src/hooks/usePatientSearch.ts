import { useQuery } from "@tanstack/react-query";
import { manageAPI, type Patient } from "../shared/services/api";

export const usePatientSearch = (query: string) => {
  return useQuery<Patient[]>({
    queryKey: ["patient-search", query],
    queryFn: () => manageAPI.searchPatients(query),
    enabled: query.trim().length >= 2,
  });
};

