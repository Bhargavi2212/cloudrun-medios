import { useQuery } from "@tanstack/react-query";
import { checkInPatient } from "../../../shared/services/manageService";
import { PortableProfile } from "../../../shared/types/api";

export const usePortableProfile = (patientId: string | null) => {
  return useQuery<PortableProfile>({
    queryKey: ["portable-profile", patientId],
    queryFn: () => checkInPatient(patientId as string),
    enabled: Boolean(patientId),
  });
};
