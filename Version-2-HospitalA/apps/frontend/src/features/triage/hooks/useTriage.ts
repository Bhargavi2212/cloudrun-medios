import { useMutation } from "@tanstack/react-query";
import { classifyTriage } from "../../../shared/services/manageService";
import { TriageRequestPayload } from "../../../shared/types/api";

export const useTriageClassification = () => {
  return useMutation({
    mutationFn: (payload: TriageRequestPayload) => classifyTriage(payload),
  });
};

