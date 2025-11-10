import { useQuery } from '@tanstack/react-query'
import { patientsAPI } from '@/services/api'
import type { PatientSearchResult } from '@/types'

export const usePatientSearch = (query: string) => {
  return useQuery<PatientSearchResult[]>({
    queryKey: ['patient-search', query],
    queryFn: () => patientsAPI.search(query),
    enabled: query.trim().length >= 2,
  })
}
