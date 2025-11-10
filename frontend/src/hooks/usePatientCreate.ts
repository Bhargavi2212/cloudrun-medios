import { useMutation } from '@tanstack/react-query'
import { patientsAPI } from '@/services/api'
import type { CreatePatientRequest, Patient } from '@/types'

export const usePatientCreate = () => {
  return useMutation<Patient, unknown, CreatePatientRequest>({
    mutationFn: (payload) => patientsAPI.create(payload),
  })
}
