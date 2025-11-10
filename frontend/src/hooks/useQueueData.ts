import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { manageAPI } from '@/services/api'
import type { QueuePatient, ManageQueueResponse, QueueStage } from '@/types'

interface QueueMetrics {
  total: number
  awaitingVitals: number
  awaitingDoctor: number
  inConsultation: number
  averageWait: number
  highPriority: number
  assigned: number
}

interface UseQueueDataResult {
  patients: QueuePatient[]
  metrics: QueueMetrics
  isLoading: boolean
  isError: boolean
  refetch: () => void
}

const stageFilter = (patients: QueuePatient[], stage: QueueStage) =>
  patients.filter((patient) => patient.status === stage)

export const useQueueData = (): UseQueueDataResult => {
  const {
    data,
    isLoading,
    isError,
    refetch,
  } = useQuery<ManageQueueResponse>({
    queryKey: ['queue-data'],
    queryFn: () => manageAPI.getQueue(),
    refetchInterval: 5000,
  })

  const patients = useMemo<QueuePatient[]>(() => {
    return data?.patients ?? []
  }, [data])

  const metrics = useMemo<QueueMetrics>(() => {
    if (!data) {
      return {
        total: 0,
        awaitingVitals: 0,
        awaitingDoctor: 0,
        inConsultation: 0,
        averageWait: 0,
        highPriority: 0,
        assigned: 0,
      }
    }

    const awaitingVitals = stageFilter(patients, 'waiting').length
    const awaitingDoctor = stageFilter(patients, 'triage').length
    const inConsultation = stageFilter(patients, 'scribe').length
    const assigned = patients.filter((patient) => Boolean(patient.assigned_doctor)).length
    const highPriority = patients.filter((patient) => (patient.triage_level ?? 3) <= 2).length

    return {
      total: patients.length,
      awaitingVitals,
      awaitingDoctor,
      inConsultation,
      averageWait: data.average_wait_time,
      highPriority,
      assigned,
    }
  }, [data, patients])

  return {
    patients,
    metrics,
    isLoading,
    isError,
    refetch,
  }
}
