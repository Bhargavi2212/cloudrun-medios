import { useEffect, useMemo, useState } from 'react'
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

interface UseQueueDataSSEResult {
  patients: QueuePatient[]
  metrics: QueueMetrics
  isLoading: boolean
  isError: boolean
  isConnected: boolean
}

const stageFilter = (patients: QueuePatient[], stage: QueueStage) =>
  patients.filter((patient) => patient.status === stage)

interface QueueStateMessage {
  patient_id: string
  patient_name?: string
  stage: QueueStage
  priority_level?: number
  chief_complaint?: string
  wait_time_seconds?: number
  assigned_to?: string
  assigned_to_name?: string
  consultation_id?: string
  created_at?: string
}

interface QueueSSEMessage {
  type: string
  data: {
    states?: QueueStateMessage[]
  }
}

/**
 * Hook for real-time queue data using Server-Sent Events (SSE).
 * Falls back to polling if SSE is not available.
 */
export const useQueueDataSSE = (): UseQueueDataSSEResult => {
  const [patients, setPatients] = useState<QueuePatient[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [isError, setIsError] = useState(false)

  // Initial data fetch
  const {
    data: initialData,
    isLoading,
    isError: queryError,
  } = useQuery<ManageQueueResponse>({
    queryKey: ['queue-data-initial'],
    queryFn: () => manageAPI.getQueue(),
    staleTime: 0,
  })

  // Initialize with initial data
  useEffect(() => {
    if (initialData?.patients) {
      setPatients(initialData.patients)
    }
  }, [initialData])

  // SSE connection for real-time updates
  useEffect(() => {
    if (!initialData) return

    const eventSource = new EventSource('/api/v1/queue/stream', {
      withCredentials: true,
    })

    eventSource.onopen = () => {
      setIsConnected(true)
      setIsError(false)
    }

    eventSource.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as QueueSSEMessage
        
        if (message.type === 'queue.snapshot' || message.type === 'queue.update') {
          const queueData = message.data
          if (queueData.states) {
            // Convert queue states to patients format
            const updatedPatients: QueuePatient[] = queueData.states.map((state: QueueStateMessage) => ({
              id: state.patient_id,
              patient_id: state.patient_id,
              patient_name: state.patient_name || 'Unknown',
              status: state.stage,
              triage_level: state.priority_level,
              chief_complaint: state.chief_complaint,
              wait_time_minutes: state.wait_time_seconds ? state.wait_time_seconds / 60 : null,
              assigned_doctor_id: state.assigned_to,
              assigned_doctor: state.assigned_to_name,
              consultation_id: state.consultation_id,
              checked_in_at: state.created_at,
            }))
            setPatients(updatedPatients)
          }
        }
      } catch (error) {
        console.error('Error parsing SSE message:', error)
      }
    }

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error)
      setIsConnected(false)
      setIsError(true)
      eventSource.close()
      
      // Fallback to polling if SSE fails
      const pollInterval = setInterval(async () => {
        try {
          const data = await manageAPI.getQueue()
          if (data?.patients) {
            setPatients(data.patients)
            setIsError(false)
          }
        } catch (err) {
          console.error('Polling error:', err)
        }
      }, 5000)
      
      return () => clearInterval(pollInterval)
    }

    return () => {
      eventSource.close()
      setIsConnected(false)
    }
  }, [initialData])

  const metrics = useMemo<QueueMetrics>(() => {
    const awaitingVitals = stageFilter(patients, 'waiting').length
    const awaitingDoctor = stageFilter(patients, 'triage').length
    const inConsultation = stageFilter(patients, 'scribe').length
    const assigned = patients.filter((patient) => Boolean(patient.assigned_doctor)).length
    const highPriority = patients.filter((patient) => (patient.triage_level ?? 3) <= 2).length

    // Calculate average wait time
    const waitTimes = patients
      .map((p) => p.wait_time_minutes)
      .filter((w): w is number => w !== null && w !== undefined)
    const averageWait = waitTimes.length > 0
      ? waitTimes.reduce((sum, time) => sum + time, 0) / waitTimes.length
      : 0

    return {
      total: patients.length,
      awaitingVitals,
      awaitingDoctor,
      inConsultation,
      averageWait,
      highPriority,
      assigned,
    }
  }, [patients])

  return {
    patients,
    metrics,
    isLoading: isLoading && patients.length === 0,
    isError: isError || queryError,
    isConnected,
  }
}

