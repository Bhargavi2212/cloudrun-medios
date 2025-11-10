import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { useQueueData } from '@/hooks/useQueueData'
import { useToast } from '@/components/ui/use-toast'
import { useAuthStore } from '@/store/authStore'
import { scribeAPI, queueAPI, manageAPI } from '@/services/api'
import type {
  ConsultationRecord,
  ProcessedAudioResult,
  QueuePatient,
  TimelineSummary,
  TimelineEventEntry,
} from '@/types'
import { StatusIndicator, type ProcessingStatus } from '@/components/ai/StatusIndicator'
import { NoteApprovalWorkflow, type NoteStatus } from '@/components/ai/NoteApprovalWorkflow'
import { ConsultationHistory, type ConsultationHistoryItem } from '@/components/consultation/ConsultationHistory'
import { downloadNote, downloadTranscription, downloadCombined, copyToClipboard } from '@/lib/export-utils'
import { Download, Copy } from 'lucide-react'

const formatMinutes = (value: number | null | undefined) => {
  if (value == null || Number.isNaN(value)) return '—'
  if (value === 0) return 'Just now'
  if (value < 60) return `${Math.round(value)} min`
  const hours = Math.floor(value / 60)
  const minutes = Math.round(value % 60)
  const parts = [hours > 0 ? `${hours}h` : null, minutes > 0 ? `${minutes}m` : null]
  return parts.filter(Boolean).join(' ')
}

const formatTime = (iso?: string | null) => {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const formatDate = (iso?: string | null) => {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleDateString()
}

const normalizeStatus = (status: string) =>
  status.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())

const formatSeconds = (totalSeconds: number) => {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

type NoteGenerationResult = {
  note: string
  confidence: number
  note_version_id?: string | null
  entities: Record<string, unknown>
}

const DoctorWorkflow = () => {
  const { toast } = useToast()
  const { user } = useAuthStore()
  const { patients, isLoading, isError, refetch } = useQueueData()
  const queryClient = useQueryClient()

  const awaitingPatients = useMemo(
    () => patients.filter((patient) => patient.status === 'triage' && !patient.assigned_doctor_id),
    [patients],
  )

  const myPatients = useMemo(() => {
    if (!user?.id) return []
    return patients.filter((patient) => patient.assigned_doctor_id === user.id)
  }, [patients, user?.id])

  const [selectedConsultationId, setSelectedConsultationId] = useState<string | null>(null)
  const [transcript, setTranscript] = useState('')
  const [noteOutput, setNoteOutput] = useState('')
  const [isEditingNote, setIsEditingNote] = useState(false)
  const [editedNoteContent, setEditedNoteContent] = useState('')
  const [noteSaved, setNoteSaved] = useState(false)
  const [entitiesPreview, setEntitiesPreview] = useState<Record<string, unknown> | null>(null)
  const [confidenceScores, setConfidenceScores] = useState<Record<string, number> | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const recordedChunksRef = useRef<Blob[]>([])
  const discardRecordingRef = useRef(false)
  const recordingTimerRef = useRef<number | null>(null)
  const recordingStartRef = useRef<number | null>(null)
  const [isMicSupported, setIsMicSupported] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [recordingError, setRecordingError] = useState<string | null>(null)
  const [isAudioProcessing, setIsAudioProcessing] = useState(false)
  const [lastAudioId, setLastAudioId] = useState<string | null>(null)
  const [audioWarnings, setAudioWarnings] = useState<string[]>([])
  const [isConsultationActive, setIsConsultationActive] = useState(false)
  const [processingStatus, setProcessingStatus] = useState<{
    transcription: ProcessingStatus
    entityExtraction: ProcessingStatus
    noteGeneration: ProcessingStatus
  }>({
    transcription: 'idle',
    entityExtraction: 'idle',
    noteGeneration: 'idle',
  })
  const [historyItems, setHistoryItems] = useState<Array<{
    id: string
    title: string
    description?: string
    timestamp: string
    status: 'completed' | 'failed' | 'pending'
    type: 'note' | 'transcription'
    content?: string
    metadata?: Record<string, unknown>
  }>>([])

  useEffect(() => {
    setIsMicSupported(Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia))
  }, [])

  // Load saved note when consultation is selected
  const { data: savedNote, refetch: refetchNote } = useQuery({
    queryKey: ['consultation-note', selectedConsultationId],
    queryFn: () => {
      if (!selectedConsultationId) return null
      return scribeAPI.getConsultationNote(selectedConsultationId).catch(() => null)
    },
    enabled: !!selectedConsultationId,
    refetchOnWindowFocus: false,
  })

  // Get note status from saved note or default to draft
  const noteStatus: NoteStatus = (savedNote?.status as NoteStatus) || 'draft'
  
  // Check if user can approve/reject (typically admins or supervisors)
  // Note: Adjust based on your actual role structure
  const userRole = user?.role || ''
  const canApprove = typeof userRole === 'string' 
    ? ['admin', 'supervisor', 'doctor'].includes(userRole.toLowerCase())
    : false
  const canReject = canApprove

  // Update noteOutput when saved note is loaded
  useEffect(() => {
    if (savedNote?.content && !noteOutput) {
      setNoteOutput(savedNote.content)
      setEditedNoteContent(savedNote.content)
      if (savedNote.is_ai_generated) {
        toast({
          title: 'Note loaded',
          description: 'Clinical note loaded from saved consultation.',
        })
      }
    }
  }, [savedNote, noteOutput, toast])

  useEffect(() => {
    if (selectedConsultationId) {
      const stillExists = patients.some((patient) => patient.consultation_id === selectedConsultationId)
      if (stillExists) {
        return
      }
    }

    if (myPatients.length > 0) {
      setSelectedConsultationId(myPatients[0].consultation_id)
      return
    }

    if (awaitingPatients.length > 0) {
      setSelectedConsultationId(awaitingPatients[0].consultation_id)
      return
    }

    setSelectedConsultationId(null)
  }, [awaitingPatients, myPatients, patients, selectedConsultationId])

  const selectedPatient = useMemo<QueuePatient | null>(
    () => patients.find((patient) => patient.consultation_id === selectedConsultationId) ?? null,
    [patients, selectedConsultationId],
  )

  useEffect(() => {
    setTranscript('')
    setNoteOutput('')
    setEntitiesPreview(null)
    setConfidenceScores(null)
    setAudioWarnings([])
    setLastAudioId(null)
    setRecordingError(null)
    setRecordingDuration(0)

    const shouldBeActive =
      selectedPatient?.status === 'scribe' && selectedPatient.assigned_doctor_id === user?.id

    if (!shouldBeActive && isRecording && mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      discardRecordingRef.current = true
      mediaRecorderRef.current.stop()
    }

    setIsConsultationActive(Boolean(shouldBeActive))
  }, [selectedConsultationId, selectedPatient, user?.id, isRecording])

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        discardRecordingRef.current = true
        mediaRecorderRef.current.stop()
      }
      if (recordingTimerRef.current !== null) {
        window.clearInterval(recordingTimerRef.current)
        recordingTimerRef.current = null
      }
    }
  }, [])

  const patientId = selectedPatient?.patient_id ?? null

  const summaryQuery = useQuery<TimelineSummary>({
    queryKey: ['patient-timeline', patientId],
    queryFn: () => manageAPI.getTimelineSummary(patientId!),
    enabled: patientId !== null,
    staleTime: 5 * 60 * 1000,
  })

  const recordsQuery = useQuery({
    queryKey: ['consultation-records', selectedPatient?.consultation_id],
    queryFn: () => manageAPI.getConsultationRecords(selectedPatient!.consultation_id),
    enabled: Boolean(selectedPatient?.consultation_id),
    staleTime: 60 * 1000,
  })
  const records = useMemo<ConsultationRecord[]>(() => recordsQuery.data?.records ?? [], [recordsQuery.data])

  const handleRefreshSummary = async () => {
    if (!patientId) return
    try {
      const summary = await manageAPI.getTimelineSummary(patientId, { forceRefresh: true })
      queryClient.setQueryData(['patient-timeline', patientId], summary)
      toast({ title: 'Summary refreshed' })
    } catch (error) {
      const message = error instanceof Error 
        ? error.message 
        : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Unable to refresh summary'
      toast({ title: 'Refresh failed', description: message, variant: 'destructive' })
    }
  }

  const handleRefreshRecords = async () => {
    if (!selectedPatient?.consultation_id) return
    try {
      const result = await manageAPI.getConsultationRecords(selectedPatient.consultation_id)
      queryClient.setQueryData(
        ['consultation-records', selectedPatient.consultation_id],
        result,
      )
      toast({ title: 'Records refreshed' })
    } catch (error) {
      const message = error instanceof Error 
        ? error.message 
        : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Unable to refresh records'
      toast({ title: 'Refresh failed', description: message, variant: 'destructive' })
    }
  }

  const generateNoteMutation = useMutation<
    NoteGenerationResult,
    unknown,
    { transcript: string; consultationId?: string }
  >({
    mutationFn: async (input) => {
      setProcessingStatus((prev) => ({ ...prev, entityExtraction: 'processing', noteGeneration: 'processing' }))
      const entityResponse = await scribeAPI.extractEntities(input.transcript)
      setProcessingStatus((prev) => ({ ...prev, entityExtraction: 'completed' }))
      const noteResponse = await scribeAPI.generateNote(
        input.transcript,
        entityResponse.entities,
        input.consultationId,
      )
      return {
        note: noteResponse.note,
        confidence: noteResponse.confidence,
        note_version_id: noteResponse.note_version_id,
        entities: entityResponse.entities,
      }
    },
    onSuccess: (result) => {
      setNoteOutput(result.note)
      setEditedNoteContent(result.note)
      setEntitiesPreview(result.entities)
      setConfidenceScores({ note_generation: result.confidence })
      setProcessingStatus((prev) => ({ ...prev, noteGeneration: 'completed' }))
      
      // Add to history
      const historyItem = {
        id: `note-${Date.now()}`,
        title: `AI Generated Note - ${selectedPatient?.patient_name || 'Patient'}`,
        description: 'AI-generated clinical note from transcript',
        timestamp: new Date().toISOString(),
        status: 'completed' as const,
        type: 'note' as const,
        content: result.note,
        metadata: {
          entities: result.entities,
          confidence: result.confidence,
        },
      }
      setHistoryItems((prev) => [historyItem, ...prev.slice(0, 9)])
      
      toast({
        title: 'AI note ready',
        description: 'Review the generated SOAP note before finalizing.',
      })
    },
    onError: (error) => {
      setProcessingStatus((prev) => ({ ...prev, noteGeneration: 'failed' }))
      const message = error instanceof Error 
        ? error.message 
        : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Unable to generate note'
      toast({ title: 'Generation failed', description: message, variant: 'destructive' })
    },
  })

  const claimPatientMutation = useMutation<void, unknown, QueuePatient>({
    mutationFn: async (patient) => {
      if (!user?.id) {
        throw new Error('User session not available')
      }
      await queueAPI.assignQueueState(patient.queue_state_id, user.id)
    },
    onSuccess: (_, patient) => {
      toast({
        title: 'Patient assigned',
        description: `${patient.patient_name} is now assigned to you.`,
      })
      setSelectedConsultationId(patient.consultation_id)
      queryClient.invalidateQueries({ queryKey: ['queue-data'] })
    },
    onError: (error) => {
      const message = error instanceof Error 
        ? error.message 
        : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Unable to assign patient'
      toast({ title: 'Assignment failed', description: message, variant: 'destructive' })
    },
  })

  const startConsultationMutation = useMutation<void, unknown, QueuePatient>({
    mutationFn: async (patient) => {
      if (!user?.id) {
        throw new Error('User session not available')
      }
      if (patient.assigned_doctor_id !== user.id) {
        await queueAPI.assignQueueState(patient.queue_state_id, user.id)
      }
      await queueAPI.advanceQueueState(patient.queue_state_id, { next_stage: 'scribe' })
    },
    onSuccess: () => {
      toast({
        title: 'Consultation started',
        description: 'Patient moved to in-consultation status.',
      })
      handleResetWorkspace()
      setIsConsultationActive(true)
      queryClient.invalidateQueries({ queryKey: ['queue-data'] })
    },
    onError: (error) => {
      const message = error instanceof Error 
        ? error.message 
        : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Unable to start consultation'
      toast({ title: 'Action failed', description: message, variant: 'destructive' })
    },
  })

  const completeConsultationMutation = useMutation<void, unknown, QueuePatient>({
    mutationFn: async (patient) => {
      await queueAPI.advanceQueueState(patient.queue_state_id, { next_stage: 'discharge' })
    },
    onSuccess: () => {
      toast({
        title: 'Consultation completed',
        description: 'Patient marked as discharged.',
      })
      if (isRecording) {
        handleStopRecording({ discard: true })
      }
      handleResetWorkspace()
      setIsConsultationActive(false)
      queryClient.invalidateQueries({ queryKey: ['queue-data'] })
    },
    onError: (error) => {
      const message = error instanceof Error 
        ? error.message 
        : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Unable to complete consultation'
      toast({ title: 'Action failed', description: message, variant: 'destructive' })
    },
  })

  const isSelectedMine = selectedPatient?.assigned_doctor_id === user?.id
  const isAwaitingAssignment =
    selectedPatient?.status === 'triage' && !selectedPatient?.assigned_doctor_id
  const canStartConsultation = selectedPatient?.status === 'triage'
  const canCompleteConsultation = selectedPatient?.status === 'scribe'
  const anyQueueActionPending =
    claimPatientMutation.isPending ||
    startConsultationMutation.isPending ||
    completeConsultationMutation.isPending

  const applyProcessedAudio = (result: ProcessedAudioResult) => {
    setTranscript(result.transcription ?? '')
    const generatedNote = result.generated_note ?? ''
    setNoteOutput(generatedNote)
    setEditedNoteContent(generatedNote)
    setEntitiesPreview(result.entities ?? null)
    setConfidenceScores(result.confidence_scores ?? null)
    setAudioWarnings(result.warnings ?? [])
    setRecordingDuration(0)
    
    // Update processing status
    if (result.transcription) {
      setProcessingStatus((prev) => ({ ...prev, transcription: 'completed' }))
    }
    if (result.entities) {
      setProcessingStatus((prev) => ({ ...prev, entityExtraction: 'completed' }))
    }
    if (result.generated_note) {
      setProcessingStatus((prev) => ({ ...prev, noteGeneration: 'completed' }))
    }
    
    // Add to history
    if (result.transcription || result.generated_note) {
      const status: 'completed' | 'failed' = (result.errors && result.errors.length > 0) ? 'failed' : 'completed'
      const type: 'note' | 'transcription' = result.generated_note ? 'note' : 'transcription'
      const historyItem = {
        id: `history-${Date.now()}`,
        title: `Consultation Note - ${selectedPatient?.patient_name || 'Patient'}`,
        description: result.generated_note ? 'AI-generated clinical note' : 'Audio transcription',
        timestamp: new Date().toISOString(),
        status,
        type,
        content: result.generated_note || result.transcription || '',
        metadata: {
          entities: result.entities,
          confidence_scores: result.confidence_scores,
        },
      }
      setHistoryItems((prev) => [historyItem, ...prev.slice(0, 9)]) // Keep last 10 items
    }
    
    // Check if note was saved
    if (result.note_saved) {
      setNoteSaved(true)
      toast({
        title: 'Note saved',
        description: 'Clinical note has been saved to the patient profile.',
      })
      // Refetch the saved note to get the latest version
      if (selectedConsultationId) {
        refetchNote()
      }
    } else if (result.note_save_error) {
      toast({
        title: 'Note not saved',
        description: result.note_save_error,
        variant: 'destructive',
      })
    }
  }

  const stopRecordingTimer = () => {
    if (recordingTimerRef.current !== null) {
      window.clearInterval(recordingTimerRef.current)
      recordingTimerRef.current = null
    }
  }

  const processAudioBlob = async (blob: Blob) => {
    if (!selectedPatient?.consultation_id) {
      toast({
        title: 'No consultation selected',
        description: 'Select a patient before processing a recording.',
        variant: 'destructive',
      })
      return
    }
    try {
      setIsAudioProcessing(true)
      setAudioWarnings([])
      const file = new File([blob], `consultation-${Date.now()}.webm`, {
        type: blob.type || 'audio/webm',
      })
      const upload = await scribeAPI.uploadAudio(file, {
        consultationId: selectedPatient.consultation_id,
      })
      setLastAudioId(upload.audio_id)
      const processed = await scribeAPI.processAudio(upload.audio_id, { variant: 'medi-os' })

      if (processed.errors && processed.errors.length > 0) {
        setAudioWarnings((processed.warnings ?? []).concat(processed.errors))
        toast({
          title: 'Processing failed',
          description: processed.errors.join('; '),
          variant: 'destructive',
        })
        return
      }

      applyProcessedAudio(processed)
      toast({
        title: 'Audio processed',
        description: 'Transcription and AI note generated from the conversation.',
      })
    } catch (error) {
      const message = error instanceof Error 
        ? error.message 
        : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Unable to process audio'
      toast({ title: 'Processing failed', description: message, variant: 'destructive' })
    } finally {
      setIsAudioProcessing(false)
      setRecordingDuration(0)
    }
  }

  const handleStartRecording = async () => {
    if (!selectedPatient?.consultation_id) {
      toast({
        title: 'No consultation selected',
        description: 'Select a patient before starting the AI scribe.',
        variant: 'destructive',
      })
      return
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setRecordingError('Microphone access is not supported in this browser.')
      toast({
        title: 'Microphone unsupported',
        description: 'Use a browser that supports microphone recording (e.g. Chrome).',
        variant: 'destructive',
      })
      return
    }
    try {
      setRecordingError(null)
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      recordedChunksRef.current = []
      discardRecordingRef.current = false

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunksRef.current.push(event.data)
        }
      }

      recorder.onerror = (event) => {
        console.error('Recorder error', event)
        setRecordingError('Microphone error encountered during recording.')
      }

      recorder.onstop = async () => {
        stopRecordingTimer()
        setIsRecording(false)
        recordingStartRef.current = null
        const chunks = recordedChunksRef.current.slice()
        recordedChunksRef.current = []
        stream.getTracks().forEach((track) => track.stop())

        if (discardRecordingRef.current) {
          discardRecordingRef.current = false
          setRecordingDuration(0)
          return
        }

        if (chunks.length === 0) {
          toast({
            title: 'No audio captured',
            description: 'Try recording again.',
            variant: 'destructive',
          })
          return
        }

        const mimeType = recorder.mimeType || 'audio/webm'
        const blob = new Blob(chunks, { type: mimeType })
        await processAudioBlob(blob)
      }

      recorder.start()
      mediaRecorderRef.current = recorder
      setIsRecording(true)
      recordingStartRef.current = Date.now()
      setRecordingDuration(0)
      stopRecordingTimer()
      recordingTimerRef.current = window.setInterval(() => {
        if (recordingStartRef.current) {
          const elapsed = Math.floor((Date.now() - recordingStartRef.current) / 1000)
          setRecordingDuration(elapsed)
        }
      }, 1000)
    } catch (error) {
      console.error('Failed to start recording', error)
      setRecordingError('Microphone access denied or unavailable.')
      toast({
        title: 'Microphone unavailable',
        description: 'Allow microphone access to enable live transcription.',
        variant: 'destructive',
      })
    }
  }

  const handleStopRecording = (options?: { discard?: boolean }) => {
    if (!mediaRecorderRef.current) return
    discardRecordingRef.current = Boolean(options?.discard)
    if (mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
  }

  const handleCancelRecording = () => {
    handleStopRecording({ discard: true })
  }

  const handleResetWorkspace = () => {
    setTranscript('')
    setNoteOutput('')
    setEditedNoteContent('')
    setEntitiesPreview(null)
    setConfidenceScores(null)
    setAudioWarnings([])
    setLastAudioId(null)
    setRecordingError(null)
    setRecordingDuration(0)
    setProcessingStatus({
      transcription: 'idle',
      entityExtraction: 'idle',
      noteGeneration: 'idle',
    })
    setIsEditingNote(false)
    setNoteSaved(false)
  }

  const handleDownloadNote = () => {
    if (!noteOutput) {
      toast({
        title: 'No note to download',
        description: 'Generate a note before downloading.',
        variant: 'destructive',
      })
      return
    }
    downloadNote(noteOutput, selectedPatient?.patient_name, selectedConsultationId ?? undefined)
    toast({
      title: 'Note downloaded',
      description: 'Clinical note has been downloaded.',
    })
  }

  const handleDownloadTranscription = () => {
    if (!transcript) {
      toast({
        title: 'No transcription to download',
        description: 'No transcription available to download.',
        variant: 'destructive',
      })
      return
    }
    downloadTranscription(transcript, selectedPatient?.patient_name, selectedConsultationId ?? undefined)
    toast({
      title: 'Transcription downloaded',
      description: 'Transcription has been downloaded.',
    })
  }

  const handleDownloadCombined = () => {
    const items: Array<{ title: string; content: string }> = []
    if (transcript) {
      items.push({ title: 'Transcription', content: transcript })
    }
    if (noteOutput) {
      items.push({ title: 'Clinical Note', content: noteOutput })
    }
    if (items.length === 0) {
      toast({
        title: 'No content to download',
        description: 'No transcription or note available.',
        variant: 'destructive',
      })
      return
    }
    const filename = `Consultation-${selectedPatient?.patient_name?.replace(/\s+/g, '-') || 'Patient'}-${new Date().toISOString().split('T')[0]}.txt`
    downloadCombined(items, filename)
    toast({
      title: 'Content downloaded',
      description: 'Consultation content has been downloaded.',
    })
  }

  const handleCopyNote = async () => {
    if (!noteOutput) {
      toast({
        title: 'No note to copy',
        description: 'Generate a note before copying.',
        variant: 'destructive',
      })
      return
    }
    const success = await copyToClipboard(noteOutput)
    if (success) {
      toast({
        title: 'Note copied',
        description: 'Clinical note has been copied to clipboard.',
      })
    } else {
      toast({
        title: 'Copy failed',
        description: 'Unable to copy note to clipboard.',
        variant: 'destructive',
      })
    }
  }

  const handleSubmitNoteForApproval = async () => {
    if (!selectedConsultationId) {
      toast({
        title: 'No consultation selected',
        description: 'Select a consultation before submitting.',
        variant: 'destructive',
      })
      return
    }
    await scribeAPI.submitNoteForApproval(selectedConsultationId)
    refetchNote()
  }

  const handleApproveNote = async () => {
    if (!selectedConsultationId) {
      toast({
        title: 'No consultation selected',
        description: 'Select a consultation before approving.',
        variant: 'destructive',
      })
      return
    }
    await scribeAPI.approveNote(selectedConsultationId)
    refetchNote()
  }

  const handleRejectNote = async (reason: string) => {
    if (!selectedConsultationId) {
      toast({
        title: 'No consultation selected',
        description: 'Select a consultation before rejecting.',
        variant: 'destructive',
      })
      return
    }
    await scribeAPI.rejectNote(selectedConsultationId, reason)
    refetchNote()
  }

  // Build consultation history items
  const consultationHistoryItems: ConsultationHistoryItem[] = useMemo(() => {
    const items: ConsultationHistoryItem[] = []
    
    // Add saved note to history
    if (savedNote) {
      items.push({
        id: savedNote.note_id,
        type: 'note',
        title: 'Clinical Note',
        description: savedNote.is_ai_generated ? 'AI-generated note' : 'Manual note',
        timestamp: savedNote.created_at || new Date().toISOString(),
        status: noteStatus === 'approved' ? 'completed' : noteStatus === 'rejected' ? 'failed' : 'pending',
        content: savedNote.content,
        metadata: {
          status: noteStatus,
          is_ai_generated: savedNote.is_ai_generated,
        },
        patientName: selectedPatient?.patient_name,
        consultationId: selectedConsultationId || undefined,
      })
    }
    
    // Add history items from processing
    historyItems.forEach((item) => {
      items.push({
        id: item.id,
        type: item.type,
        title: item.title,
        description: item.description,
        timestamp: item.timestamp,
        status: item.status,
        content: item.content,
        metadata: item.metadata,
        patientName: selectedPatient?.patient_name,
        consultationId: selectedConsultationId || undefined,
      })
    })
    
    return items.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
  }, [savedNote, historyItems, noteStatus, selectedPatient, selectedConsultationId])

  const handleClaimPatient = (patient: QueuePatient) => {
    claimPatientMutation.mutate(patient)
  }

  const handleBeginConsultation = () => {
    if (!selectedPatient) {
      toast({
        title: 'No consultation selected',
        description: 'Select or claim a patient before starting a consultation.',
        variant: 'destructive',
      })
      return
    }
    startConsultationMutation.mutate(selectedPatient)
  }

  const handleFinalizeConsultation = () => {
    if (!selectedPatient) {
      toast({
        title: 'No consultation selected',
        description: 'Select a patient before completing a consultation.',
        variant: 'destructive',
      })
      return
    }
    completeConsultationMutation.mutate(selectedPatient)
  }

  const handleGenerateNote = () => {
    if (!transcript.trim()) {
      toast({
        title: 'Transcript required',
        description: 'Provide a transcript snippet before generating a note.',
        variant: 'destructive',
      })
      return
    }

    generateNoteMutation.mutate({
      transcript: transcript.trim(),
      consultationId: selectedPatient?.consultation_id,
    })
  }

  const metrics = useMemo(() => {
    const highPriority = myPatients.filter((patient) => (patient.triage_level ?? 3) <= 2).length
    const ready = myPatients.filter((patient) => patient.status === 'scribe').length
    const completed = myPatients.filter((patient) => patient.status === 'discharge').length
    return {
      assigned: myPatients.length,
      ready,
      highPriority,
      completed,
    }
  }, [myPatients])

  const renderSummary = (summary?: TimelineSummary) => {
    if (!summary) {
      return <p className="text-sm text-gray-500">Select a patient to view the AI summary.</p>
    }

    const events: TimelineEventEntry[] = summary.timeline?.events ?? []

    return (
      <div className="space-y-4">
        <div className="flex items-center gap-3 text-sm text-gray-500">
          <span>Generated {formatDate(summary.generated_at)} at {formatTime(summary.generated_at)}</span>
          {summary.cached && <Badge variant="outline">Cached</Badge>}
          {summary.model && <Badge variant="secondary">{summary.model}</Badge>}
        </div>
        <pre className="whitespace-pre-wrap text-sm text-gray-800 bg-gray-50 p-4 rounded-md border border-gray-100 overflow-x-auto">
          {summary.summary}
        </pre>
        {summary.highlights.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-semibold text-gray-700">Key highlights</h4>
            <ul className="list-disc pl-5 text-sm text-gray-600 space-y-1">
              {summary.highlights.slice(0, 5).map((highlight, index) => (
                <li key={index}>{highlight}</li>
              ))}
            </ul>
          </div>
        )}
        {events.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-semibold text-gray-700">Timeline highlights</h4>
            <div className="space-y-2">
              {events.slice(0, 5).map((event) => (
                <div
                  key={event.id}
                  className="rounded border border-gray-200 bg-white shadow-sm px-3 py-2 space-y-1"
                >
                  <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
                    <Badge variant="outline" className="uppercase">
                      {event.event_type}
                    </Badge>
                    <span>
                      {formatDate(event.event_date)} • {formatTime(event.event_date)}
                    </span>
                    <Badge variant="outline" className="uppercase text-gray-600">
                      {event.status}
                    </Badge>
                    {typeof event.confidence === 'number' && (
                      <Badge variant="secondary">
                        {Math.round(event.confidence * 100)}% confidence
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm font-medium text-gray-800">{event.title}</p>
                  {event.summary && (
                    <p className="text-sm text-gray-600">{event.summary}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <MetricCard title="Assigned" value={metrics.assigned} accent="bg-emerald-100 text-emerald-700" />
        <MetricCard title="Ready for Review" value={metrics.ready} accent="bg-blue-100 text-blue-700" />
        <MetricCard title="High Priority" value={metrics.highPriority} accent="bg-amber-100 text-amber-700" />
        <MetricCard title="Recently Completed" value={metrics.completed} accent="bg-gray-100 text-gray-700" />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-1 space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>My Patients</CardTitle>
                <Button size="sm" variant="ghost" onClick={() => refetch()} disabled={isLoading}>
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {isError && (
                <p className="text-sm text-red-500">
                  Unable to load queue data. Please try again shortly.
                </p>
              )}
              {isLoading ? (
                <p className="text-sm text-gray-500">Loading queue…</p>
              ) : myPatients.length === 0 ? (
                <div className="rounded border border-dashed p-6 text-center text-sm text-gray-500">
                  No patients are currently assigned to you.
                </div>
              ) : (
                <div className="space-y-3">
                  {myPatients.map((patient) => {
                    const isSelected = patient.consultation_id === selectedConsultationId
                    return (
                      <button
                        key={patient.consultation_id}
                        onClick={() => setSelectedConsultationId(patient.consultation_id)}
                        className={`w-full text-left rounded-lg border p-4 transition ${
                          isSelected
                            ? 'border-emerald-500 bg-emerald-50/60'
                            : 'border-gray-200 hover:border-emerald-300 hover:bg-emerald-50/40'
                        }`}
                      >
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-semibold text-gray-900">{patient.patient_name}</span>
                          <Badge variant="secondary">
                            {patient.age != null ? `Age ${patient.age}` : 'Age n/a'}
                          </Badge>
                          <Badge variant="outline">{`ESI ${patient.triage_level ?? '—'}`}</Badge>
                          <Badge variant="outline" className="capitalize">
                            {normalizeStatus(patient.status)}
                          </Badge>
                        </div>
                        <p className="mt-1 text-sm text-gray-600 line-clamp-2">
                          {patient.chief_complaint || 'No chief complaint recorded.'}
                        </p>
                        <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-500">
                          <span>Waiting {formatMinutes(patient.wait_time_minutes)}</span>
                          <span>Checked in at {formatTime(patient.check_in_time)}</span>
                          {patient.estimated_wait_minutes != null && (
                            <span>ETA {formatMinutes(patient.estimated_wait_minutes)}</span>
                          )}
                        </div>
                      </button>
                    )
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Awaiting Assignment</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {awaitingPatients.length === 0 ? (
                <p className="text-sm text-gray-500">No patients are waiting for assignment.</p>
              ) : (
                <div className="space-y-3">
                  {awaitingPatients.map((patient) => {
                    const isSelected = patient.consultation_id === selectedConsultationId
                    return (
                      <div
                        key={patient.consultation_id}
                        className={`rounded-lg border p-4 ${isSelected ? 'border-blue-400 bg-blue-50/60' : 'border-gray-200'}`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div
                            role="button"
                            tabIndex={0}
                            onClick={() => setSelectedConsultationId(patient.consultation_id)}
                            onKeyDown={(event) => {
                              if (event.key === 'Enter' || event.key === ' ') {
                                setSelectedConsultationId(patient.consultation_id)
                              }
                            }}
                            className="flex-1 cursor-pointer"
                          >
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="font-semibold text-gray-900">{patient.patient_name}</span>
                              <Badge variant="secondary">
                                {patient.age != null ? `Age ${patient.age}` : 'Age n/a'}
                              </Badge>
                              <Badge variant="outline">{`ESI ${patient.triage_level ?? '—'}`}</Badge>
                            </div>
                            <p className="mt-1 text-sm text-gray-600 line-clamp-2">
                              {patient.chief_complaint || 'No chief complaint recorded.'}
                            </p>
                            <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-500">
                              <span>Waiting {formatMinutes(patient.wait_time_minutes)}</span>
                              <span>Checked in at {formatTime(patient.check_in_time)}</span>
                            </div>
                          </div>
                          <Button
                            size="sm"
                            disabled={claimPatientMutation.isPending || anyQueueActionPending}
                            onClick={() => handleClaimPatient(patient)}
                          >
                            {claimPatientMutation.isPending ? 'Claiming…' : 'Claim patient'}
                          </Button>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6 lg:col-span-2">
    <Card>
      <CardHeader>
              <CardTitle>Patient Overview</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {selectedPatient ? (
                <>
                  <div className="flex flex-wrap items-center gap-3">
                    <h3 className="text-lg font-semibold text-gray-900">
                      {selectedPatient.patient_name}
                    </h3>
                    <Badge variant="secondary">
                      {selectedPatient.age != null ? `Age ${selectedPatient.age}` : 'Age n/a'}
                    </Badge>
                    <Badge>{`ESI ${selectedPatient.triage_level ?? '—'}`}</Badge>
                    <Badge variant="outline" className="capitalize">
                      {normalizeStatus(selectedPatient.status)}
                    </Badge>
                  </div>
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <p className="text-sm text-gray-500">
                      Assigned doctor: {selectedPatient.assigned_doctor ?? 'Unassigned'}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {isAwaitingAssignment && (
                        <Button
                          size="sm"
                          onClick={() => handleClaimPatient(selectedPatient)}
                          disabled={anyQueueActionPending}
                        >
                          {claimPatientMutation.isPending ? 'Claiming…' : 'Claim patient'}
                        </Button>
                      )}
                      {isSelectedMine && canStartConsultation && (
                        <Button
                          size="sm"
                          onClick={handleBeginConsultation}
                          disabled={
                            anyQueueActionPending ||
                            isAudioProcessing
                          }
                        >
                          {startConsultationMutation.isPending ? 'Starting…' : 'Begin consultation'}
                        </Button>
                      )}
                      {isSelectedMine && canCompleteConsultation && (
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={handleFinalizeConsultation}
                          disabled={
                            anyQueueActionPending ||
                            isAudioProcessing ||
                            isRecording
                          }
                        >
                          {completeConsultationMutation.isPending ? 'Completing…' : 'Complete consultation'}
                        </Button>
                      )}
                    </div>
                  </div>
                  <p className="text-sm text-gray-600">
                    Chief complaint: {selectedPatient.chief_complaint ?? 'Not provided'}
                  </p>
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    <InfoRow label="Queue status" value={normalizeStatus(selectedPatient.status)} />
                    <InfoRow
                      label="Wait time"
                      value={formatMinutes(selectedPatient.wait_time_minutes)}
                    />
                    <InfoRow
                      label="Estimated remaining"
                      value={formatMinutes(selectedPatient.estimated_wait_minutes)}
                    />
                    <InfoRow label="Checked in" value={formatTime(selectedPatient.check_in_time)} />
                  </div>
                  <VitalsPanel vitals={selectedPatient.vitals ?? null} />
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h4 className="text-sm font-semibold text-gray-700">Uploaded records</h4>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={handleRefreshRecords}
                      disabled={!selectedPatient || recordsQuery.isFetching}
                    >
                      {recordsQuery.isFetching ? 'Refreshing…' : 'Refresh records'}
                    </Button>
                  </div>
                  {recordsQuery.isLoading ? (
                    <p className="text-sm text-gray-500">Loading records…</p>
                  ) : recordsQuery.isError ? (
                    <p className="text-sm text-red-500">
                      Unable to load records. Try refreshing.
                    </p>
                  ) : records.length === 0 ? (
                    <p className="text-sm text-gray-500">
                      No records have been uploaded for this consultation.
                    </p>
                  ) : (
                    <ul className="space-y-2 rounded-lg border border-gray-200 bg-gray-50 p-3 text-sm text-gray-700">
                      {records.map((record) => {
                        return (
                          <li key={record.id} className="rounded border border-gray-200 bg-white p-3">
                            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
                              <div className="space-y-1">
                                <div className="flex flex-wrap items-center gap-2">
                                  <span className="font-semibold text-gray-800">
                                    {record.original_filename ?? 'Record'}
                                  </span>
                                  <Badge variant="outline" className="uppercase">
                                    {record.status}
                                  </Badge>
                                  {record.needs_review && (
                                    <Badge variant="destructive" className="uppercase">
                                      Needs review
                                    </Badge>
                                  )}
                                </div>
                                <span className="block text-xs text-gray-500">
                                  Uploaded {new Date(record.uploaded_at).toLocaleString()} by{' '}
                                  {record.uploaded_by ?? 'Unknown'}
                                </span>
                                {record.processed_at && (
                                  <span className="block text-xs text-gray-500">
                                    Processed {new Date(record.processed_at).toLocaleString()}
                                  </span>
                                )}
                                {record.processing_notes && (
                                  <p className="text-xs text-gray-600">{record.processing_notes}</p>
                                )}
                              </div>
                              <div className="flex flex-col items-start gap-2 sm:items-end">
                                <button
                                  type="button"
                                  onClick={async () => {
                                    try {
                                      // If we have a signed URL (from GCS), use it directly
                                      if (record.signed_url) {
                                        window.open(record.signed_url, '_blank', 'noopener,noreferrer')
                                        return
                                      }
                                      
                                      // Otherwise, fetch with authentication
                                      const blob = await manageAPI.downloadDocument(record.id)
                                      const url = URL.createObjectURL(blob)
                                      const newWindow = window.open(url, '_blank', 'noopener,noreferrer')
                                      
                                      // Clean up the blob URL after a delay (browser should have loaded it)
                                      setTimeout(() => {
                                        URL.revokeObjectURL(url)
                                      }, 1000)
                                      
                                      if (!newWindow) {
                                        toast({
                                          title: 'Failed to open document',
                                          description: 'Please allow pop-ups for this site to view documents.',
                                          variant: 'destructive',
                                        })
                                      }
                                    } catch (error) {
                                      toast({
                                        title: 'Failed to load document',
                                        description: error instanceof Error ? error.message : 'An error occurred',
                                        variant: 'destructive',
                                      })
                                    }
                                  }}
                                  className="text-sm font-medium text-emerald-600 hover:text-emerald-800 cursor-pointer"
                                >
                                  View
                                </button>
                                <div className="flex flex-wrap gap-2 text-xs text-gray-500">
                                  {typeof record.confidence === 'number' && (
                                    <Badge variant="secondary">
                                      {Math.round(record.confidence * 100)}% confidence
                                    </Badge>
                                  )}
                                  {record.document_type && (
                                    <Badge variant="outline">{record.document_type}</Badge>
                                  )}
                                </div>
                              </div>
                            </div>
                          </li>
                        )
                      })}
                    </ul>
                  )}
                </div>
                </>
              ) : (
                <p className="text-sm text-gray-500">Select a patient to view details.</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
              <div>
                <CardTitle>AI Patient Summary</CardTitle>
                <p className="text-sm text-gray-500">
                  Review recent history first. Start the consultation when you&rsquo;re ready to document.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={handleRefreshSummary}
                  disabled={!selectedPatient || summaryQuery.isFetching}
                >
                  Refresh summary
                </Button>
                <Button
                  size="sm"
                  onClick={isConsultationActive ? handleFinalizeConsultation : handleBeginConsultation}
                  disabled={
                    !selectedPatient ||
                    anyQueueActionPending ||
                    isAudioProcessing ||
                    (isConsultationActive ? !canCompleteConsultation : !canStartConsultation)
                  }
                  variant={isConsultationActive ? 'destructive' : 'default'}
                >
                  {isConsultationActive ? 'Wrap Consultation' : 'Begin Consultation'}
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {summaryQuery.isLoading && (
                <p className="text-sm text-gray-500">Loading summary…</p>
              )}
              {summaryQuery.isError && (
                <p className="text-sm text-red-500">
                  Unable to fetch timeline summary. Try refreshing in a moment.
                </p>
              )}
              {!summaryQuery.isLoading && !summaryQuery.isError && renderSummary(summaryQuery.data)}
            </CardContent>
          </Card>

          {isConsultationActive && (
            <Card>
            <CardHeader className="flex flex-col gap-1 md:flex-row md:items-center md:justify-between">
              <div>
                <CardTitle>AI Scribe Workspace</CardTitle>
                <p className="text-sm text-gray-500">
                  Launch a live capture to transcribe the consultation and draft SOAP notes.
                </p>
              </div>
              <Button
                size="sm"
                onClick={handleGenerateNote}
                disabled={
                  generateNoteMutation.isPending ||
                  !selectedPatient ||
                  isRecording ||
                  isAudioProcessing ||
                  !transcript.trim()
                }
              >
                {generateNoteMutation.isPending ? 'Generating…' : 'Generate AI Note'}
              </Button>
      </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-gray-700">Live transcription</label>
                  <div className="flex items-center gap-2">
                    <StatusIndicator
                      status={isRecording ? 'processing' : processingStatus.transcription}
                      message={isRecording ? 'Recording...' : processingStatus.transcription === 'completed' ? 'Transcribed' : undefined}
                      size="sm"
                    />
                    {transcript && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleDownloadTranscription}
                        title="Download transcription"
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    onClick={handleStartRecording}
                    disabled={
                      !selectedPatient ||
                      isRecording ||
                      isAudioProcessing ||
                      generateNoteMutation.isPending ||
                      !isMicSupported
                    }
                  >
                    Start recording
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => handleStopRecording()}
                    disabled={!isRecording || isAudioProcessing}
                  >
                    Stop & process
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleCancelRecording}
                    disabled={!isRecording || isAudioProcessing}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleResetWorkspace}
                    disabled={isRecording || isAudioProcessing || (!noteOutput && !transcript)}
                  >
                    Reset workspace
                  </Button>
                </div>
                {isRecording && (
                  <div className="flex items-center gap-2 text-sm text-emerald-700">
                    <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
                    <span>Recording… {formatSeconds(recordingDuration)}</span>
                  </div>
                )}
                {isAudioProcessing && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm text-blue-600">
                      <StatusIndicator status="processing" message="Processing audio..." size="sm" />
                    </div>
                    <div className="flex items-center gap-4 text-xs text-gray-600">
                      <StatusIndicator
                        status={processingStatus.transcription}
                        message="Transcription"
                        size="sm"
                      />
                      <StatusIndicator
                        status={processingStatus.entityExtraction}
                        message="Entity Extraction"
                        size="sm"
                      />
                      <StatusIndicator
                        status={processingStatus.noteGeneration}
                        message="Note Generation"
                        size="sm"
                      />
                    </div>
                  </div>
                )}
                {recordingError && (
                  <div className="flex items-center gap-2">
                    <StatusIndicator status="failed" message={recordingError} size="sm" />
                  </div>
                )}
                {!isMicSupported && (
                  <div className="flex items-center gap-2">
                    <StatusIndicator
                      status="warning"
                      message="Microphone access is not available in this browser"
                      size="sm"
                    />
                  </div>
                )}
                {lastAudioId && (
                  <div className="text-xs text-gray-500">
                    <p>Last processed audio ID: {lastAudioId}</p>
                  </div>
                )}
                {audioWarnings.length > 0 && (
                  <div className="rounded-md border border-amber-300 bg-amber-50 p-3 text-xs text-amber-700">
                    <p className="font-medium">Pipeline warnings</p>
                    <ul className="list-disc pl-4">
                      {audioWarnings.map((warning, index) => (
                        <li key={`${warning}-${index}`}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              <div className="border-t border-dashed border-gray-200" />

              <div className="space-y-3">
                <label className="text-sm font-medium text-gray-700">Transcript snippet</label>
                <Textarea
                  className="h-40"
                  placeholder="Transcription will appear here. You can also add manual notes."
                  value={transcript}
                  onChange={(event) => setTranscript(event.target.value)}
                  disabled={!selectedPatient || generateNoteMutation.isPending || isAudioProcessing}
                />
              </div>

              {noteOutput && (
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-semibold text-gray-700">Clinical Note</h4>
                      <StatusIndicator
                        status={processingStatus.noteGeneration}
                        message={processingStatus.noteGeneration === 'completed' ? 'Generated' : undefined}
                        size="sm"
                      />
                      {noteSaved && (
                        <Badge variant="outline" className="text-xs text-green-600">
                          Saved
                        </Badge>
                      )}
                      {savedNote?.is_ai_generated === false && (
                        <Badge variant="outline" className="text-xs text-blue-600">
                          Edited
                        </Badge>
                      )}
                    </div>
                    <div className="flex gap-2">
                      {!isEditingNote && (
                        <>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={handleDownloadNote}
                            title="Download note"
                          >
                            <Download className="h-4 w-4 mr-1" />
                            Download
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={handleCopyNote}
                            title="Copy note to clipboard"
                          >
                            <Copy className="h-4 w-4 mr-1" />
                            Copy
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => {
                              setIsEditingNote(true)
                              setEditedNoteContent(noteOutput)
                            }}
                          >
                            Edit Note
                          </Button>
                        </>
                      )}
                      {isEditingNote && (
                        <>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => {
                              setIsEditingNote(false)
                              setEditedNoteContent(noteOutput)
                            }}
                          >
                            Cancel
                          </Button>
                          <Button
                            size="sm"
                            onClick={async () => {
                              if (!selectedConsultationId) {
                                toast({
                                  title: 'Error',
                                  description: 'No consultation selected',
                                  variant: 'destructive',
                                })
                                return
                              }
                              try {
                                await scribeAPI.updateConsultationNote(
                                  selectedConsultationId,
                                  editedNoteContent,
                                )
                                setNoteOutput(editedNoteContent)
                                setIsEditingNote(false)
                                setNoteSaved(true)
                                refetchNote()
                                toast({
                                  title: 'Note updated',
                                  description: 'Clinical note has been saved.',
                                })
                              } catch (error) {
                                const errorMessage = error instanceof Error ? error.message : 'Failed to update note'
                                toast({
                                  title: 'Update failed',
                                  description: errorMessage,
                                  variant: 'destructive',
                                })
                              }
                            }}
                          >
                            Save Changes
                          </Button>
                        </>
                      )}
                    </div>
                  </div>
                  <Textarea
                    className="h-64 font-mono text-sm"
                    value={isEditingNote ? editedNoteContent : noteOutput}
                    onChange={(e) => isEditingNote && setEditedNoteContent(e.target.value)}
                    readOnly={!isEditingNote}
                    placeholder="Clinical note will appear here..."
                  />
                  <div className="flex items-center justify-between">
                    {savedNote?.created_at && (
                      <p className="text-xs text-gray-500">
                        Last updated: {new Date(savedNote.created_at).toLocaleString()}
                      </p>
                    )}
                    {(transcript || noteOutput) && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={handleDownloadCombined}
                        className="text-xs"
                      >
                        <Download className="h-3 w-3 mr-1" />
                        Download All
                      </Button>
                    )}
                  </div>
                </div>
              )}

              {confidenceScores && Object.keys(confidenceScores).length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-gray-700">Confidence scores</h4>
                  <ul className="text-xs text-gray-600">
                    {Object.entries(confidenceScores).map(([stage, value]) => (
                      <li key={stage}>
                        <span className="font-medium capitalize">{stage.replace(/_/g, ' ')}:</span>{' '}
                        {(value * 100).toFixed(0)}%
                      </li>
                    ))}
        </ul>
                </div>
              )}

              {entitiesPreview && (
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-gray-700">Extracted entities</h4>
                  <pre className="max-h-48 overflow-y-auto rounded-md border border-gray-200 bg-gray-50 p-3 text-xs text-gray-700">
                    {JSON.stringify(entitiesPreview, null, 2)}
                  </pre>
                </div>
              )}
      </CardContent>
    </Card>
          )}

          {noteOutput && selectedConsultationId && (
            <NoteApprovalWorkflow
              noteStatus={noteStatus}
              onSubmit={handleSubmitNoteForApproval}
              onApprove={handleApproveNote}
              onReject={handleRejectNote}
              canApprove={canApprove}
              canReject={canReject}
              canSubmit={noteStatus === 'draft' || noteStatus === 'rejected'}
            />
          )}

          {selectedConsultationId && (
            <ConsultationHistory
              items={consultationHistoryItems}
              patientName={selectedPatient?.patient_name}
              consultationId={selectedConsultationId}
              title="Consultation History"
              emptyMessage="No history available for this consultation"
            />
          )}
        </div>
      </div>
    </div>
  )
}

interface MetricCardProps {
  title: string
  value: number | string
  accent: string
}

const MetricCard = ({ title, value, accent }: MetricCardProps) => (
  <Card>
    <CardHeader>
      <CardTitle className="text-sm font-medium text-gray-500">{title}</CardTitle>
    </CardHeader>
    <CardContent>
      <p className={`text-2xl font-semibold ${accent}`}>{value}</p>
    </CardContent>
  </Card>
)

const InfoRow = ({ label, value }: { label: string; value: string }) => (
  <div>
    <div className="text-xs uppercase tracking-wide text-gray-400">{label}</div>
    <div className="text-sm text-gray-700">{value}</div>
  </div>
)

const VitalsPanel = ({ vitals }: { vitals: Record<string, unknown> | null }) => {
  if (!vitals || Object.keys(vitals).length === 0) {
    return <p className="text-sm text-gray-500">No vitals recorded for this consultation yet.</p>
  }

  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
      <h4 className="mb-2 text-sm font-semibold text-gray-700">Latest vitals</h4>
      <dl className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {Object.entries(vitals).map(([key, rawValue]) => {
          if (rawValue === null || rawValue === undefined) return null
          const value =
            typeof rawValue === 'number' || typeof rawValue === 'string'
              ? rawValue
              : JSON.stringify(rawValue)
          const label = key.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())
          return (
            <div key={key} className="text-sm text-gray-700">
              <span className="font-medium text-gray-600">{label}: </span>
              <span>{value}</span>
            </div>
          )
        })}
      </dl>
    </div>
  )
}

export default DoctorWorkflow