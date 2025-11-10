import { type ChangeEvent, useMemo, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { useToast } from '@/components/ui/use-toast'
import { useQueueData } from '@/hooks/useQueueData'
import type { ConsultationRecord, QueuePatient } from '@/types'
import { manageAPI } from '@/services/api'
import { Camera, Loader2 } from 'lucide-react'

type VitalsFormState = {
  heart_rate: string
  blood_pressure_systolic: string
  blood_pressure_diastolic: string
  respiratory_rate: string
  temperature_celsius: string
  oxygen_saturation: string
  weight_kg: string
}

type PatientVitalsSnapshot = {
  heart_rate?: number | string | null
  blood_pressure_systolic?: number | string | null
  blood_pressure_diastolic?: number | string | null
  respiratory_rate?: number | string | null
  temperature?: number | string | null
  temperature_celsius?: number | string | null
  oxygen_saturation?: number | string | null
  weight_kg?: number | string | null
  extra?: Record<string, unknown> | null
}

const initialVitalsState: VitalsFormState = {
  heart_rate: '',
  blood_pressure_systolic: '',
  blood_pressure_diastolic: '',
  respiratory_rate: '',
  temperature_celsius: '',
  oxygen_saturation: '',
  weight_kg: '',
}

const formatMinutes = (value: number | null | undefined): string => {
  if (value == null || Number.isNaN(value)) return '—'
  if (value === 0) return 'Just now'
  if (value < 60) return `${Math.round(value)} min`
  const hours = Math.floor(value / 60)
  const minutes = Math.round(value % 60)
  const parts = [hours > 0 ? `${hours}h` : null, minutes > 0 ? `${minutes}m` : null]
  return parts.filter(Boolean).join(' ')
}

const formatTime = (iso?: string | null): string => {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const NurseDashboard = () => {
  const { toast } = useToast()
  const {
    patients,
    metrics,
    isLoading,
    isError,
    refetch,
  } = useQueueData()

  const queryClient = useQueryClient()

  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [selectedPatient, setSelectedPatient] = useState<QueuePatient | null>(null)
  const [vitals, setVitals] = useState<VitalsFormState>(initialVitalsState)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [attachments, setAttachments] = useState<File[]>([])
  const [attachmentNotes, setAttachmentNotes] = useState('')
  const [records, setRecords] = useState<ConsultationRecord[]>([])
  const [recordActionLoading, setRecordActionLoading] = useState<Record<string, boolean>>({})
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const cameraInputRef = useRef<HTMLInputElement | null>(null)

  const awaitingVitals = useMemo(
    () => patients.filter((patient) => patient.status === 'waiting'),
    [patients],
  )

  const inTriage = useMemo(
    () => patients.filter((patient) => patient.status === 'triage'),
    [patients],
  )

  const longestWait = awaitingVitals.reduce(
    (acc, patient) => Math.max(acc, patient.wait_time_minutes ?? 0),
    0,
  )

  const mapPatientVitalsToForm = (patient: QueuePatient | null): VitalsFormState => {
    if (!patient?.vitals) {
      return initialVitalsState
    }

    const raw = patient.vitals as PatientVitalsSnapshot
    const extra = (raw.extra as Record<string, unknown>) ?? {}

    const asString = (value: unknown): string =>
      value === null || value === undefined || value === '' ? '' : String(value)

    return {
      heart_rate: asString(raw.heart_rate),
      blood_pressure_systolic: asString(raw.blood_pressure_systolic),
      blood_pressure_diastolic: asString(raw.blood_pressure_diastolic),
      respiratory_rate: asString(raw.respiratory_rate),
      temperature_celsius: asString(raw.temperature ?? raw.temperature_celsius),
      oxygen_saturation: asString(raw.oxygen_saturation),
      weight_kg: asString(extra.weight_kg ?? raw.weight_kg),
    }
  }

  const handleOpenVitals = (patient: QueuePatient) => {
    setSelectedPatient(patient)
    setVitals(mapPatientVitalsToForm(patient))
    setAttachments([])
    setAttachmentNotes('')
    setRecords([])
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
    if (cameraInputRef.current) {
      cameraInputRef.current.value = ''
    }
    setIsDialogOpen(true)
    if (patient.consultation_id) {
      queryClient
        .fetchQuery({
          queryKey: ['consultation-records', patient.consultation_id],
          queryFn: () => manageAPI.getConsultationRecords(patient.consultation_id!),
        })
        .then((response) => {
          setRecords(response.records ?? [])
        })
        .catch(() => {
          // ignore errors; records can be refreshed later
        })
    }
  }

  const handleCloseDialog = () => {
    setIsDialogOpen(false)
    setSelectedPatient(null)
    setVitals(initialVitalsState)
    setAttachments([])
    setAttachmentNotes('')
    setRecords([])
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
    if (cameraInputRef.current) {
      cameraInputRef.current.value = ''
    }
  }

  const handleInputChange = (field: keyof VitalsFormState, value: string) => {
    setVitals((prev) => ({ ...prev, [field]: value }))
  }

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const filesArray = event.target.files ? Array.from(event.target.files) : []
    if (filesArray.length > 0) {
      setAttachments((prev) => [...prev, ...filesArray])
    }
  }

  const handleCameraCapture = (event: ChangeEvent<HTMLInputElement>) => {
    const filesArray = event.target.files ? Array.from(event.target.files) : []
    if (filesArray.length > 0) {
      setAttachments((prev) => [...prev, ...filesArray])
    }
  }

  const triggerCameraCapture = () => {
    cameraInputRef.current?.click()
  }

  const handleRemoveAttachment = (index: number) => {
    setAttachments((prev) => prev.filter((_, itemIndex) => itemIndex !== index))
  }

  const setRecordLoadingState = (id: string, value: boolean) => {
    setRecordActionLoading((prev) => ({
      ...prev,
      [id]: value,
    }))
  }

  const mergeRecord = (updated: ConsultationRecord) => {
    setRecords((prev) =>
      prev.map((record) => (record.id === updated.id ? { ...record, ...updated } : record)),
    )
  }

  const handleRefreshRecord = async (record: ConsultationRecord) => {
    try {
      setRecordLoadingState(record.id, true)
      const status = await manageAPI.getRecordStatus(record.id)
      mergeRecord({
        ...record,
        status: status.status,
        processed_at: status.processed_at ?? record.processed_at,
        confidence: status.confidence ?? record.confidence,
        needs_review: status.needs_review,
        processing_notes: status.processing_notes ?? record.processing_notes,
        processing_metadata: status.metadata ?? record.processing_metadata,
        timeline_event_ids: status.timeline_event_ids ?? record.timeline_event_ids,
      })
      toast({
        title: 'Record status updated',
        description: `${record.original_filename ?? 'Document'} is now ${status.status}.`,
      })
    } catch (error) {
      toast({
        title: 'Unable to refresh status',
        description:
          error instanceof Error ? error.message : 'Please try again in a few moments.',
        variant: 'destructive',
      })
    } finally {
      setRecordLoadingState(record.id, false)
    }
  }

  const handleReviewRecord = async (
    record: ConsultationRecord,
    resolution: 'approved' | 'needs_review',
  ) => {
    try {
      setRecordLoadingState(record.id, true)
      const updated = await manageAPI.reviewConsultationRecord(record.id, {
        resolution,
      })
      mergeRecord(updated)
      toast({
        title: resolution === 'approved' ? 'Record cleared' : 'Record flagged',
        description:
          resolution === 'approved'
            ? 'The document no longer requires review.'
            : 'The document has been flagged for follow-up.',
      })
    } catch (error) {
      toast({
        title: 'Unable to update record',
        description:
          error instanceof Error ? error.message : 'Please try again in a few moments.',
        variant: 'destructive',
      })
    } finally {
      setRecordLoadingState(record.id, false)
    }
  }

  const validateVitals = (): string | null => {
    const requiredFields: Array<keyof VitalsFormState> = [
      'heart_rate',
      'blood_pressure_systolic',
      'blood_pressure_diastolic',
      'respiratory_rate',
      'temperature_celsius',
      'oxygen_saturation',
    ]

    for (const key of requiredFields) {
      if (!vitals[key]) {
        return 'All vital fields except weight are required.'
      }
    }

    const numberChecks: Array<[keyof VitalsFormState, number, number]> = [
      ['heart_rate', 20, 250],
      ['blood_pressure_systolic', 50, 300],
      ['blood_pressure_diastolic', 20, 200],
      ['respiratory_rate', 5, 80],
      ['temperature_celsius', 25, 45],
      ['oxygen_saturation', 50, 100],
    ]

    for (const [field, min, max] of numberChecks) {
      const parsed = Number(vitals[field])
      if (Number.isNaN(parsed) || parsed < min || parsed > max) {
        return `Please enter a valid value for ${field.replace(/_/g, ' ')}.`
      }
    }

    if (vitals.weight_kg) {
      const weight = Number(vitals.weight_kg)
      if (Number.isNaN(weight) || weight < 2 || weight > 400) {
        return 'Please enter a valid weight in kilograms.'
      }
    }

    return null
  }

  const handleSubmitVitals = async () => {
    if (!selectedPatient?.consultation_id) {
      toast({
        title: 'No consultation',
        description: 'Unable to determine consultation for this patient.',
        variant: 'destructive',
      })
      return
    }

    const validationError = validateVitals()
    if (validationError) {
      toast({ title: 'Invalid vitals', description: validationError, variant: 'destructive' })
      return
    }

    setIsSubmitting(true)
    try {
      const wasInTriage = selectedPatient.status === 'triage'
      await manageAPI.submitVitals(selectedPatient.consultation_id, {
        heart_rate: Number(vitals.heart_rate),
        blood_pressure_systolic: Number(vitals.blood_pressure_systolic),
        blood_pressure_diastolic: Number(vitals.blood_pressure_diastolic),
        respiratory_rate: Number(vitals.respiratory_rate),
        temperature_celsius: Number(vitals.temperature_celsius),
        oxygen_saturation: Number(vitals.oxygen_saturation),
        weight_kg: vitals.weight_kg ? Number(vitals.weight_kg) : undefined,
      })

      if (attachments.length > 0) {
        try {
          const uploadResult = await manageAPI.uploadConsultationRecords(
            selectedPatient.consultation_id,
            attachments,
            attachmentNotes.trim() || undefined,
          )
          setRecords(uploadResult.records ?? [])
          queryClient.setQueryData(
            ['consultation-records', selectedPatient.consultation_id],
            uploadResult,
          )
          setAttachments([])
          setAttachmentNotes('')
          if (fileInputRef.current) {
            fileInputRef.current.value = ''
          }
          if (cameraInputRef.current) {
            cameraInputRef.current.value = ''
          }
          toast({
            title: 'Records uploaded',
            description: `${uploadResult.records?.length ?? attachments.length} file(s) attached to consultation.`,
          })
        } catch (uploadError) {
          const message = uploadError instanceof Error 
            ? uploadError.message 
            : (uploadError as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Failed to upload records.'
          toast({
            title: 'Record upload failed',
            description: message,
            variant: 'destructive',
          })
          setIsSubmitting(false)
          return
        }
      }

      toast({
        title: wasInTriage ? 'Vitals updated' : 'Vitals recorded',
        description: wasInTriage
          ? `${selectedPatient.patient_name}'s vitals have been updated.`
          : `${selectedPatient.patient_name} moved to triage.`,
      })

      handleCloseDialog()
      refetch()
    } catch (error) {
      const message = error instanceof Error 
        ? error.message 
        : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Failed to submit vitals.'
      toast({ title: 'Submission failed', description: message, variant: 'destructive' })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-gray-500">
              Awaiting Vitals
            </CardTitle>
          </CardHeader>
          <CardContent className="text-2xl font-semibold">
            {awaitingVitals.length}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-gray-500">
              Average Wait
            </CardTitle>
          </CardHeader>
          <CardContent className="text-2xl font-semibold">
            {formatMinutes(metrics.averageWait)}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-gray-500">
              Longest Wait
            </CardTitle>
          </CardHeader>
          <CardContent className="text-2xl font-semibold">
            {formatMinutes(longestWait)}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-gray-500">
              In Triage
            </CardTitle>
          </CardHeader>
          <CardContent className="text-2xl font-semibold">
            {inTriage.length}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Patients Awaiting Vitals</CardTitle>
        </CardHeader>
        <CardContent>
          {isError ? (
            <p className="text-sm text-red-500">
              Unable to load queue data. Please try again shortly.
            </p>
          ) : isLoading ? (
            <p className="text-sm text-gray-500">Loading queue…</p>
          ) : (
            <div className="space-y-6">
              <section>
                <h3 className="mb-3 text-base font-semibold">Awaiting Vitals</h3>
                {awaitingVitals.length === 0 ? (
                  <div className="rounded border border-dashed p-8 text-center text-sm text-gray-500">
                    No patients are currently waiting for vitals.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {awaitingVitals
                      .slice()
                      .sort(
                        (a, b) =>
                          (b.triage_level ?? 3) - (a.triage_level ?? 3) ||
                          (b.wait_time_minutes ?? 0) - (a.wait_time_minutes ?? 0),
                      )
                      .map((patient) => (
                        <div
                          key={patient.consultation_id}
                          className="flex flex-col gap-3 rounded-lg border p-4 sm:flex-row sm:items-center sm:justify-between"
                        >
                          <div className="space-y-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="text-lg font-semibold">
                                {patient.patient_name}
                              </span>
                              <Badge variant="secondary">
                                {patient.age != null ? `Age ${patient.age}` : 'Age n/a'}
                              </Badge>
                              <Badge>{`ESI ${patient.triage_level ?? '—'}`}</Badge>
                            </div>
                            <p className="text-sm text-gray-600">
                              Chief complaint: {patient.chief_complaint ?? 'Not provided'}
                            </p>
                            <div className="flex flex-wrap gap-3 text-xs text-gray-500">
                              <span>Waiting {formatMinutes(patient.wait_time_minutes)}</span>
                              <span>Checked in at {formatTime(patient.check_in_time)}</span>
                              {patient.estimated_wait_minutes != null && (
                                <span>
                                  Est. remaining {formatMinutes(patient.estimated_wait_minutes)}
                                </span>
                              )}
                            </div>
                          </div>
                          <Button
                            onClick={() => handleOpenVitals(patient)}
                            className="w-full sm:w-auto"
                          >
                            Record Vitals
                          </Button>
                        </div>
                      ))}
                  </div>
                )}
              </section>

              <section>
                <h3 className="mb-3 text-base font-semibold">In Triage</h3>
                {inTriage.length === 0 ? (
                  <div className="rounded border border-dashed p-8 text-center text-sm text-gray-500">
                    No patients are currently being triaged.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {inTriage
                      .slice()
                      .sort(
                        (a, b) =>
                          (b.triage_level ?? 3) - (a.triage_level ?? 3) ||
                          (b.wait_time_minutes ?? 0) - (a.wait_time_minutes ?? 0),
                      )
                      .map((patient) => (
                        <div
                          key={patient.consultation_id}
                          className="flex flex-col gap-3 rounded-lg border border-blue-100 bg-blue-50/40 p-4 sm:flex-row sm:items-center sm:justify-between"
                        >
                          <div className="space-y-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="text-lg font-semibold">
                                {patient.patient_name}
                              </span>
                              <Badge variant="outline" className="border-blue-500 text-blue-600">
                                In triage
                              </Badge>
                              <Badge>{`ESI ${patient.triage_level ?? '—'}`}</Badge>
                            </div>
                            <p className="text-sm text-gray-600">
                              Chief complaint: {patient.chief_complaint ?? 'Not provided'}
                            </p>
                            <div className="flex flex-wrap gap-3 text-xs text-gray-500">
                              <span>Waiting {formatMinutes(patient.wait_time_minutes)}</span>
                              {patient.assigned_doctor && <span>{patient.assigned_doctor}</span>}
                              {patient.estimated_wait_minutes != null && (
                                <span>
                                  Est. remaining {formatMinutes(patient.estimated_wait_minutes)}
                                </span>
                              )}
                            </div>
                          </div>
                          <Button
                            onClick={() => handleOpenVitals(patient)}
                            variant="outline"
                            className="w-full sm:w-auto"
                          >
                            Edit Vitals
                          </Button>
                        </div>
                      ))}
                  </div>
                )}
              </section>
            </div>
          )}
        </CardContent>
      </Card>

      <Dialog open={isDialogOpen} onOpenChange={handleCloseDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>
              {selectedPatient && selectedPatient.status === 'triage'
                ? `Update Vitals – ${selectedPatient.patient_name}`
                : `Record Vitals ${
                    selectedPatient ? `– ${selectedPatient.patient_name}` : ''
                  }`}
            </DialogTitle>
          </DialogHeader>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="heart_rate">Heart rate (bpm)</Label>
              <Input
                id="heart_rate"
                type="number"
                min={20}
                max={250}
                value={vitals.heart_rate}
                onChange={(event) => handleInputChange('heart_rate', event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="blood_pressure_systolic">Blood pressure (systolic)</Label>
              <Input
                id="blood_pressure_systolic"
                type="number"
                min={50}
                max={300}
                value={vitals.blood_pressure_systolic}
                onChange={(event) =>
                  handleInputChange('blood_pressure_systolic', event.target.value)
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="blood_pressure_diastolic">Blood pressure (diastolic)</Label>
              <Input
                id="blood_pressure_diastolic"
                type="number"
                min={20}
                max={200}
                value={vitals.blood_pressure_diastolic}
                onChange={(event) =>
                  handleInputChange('blood_pressure_diastolic', event.target.value)
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="respiratory_rate">Respiratory rate (breaths/min)</Label>
              <Input
                id="respiratory_rate"
                type="number"
                min={5}
                max={80}
                value={vitals.respiratory_rate}
                onChange={(event) =>
                  handleInputChange('respiratory_rate', event.target.value)
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="temperature_celsius">Temperature (°C)</Label>
              <Input
                id="temperature_celsius"
                type="number"
                step="0.1"
                min={25}
                max={45}
                value={vitals.temperature_celsius}
                onChange={(event) =>
                  handleInputChange('temperature_celsius', event.target.value)
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="oxygen_saturation">Oxygen saturation (%)</Label>
              <Input
                id="oxygen_saturation"
                type="number"
                min={50}
                max={100}
                value={vitals.oxygen_saturation}
                onChange={(event) =>
                  handleInputChange('oxygen_saturation', event.target.value)
                }
              />
            </div>
            <div className="space-y-2 md:col-span-2">
              <Label htmlFor="weight_kg">Weight (kg, optional)</Label>
              <Input
                id="weight_kg"
                type="number"
                min={2}
                max={400}
                value={vitals.weight_kg}
                onChange={(event) =>
                  handleInputChange('weight_kg', event.target.value)
                }
              />
            </div>
          </div>

          <div className="space-y-3">
            <div className="space-y-2">
              <Label htmlFor="record_files">Attach past records or reports</Label>
              <Input
                id="record_files"
                type="file"
                accept=".pdf,.doc,.docx,.jpg,.jpeg,.png,.tiff,.txt"
                multiple
                ref={fileInputRef}
                onChange={handleFileChange}
                disabled={isSubmitting}
              />
              <Input
                id="record_camera"
                type="file"
                accept="image/*"
                capture="environment"
                ref={cameraInputRef}
                onChange={handleCameraCapture}
                disabled={isSubmitting}
                className="hidden"
              />
              <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
                <span>Files will be shared with the care team as part of the consultation record.</span>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="text-xs"
                  onClick={triggerCameraCapture}
                  disabled={isSubmitting}
                >
                  <Camera className="mr-2 h-4 w-4" />
                  Capture image
                </Button>
              </div>
            </div>

            {attachments.length > 0 && (
              <div className="rounded-md border border-gray-200 bg-gray-50 p-3 space-y-2">
                <p className="text-sm font-semibold text-gray-700">Pending uploads</p>
                <ul className="space-y-1 text-sm text-gray-600">
                  {attachments.map((file, index) => (
                    <li key={`${file.name}-${index}`} className="flex items-center justify-between gap-3">
                      <span className="truncate">{file.name}</span>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => handleRemoveAttachment(index)}
                        disabled={isSubmitting}
                      >
                        Remove
                      </Button>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="record_notes">Notes for attached records (optional)</Label>
              <Textarea
                id="record_notes"
                value={attachmentNotes}
                onChange={(event) => setAttachmentNotes(event.target.value)}
                placeholder="Add context about the uploaded documents (e.g. 'Previous lab results from last visit')."
                rows={3}
                disabled={isSubmitting}
              />
            </div>
          </div>

          {records.length > 0 && (
            <div className="rounded-md border border-emerald-200 bg-emerald-50/50 p-3 space-y-2">
              <p className="text-sm font-semibold text-emerald-800">Existing records for this consultation</p>
              <ul className="space-y-2 text-sm text-emerald-900">
                {records.map((record) => {
                  const loading = Boolean(recordActionLoading[record.id])
                  const confidence =
                    typeof record.confidence === 'number'
                      ? `${Math.round(record.confidence * 100)}% confidence`
                      : null
                  return (
                    <li key={record.id} className="rounded border border-emerald-200/60 bg-white/70 p-3">
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
                        <div className="space-y-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="font-medium">{record.original_filename ?? 'Record'}</span>
                            <Badge variant="outline" className="uppercase tracking-wide">
                              {record.status}
                            </Badge>
                            {record.needs_review && (
                              <Badge variant="destructive" className="uppercase">
                                Needs review
                              </Badge>
                            )}
                            {confidence && <Badge variant="secondary">{confidence}</Badge>}
                          </div>
                          <span className="block text-xs text-emerald-700">
                            Uploaded {new Date(record.uploaded_at).toLocaleString()} by{' '}
                            {record.uploaded_by ?? 'Unknown'}
                          </span>
                          {record.processed_at && (
                            <span className="block text-xs text-emerald-700">
                              Processed {new Date(record.processed_at).toLocaleString()}
                            </span>
                          )}
                          {record.processing_notes && (
                            <p className="text-xs text-emerald-800">{record.processing_notes}</p>
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
                            className="text-sm font-medium text-emerald-700 hover:text-emerald-900 cursor-pointer"
                          >
                            View document
                          </button>
                          <div className="flex flex-wrap gap-2">
                            <Button
                              type="button"
                              size="sm"
                              variant="outline"
                              onClick={() => handleRefreshRecord(record)}
                              disabled={loading}
                            >
                              {loading ? (
                                <>
                                  <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
                                  Updating…
                                </>
                              ) : (
                                'Refresh status'
                              )}
                            </Button>
                            {record.needs_review ? (
                              <Button
                                type="button"
                                size="sm"
                                onClick={() => handleReviewRecord(record, 'approved')}
                                disabled={loading}
                              >
                                Mark reviewed
                              </Button>
                            ) : (
                              <Button
                                type="button"
                                size="sm"
                                variant="secondary"
                                onClick={() => handleReviewRecord(record, 'needs_review')}
                                disabled={loading}
                              >
                                Flag for review
                              </Button>
                            )}
                          </div>
                        </div>
                      </div>
                    </li>
                  )
                })}
              </ul>
            </div>
          )}

          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={handleCloseDialog}>
              Cancel
            </Button>
            <Button onClick={handleSubmitVitals} disabled={isSubmitting}>
              {isSubmitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Saving…
                </>
              ) : (
                'Submit vitals'
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default NurseDashboard