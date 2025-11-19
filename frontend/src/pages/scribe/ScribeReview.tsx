import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { aiScribeAPI } from '@/services/api'
import type { ScribeSessionDetails, SoapNote, SoapNoteContent, TriagePredictionSnapshot } from '@/types'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardHeader, CardContent, CardTitle } from '@/components/ui/card'
import { useToast } from '@/components/ui/use-toast'

const ScribeReviewPage = () => {
  const params = useParams<{ sessionId?: string }>()
  const [lookupId, setLookupId] = useState(params.sessionId ?? '')
  const [details, setDetails] = useState<ScribeSessionDetails | null>(null)
  const [selectedNote, setSelectedNote] = useState<SoapNote | null>(null)
  const [noteContent, setNoteContent] = useState<SoapNoteContent | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const { toast } = useToast()

  useEffect(() => {
    if (params.sessionId) {
      void fetchDetails(params.sessionId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.sessionId])

  const fetchDetails = async (sessionId: string) => {
    setIsLoading(true)
    try {
      const payload = await aiScribeAPI.getSession(sessionId)
      setDetails(payload)
      const newestNote = payload.notes[0] ?? null
      setSelectedNote(newestNote)
      setNoteContent(newestNote?.content ?? null)
    } catch (error) {
      console.error(error)
      toast({ title: 'Unable to load session', description: 'Please verify the session ID.', variant: 'destructive' })
    } finally {
      setIsLoading(false)
    }
  }

  const handleSaveNote = async () => {
    if (!selectedNote || !noteContent) return
    setIsSaving(true)
    try {
      const response = await aiScribeAPI.updateNote(selectedNote.id, { content: noteContent })
      setSelectedNote(response.note)
      toast({ title: 'Note updated', description: 'Edits saved successfully.' })
    } catch (error) {
      console.error(error)
      toast({ title: 'Failed to save note', description: 'Please try again.', variant: 'destructive' })
    } finally {
      setIsSaving(false)
    }
  }

  const handleDownloadPdf = async () => {
    if (!details) return
    const blob = await aiScribeAPI.downloadPdf(details.session.id)
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `soap-note-${details.session.id}.pdf`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const handleExportFhir = async () => {
    if (!details) return
    const document = await aiScribeAPI.exportFhir(details.session.id)
    const blob = new Blob([JSON.stringify(document, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `soap-note-${details.session.id}.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const renderSection = (section: keyof SoapNoteContent, label: string) => {
    const summary = (noteContent?.[section] as Record<string, unknown> | undefined)?.summary as string | undefined
    return (
      <div key={section} className="space-y-2">
        <p className="text-sm font-medium">{label}</p>
        <Textarea
          value={summary ?? ''}
          onChange={(event) => {
            setNoteContent((prev) => ({
              ...(prev ?? {}),
              [section]: {
                ...(prev?.[section] as Record<string, unknown> | undefined),
                summary: event.target.value,
              },
            }))
          }}
          rows={4}
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-4">
        <Input placeholder="Session ID" value={lookupId} onChange={(e) => setLookupId(e.target.value)} className="max-w-sm" />
        <Button onClick={() => fetchDetails(lookupId)} disabled={!lookupId || isLoading}>
          {isLoading ? 'Loading…' : 'Load session'}
        </Button>
      </div>

      {details && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Session Overview</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-500">Session ID</p>
                <p className="font-medium">{details.session.id}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Status</p>
                <p className="font-medium capitalize">{details.session.status}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Transcript lines</p>
                <p className="font-medium">{details.segments.length}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Vitals recorded</p>
                <p className="font-medium">{details.vitals.length}</p>
              </div>
            </CardContent>
          </Card>

          {selectedNote && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>SOAP Note Draft</CardTitle>
                <div className="space-x-2">
                  <Button variant="outline" onClick={handleDownloadPdf}>
                    Download PDF
                  </Button>
                  <Button variant="outline" onClick={handleExportFhir}>
                    Export FHIR
                  </Button>
                  <Button onClick={handleSaveNote} disabled={isSaving}>
                    {isSaving ? 'Saving…' : 'Save Edits'}
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {['subjective', 'objective', 'assessment', 'plan'].map((section) =>
                  renderSection(section as keyof SoapNoteContent, section.toUpperCase()),
                )}
              </CardContent>
            </Card>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Vitals Timeline</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 max-h-64 overflow-y-auto">
                {details.vitals.map((vital) => (
                  <div key={vital.id} className="text-sm border-b pb-2">
                    <p className="font-medium">{new Date(vital.recorded_at).toLocaleString()}</p>
                    <p>
                      HR {vital.heart_rate ?? '--'} · RR {vital.respiratory_rate ?? '--'} · BP {vital.systolic_bp ?? '--'}/
                      {vital.diastolic_bp ?? '--'}
                    </p>
                  </div>
                ))}
                {details.vitals.length === 0 && <p className="text-sm text-gray-500">No vitals available.</p>}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Triage Predictions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {details.triage_predictions.map((prediction: TriagePredictionSnapshot) => (
                  <div key={prediction.id} className="text-sm border-b pb-2">
                    <p className="font-medium">
                      ESI {prediction.esi_level} · {new Date(prediction.created_at).toLocaleTimeString()}
                    </p>
                    <p>Flagged: {prediction.flagged ? 'Yes' : 'No'}</p>
                    {prediction.probabilities && (
                      <pre className="text-xs bg-gray-50 rounded p-2 mt-1">
                        {JSON.stringify(prediction.probabilities, null, 2)}
                      </pre>
                    )}
                  </div>
                ))}
                {details.triage_predictions.length === 0 && <p className="text-sm text-gray-500">No predictions yet.</p>}
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}

export default ScribeReviewPage

