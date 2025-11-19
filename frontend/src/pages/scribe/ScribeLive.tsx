import { useState } from 'react'
import { Link } from 'react-router-dom'
import { aiScribeAPI } from '@/services/api'
import type { ScribeSession } from '@/types'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useToast } from '@/components/ui/use-toast'
import { VitalsPanel } from '@/components/scribe/VitalsPanel'
import { useScribeStreaming } from '@/hooks/useScribeStreaming'

const ScribeLivePage = () => {
  const [session, setSession] = useState<ScribeSession | null>(null)
  const [consultationId, setConsultationId] = useState('')
  const [patientId, setPatientId] = useState('')
  const [isCreating, setIsCreating] = useState(false)
  const [isFinalizing, setIsFinalizing] = useState(false)
  const { toast } = useToast()
  const stream = useScribeStreaming({ sessionId: session?.id })

  const handleCreateSession = async () => {
    if (!consultationId || !patientId) return
    setIsCreating(true)
    try {
      const response = await aiScribeAPI.createSession({
        consultation_id: consultationId,
        patient_id: patientId,
        language: 'en',
      })
      setSession(response.session)
      toast({ title: 'Session created', description: 'You can now start streaming audio and vitals.' })
    } catch (error) {
      console.error(error)
      toast({ title: 'Failed to create session', description: 'Please verify the IDs and try again.', variant: 'destructive' })
    } finally {
      setIsCreating(false)
    }
  }

  const handleFinalize = async () => {
    if (!session) return
    setIsFinalizing(true)
    try {
      await aiScribeAPI.finalizeSession(session.id)
      toast({ title: 'SOAP note generated', description: 'Review the draft in the Scribe Review workspace.' })
    } catch (error) {
      console.error(error)
      toast({ title: 'Finalization failed', description: 'Ensure transcripts exist before finalizing.', variant: 'destructive' })
    } finally {
      setIsFinalizing(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">AI Scribe Live</h1>
        {session && (
          <div className="space-x-3">
            <Button variant="outline" onClick={stream.isStreaming ? stream.stopStreaming : stream.startStreaming}>
              {stream.isStreaming ? 'Stop Streaming' : 'Start Streaming'}
            </Button>
            <Button onClick={handleFinalize} disabled={isFinalizing}>
              {isFinalizing ? 'Generating…' : 'Finalize Session'}
            </Button>
            <Button variant="secondary" asChild>
              <Link to={`/scribe/review/${session.id}`}>Open Review</Link>
            </Button>
          </div>
        )}
      </div>

      {!session ? (
        <Card>
          <CardHeader>
            <CardTitle>Start a new encounter</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="consultationId">Consultation ID</Label>
                <Input id="consultationId" value={consultationId} onChange={(e) => setConsultationId(e.target.value)} />
              </div>
              <div>
                <Label htmlFor="patientId">Patient ID</Label>
                <Input id="patientId" value={patientId} onChange={(e) => setPatientId(e.target.value)} />
              </div>
            </div>
            <Button onClick={handleCreateSession} disabled={isCreating || !consultationId || !patientId}>
              {isCreating ? 'Creating…' : 'Create Session'}
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>Live Transcript</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between pb-4">
                <div className="space-x-2">
                  <Button size="sm" variant={stream.speaker === 'doctor' ? 'default' : 'outline'} onClick={() => stream.setSpeaker('doctor')}>
                    Doctor
                  </Button>
                  <Button size="sm" variant={stream.speaker === 'patient' ? 'default' : 'outline'} onClick={() => stream.setSpeaker('patient')}>
                    Patient
                  </Button>
                </div>
                <span className="text-sm text-muted-foreground">
                  {stream.isStreaming ? 'Streaming audio…' : 'Audio idle'}
                </span>
              </div>
              <div className="h-72 overflow-y-auto space-y-3 rounded-md border p-3 bg-gray-50">
                {stream.segments.slice(-50).map((segment) => (
                  <div key={segment.id}>
                    <p className="text-xs uppercase text-gray-500">
                      {segment.speaker_label ?? 'Unknown'} · {(segment.confidence ?? 0).toFixed(2)}
                    </p>
                    <p className="text-sm">{segment.text}</p>
                  </div>
                ))}
                {stream.segments.length === 0 && <p className="text-sm text-gray-500">Waiting for audio…</p>}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Manual Vitals</CardTitle>
            </CardHeader>
            <CardContent>
              <VitalsPanel sessionId={session.id} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recent Vitals</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {stream.vitals.slice(-5).map((vital) => (
                <div key={vital.id} className="text-sm border-b pb-2">
                  <p className="font-medium">{new Date(vital.recorded_at).toLocaleTimeString()}</p>
                  <p>HR: {vital.heart_rate ?? '--'} bpm · RR: {vital.respiratory_rate ?? '--'}</p>
                  <p>
                    BP: {vital.systolic_bp ?? '--'}/{vital.diastolic_bp ?? '--'} · Temp: {vital.temperature_c ?? '--'}
                  </p>
                </div>
              ))}
              {stream.vitals.length === 0 && <p className="text-sm text-gray-500">No vitals yet.</p>}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>ESI Prediction</CardTitle>
            </CardHeader>
            <CardContent>
              {stream.triage ? (
                <div>
                  <p className="text-3xl font-semibold">ESI {stream.triage.esi_level}</p>
                  <p className="text-sm text-gray-500">
                    Flagged: {stream.triage.flagged ? 'Yes' : 'No'}
                  </p>
                  <pre className="text-xs bg-gray-50 rounded p-2 mt-2">
                    {JSON.stringify(stream.triage.probabilities, null, 2)}
                  </pre>
                </div>
              ) : (
                <p className="text-sm text-gray-500">Run finalization to view predictions.</p>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

export default ScribeLivePage

