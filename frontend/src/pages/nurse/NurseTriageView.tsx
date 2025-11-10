import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

const NurseTriageView = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Triage Workflow (coming soon)</CardTitle>
      </CardHeader>
      <CardContent className="text-sm text-gray-600 space-y-2">
        <p>This page will guide the nurse through vitals capture and AI-assisted prioritisation once the workflow API is connected.</p>
        <p>Future enhancements will include:</p>
        <ul className="list-disc list-inside">
          <li>Real-time queue of patients awaiting triage.</li>
          <li>Guided vitals capture with normal range warnings.</li>
          <li>Automatic handoff to AI triage and physician queue.</li>
        </ul>
      </CardContent>
    </Card>
  )
}

export default NurseTriageView