import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

const VitalsForm = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Vitals Capture (coming soon)</CardTitle>
      </CardHeader>
      <CardContent className="text-sm text-gray-600 space-y-2">
        <p>This screen will support guided vitals entry, normal range alerts, and AI-assisted triage hand-off once the workflow is implemented.</p>
      </CardContent>
    </Card>
  )
}

export default VitalsForm