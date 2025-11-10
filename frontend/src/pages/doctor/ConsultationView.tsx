import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

const ConsultationView = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Consultation View (coming soon)</CardTitle>
      </CardHeader>
      <CardContent className="text-sm text-gray-600 space-y-2">
        <p>This page will show detailed consultation information, AI transcription, and note editing workflows once connected to the backend services.</p>
      </CardContent>
    </Card>
  )
}

export default ConsultationView