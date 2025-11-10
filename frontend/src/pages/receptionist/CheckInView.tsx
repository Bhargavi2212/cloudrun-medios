import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

const CheckInView = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Patient Check-In (coming soon)</CardTitle>
      </CardHeader>
      <CardContent className="text-sm text-gray-600 space-y-2">
        <p>Once connected to backend services, this interface will guide receptionists through patient selection, complaint capture, and queue placement.</p>
      </CardContent>
    </Card>
  )
}

export default CheckInView