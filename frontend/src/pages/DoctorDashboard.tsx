import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useAuthStore } from '@/store/authStore'

const DoctorDashboard = () => {
  const { user } = useAuthStore()

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Welcome, {user?.full_name ?? 'Doctor'}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-gray-600">
          <p>An interactive list of assigned consultations, AI notes, and patient summaries will be shown here once the integrations are finished.</p>
          <p>Sample workflow steps:</p>
          <ul className="list-disc list-inside">
            <li>Review incoming consultations with AI-generated notes.</li>
            <li>Approve or edit AI scribe output before signing.</li>
            <li>Access longitudinal patient history summaries.</li>
          </ul>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Consultation Queue (coming soon)</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-gray-500">
          Once wired, this section will display real-time consultant workload, SLA timers, and escalation notifications.
        </CardContent>
      </Card>
    </div>
  )
}

export default DoctorDashboard