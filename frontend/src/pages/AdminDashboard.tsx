import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useAuthStore } from '@/store/authStore'

const AdminDashboard = () => {
  const { user } = useAuthStore()

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Welcome, {user?.full_name ?? 'Administrator'}</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-gray-600 space-y-2">
          <p>This dashboard will surface organisation-level metrics, AI usage, and compliance alerts once reporting endpoints are ready.</p>
          <p>Example metrics:</p>
          <ul className="list-disc list-inside">
            <li>AI scribe usage and cost controls</li>
            <li>Triage SLA compliance</li>
            <li>Patient throughput and queue statistics</li>
          </ul>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>System Health Overview (coming soon)</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-gray-500">
          This section will show live service status, background job queues, and audit insights once the observability stack is connected.
        </CardContent>
      </Card>
    </div>
  )
}

export default AdminDashboard
