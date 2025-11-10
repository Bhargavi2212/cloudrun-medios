import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

const AdminSettingsPage = () => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Administration Settings (coming soon)</CardTitle>
      </CardHeader>
      <CardContent className="text-sm text-gray-600 space-y-2">
        <p>This page will allow administrators to manage facility settings, user permissions, and AI configurations.</p>
        <p>Planned capabilities include:</p>
        <ul className="list-disc list-inside">
          <li>User and role provisioning</li>
          <li>AI usage and cost controls</li>
          <li>Operational alert thresholds</li>
        </ul>
      </CardContent>
    </Card>
  )
}

export default AdminSettingsPage
