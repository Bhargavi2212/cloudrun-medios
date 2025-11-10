import { useMemo, useState } from 'react'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { useQueueData } from '@/hooks/useQueueData'
import { usePatientSearch } from '@/hooks/usePatientSearch'
import { usePatientCreate } from '@/hooks/usePatientCreate'
import { manageAPI } from '@/services/api'
import type { QueuePatient, Patient, QueueStage } from '@/types'
import {
  ResponsiveContainer,
  Pie,
  PieChart,
  Cell,
  Tooltip,
} from 'recharts'
import { useToast } from '@/components/ui/use-toast'
import { MetricCard } from '@/components/dashboard/MetricCard'
import { QueueFilters } from '@/components/dashboard/QueueFilters'
import { MetricsChart } from '@/components/dashboard/MetricsChart'
import { Wifi } from 'lucide-react'

const TRIAGE_COLORS = ['#ef4444', '#f97316', '#facc15', '#22c55e', '#3b82f6']

const normalizeStatus = (status: string) => status.replace(/_/g, ' ')

const getTriageLabel = (level?: number | null) => {
  if (!level) return 'Unknown'
  return `ESI ${level}`
}

const formatMinutes = (minutes: number | null | undefined) =>
  typeof minutes === 'number' ? `${Math.round(minutes)} min` : '—'

const buildTriageData = (patients: QueuePatient[]) => {
  const buckets = new Map<number, number>()
  patients.forEach((patient) => {
    if (!patient.triage_level) return
    buckets.set(patient.triage_level, (buckets.get(patient.triage_level) ?? 0) + 1)
  })

  return Array.from(buckets.entries())
    .sort(([a], [b]) => a - b)
    .map(([level, count]) => ({ level, count }))
}

const ReceptionistDashboard = () => {
  const { patients, metrics, isLoading, isError, refetch } = useQueueData()
  const triageData = useMemo(() => buildTriageData(patients), [patients])
  const { toast } = useToast()

  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<QueueStage | 'all'>('all')
  const [triageFilter, setTriageFilter] = useState<number | 'all'>('all')
  const [tableSearchQuery, setTableSearchQuery] = useState('')
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null)
  const [chiefComplaint, setChiefComplaint] = useState('')
  const [isCheckInPending, setIsCheckInPending] = useState(false)

  const [isNewPatientOpen, setIsNewPatientOpen] = useState(false)
  const [newPatientForm, setNewPatientForm] = useState({
    first_name: '',
    last_name: '',
    date_of_birth: '',
    sex: 'M' as 'M' | 'F' | 'Other',
    contact_phone: '',
    contact_email: '',
  })

  const { data: searchResults, isFetching: isSearching } = usePatientSearch(searchQuery)
  const createPatientMutation = usePatientCreate()

  // Filter patients based on filters
  const filteredPatients = useMemo(() => {
    let filtered = patients

    // Status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter((p) => p.status === statusFilter)
    }

    // Triage filter
    if (triageFilter !== 'all') {
      filtered = filtered.filter((p) => p.triage_level === triageFilter)
    }

    // Search filter
    if (tableSearchQuery.trim()) {
      const query = tableSearchQuery.toLowerCase()
      filtered = filtered.filter(
        (p) =>
          p.patient_name?.toLowerCase().includes(query) ||
          p.chief_complaint?.toLowerCase().includes(query)
      )
    }

    return filtered
  }, [patients, statusFilter, triageFilter, tableSearchQuery])

  // Chart data
  const statusChartData = useMemo(() => {
    const statusCounts = new Map<string, number>()
    patients.forEach((p) => {
      const status = normalizeStatus(p.status)
      statusCounts.set(status, (statusCounts.get(status) || 0) + 1)
    })
    return Array.from(statusCounts.entries()).map(([name, value]) => ({ name, value }))
  }, [patients])

  const waitTimeChartData = useMemo(() => {
    const buckets = ['0-10', '10-20', '20-30', '30-60', '60+']
    const counts = buckets.map(() => 0)
    patients.forEach((p) => {
      const wait = p.wait_time_minutes || 0
      if (wait < 10) counts[0]++
      else if (wait < 20) counts[1]++
      else if (wait < 30) counts[2]++
      else if (wait < 60) counts[3]++
      else counts[4]++
    })
    return buckets.map((name, index) => ({ name, value: counts[index] }))
  }, [patients])

  const activeFiltersCount = useMemo(() => {
    let count = 0
    if (statusFilter !== 'all') count++
    if (triageFilter !== 'all') count++
    if (tableSearchQuery.trim()) count++
    return count
  }, [statusFilter, triageFilter, tableSearchQuery])

  const handleClearFilters = () => {
    setStatusFilter('all')
    setTriageFilter('all')
    setTableSearchQuery('')
  }

  const handlePatientSelect = (patient: Patient) => {
    setSelectedPatient(patient)
    setSearchQuery('')
  }

  const handleCheckIn = async () => {
    if (!selectedPatient || !chiefComplaint.trim()) {
      toast({ title: 'Check-in requires a patient and chief complaint.', variant: 'destructive' })
      return
    }

    try {
      setIsCheckInPending(true)
      await manageAPI.checkInPatient({
        patient_id: selectedPatient.id,
        chief_complaint: chiefComplaint.trim(),
      })
      toast({
        title: 'Patient checked in',
        description: `${selectedPatient.first_name} ${selectedPatient.last_name} added to queue.`,
      })
      setSelectedPatient(null)
      setChiefComplaint('')
      refetch()
    } catch (error) {
      const message = error instanceof Error 
        ? error.message 
        : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Unable to check in patient'
      toast({ title: 'Check-in failed', description: message, variant: 'destructive' })
    } finally {
      setIsCheckInPending(false)
    }
  }

  const handleCreatePatient = async () => {
    if (!newPatientForm.first_name || !newPatientForm.last_name || !newPatientForm.date_of_birth) {
      toast({ title: 'First name, last name, and DOB are required', variant: 'destructive' })
      return
    }

    try {
      const payload = {
        first_name: newPatientForm.first_name.trim(),
        last_name: newPatientForm.last_name.trim(),
        date_of_birth: newPatientForm.date_of_birth,
        sex: newPatientForm.sex,
        contact_phone: newPatientForm.contact_phone || null,
        contact_email: newPatientForm.contact_email || null,
      }
      const patient = await createPatientMutation.mutateAsync(payload)
      toast({ title: 'Patient created', description: `${patient.first_name} ${patient.last_name}` })
      setSelectedPatient(patient)
      setIsNewPatientOpen(false)
      setNewPatientForm({
        first_name: '',
        last_name: '',
        date_of_birth: '',
        sex: 'M',
        contact_phone: '',
        contact_email: '',
      })
    } catch (error) {
      const message = error instanceof Error 
        ? error.message 
        : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Unable to create patient'
      toast({ title: 'Creation failed', description: message, variant: 'destructive' })
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Receptionist Dashboard</h1>
          <p className="text-sm text-gray-500">Real-time patient queue management</p>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <Wifi className="h-4 w-4" />
          <span>Live updates enabled</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard title="Total in Queue" value={metrics.total} accent="bg-blue-100 text-blue-700" />
        <MetricCard title="Awaiting Vitals" value={metrics.awaitingVitals} accent="bg-yellow-100 text-yellow-700" />
        <MetricCard title="In Consultation" value={metrics.inConsultation} accent="bg-green-100 text-green-700" />
        <MetricCard title="Average Wait" value={`${Math.round(metrics.averageWait)} min`} accent="bg-purple-100 text-purple-700" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MetricsChart title="Patients by Status" data={statusChartData} type="bar" color="#3b82f6" />
        <MetricsChart title="Wait Time Distribution" data={waitTimeChartData} type="bar" color="#10b981" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-6 lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Patient Check-In</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium text-gray-700">Search existing patient</label>
                <Input
                  placeholder="Type at least 2 characters…"
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                />
                {searchQuery.trim().length >= 2 && (
                  <div className="mt-2 border rounded-md max-h-48 overflow-y-auto">
                    {isSearching ? (
                      <p className="p-3 text-sm text-gray-500">Searching…</p>
                    ) : (searchResults?.length ?? 0) === 0 ? (
                      <p className="p-3 text-sm text-gray-500">No matches found.</p>
                    ) : (
                      searchResults!.map((patient) => (
                        <button
                          key={patient.id}
                          onClick={() => handlePatientSelect(patient)}
                          className="w-full text-left px-3 py-2 hover:bg-blue-50"
                        >
                          <div className="font-medium text-gray-900">
                            {patient.first_name} {patient.last_name}
                          </div>
                          <div className="text-xs text-gray-500">
                            DOB:{' '}
                            {patient.date_of_birth
                              ? new Date(patient.date_of_birth).toLocaleDateString()
                              : 'Unknown'}
                          </div>
                        </button>
                      ))
                    )}
                  </div>
                )}
              </div>

              <Button variant="outline" onClick={() => setIsNewPatientOpen(true)}>
                Register new patient
              </Button>

              {selectedPatient && (
                <div className="border rounded-md p-3 bg-blue-50 space-y-2">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-gray-800">
                        {selectedPatient.first_name} {selectedPatient.last_name}
                      </p>
                      <p className="text-xs text-gray-500">
                        DOB:{' '}
                        {selectedPatient.date_of_birth
                          ? new Date(selectedPatient.date_of_birth).toLocaleDateString()
                          : 'Unknown'}
                      </p>
                    </div>
                    <Button variant="ghost" size="sm" onClick={() => setSelectedPatient(null)}>
                      Clear
                    </Button>
                  </div>

                  <div>
                    <label className="text-sm font-medium text-gray-700">Chief Complaint</label>
                    <Textarea
                      rows={3}
                      value={chiefComplaint}
                      onChange={(event) => setChiefComplaint(event.target.value)}
                      placeholder="Describe the primary reason for the visit"
                    />
                  </div>

                  <Button
                    onClick={handleCheckIn}
                    disabled={!chiefComplaint.trim() || isCheckInPending}
                    className="w-full"
                  >
                    {isCheckInPending ? 'Checking in…' : 'Complete Check-In'}
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between gap-4">
            <CardTitle>Live Patient Queue</CardTitle>
            <button
              onClick={() => refetch()}
              className="text-sm text-blue-600 hover:underline"
            >
              Refresh
            </button>
          </CardHeader>
          <CardContent className="space-y-4">
            <QueueFilters
              searchQuery={tableSearchQuery}
              onSearchChange={setTableSearchQuery}
              statusFilter={statusFilter}
              onStatusFilterChange={setStatusFilter}
              triageFilter={triageFilter}
              onTriageFilterChange={setTriageFilter}
              activeFiltersCount={activeFiltersCount}
              onClearFilters={handleClearFilters}
            />

            {isError && (
              <p className="text-sm text-red-500">Unable to load queue data. Please try again shortly.</p>
            )}

            {isLoading ? (
              <p className="text-sm text-gray-500">Loading queue…</p>
            ) : (
              <>
                <div className="text-sm text-gray-600">
                  Showing {filteredPatients.length} of {patients.length} patients
                </div>
                <div className="border rounded-lg overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Patient</TableHead>
                        <TableHead>Chief Complaint</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Triage</TableHead>
                        <TableHead>Wait</TableHead>
                        <TableHead>Est. Wait</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredPatients.length > 0 ? (
                        filteredPatients.map((patient) => (
                          <TableRow key={patient.consultation_id}>
                            <TableCell className="font-medium">{patient.patient_name}</TableCell>
                            <TableCell className="max-w-[200px] truncate" title={patient.chief_complaint ?? undefined}>
                              {patient.chief_complaint ?? '—'}
                            </TableCell>
                            <TableCell>
                              <Badge variant="outline">{normalizeStatus(patient.status)}</Badge>
                            </TableCell>
                            <TableCell>
                              <Badge>{getTriageLabel(patient.triage_level)}</Badge>
                            </TableCell>
                            <TableCell>{formatMinutes(patient.wait_time_minutes)}</TableCell>
                            <TableCell>{formatMinutes(patient.estimated_wait_minutes ?? null)}</TableCell>
                          </TableRow>
                        ))
                      ) : (
                        <TableRow>
                          <TableCell colSpan={6} className="text-center text-sm text-gray-500 py-8">
                            {patients.length === 0
                              ? 'No patients currently in queue.'
                              : 'No patients match the current filters.'}
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Triage Distribution</CardTitle>
        </CardHeader>
        <CardContent className="h-72">
          {triageData.length === 0 ? (
            <p className="text-sm text-gray-500 text-center py-16">No triage data available yet.</p>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={triageData}
                  dataKey="count"
                  nameKey="level"
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={100}
                  label={({ level, count }) => `ESI ${level}: ${count}`}
                >
                  {triageData.map((entry) => (
                    <Cell key={`cell-${entry.level}`} fill={TRIAGE_COLORS[(entry.level - 1) % TRIAGE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value: number, name: string) => [
                    `${value} patient${value !== 1 ? 's' : ''}`,
                    `ESI Level ${name}`,
                  ]}
                />
              </PieChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      <Dialog open={isNewPatientOpen} onOpenChange={setIsNewPatientOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Register New Patient</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <Input
                placeholder="First name"
                value={newPatientForm.first_name}
                onChange={(event) => setNewPatientForm((prev) => ({ ...prev, first_name: event.target.value }))}
              />
              <Input
                placeholder="Last name"
                value={newPatientForm.last_name}
                onChange={(event) => setNewPatientForm((prev) => ({ ...prev, last_name: event.target.value }))}
              />
              <Input
                type="date"
                value={newPatientForm.date_of_birth}
                onChange={(event) => setNewPatientForm((prev) => ({ ...prev, date_of_birth: event.target.value }))}
              />
              <select
                className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={newPatientForm.sex}
                onChange={(event) =>
                  setNewPatientForm((prev) => ({ ...prev, sex: event.target.value as 'M' | 'F' | 'Other' }))
                }
              >
                <option value="M">Male</option>
                <option value="F">Female</option>
                <option value="Other">Other</option>
              </select>
              <Input
                placeholder="Phone"
                value={newPatientForm.contact_phone}
                onChange={(event) =>
                  setNewPatientForm((prev) => ({ ...prev, contact_phone: event.target.value }))
                }
              />
              <Input
                placeholder="Email"
                value={newPatientForm.contact_email}
                onChange={(event) =>
                  setNewPatientForm((prev) => ({ ...prev, contact_email: event.target.value }))
                }
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsNewPatientOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreatePatient} disabled={createPatientMutation.isPending}>
              {createPatientMutation.isPending ? 'Creating…' : 'Create'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default ReceptionistDashboard