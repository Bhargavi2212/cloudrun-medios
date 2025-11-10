import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/hooks/useAuth";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { HealthcareCard } from "@/components/ui/healthcare-card";
import { 
  Activity, 
  AlertTriangle, 
  Bell, 
  Clipboard, 
  Clock, 
  Heart, 
  MessageSquare, 
  Pill, 
  PlusCircle, 
  Thermometer, 
  Users, 
  CheckCircle2, 
  AlertCircle,
  Stethoscope,
  Bed,
  Syringe,
  FileText,
  UserCheck
} from "lucide-react";

export default function NurseDashboard() {
  const { user } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  const [selectedPatient, setSelectedPatient] = useState<any>(null);
  const [vitalsDialog, setVitalsDialog] = useState(false);
  const [medicationDialog, setMedicationDialog] = useState(false);
  const [vitalsForm, setVitalsForm] = useState({
    temperature: '',
    heartRate: '',
    bloodPressure: '',
    oxygenSaturation: '',
    notes: ''
  });

  // Fetch nurse queue - patients waiting for vitals
  const { data: nurseQueue = [], isLoading: queueLoading } = useQuery({
    queryKey: ["/api/nurse/queue"],
  });

  // Fetch assigned patients
  const { data: assignedPatients = [], isLoading: patientsLoading } = useQuery({
    queryKey: ["/api/nurse/patients"],
  });

  // Fetch daily tasks
  const { data: tasks = [], isLoading: tasksLoading } = useQuery({
    queryKey: ["/api/nurse/tasks"],
  });

  // Fetch medication schedule
  const { data: medications = [], isLoading: medLoading } = useQuery({
    queryKey: ["/api/nurse/medications"],
  });

  // Fetch vitals monitoring
  const { data: vitalsAlerts = [], isLoading: vitalsLoading } = useQuery({
    queryKey: ["/api/nurse/vitals-alerts"],
  });

  // Mutations for vitals recording and patient transfer
  const recordVitalsMutation = useMutation({
    mutationFn: async (data: any) => apiRequest('/api/nurse/record-vitals', 'POST', data),
    onSuccess: () => {
      toast({
        title: "Vitals Recorded",
        description: "Patient vitals have been recorded and patient transferred to doctor queue.",
      });
      setVitalsDialog(false);
      setVitalsForm({ temperature: '', heartRate: '', bloodPressure: '', oxygenSaturation: '', notes: '' });
      queryClient.invalidateQueries({ queryKey: ['/api/nurse/queue'] });
      queryClient.invalidateQueries({ queryKey: ['/api/nurse/patients'] });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to record vitals. Please try again.",
        variant: "destructive",
      });
    }
  });

  const handleRecordVitals = () => {
    const temp = parseFloat(vitalsForm.temperature);
    const hr = parseFloat(vitalsForm.heartRate);
    const o2 = parseFloat(vitalsForm.oxygenSaturation);
    
    // Check for emergency conditions
    const isEmergency = temp > 101 || hr > 100 || o2 < 90;
    
    recordVitalsMutation.mutate({
      patientId: selectedPatient.id,
      temperature: temp,
      heartRate: hr,
      bloodPressure: vitalsForm.bloodPressure,
      oxygenSaturation: o2,
      notes: vitalsForm.notes,
      isEmergency,
      transferToDoctor: true // Always transfer after vitals
    });
  };

  const urgentTasks = tasks.filter((t: any) => t.priority === 'high' || t.priority === 'critical');
  const medicationsDue = medications.filter((m: any) => new Date(m.dueTime) <= new Date(Date.now() + 60 * 60 * 1000));
  const emergencyAlerts = assignedPatients.filter((p: any) => 
    (p.temperature > 101) || (p.heartRate > 100) || (p.oxygenSaturation < 90)
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50 via-white to-green-50">
      {/* Header */}
      <div className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Nurse {user?.firstName} {user?.lastName}
              </h1>
              <p className="text-gray-600">Patient Care & Monitoring Dashboard</p>
            </div>
            <div className="flex items-center gap-4">
              <Badge variant="outline" className="bg-teal-100 text-teal-800">
                <Heart className="w-4 h-4 mr-1" />
                On Duty
              </Badge>
              <Button onClick={() => window.location.href = "/"} variant="ghost">
                Switch Role
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Emergency Alerts - Auto-flagged patients */}
        {emergencyAlerts.length > 0 && (
          <div className="mb-6">
            <Card className="border-red-200 bg-red-50 animate-pulse">
              <CardHeader className="pb-3">
                <CardTitle className="text-red-800 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 animate-bounce" />
                  EMERGENCY ALERTS - Auto-Transferred to Doctor Queue
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3">
                  {emergencyAlerts.map((patient: any) => (
                    <div key={patient.id} className="flex items-center justify-between p-3 bg-white rounded-lg border border-red-200">
                      <div className="flex items-center gap-3">
                        <Heart className="w-5 h-5 text-red-600" />
                        <div>
                          <p className="font-semibold">{patient.name}</p>
                          <div className="text-sm text-red-600 space-y-1">
                            {patient.temperature > 101 && <p>• High Fever: {patient.temperature}°F</p>}
                            {patient.heartRate > 100 && <p>• High Heart Rate: {patient.heartRate} bpm</p>}
                            {patient.oxygenSaturation < 90 && <p>• Low O2 Saturation: {patient.oxygenSaturation}%</p>}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge className="bg-red-100 text-red-800">EMERGENCY</Badge>
                        <Button size="sm">Call Doctor</Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Critical Alerts */}
        {(urgentTasks.length > 0 || medicationsDue.length > 0 || vitalsAlerts.length > 0) && (
          <div className="mb-6">
            <Card className="border-red-200 bg-red-50">
              <CardHeader className="pb-3">
                <CardTitle className="text-red-800 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5" />
                  Priority Alerts - Immediate Attention Required
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3">
                  {urgentTasks.map((task: any) => (
                    <div key={task.id} className="flex items-center justify-between p-3 bg-white rounded-lg border border-red-200">
                      <div className="flex items-center gap-3">
                        <Clipboard className="w-5 h-5 text-red-600" />
                        <div>
                          <p className="font-semibold">{task.description}</p>
                          <p className="text-sm text-gray-600">Patient: {task.patientName}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge className="bg-red-100 text-red-800">{task.priority.toUpperCase()}</Badge>
                        <Button size="sm">Complete</Button>
                      </div>
                    </div>
                  ))}
                  
                  {medicationsDue.map((med: any) => (
                    <div key={med.id} className="flex items-center justify-between p-3 bg-white rounded-lg border border-orange-200">
                      <div className="flex items-center gap-3">
                        <Pill className="w-5 h-5 text-orange-600" />
                        <div>
                          <p className="font-semibold">{med.medicationName}</p>
                          <p className="text-sm text-gray-600">Patient: {med.patientName} • Due: {new Date(med.dueTime).toLocaleTimeString()}</p>
                        </div>
                      </div>
                      <Button size="sm" variant="outline">Administer</Button>
                    </div>
                  ))}
                  
                  {vitalsAlerts.map((alert: any) => (
                    <div key={alert.id} className="flex items-center justify-between p-3 bg-white rounded-lg border border-yellow-200">
                      <div className="flex items-center gap-3">
                        <Activity className="w-5 h-5 text-yellow-600" />
                        <div>
                          <p className="font-semibold">{alert.alertType}</p>
                          <p className="text-sm text-gray-600">{alert.patientName} • {alert.value}</p>
                        </div>
                      </div>
                      <Button size="sm" variant="outline">Check</Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Nurse Queue - Patients Waiting for Vitals */}
        <div className="mb-6">
          <Card className="border-blue-200 bg-blue-50">
            <CardHeader className="pb-3">
              <CardTitle className="text-blue-800 flex items-center gap-2">
                <Clock className="w-5 h-5" />
                Nurse Queue - Patients Waiting for Vitals ({nurseQueue.length || 6})
              </CardTitle>
              <CardDescription className="text-blue-700">
                Checked-in patients awaiting vital signs collection and triage
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3">
                {queueLoading ? (
                  <div className="text-center py-8 text-gray-500">Loading nurse queue...</div>
                ) : nurseQueue.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">No patients waiting for vitals</div>
                ) : (
                  nurseQueue.concat([
                    { id: 'demo1', name: 'Sarah Johnson', checkedInAt: new Date(Date.now() - 15*60*1000), chiefComplaint: 'Fever and headache', waitTime: 15 },
                    { id: 'demo2', name: 'Mike Chen', checkedInAt: new Date(Date.now() - 8*60*1000), chiefComplaint: 'Chest pain', waitTime: 8 },
                    { id: 'demo3', name: 'Lisa Rodriguez', checkedInAt: new Date(Date.now() - 25*60*1000), chiefComplaint: 'Dizziness', waitTime: 25 },
                    { id: 'demo4', name: 'David Kim', checkedInAt: new Date(Date.now() - 5*60*1000), chiefComplaint: 'Shortness of breath', waitTime: 5 },
                    { id: 'demo5', name: 'Emma Wilson', checkedInAt: new Date(Date.now() - 12*60*1000), chiefComplaint: 'Stomach pain', waitTime: 12 },
                    { id: 'demo6', name: 'John Adams', checkedInAt: new Date(Date.now() - 18*60*1000), chiefComplaint: 'Back injury', waitTime: 18 }
                  ]).map((patient: any) => (
                    <div key={patient.id} className="flex items-center justify-between p-4 bg-white rounded-lg border border-blue-200 hover:bg-blue-50 transition-colors">
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                          <UserCheck className="w-6 h-6 text-blue-600" />
                        </div>
                        <div>
                          <p className="font-semibold">{patient.name}</p>
                          <p className="text-sm text-gray-600">Chief Complaint: {patient.chiefComplaint}</p>
                          <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                            <span>Checked in: {patient.checkedInAt ? new Date(patient.checkedInAt).toLocaleTimeString() : '10:30 AM'}</span>
                            <span>•</span>
                            <span className={patient.waitTime > 20 ? 'text-red-600 font-semibold' : patient.waitTime > 10 ? 'text-orange-600' : 'text-gray-500'}>
                              Waiting: {patient.waitTime || 10} min
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge 
                          variant="outline" 
                          className={
                            patient.waitTime > 20 ? 'border-red-500 text-red-700' :
                            patient.waitTime > 10 ? 'border-orange-500 text-orange-700' :
                            'border-blue-500 text-blue-700'
                          }
                        >
                          {patient.waitTime > 20 ? 'HIGH PRIORITY' : 'WAITING'}
                        </Badge>
                        <Button 
                          size="sm" 
                          onClick={() => {
                            setSelectedPatient(patient);
                            setVitalsDialog(true);
                          }}
                          className="bg-blue-600 hover:bg-blue-700"
                        >
                          <Stethoscope className="w-4 h-4 mr-1" />
                          Take Vitals
                        </Button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Ward Management & Patients */}
          <div className="lg:col-span-2 space-y-6">
            {/* Ward Management */}
            <HealthcareCard title="Ward Management" description="Your assigned patients and their status">
              <div className="space-y-3">
                {patientsLoading ? (
                  <div className="text-center py-8 text-gray-500">Loading assigned patients...</div>
                ) : assignedPatients.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">No patients assigned</div>
                ) : (
                  assignedPatients.map((patient: any) => (
                    <div key={patient.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 bg-teal-100 rounded-full flex items-center justify-center">
                          <Bed className="w-6 h-6 text-teal-600" />
                        </div>
                        <div>
                          <p className="font-semibold">{patient.name}</p>
                          <p className="text-sm text-gray-600">Room {patient.roomNumber} • Bed {patient.bedNumber}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge 
                              variant="outline" 
                              className={
                                (patient.temperature > 101 || patient.heartRate > 100 || patient.oxygenSaturation < 90) ? 'border-red-500 text-red-700 animate-pulse' :
                                patient.condition === 'critical' ? 'border-red-500 text-red-700' :
                                patient.condition === 'stable' ? 'border-green-500 text-green-700' :
                                'border-yellow-500 text-yellow-700'
                              }
                            >
                              {(patient.temperature > 101 || patient.heartRate > 100 || patient.oxygenSaturation < 90) ? 'EMERGENCY' :
                               patient.condition?.toUpperCase() || 'STABLE'}
                            </Badge>
                            {patient.lastVitalsTime && (
                              <span className="text-xs text-gray-500">
                                Last vitals: {new Date(patient.lastVitalsTime).toLocaleTimeString()}
                              </span>
                            )}
                            {(patient.temperature > 101 || patient.heartRate > 100 || patient.oxygenSaturation < 90) && (
                              <Badge className="bg-red-100 text-red-800 text-xs animate-pulse">
                                AUTO-FLAGGED
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button size="sm" variant="outline">
                          <Thermometer className="w-4 h-4 mr-1" />
                          Vitals
                        </Button>
                        <Button size="sm">
                          <FileText className="w-4 h-4 mr-1" />
                          Chart
                        </Button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </HealthcareCard>

            {/* Task Management */}
            <HealthcareCard title="Daily Care Tasks" description="Your scheduled activities and responsibilities">
              <div className="space-y-3">
                {tasksLoading ? (
                  <div className="text-center py-4 text-gray-500">Loading tasks...</div>
                ) : tasks.length === 0 ? (
                  <div className="text-center py-4 text-gray-500">No tasks assigned</div>
                ) : (
                  tasks.map((task: any) => (
                    <div key={task.id} className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50">
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                          task.status === 'completed' ? 'bg-green-100' : 
                          task.priority === 'high' ? 'bg-red-100' : 'bg-blue-100'
                        }`}>
                          {task.status === 'completed' ? (
                            <CheckCircle2 className="w-4 h-4 text-green-600" />
                          ) : task.taskType === 'medication' ? (
                            <Pill className="w-4 h-4 text-blue-600" />
                          ) : task.taskType === 'vitals' ? (
                            <Activity className="w-4 h-4 text-blue-600" />
                          ) : (
                            <Clipboard className="w-4 h-4 text-blue-600" />
                          )}
                        </div>
                        <div>
                          <p className="font-medium">{task.description}</p>
                          <p className="text-sm text-gray-600">Patient: {task.patientName}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge 
                              variant="outline" 
                              className={
                                task.priority === 'critical' ? 'border-red-500 text-red-700' :
                                task.priority === 'high' ? 'border-orange-500 text-orange-700' :
                                'border-blue-500 text-blue-700'
                              }
                            >
                              {task.priority?.toUpperCase() || 'NORMAL'}
                            </Badge>
                            {task.dueTime && (
                              <span className="text-xs text-gray-500">
                                Due: {new Date(task.dueTime).toLocaleTimeString()}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {task.status !== 'completed' && (
                          <Button size="sm">
                            <CheckCircle2 className="w-4 h-4 mr-1" />
                            Complete
                          </Button>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </HealthcareCard>
          </div>

          {/* Right Column - Medication & Communication */}
          <div className="space-y-6">
            {/* Quick Stats */}
            <div className="grid grid-cols-2 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <Users className="w-5 h-5 text-teal-600" />
                    <div>
                      <p className="text-2xl font-bold">{assignedPatients.length}</p>
                      <p className="text-xs text-gray-600">Assigned Patients</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <Clipboard className="w-5 h-5 text-blue-600" />
                    <div>
                      <p className="text-2xl font-bold">{tasks.filter((t: any) => t.status !== 'completed').length}</p>
                      <p className="text-xs text-gray-600">Pending Tasks</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Medication Schedule */}
            <HealthcareCard title="Medication Schedule" description="Upcoming medication administrations">
              <div className="space-y-3">
                {medLoading ? (
                  <div className="text-center py-4 text-gray-500">Loading medications...</div>
                ) : medications.length === 0 ? (
                  <div className="text-center py-4 text-gray-500">No medications scheduled</div>
                ) : (
                  medications.slice(0, 6).map((med: any) => (
                    <div key={med.id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <Pill className="w-5 h-5 text-green-600" />
                        <div>
                          <p className="font-medium">{med.medicationName}</p>
                          <p className="text-sm text-gray-600">{med.patientName}</p>
                          <p className="text-xs text-gray-500">{med.dosage} • {med.frequency}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium">
                          {new Date(med.dueTime).toLocaleTimeString()}
                        </p>
                        <Badge 
                          variant="outline"
                          className={
                            new Date(med.dueTime) <= new Date() ? 'border-red-500 text-red-700' :
                            new Date(med.dueTime) <= new Date(Date.now() + 30 * 60 * 1000) ? 'border-yellow-500 text-yellow-700' :
                            'border-green-500 text-green-700'
                          }
                        >
                          {new Date(med.dueTime) <= new Date() ? 'OVERDUE' :
                           new Date(med.dueTime) <= new Date(Date.now() + 30 * 60 * 1000) ? 'DUE SOON' : 'SCHEDULED'}
                        </Badge>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </HealthcareCard>

            {/* Communication Center */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <MessageSquare className="w-5 h-5" />
                  Communication Center
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <Button className="w-full justify-start" variant="outline">
                    <Bell className="w-4 h-4 mr-2" />
                    Doctor Instructions (3)
                  </Button>
                  <Button className="w-full justify-start" variant="outline">
                    <UserCheck className="w-4 h-4 mr-2" />
                    Shift Handover
                  </Button>
                  <Button className="w-full justify-start" variant="outline">
                    <AlertCircle className="w-4 h-4 mr-2" />
                    Report Incident
                  </Button>
                  <Button className="w-full justify-start" variant="outline">
                    <MessageSquare className="w-4 h-4 mr-2" />
                    Family Communications
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Quick Actions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3">
                  <Button className="w-full justify-start" variant="outline">
                    <PlusCircle className="w-4 h-4 mr-2" />
                    Record Vitals
                  </Button>
                  <Button className="w-full justify-start" variant="outline">
                    <Syringe className="w-4 h-4 mr-2" />
                    Administer Medication
                  </Button>
                  <Button className="w-full justify-start" variant="outline">
                    <Clock className="w-4 h-4 mr-2" />
                    Update Care Plan
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Real-time Patient Status */}
        <div className="mt-6">
          <HealthcareCard title="Real-time Patient Status" description="Live monitoring and vital sign trends">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {assignedPatients.slice(0, 6).map((patient: any) => (
                <div key={patient.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-semibold">{patient.name}</h4>
                    <Badge 
                      variant="outline"
                      className={
                        patient.condition === 'critical' ? 'border-red-500 text-red-700' :
                        patient.condition === 'stable' ? 'border-green-500 text-green-700' :
                        'border-yellow-500 text-yellow-700'
                      }
                    >
                      {patient.condition?.toUpperCase() || 'STABLE'}
                    </Badge>
                  </div>
                  
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Heart Rate:</span>
                      <span className={
                        patient.heartRate > 100 ? 'text-red-600 font-semibold animate-pulse' : 
                        patient.heartRate < 60 ? 'text-red-600 font-semibold' : 
                        'text-green-600'
                      }>
                        {patient.heartRate || '--'} bpm
                        {patient.heartRate > 100 && <span className="ml-1 text-xs">⚠️</span>}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Blood Pressure:</span>
                      <span className="text-gray-700">{patient.bloodPressure || '--'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Temperature:</span>
                      <span className={
                        patient.temperature > 101 ? 'text-red-600 font-semibold animate-pulse' :
                        patient.temperature > 99.5 ? 'text-orange-600 font-semibold' : 
                        'text-green-600'
                      }>
                        {patient.temperature ? `${patient.temperature}°F` : '--'}
                        {patient.temperature > 101 && <span className="ml-1 text-xs">🚨</span>}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>O2 Saturation:</span>
                      <span className={
                        patient.oxygenSaturation < 90 ? 'text-red-600 font-semibold animate-pulse' :
                        patient.oxygenSaturation < 95 ? 'text-orange-600 font-semibold' : 
                        'text-green-600'
                      }>
                        {patient.oxygenSaturation ? `${patient.oxygenSaturation}%` : '--'}
                        {patient.oxygenSaturation < 90 && <span className="ml-1 text-xs">🚨</span>}
                      </span>
                    </div>
                  </div>
                  
                  <div className="mt-3 pt-3 border-t">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-500">
                        Last updated: {patient.lastVitalsTime ? new Date(patient.lastVitalsTime).toLocaleTimeString() : 'Never'}
                      </span>
                      <Button size="sm" variant="outline">
                        Update Vitals
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </HealthcareCard>
        </div>
      </div>

      {/* Vitals Recording Dialog */}
      <Dialog open={vitalsDialog} onOpenChange={setVitalsDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Stethoscope className="w-5 h-5 text-teal-600" />
              Record Vitals - {selectedPatient?.name}
            </DialogTitle>
            <DialogDescription>
              Record vital signs and triage the patient
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 pt-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">Temperature (°F)</label>
                <Input
                  type="number"
                  step="0.1"
                  placeholder="98.6"
                  value={vitalsForm.temperature}
                  onChange={(e) => setVitalsForm({...vitalsForm, temperature: e.target.value})}
                />
              </div>
              <div>
                <label className="text-sm font-medium">Heart Rate (bpm)</label>
                <Input
                  type="number"
                  placeholder="72"
                  value={vitalsForm.heartRate}
                  onChange={(e) => setVitalsForm({...vitalsForm, heartRate: e.target.value})}
                />
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium">Blood Pressure</label>
                <Input
                  placeholder="120/80"
                  value={vitalsForm.bloodPressure}
                  onChange={(e) => setVitalsForm({...vitalsForm, bloodPressure: e.target.value})}
                />
              </div>
              <div>
                <label className="text-sm font-medium">O2 Saturation (%)</label>
                <Input
                  type="number"
                  placeholder="98"
                  value={vitalsForm.oxygenSaturation}
                  onChange={(e) => setVitalsForm({...vitalsForm, oxygenSaturation: e.target.value})}
                />
              </div>
            </div>
            
            <div>
              <label className="text-sm font-medium">Nursing Notes</label>
              <Textarea
                placeholder="Additional observations or concerns..."
                value={vitalsForm.notes}
                onChange={(e) => setVitalsForm({...vitalsForm, notes: e.target.value})}
                rows={3}
              />
            </div>

            {/* Emergency Alert Preview */}
            {(parseFloat(vitalsForm.temperature) > 101 || parseFloat(vitalsForm.heartRate) > 100 || parseFloat(vitalsForm.oxygenSaturation) < 90) && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-center gap-2 text-red-800 font-semibold mb-2">
                  <AlertTriangle className="w-4 h-4" />
                  EMERGENCY DETECTED
                </div>
                <div className="text-sm text-red-700 space-y-1">
                  {parseFloat(vitalsForm.temperature) > 101 && <p>• High fever detected (&gt;101°F)</p>}
                  {parseFloat(vitalsForm.heartRate) > 100 && <p>• High heart rate detected (&gt;100 bpm)</p>}
                  {parseFloat(vitalsForm.oxygenSaturation) < 90 && <p>• Low oxygen saturation (&lt;90%)</p>}
                  <p className="font-medium mt-2">Patient will be automatically transferred to doctor queue with EMERGENCY priority.</p>
                </div>
              </div>
            )}
            
            <div className="flex gap-3 pt-4">
              <Button 
                variant="outline" 
                onClick={() => setVitalsDialog(false)}
                className="flex-1"
              >
                Cancel
              </Button>
              <Button 
                onClick={handleRecordVitals}
                disabled={recordVitalsMutation.isPending || !vitalsForm.temperature || !vitalsForm.heartRate}
                className="flex-1 bg-teal-600 hover:bg-teal-700"
              >
                {recordVitalsMutation.isPending ? (
                  <>
                    <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"></div>
                    Recording...
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-4 h-4 mr-2" />
                    Record & Transfer to Doctor
                  </>
                )}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}