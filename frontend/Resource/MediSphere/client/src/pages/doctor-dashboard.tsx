import { useState } from 'react';
import DashboardLayout from '../components/layout/dashboard-layout';
import { usePatientStore } from '../store/patientStore';
import { useAuthStore } from '../store/authStore';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { Stethoscope, Clock, AlertTriangle, Mic, MicOff, FileText, Heart, Thermometer } from 'lucide-react';

export default function DoctorDashboard() {
  const { user } = useAuthStore();
  const { patients, updatePatientStatus } = usePatientStore();
  const [selectedPatient, setSelectedPatient] = useState<any>(null);
  const [consultationDialogOpen, setConsultationDialogOpen] = useState(false);
  const [consultationStarted, setConsultationStarted] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [clinicalNotes, setClinicalNotes] = useState('');
  const [transcript, setTranscript] = useState('');

  // Get patients assigned to this doctor
  const myPatients = patients.filter(patient => 
    patient.assignedDoctorId === user?.id && 
    (patient.status === 'AWAITING_DOCTOR' || patient.status === 'IN_CONSULTATION')
  );

  const handleStartConsultation = (patient: any) => {
    setSelectedPatient(patient);
    setConsultationDialogOpen(true);
    setConsultationStarted(false);
    setClinicalNotes('');
    setTranscript('');
    updatePatientStatus(patient.id, 'IN_CONSULTATION');
  };

  const handleBeginConsultation = () => {
    setConsultationStarted(true);
  };

  const handleToggleRecording = () => {
    setIsRecording(!isRecording);
    // Simulate AI transcription
    if (!isRecording) {
      setTimeout(() => {
        setTranscript(prev => prev + 'Patient reports chest pain that started 2 hours ago. Pain is described as sharp, 7/10 intensity, located in the center of the chest. ');
        setClinicalNotes(prev => prev + 'Chief Complaint: Chest pain, 7/10, sharp, central\nHistory of Present Illness: Acute onset chest pain 2 hours prior to presentation...\n');
      }, 2000);
    }
  };

  const handleCompleteConsultation = () => {
    if (selectedPatient) {
      updatePatientStatus(selectedPatient.id, 'AWAITING_DISCHARGE');
      setConsultationDialogOpen(false);
      setConsultationStarted(false);
    }
  };

  const getWaitTime = (checkinTime: Date) => {
    const now = new Date();
    const diff = Math.floor((now.getTime() - checkinTime.getTime()) / (1000 * 60));
    return `${diff} min`;
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getTriageColor = (level?: string) => {
    switch (level) {
      case 'CRITICAL': return 'bg-red-100 text-red-800';
      case 'HIGH': return 'bg-orange-100 text-orange-800';
      case 'MEDIUM': return 'bg-yellow-100 text-yellow-800';
      case 'LOW': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <DashboardLayout>
      <div className="p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">My Patients</h1>
          <p className="text-gray-600 mt-1">Your assigned patient queue</p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <Stethoscope className="w-6 h-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-2xl font-bold">{myPatients.filter(p => p.status === 'AWAITING_DOCTOR').length}</p>
                  <p className="text-sm text-gray-600">Awaiting Consultation</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-green-100 rounded-lg">
                  <FileText className="w-6 h-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-2xl font-bold">{myPatients.filter(p => p.status === 'IN_CONSULTATION').length}</p>
                  <p className="text-sm text-gray-600">In Consultation</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-orange-100 rounded-lg">
                  <AlertTriangle className="w-6 h-6 text-orange-600" />
                </div>
                <div className="ml-4">
                  <p className="text-2xl font-bold">{myPatients.filter(p => p.triageLevel === 'HIGH' || p.triageLevel === 'CRITICAL').length}</p>
                  <p className="text-sm text-gray-600">High Priority</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Patient Queue */}
        <Card>
          <CardHeader>
            <CardTitle>Your Patient Queue</CardTitle>
          </CardHeader>
          <CardContent>
            {myPatients.length === 0 ? (
              <div className="text-center py-8">
                <Stethoscope className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">No patients assigned to you</p>
              </div>
            ) : (
              <div className="space-y-4">
                {myPatients
                  .sort((a, b) => {
                    // Sort by triage level first, then by wait time
                    const triageOrder = { 'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1 };
                    const aLevel = triageOrder[a.triageLevel as keyof typeof triageOrder] || 0;
                    const bLevel = triageOrder[b.triageLevel as keyof typeof triageOrder] || 0;
                    
                    if (aLevel !== bLevel) return bLevel - aLevel;
                    return a.checkinTime.getTime() - b.checkinTime.getTime();
                  })
                  .map((patient) => (
                  <div key={patient.id} className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50" data-testid={`doctor-patient-${patient.id}`}>
                    <div className="flex-1">
                      <div className="flex items-center space-x-4">
                        <div>
                          <h3 className="font-semibold text-lg">{patient.name}</h3>
                          <p className="text-sm text-gray-600">Age {patient.age} • Waiting {getWaitTime(patient.checkinTime)}</p>
                        </div>
                        <div className="flex space-x-2">
                          {patient.triageLevel && (
                            <Badge className={getTriageColor(patient.triageLevel)}>
                              {patient.triageLevel}
                            </Badge>
                          )}
                          <Badge variant={patient.status === 'IN_CONSULTATION' ? 'default' : 'secondary'}>
                            {patient.status === 'IN_CONSULTATION' ? 'In Progress' : 'Ready'}
                          </Badge>
                        </div>
                      </div>
                      <div className="mt-2">
                        <p className="text-sm"><strong>Chief Complaint:</strong> {patient.chiefComplaint}</p>
                        {patient.vitals && (
                          <div className="flex space-x-4 mt-1 text-xs text-gray-600">
                            <span>HR: {patient.vitals.heartRate}</span>
                            <span>BP: {patient.vitals.bloodPressure}</span>
                            <span>Temp: {patient.vitals.temperature}°F</span>
                            <span>O2: {patient.vitals.oxygenSaturation}%</span>
                          </div>
                        )}
                      </div>
                    </div>
                    <Button 
                      onClick={() => handleStartConsultation(patient)}
                      variant={patient.status === 'IN_CONSULTATION' ? 'default' : 'outline'}
                      data-testid={`button-consult-${patient.id}`}
                    >
                      {patient.status === 'IN_CONSULTATION' ? 'Continue' : 'Start Consultation'}
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Consultation Dialog */}
        <Dialog open={consultationDialogOpen} onOpenChange={setConsultationDialogOpen}>
          <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Consultation - {selectedPatient?.name}</DialogTitle>
            </DialogHeader>
            
            <Tabs defaultValue="summary" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="summary">Pre-Consultation Summary</TabsTrigger>
                <TabsTrigger value="consultation" disabled={!consultationStarted}>AI Scribe</TabsTrigger>
              </TabsList>
              
              <TabsContent value="summary" className="space-y-6">
                <div className="grid grid-cols-2 gap-6">
                  {/* Patient Info */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Patient Information</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <p><strong>Name:</strong> {selectedPatient?.name}</p>
                        <p><strong>Age:</strong> {selectedPatient?.age}</p>
                        <p><strong>Phone:</strong> {selectedPatient?.phone}</p>
                        <p><strong>Check-in Time:</strong> {selectedPatient && formatTime(selectedPatient.checkinTime)}</p>
                        <p><strong>Chief Complaint:</strong> {selectedPatient?.chiefComplaint}</p>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Vitals */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg flex items-center">
                        <Heart className="w-5 h-5 mr-2 text-red-500" />
                        Vital Signs
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {selectedPatient?.vitals ? (
                        <div className="space-y-2">
                          <p><strong>Heart Rate:</strong> {selectedPatient.vitals.heartRate} BPM</p>
                          <p><strong>Blood Pressure:</strong> {selectedPatient.vitals.bloodPressure}</p>
                          <p><strong>Temperature:</strong> {selectedPatient.vitals.temperature}°F</p>
                          <p><strong>Weight:</strong> {selectedPatient.vitals.weight} lbs</p>
                          <p><strong>Oxygen Saturation:</strong> {selectedPatient.vitals.oxygenSaturation}%</p>
                          {selectedPatient.triageLevel && (
                            <div className="pt-2">
                              <Badge className={getTriageColor(selectedPatient.triageLevel)}>
                                Triage: {selectedPatient.triageLevel}
                              </Badge>
                            </div>
                          )}
                        </div>
                      ) : (
                        <p className="text-gray-500">No vitals recorded</p>
                      )}
                    </CardContent>
                  </Card>
                </div>
                
                <div className="flex justify-end">
                  <Button onClick={handleBeginConsultation} data-testid="button-start-consultation">
                    Start Consultation
                  </Button>
                </div>
              </TabsContent>
              
              <TabsContent value="consultation" className="space-y-6">
                <div className="grid grid-cols-2 gap-6">
                  {/* Recording Controls */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg flex items-center">
                        <Mic className="w-5 h-5 mr-2" />
                        AI Scribe
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-center space-y-4">
                        <Button
                          onClick={handleToggleRecording}
                          variant={isRecording ? "destructive" : "default"}
                          size="lg"
                          className="w-24 h-24 rounded-full"
                          data-testid="button-toggle-recording"
                        >
                          {isRecording ? <MicOff className="w-8 h-8" /> : <Mic className="w-8 h-8" />}
                        </Button>
                        <p className="text-sm text-gray-600">
                          {isRecording ? 'Recording... Click to stop' : 'Click to start recording'}
                        </p>
                      </div>
                      
                      {transcript && (
                        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                          <h4 className="font-medium text-sm mb-2">Live Transcript:</h4>
                          <p className="text-sm text-gray-700">{transcript}</p>
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {/* Clinical Notes */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg flex items-center">
                        <FileText className="w-5 h-5 mr-2" />
                        Clinical Notes
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Textarea
                        value={clinicalNotes}
                        onChange={(e) => setClinicalNotes(e.target.value)}
                        placeholder="AI-generated clinical notes will appear here..."
                        rows={12}
                        data-testid="textarea-clinical-notes"
                      />
                    </CardContent>
                  </Card>
                </div>
                
                <div className="flex justify-end space-x-2">
                  <Button variant="outline" onClick={() => setConsultationDialogOpen(false)}>
                    Save & Exit
                  </Button>
                  <Button onClick={handleCompleteConsultation} data-testid="button-complete-consultation">
                    Complete Consultation
                  </Button>
                </div>
              </TabsContent>
            </Tabs>
          </DialogContent>
        </Dialog>
      </div>
    </DashboardLayout>
  );
}