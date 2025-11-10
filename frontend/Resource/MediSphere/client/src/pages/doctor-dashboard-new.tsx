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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { HealthcareCard } from "@/components/ui/healthcare-card";
import { 
  Activity, 
  Calendar, 
  FileText, 
  FlaskConical, 
  Heart, 
  MessageCircle, 
  PlusCircle, 
  Search,
  Stethoscope,
  Users,
  AlertTriangle,
  TrendingUp,
  Clock,
  CheckCircle,
  Coffee,
  MapPin,
  Mic,
  MicOff,
  Bot,
  Clipboard,
  PlayCircle,
  PauseCircle,
  StopCircle,
  Volume2,
  Pill,
  TestTube,
  Brain,
  Zap,
  UserPlus
} from "lucide-react";

export default function DoctorDashboard() {
  const { user } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // State management for doctor features
  const [doctorStatus, setDoctorStatus] = useState("available");
  const [isRecording, setIsRecording] = useState(false);
  const [selectedPatient, setSelectedPatient] = useState<any>(null);
  const [prescriptionDialog, setPrescriptionDialog] = useState(false);
  const [labOrderDialog, setLabOrderDialog] = useState(false);
  const [aiConsultDialog, setAiConsultDialog] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [patientDetailsDialog, setPatientDetailsDialog] = useState(false);
  const [selectedPatientDetails, setSelectedPatientDetails] = useState<any>(null);
  
  // Patient details queries
  const { data: patientVitals, isLoading: vitalsLoading } = useQuery({
    queryKey: [`/api/patient/${selectedPatientDetails?.id || selectedPatientDetails?.patientId}/vitals`],
    enabled: !!selectedPatientDetails && patientDetailsDialog,
  });
  
  const { data: patientMedicalHistory, isLoading: historyLoading } = useQuery({
    queryKey: [`/api/patient/${selectedPatientDetails?.id || selectedPatientDetails?.patientId}/medical-history`],
    enabled: !!selectedPatientDetails && patientDetailsDialog,
  });
  
  const { data: patientLabResults, isLoading: labResultsLoading } = useQuery({
    queryKey: [`/api/patient/${selectedPatientDetails?.id || selectedPatientDetails?.patientId}/lab-results`],
    enabled: !!selectedPatientDetails && patientDetailsDialog,
  });
  
  const { data: patientMedications, isLoading: medicationsLoading } = useQuery({
    queryKey: [`/api/patient/${selectedPatientDetails?.id || selectedPatientDetails?.patientId}/medications`],
    enabled: !!selectedPatientDetails && patientDetailsDialog,
  });
  
  const { data: patientConsultations, isLoading: patientConsultationsLoading } = useQuery({
    queryKey: [`/api/patient/${selectedPatientDetails?.id || selectedPatientDetails?.patientId}/consultations`],
    enabled: !!selectedPatientDetails && patientDetailsDialog,
  });
  
  // Form states
  const [prescriptionForm, setPrescriptionForm] = useState({
    medications: [
      {
        medication: "",
        strength: "",
        form: "",
        dosage: "",
        frequency: "",
        duration: "",
        route: "",
        timings: "",
        beforeAfterMeals: "",
        specialInstructions: ""
      }
    ]
  });
  
  const [labOrderForm, setLabOrderForm] = useState({
    bloodTests: [],
    imagingTests: [],
    otherTests: [],
    priority: "routine",
    fasting: false,
    clinicalNotes: "",
    indicationsForTesting: ""
  });

  // Fetch doctor's queue
  const { data: patientQueue = [], isLoading: queueLoading } = useQuery({
    queryKey: ["/api/doctor/queue"],
  });

  // Fetch today's appointments
  const { data: appointments = [], isLoading: appointmentsLoading } = useQuery({
    queryKey: ["/api/doctor/appointments"],
  });

  // Fetch pending lab results
  const { data: labResults = [], isLoading: labLoading } = useQuery({
    queryKey: ["/api/doctor/lab-results"],
  });

  // Fetch recent consultations
  const { data: consultations = [], isLoading: consultationsLoading } = useQuery({
    queryKey: ["/api/doctor/consultations"],
  });

  // Fetch doctor's schedule/calendar
  const { data: schedule = [], isLoading: scheduleLoading } = useQuery({
    queryKey: ["/api/doctor/schedule"],
  });

  // Mutations for doctor actions
  const prescriptionMutation = useMutation({
    mutationFn: async (data: any) => apiRequest('/api/doctor/prescriptions', 'POST', data),
    onSuccess: () => {
      toast({ title: "Prescription Created", description: "Prescription has been successfully created and sent to pharmacy." });
      setPrescriptionDialog(false);
      setPrescriptionForm({ 
        medication: "", strength: "", form: "", dosage: "", frequency: "", 
        duration: "", route: "", timings: "", beforeAfterMeals: "", specialInstructions: "" 
      });
    }
  });

  const labOrderMutation = useMutation({
    mutationFn: async (data: any) => apiRequest('/api/doctor/lab-orders', 'POST', data),
    onSuccess: () => {
      toast({ title: "Lab Order Submitted", description: "Lab tests have been ordered successfully." });
      setLabOrderDialog(false);
      setLabOrderForm({ 
        bloodTests: [], imagingTests: [], otherTests: [], priority: "routine", 
        fasting: false, clinicalNotes: "", indicationsForTesting: "" 
      });
    }
  });

  const statusUpdateMutation = useMutation({
    mutationFn: async (status: string) => apiRequest('/api/doctor/status', 'PUT', { status }),
    onSuccess: () => {
      toast({ title: "Status Updated", description: `Status changed to ${doctorStatus}` });
    }
  });

  // AI Scribe functionality
  const toggleRecording = () => {
    if (!isRecording) {
      setIsRecording(true);
      setTranscription("Recording started...");
      // Simulate speech recognition
      setTimeout(() => {
        setTranscription("Patient reports chest pain for 2 days. No shortness of breath. Pain is sharp, intermittent. No radiation. Vital signs stable. Physical exam reveals tenderness in left chest wall...");
      }, 3000);
    } else {
      setIsRecording(false);
      toast({ title: "Recording Saved", description: "Consultation notes have been transcribed and saved." });
    }
  };

  const priorityQueue = patientQueue.filter((p: any) => p.priority === 'high' || p.priority === 'critical' || p.priority === 'emergency');
  const criticalCases = patientQueue.filter((p: any) => p.priority === 'critical' || p.priority === 'emergency');

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-teal-50">
      {/* Header */}
      <div className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Dr. {user?.firstName} {user?.lastName}
              </h1>
              <p className="text-gray-600">Medical Dashboard - {new Date().toLocaleDateString()}</p>
            </div>
            <div className="flex items-center gap-4">
              {/* Doctor Status Management */}
              <div className="flex items-center gap-2">
                <Select value={doctorStatus} onValueChange={(value) => {
                  setDoctorStatus(value);
                  statusUpdateMutation.mutate(value);
                }}>
                  <SelectTrigger className="w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="available">
                      <div className="flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-600" />
                        Available
                      </div>
                    </SelectItem>
                    <SelectItem value="on_break">
                      <div className="flex items-center gap-2">
                        <Coffee className="w-4 h-4 text-orange-600" />
                        On Break
                      </div>
                    </SelectItem>
                    <SelectItem value="in_rounds">
                      <div className="flex items-center gap-2">
                        <MapPin className="w-4 h-4 text-blue-600" />
                        In Rounds
                      </div>
                    </SelectItem>
                    <SelectItem value="busy">
                      <div className="flex items-center gap-2">
                        <Clock className="w-4 h-4 text-red-600" />
                        Busy
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
                
                {/* AI Scribe Toggle */}
                <Button
                  variant={isRecording ? "destructive" : "outline"}
                  size="sm"
                  onClick={toggleRecording}
                  className="flex items-center gap-2"
                >
                  {isRecording ? (
                    <>
                      <MicOff className="w-4 h-4" />
                      Stop Recording
                    </>
                  ) : (
                    <>
                      <Mic className="w-4 h-4" />
                      Start Scribe
                    </>
                  )}
                </Button>
              </div>
              
              <Button onClick={() => window.location.href = "/api/logout"} variant="ghost">
                Logout
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Critical Alerts */}
        {criticalCases.length > 0 && (
          <div className="mb-6">
            <Card className="border-red-200 bg-red-50">
              <CardHeader className="pb-3">
                <CardTitle className="text-red-800 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5" />
                  Critical Cases Requiring Immediate Attention
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3">
                  {criticalCases.map((patient: any) => (
                    <div key={patient.id} className="flex items-center justify-between p-3 bg-white rounded-lg border border-red-200">
                      <div>
                        <p className="font-semibold">{patient.name}</p>
                        <p className="text-sm text-gray-600">{patient.reasonForVisit}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge className="bg-red-100 text-red-800">CRITICAL</Badge>
                        <Button size="sm">See Patient</Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Doctor Action Bar */}
        <div className="bg-white rounded-xl border shadow-sm p-4 mb-6">
          <div className="flex flex-wrap gap-3">
            <Dialog open={prescriptionDialog} onOpenChange={setPrescriptionDialog}>
              <DialogTrigger asChild>
                <Button className="flex items-center gap-2">
                  <Pill className="w-4 h-4" />
                  Write Prescription
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Create Prescription</DialogTitle>
                  <DialogDescription>
                    Order prescription for {selectedPatient?.name}
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-6">
                  <div className="flex justify-between items-center">
                    <h3 className="text-lg font-semibold">Prescription Details</h3>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const newMed = {
                          medication: "",
                          strength: "",
                          form: "",
                          dosage: "",
                          frequency: "",
                          duration: "",
                          route: "",
                          timings: "",
                          beforeAfterMeals: "",
                          specialInstructions: ""
                        };
                        setPrescriptionForm({
                          medications: [...prescriptionForm.medications, newMed]
                        });
                      }}
                    >
                      + Add Another Medication
                    </Button>
                  </div>

                  {prescriptionForm.medications.map((medication, index) => {
                    const updateMedication = (field: string, value: string) => {
                      const updatedMedications = [...prescriptionForm.medications];
                      updatedMedications[index] = { ...updatedMedications[index], [field]: value };
                      setPrescriptionForm({ medications: updatedMedications });
                    };

                    return (
                      <div key={index} className="border rounded-lg p-4 bg-white">
                        <div className="flex justify-between items-center mb-4">
                          <h4 className="font-medium text-gray-900">Medication {index + 1}</h4>
                          {prescriptionForm.medications.length > 1 && (
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                const updatedMedications = prescriptionForm.medications.filter((_, i) => i !== index);
                                setPrescriptionForm({ medications: updatedMedications });
                              }}
                              className="text-red-600 hover:text-red-700"
                            >
                              Remove
                            </Button>
                          )}
                        </div>

                        {/* Basic Medication Info */}
                        <div className="grid grid-cols-2 gap-4 mb-4">
                          <div>
                            <label className="text-sm font-medium">Medication Name</label>
                            <div className="relative">
                              <Input
                                placeholder="Type or select medication..."
                                value={medication.medication}
                                onChange={(e) => updateMedication('medication', e.target.value)}
                                className="mb-2"
                              />
                              <div className="flex flex-wrap gap-2">
                                {["Amoxicillin", "Paracetamol", "Ibuprofen", "Metformin", "Atorvastatin", "Lisinopril"].map((med) => (
                                  <Button
                                    key={med}
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => updateMedication('medication', med)}
                                    className="text-xs"
                                  >
                                    {med}
                                  </Button>
                                ))}
                              </div>
                            </div>
                          </div>
                          <div>
                            <label className="text-sm font-medium">Strength & Form</label>
                            <div className="space-y-2">
                              <div className="flex gap-2">
                                <Input
                                  placeholder="e.g., 500mg"
                                  value={medication.strength}
                                  onChange={(e) => updateMedication('strength', e.target.value)}
                                  className="flex-1"
                                />
                                <Select onValueChange={(value) => updateMedication('form', value)}>
                                  <SelectTrigger className="w-32">
                                    <SelectValue placeholder="Form" />
                                  </SelectTrigger>
                                  <SelectContent>
                                    <SelectItem value="tablet">Tablet</SelectItem>
                                    <SelectItem value="capsule">Capsule</SelectItem>
                                    <SelectItem value="syrup">Syrup</SelectItem>
                                    <SelectItem value="injection">Injection</SelectItem>
                                    <SelectItem value="cream">Cream</SelectItem>
                                    <SelectItem value="drops">Drops</SelectItem>
                                  </SelectContent>
                                </Select>
                              </div>
                              <div className="flex flex-wrap gap-1">
                                {["10mg", "25mg", "50mg", "100mg", "250mg", "500mg", "1g", "5ml", "10ml"].map((strength) => (
                                  <Button
                                    key={strength}
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => updateMedication('strength', strength)}
                                    className="text-xs px-2 py-1 h-6"
                                  >
                                    {strength}
                                  </Button>
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Dosing Schedule Table */}
                        <div className="bg-gray-50 rounded-lg p-4 mb-4">
                          <h3 className="font-medium mb-3">Dosing Schedule</h3>
                          <div className="grid grid-cols-4 gap-3">
                            <div className="text-center">
                              <label className="text-sm font-medium block mb-2">Morning</label>
                              <div className="space-y-2">
                                <div className="flex items-center justify-center space-x-2">
                                  <input type="checkbox" id={`morning-before-${index}`} className="rounded" />
                                  <label htmlFor={`morning-before-${index}`} className="text-xs">Before meal</label>
                                </div>
                                <div className="flex items-center justify-center space-x-2">
                                  <input type="checkbox" id={`morning-after-${index}`} className="rounded" />
                                  <label htmlFor={`morning-after-${index}`} className="text-xs">After meal</label>
                                </div>
                                <Input placeholder="Dose" className="text-center text-sm" />
                              </div>
                            </div>
                            
                            <div className="text-center">
                              <label className="text-sm font-medium block mb-2">Afternoon</label>
                              <div className="space-y-2">
                                <div className="flex items-center justify-center space-x-2">
                                  <input type="checkbox" id={`afternoon-before-${index}`} className="rounded" />
                                  <label htmlFor={`afternoon-before-${index}`} className="text-xs">Before meal</label>
                                </div>
                                <div className="flex items-center justify-center space-x-2">
                                  <input type="checkbox" id={`afternoon-after-${index}`} className="rounded" />
                                  <label htmlFor={`afternoon-after-${index}`} className="text-xs">After meal</label>
                                </div>
                                <Input placeholder="Dose" className="text-center text-sm" />
                              </div>
                            </div>
                            
                            <div className="text-center">
                              <label className="text-sm font-medium block mb-2">Evening</label>
                              <div className="space-y-2">
                                <div className="flex items-center justify-center space-x-2">
                                  <input type="checkbox" id={`evening-before-${index}`} className="rounded" />
                                  <label htmlFor={`evening-before-${index}`} className="text-xs">Before meal</label>
                                </div>
                                <div className="flex items-center justify-center space-x-2">
                                  <input type="checkbox" id={`evening-after-${index}`} className="rounded" />
                                  <label htmlFor={`evening-after-${index}`} className="text-xs">After meal</label>
                                </div>
                                <Input placeholder="Dose" className="text-center text-sm" />
                              </div>
                            </div>
                            
                            <div className="text-center">
                              <label className="text-sm font-medium block mb-2">Bedtime</label>
                              <div className="space-y-2">
                                <div className="flex items-center justify-center space-x-2">
                                  <input type="checkbox" id={`bedtime-empty-${index}`} className="rounded" />
                                  <label htmlFor={`bedtime-empty-${index}`} className="text-xs">Empty stomach</label>
                                </div>
                                <div className="flex items-center justify-center space-x-2">
                                  <input type="checkbox" id={`bedtime-food-${index}`} className="rounded" />
                                  <label htmlFor={`bedtime-food-${index}`} className="text-xs">With food</label>
                                </div>
                                <Input placeholder="Dose" className="text-center text-sm" />
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Duration and Route */}
                        <div className="grid grid-cols-2 gap-4 mb-4">
                          <div>
                            <label className="text-sm font-medium">Duration</label>
                            <div className="flex gap-2">
                              <Input
                                placeholder="Number"
                                className="w-20"
                                value={medication.duration}
                                onChange={(e) => updateMedication('duration', e.target.value)}
                              />
                              <Select>
                                <SelectTrigger className="flex-1">
                                  <SelectValue placeholder="Time unit" />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="days">Days</SelectItem>
                                  <SelectItem value="weeks">Weeks</SelectItem>
                                  <SelectItem value="months">Months</SelectItem>
                                  <SelectItem value="as_needed">As needed</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                          </div>
                          <div>
                            <label className="text-sm font-medium">Route</label>
                            <Select onValueChange={(value) => updateMedication('route', value)}>
                              <SelectTrigger>
                                <SelectValue placeholder="Administration route" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="oral">Oral</SelectItem>
                                <SelectItem value="topical">Topical</SelectItem>
                                <SelectItem value="injection">Injection</SelectItem>
                                <SelectItem value="inhalation">Inhalation</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                        </div>

                        {/* Special Instructions */}
                        <div>
                          <label className="text-sm font-medium">Special Instructions for this medication</label>
                          <Textarea
                            placeholder="Additional instructions, warnings, or precautions for this medication..."
                            value={medication.specialInstructions}
                            onChange={(e) => updateMedication('specialInstructions', e.target.value)}
                            rows={2}
                          />
                        </div>
                      </div>
                    );
                  })}
                  <Button 
                    onClick={() => prescriptionMutation.mutate({
                      ...prescriptionForm,
                      patientId: selectedPatient?.id
                    })}
                    disabled={prescriptionMutation.isPending}
                    className="w-full"
                  >
                    {prescriptionMutation.isPending ? "Creating..." : "Create Prescription"}
                  </Button>
                </div>
              </DialogContent>
            </Dialog>

            <Dialog open={labOrderDialog} onOpenChange={setLabOrderDialog}>
              <DialogTrigger asChild>
                <Button variant="outline" className="flex items-center gap-2">
                  <TestTube className="w-4 h-4" />
                  Order Lab Tests
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Order Lab Tests</DialogTitle>
                  <DialogDescription>
                    Order lab tests for {selectedPatient?.name}
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  {/* Blood Tests */}
                  <div>
                    <label className="text-sm font-medium">Blood Tests</label>
                    <Select onValueChange={(value) => {
                      if (!labOrderForm.bloodTests.includes(value)) {
                        setLabOrderForm({
                          ...labOrderForm, 
                          bloodTests: [...labOrderForm.bloodTests, value]
                        });
                      }
                    }}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select blood tests" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="CBC">Complete Blood Count (CBC)</SelectItem>
                        <SelectItem value="BMP">Basic Metabolic Panel (BMP)</SelectItem>
                        <SelectItem value="CMP">Comprehensive Metabolic Panel (CMP)</SelectItem>
                        <SelectItem value="Lipid Panel">Lipid Panel</SelectItem>
                        <SelectItem value="HbA1c">Hemoglobin A1c</SelectItem>
                        <SelectItem value="TSH">Thyroid Stimulating Hormone (TSH)</SelectItem>
                        <SelectItem value="T3T4">Free T3 & T4</SelectItem>
                        <SelectItem value="Liver Function">Liver Function Tests</SelectItem>
                        <SelectItem value="Kidney Function">Kidney Function Tests</SelectItem>
                        <SelectItem value="ESR">Erythrocyte Sedimentation Rate (ESR)</SelectItem>
                        <SelectItem value="CRP">C-Reactive Protein (CRP)</SelectItem>
                        <SelectItem value="PT/INR">Prothrombin Time/INR</SelectItem>
                      </SelectContent>
                    </Select>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {labOrderForm.bloodTests.map((test, index) => (
                        <Badge key={index} variant="secondary" className="flex items-center gap-1">
                          {test}
                          <button onClick={() => setLabOrderForm({
                            ...labOrderForm,
                            bloodTests: labOrderForm.bloodTests.filter((_, i) => i !== index)
                          })}>×</button>
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Imaging Tests */}
                  <div>
                    <label className="text-sm font-medium">Imaging Tests</label>
                    <Select onValueChange={(value) => {
                      if (!labOrderForm.imagingTests.includes(value)) {
                        setLabOrderForm({
                          ...labOrderForm, 
                          imagingTests: [...labOrderForm.imagingTests, value]
                        });
                      }
                    }}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select imaging tests" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Chest X-Ray">Chest X-Ray</SelectItem>
                        <SelectItem value="Abdominal X-Ray">Abdominal X-Ray</SelectItem>
                        <SelectItem value="CT Chest">CT Scan - Chest</SelectItem>
                        <SelectItem value="CT Abdomen">CT Scan - Abdomen & Pelvis</SelectItem>
                        <SelectItem value="CT Head">CT Scan - Head</SelectItem>
                        <SelectItem value="MRI Brain">MRI - Brain</SelectItem>
                        <SelectItem value="MRI Spine">MRI - Spine</SelectItem>
                        <SelectItem value="Ultrasound Abdomen">Ultrasound - Abdomen</SelectItem>
                        <SelectItem value="Echo">Echocardiogram</SelectItem>
                        <SelectItem value="ECG">Electrocardiogram (ECG)</SelectItem>
                      </SelectContent>
                    </Select>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {labOrderForm.imagingTests.map((test, index) => (
                        <Badge key={index} variant="outline" className="flex items-center gap-1">
                          {test}
                          <button onClick={() => setLabOrderForm({
                            ...labOrderForm,
                            imagingTests: labOrderForm.imagingTests.filter((_, i) => i !== index)
                          })}>×</button>
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Other Tests */}
                  <div>
                    <label className="text-sm font-medium">Other Tests</label>
                    <Select onValueChange={(value) => {
                      if (!labOrderForm.otherTests.includes(value)) {
                        setLabOrderForm({
                          ...labOrderForm, 
                          otherTests: [...labOrderForm.otherTests, value]
                        });
                      }
                    }}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select other tests" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Urine Analysis">Urine Analysis</SelectItem>
                        <SelectItem value="Stool Test">Stool Analysis</SelectItem>
                        <SelectItem value="Blood Sugar">Random Blood Sugar</SelectItem>
                        <SelectItem value="Fasting Sugar">Fasting Blood Sugar</SelectItem>
                        <SelectItem value="OGTT">Oral Glucose Tolerance Test</SelectItem>
                        <SelectItem value="Culture">Blood/Urine Culture</SelectItem>
                        <SelectItem value="Biopsy">Tissue Biopsy</SelectItem>
                      </SelectContent>
                    </Select>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {labOrderForm.otherTests.map((test, index) => (
                        <Badge key={index} variant="destructive" className="flex items-center gap-1">
                          {test}
                          <button onClick={() => setLabOrderForm({
                            ...labOrderForm,
                            otherTests: labOrderForm.otherTests.filter((_, i) => i !== index)
                          })}>×</button>
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium">Priority</label>
                      <Select value={labOrderForm.priority} onValueChange={(value) => 
                        setLabOrderForm({...labOrderForm, priority: value})
                      }>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="routine">Routine (24-48 hours)</SelectItem>
                          <SelectItem value="urgent">Urgent (4-6 hours)</SelectItem>
                          <SelectItem value="stat">STAT (Immediate)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-center space-x-2 pt-6">
                      <input
                        type="checkbox"
                        id="fasting"
                        checked={labOrderForm.fasting}
                        onChange={(e) => setLabOrderForm({...labOrderForm, fasting: e.target.checked})}
                        className="rounded"
                      />
                      <label htmlFor="fasting" className="text-sm font-medium">
                        Fasting Required (8-12 hours)
                      </label>
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium">Clinical Indications</label>
                    <Textarea
                      placeholder="Why are these tests being ordered? Clinical symptoms, differential diagnosis..."
                      value={labOrderForm.indicationsForTesting}
                      onChange={(e) => setLabOrderForm({...labOrderForm, indicationsForTesting: e.target.value})}
                      rows={2}
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium">Additional Clinical Notes</label>
                    <Textarea
                      placeholder="Special instructions for lab technicians, patient preparation notes..."
                      value={labOrderForm.clinicalNotes}
                      onChange={(e) => setLabOrderForm({...labOrderForm, clinicalNotes: e.target.value})}
                      rows={2}
                    />
                  </div>
                  <Button 
                    onClick={() => labOrderMutation.mutate({
                      ...labOrderForm,
                      patientId: selectedPatient?.id
                    })}
                    disabled={labOrderMutation.isPending}
                    className="w-full"
                  >
                    {labOrderMutation.isPending ? "Ordering..." : "Order Tests"}
                  </Button>
                </div>
              </DialogContent>
            </Dialog>

            <Dialog open={aiConsultDialog} onOpenChange={setAiConsultDialog}>
              <DialogTrigger asChild>
                <Button variant="outline" className="flex items-center gap-2">
                  <Bot className="w-4 h-4" />
                  AI Consult
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>AI Medical Consultation</DialogTitle>
                  <DialogDescription>
                    Get AI assistance for diagnosis and treatment recommendations
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Patient</label>
                    <Select onValueChange={(value) => setSelectedPatient({id: value, name: value})}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select patient" />
                      </SelectTrigger>
                      <SelectContent>
                        {(patientQueue || []).map((patient: any) => (
                          <SelectItem key={patient.id} value={patient.id}>
                            {patient.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Consultation Type</label>
                    <Select defaultValue="diagnosis">
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="diagnosis">Diagnosis Assistance</SelectItem>
                        <SelectItem value="treatment">Treatment Options</SelectItem>
                        <SelectItem value="drug_interaction">Drug Interactions</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Query</label>
                    <Textarea
                      placeholder="Describe symptoms, patient history, or your medical question..."
                      rows={4}
                    />
                  </div>
                  <Button className="w-full">
                    <Brain className="w-4 h-4 mr-2" />
                    Get AI Consultation
                  </Button>
                </div>
              </DialogContent>
            </Dialog>

            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" className="flex items-center gap-2">
                  <UserPlus className="w-4 h-4" />
                  Admit Patient
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Admit Patient to Hospital</DialogTitle>
                  <DialogDescription>
                    Admit patient for inpatient care and assign room
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Patient</label>
                    <Select>
                      <SelectTrigger>
                        <SelectValue placeholder="Select patient to admit" />
                      </SelectTrigger>
                      <SelectContent>
                        {(patientQueue || []).map((patient: any) => (
                          <SelectItem key={patient.id} value={patient.id}>
                            {patient.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Admission Type</label>
                    <Select>
                      <SelectTrigger>
                        <SelectValue placeholder="Select admission type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="emergency">Emergency Admission</SelectItem>
                        <SelectItem value="elective">Elective Admission</SelectItem>
                        <SelectItem value="urgent">Urgent Admission</SelectItem>
                        <SelectItem value="observation">Observation</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Department</label>
                    <Select>
                      <SelectTrigger>
                        <SelectValue placeholder="Select department" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="general">General Medicine</SelectItem>
                        <SelectItem value="surgery">Surgery</SelectItem>
                        <SelectItem value="cardiology">Cardiology</SelectItem>
                        <SelectItem value="orthopedics">Orthopedics</SelectItem>
                        <SelectItem value="icu">Intensive Care Unit</SelectItem>
                        <SelectItem value="pediatrics">Pediatrics</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Room Assignment</label>
                    <Select>
                      <SelectTrigger>
                        <SelectValue placeholder="Assign room" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="101">Room 101 - General Ward</SelectItem>
                        <SelectItem value="201">Room 201 - Private Room</SelectItem>
                        <SelectItem value="301">Room 301 - ICU</SelectItem>
                        <SelectItem value="401">Room 401 - Surgery Recovery</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium">Admission Diagnosis</label>
                    <Textarea
                      placeholder="Primary diagnosis for admission..."
                      rows={2}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium">Clinical Notes</label>
                    <Textarea
                      placeholder="Additional clinical notes, care instructions..."
                      rows={3}
                    />
                  </div>
                  <Button className="w-full">
                    <UserPlus className="w-4 h-4 mr-2" />
                    Admit Patient
                  </Button>
                </div>
              </DialogContent>
            </Dialog>

            <Button variant="outline" className="flex items-center gap-2">
              <Calendar className="w-4 h-4" />
              Schedule Followup
            </Button>
          </div>
        </div>

        {/* AI Scribe Panel */}
        {isRecording && (
          <div className="bg-gradient-to-r from-red-50 to-pink-50 border border-red-200 rounded-xl p-4 mb-6">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                  <span className="font-medium text-red-800">AI Scribe Recording</span>
                </div>
                <Volume2 className="w-4 h-4 text-red-600" />
              </div>
              <Button size="sm" variant="outline" onClick={toggleRecording}>
                <StopCircle className="w-4 h-4 mr-2" />
                Save & Stop
              </Button>
            </div>
            <div className="bg-white rounded-lg p-3 border">
              <p className="text-sm text-gray-700 italic">
                {transcription}
              </p>
            </div>
          </div>
        )}

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Patient Queue & Appointments */}
          <div className="lg:col-span-2 space-y-6">
            {/* Patient Queue */}
            <HealthcareCard title="Patient Queue" description="Your assigned patients today">
              <div className="space-y-3">
                {queueLoading ? (
                  <div className="text-center py-8 text-gray-500">Loading patient queue...</div>
                ) : patientQueue.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">No patients in queue</div>
                ) : (
                  (patientQueue || []).map((patient: any) => (
                    <div key={patient.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                          <Users className="w-5 h-5 text-blue-600" />
                        </div>
                        <div>
                          <p className="font-semibold">{patient.name || `Queue #${patient.queueNumber}`}</p>
                          <p className="text-sm text-gray-600">{patient.reasonForVisit}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge 
                              variant="outline" 
                              className={
                                patient.priority === 'critical' ? 'border-red-500 text-red-700' :
                                patient.priority === 'high' ? 'border-orange-500 text-orange-700' :
                                'border-green-500 text-green-700'
                              }
                            >
                              {patient.priority?.toUpperCase() || 'NORMAL'}
                            </Badge>
                            <span className="text-xs text-gray-500">
                              {patient.estimatedWaitTime ? `~${patient.estimatedWaitTime}min` : 'Waiting'}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button 
                          size="sm" 
                          variant="outline"
                          onClick={() => {
                            setSelectedPatientDetails(patient);
                            setPatientDetailsDialog(true);
                          }}
                        >
                          <FileText className="w-4 h-4 mr-1" />
                          View Details
                        </Button>
                        <Button 
                          size="sm"
                          onClick={() => {
                            setSelectedPatient(patient);
                            setSelectedPatientDetails(patient);
                            setPatientDetailsDialog(true);
                          }}
                        >
                          <Stethoscope className="w-4 h-4 mr-1" />
                          Consult
                        </Button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </HealthcareCard>

            {/* Appointments */}
            <HealthcareCard title="Today's Appointments" description="Scheduled consultations">
              <div className="space-y-3">
                {appointmentsLoading ? (
                  <div className="text-center py-4 text-gray-500">Loading appointments...</div>
                ) : appointments.length === 0 ? (
                  <div className="text-center py-4 text-gray-500">No appointments scheduled</div>
                ) : (
                  appointments.map((appointment: any) => (
                    <div 
                      key={appointment.id} 
                      className="flex items-center justify-between p-3 border rounded-lg cursor-pointer hover:bg-gray-50 transition-colors"
                      onClick={() => {
                        setSelectedPatientDetails(appointment);
                        setPatientDetailsDialog(true);
                      }}
                    >
                      <div className="flex items-center gap-3">
                        <Calendar className="w-5 h-5 text-blue-600" />
                        <div>
                          <p className="font-medium">{appointment.patientName}</p>
                          <p className="text-sm text-gray-600">{appointment.appointmentType}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="font-medium">{new Date(appointment.scheduledTime).toLocaleTimeString()}</p>
                        <Badge 
                          variant="outline"
                          className={
                            appointment.status === 'completed' ? 'border-green-500 text-green-700' :
                            appointment.status === 'in_progress' ? 'border-yellow-500 text-yellow-700' :
                            'border-blue-500 text-blue-700'
                          }
                        >
                          {appointment.status}
                        </Badge>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </HealthcareCard>
          </div>

          {/* Right Column - Lab Results & Quick Actions */}
          <div className="space-y-6">
            {/* Quick Stats */}
            <div className="grid grid-cols-2 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <Users className="w-5 h-5 text-blue-600" />
                    <div>
                      <p className="text-2xl font-bold">{patientQueue.length}</p>
                      <p className="text-xs text-gray-600">Patients Today</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <FlaskConical className="w-5 h-5 text-green-600" />
                    <div>
                      <p className="text-2xl font-bold">{labResults.length}</p>
                      <p className="text-xs text-gray-600">Lab Reports</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Lab Results */}
            <HealthcareCard title="Laboratory Results" description="Recent test results">
              <div className="space-y-3">
                {labLoading ? (
                  <div className="text-center py-4 text-gray-500">Loading lab results...</div>
                ) : labResults.length === 0 ? (
                  <div className="text-center py-4 text-gray-500">No recent lab results</div>
                ) : (
                  labResults.slice(0, 5).map((result: any) => (
                    <div key={result.id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <p className="font-medium">{result.testName}</p>
                        <p className="text-sm text-gray-600">{result.patientName}</p>
                      </div>
                      <div className="text-right">
                        <Badge 
                          variant="outline"
                          className={
                            result.status === 'abnormal' ? 'border-red-500 text-red-700' :
                            result.status === 'critical' ? 'border-red-600 text-red-800 bg-red-50' :
                            'border-green-500 text-green-700'
                          }
                        >
                          {result.status}
                        </Badge>
                        <p className="text-xs text-gray-500 mt-1">
                          {new Date(result.reportedAt).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </HealthcareCard>

            {/* Quick Actions */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Quick Actions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3">
                  <Button className="w-full justify-start" variant="outline">
                    <PlusCircle className="w-4 h-4 mr-2" />
                    New Consultation
                  </Button>
                  <Button className="w-full justify-start" variant="outline">
                    <FlaskConical className="w-4 h-4 mr-2" />
                    Order Lab Tests
                  </Button>
                  <Button className="w-full justify-start" variant="outline">
                    <Search className="w-4 h-4 mr-2" />
                    Search Patient Records
                  </Button>
                  <Button className="w-full justify-start" variant="outline">
                    <MessageCircle className="w-4 h-4 mr-2" />
                    Consult Colleague
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Recent Consultations */}
        <div className="mt-6">
          <HealthcareCard title="Recent Consultations" description="Patient interactions and treatments">
            <div className="grid gap-4">
              {consultationsLoading ? (
                <div className="text-center py-8 text-gray-500">Loading consultations...</div>
              ) : consultations.length === 0 ? (
                <div className="text-center py-8 text-gray-500">No recent consultations</div>
              ) : (
                (consultations || []).slice(0, 6).map((consultation: any) => (
                  <div key={consultation.id} className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 bg-teal-100 rounded-full flex items-center justify-center">
                        <Heart className="w-5 h-5 text-teal-600" />
                      </div>
                      <div>
                        <p className="font-semibold">{consultation.patientName}</p>
                        <p className="text-sm text-gray-600">{consultation.notes?.substring(0, 60)}...</p>
                        <p className="text-xs text-gray-500">
                          {new Date(consultation.consultationTime).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge 
                        variant="outline"
                        className={
                          consultation.status === 'completed' ? 'border-green-500 text-green-700' :
                          consultation.status === 'in_progress' ? 'border-yellow-500 text-yellow-700' :
                          'border-blue-500 text-blue-700'
                        }
                      >
                        {consultation.status}
                      </Badge>
                      <Button size="sm" variant="outline">View Details</Button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </HealthcareCard>
        </div>

        {/* Patient Details Dialog */}
        <Dialog open={patientDetailsDialog} onOpenChange={setPatientDetailsDialog}>
          <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Patient Details - {selectedPatientDetails?.name || selectedPatientDetails?.patientName}</DialogTitle>
              <DialogDescription>
                Complete patient information, vitals, and medical history
              </DialogDescription>
            </DialogHeader>
            
            {selectedPatientDetails && (
              <div className="space-y-6">
                {/* Patient Info Header */}
                <div className="bg-blue-50 rounded-lg p-4">
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <label className="text-sm font-medium text-gray-600">Patient Name</label>
                      <p className="font-semibold">{selectedPatientDetails.name || selectedPatientDetails.patientName}</p>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-gray-600">Age</label>
                      <p className="font-semibold">{selectedPatientDetails.age || '32 years'}</p>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-gray-600">Gender</label>
                      <p className="font-semibold">{selectedPatientDetails.gender || 'Female'}</p>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 mt-3">
                    <div>
                      <label className="text-sm font-medium text-gray-600">Reason for Visit</label>
                      <p className="font-semibold">{selectedPatientDetails.reasonForVisit || selectedPatientDetails.appointmentType || 'General Consultation'}</p>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-gray-600">Priority</label>
                      <Badge 
                        variant="outline" 
                        className={
                          selectedPatientDetails.priority === 'critical' ? 'border-red-500 text-red-700' :
                          selectedPatientDetails.priority === 'high' ? 'border-orange-500 text-orange-700' :
                          'border-green-500 text-green-700'
                        }
                      >
                        {selectedPatientDetails.priority?.toUpperCase() || 'NORMAL'}
                      </Badge>
                    </div>
                  </div>
                </div>

                {/* Current Vitals */}
                <div>
                  <h3 className="text-lg font-semibold mb-3">Current Vitals</h3>
                  {vitalsLoading ? (
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="text-center py-4 text-gray-500">Loading vitals...</div>
                    </div>
                  ) : patientVitals && patientVitals.length > 0 ? (
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="grid grid-cols-4 gap-4">
                        <div className="text-center">
                          <Heart className="w-8 h-8 text-red-500 mx-auto mb-2" />
                          <p className="text-2xl font-bold text-red-600">{patientVitals[0].heartRate || '--'}</p>
                          <p className="text-sm text-gray-600">Heart Rate</p>
                          <p className="text-xs text-gray-500">bpm</p>
                        </div>
                        <div className="text-center">
                          <Activity className="w-8 h-8 text-blue-500 mx-auto mb-2" />
                          <p className="text-2xl font-bold text-blue-600">{patientVitals[0].bloodPressure || '--'}</p>
                          <p className="text-sm text-gray-600">Blood Pressure</p>
                          <p className="text-xs text-gray-500">mmHg</p>
                        </div>
                        <div className="text-center">
                          <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-2">
                            <span className="text-green-600 font-bold">°</span>
                          </div>
                          <p className="text-2xl font-bold text-green-600">{patientVitals[0].temperatureC || '--'}</p>
                          <p className="text-sm text-gray-600">Temperature</p>
                          <p className="text-xs text-gray-500">°C</p>
                        </div>
                        <div className="text-center">
                          <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-2">
                            <span className="text-purple-600 font-bold">O₂</span>
                          </div>
                          <p className="text-2xl font-bold text-purple-600">{patientVitals[0].oxygenSaturation || '--'}</p>
                          <p className="text-sm text-gray-600">Oxygen Sat</p>
                          <p className="text-xs text-gray-500">%</p>
                        </div>
                      </div>
                      <div className="mt-3 text-xs text-gray-500 text-center">
                        Recorded: {patientVitals[0].recordedAt ? new Date(patientVitals[0].recordedAt).toLocaleString() : 'Unknown'}
                      </div>
                    </div>
                  ) : (
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="text-center py-4 text-gray-500">No recent vitals recorded</div>
                    </div>
                  )}
                </div>

                {/* Medical History */}
                <div>
                  <h3 className="text-lg font-semibold mb-3">Medical History</h3>
                  {historyLoading ? (
                    <div className="text-center py-4 text-gray-500">Loading medical history...</div>
                  ) : patientMedicalHistory && patientMedicalHistory.length > 0 ? (
                    <div className="space-y-3">
                      {patientMedicalHistory.map((record: any) => (
                        <div key={record.id} className="border rounded-lg p-3">
                          <div className="flex justify-between items-start">
                            <div>
                              <p className="font-medium">{record.recordType}</p>
                              <p className="text-sm text-gray-600">
                                {record.createdAt ? new Date(record.createdAt).toLocaleDateString() : 'Date unknown'}
                              </p>
                            </div>
                            <Badge variant="outline" className="border-blue-500 text-blue-700">Medical Record</Badge>
                          </div>
                          <p className="text-sm text-gray-700 mt-2">{record.content}</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-4 text-gray-500">No medical history available</div>
                  )}
                </div>

                {/* Recent Lab Results */}
                <div>
                  <h3 className="text-lg font-semibold mb-3">Recent Lab Results</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse border border-gray-200">
                      <thead>
                        <tr className="bg-gray-50">
                          <th className="border border-gray-200 px-3 py-2 text-left">Test</th>
                          <th className="border border-gray-200 px-3 py-2 text-left">Result</th>
                          <th className="border border-gray-200 px-3 py-2 text-left">Reference Range</th>
                          <th className="border border-gray-200 px-3 py-2 text-left">Status</th>
                          <th className="border border-gray-200 px-3 py-2 text-left">Date</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td className="border border-gray-200 px-3 py-2">HbA1c</td>
                          <td className="border border-gray-200 px-3 py-2 font-medium">7.2%</td>
                          <td className="border border-gray-200 px-3 py-2">4.0-5.6%</td>
                          <td className="border border-gray-200 px-3 py-2">
                            <Badge variant="outline" className="border-orange-500 text-orange-700">Elevated</Badge>
                          </td>
                          <td className="border border-gray-200 px-3 py-2">Dec 15, 2024</td>
                        </tr>
                        <tr>
                          <td className="border border-gray-200 px-3 py-2">Total Cholesterol</td>
                          <td className="border border-gray-200 px-3 py-2 font-medium">195 mg/dL</td>
                          <td className="border border-gray-200 px-3 py-2">&lt;200 mg/dL</td>
                          <td className="border border-gray-200 px-3 py-2">
                            <Badge variant="outline" className="border-green-500 text-green-700">Normal</Badge>
                          </td>
                          <td className="border border-gray-200 px-3 py-2">Dec 15, 2024</td>
                        </tr>
                        <tr>
                          <td className="border border-gray-200 px-3 py-2">Creatinine</td>
                          <td className="border border-gray-200 px-3 py-2 font-medium">1.0 mg/dL</td>
                          <td className="border border-gray-200 px-3 py-2">0.6-1.2 mg/dL</td>
                          <td className="border border-gray-200 px-3 py-2">
                            <Badge variant="outline" className="border-green-500 text-green-700">Normal</Badge>
                          </td>
                          <td className="border border-gray-200 px-3 py-2">Dec 15, 2024</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Current Medications */}
                <div>
                  <h3 className="text-lg font-semibold mb-3">Current Medications</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                      <div>
                        <p className="font-medium">Metformin 500mg</p>
                        <p className="text-sm text-gray-600">Twice daily with meals</p>
                      </div>
                      <Badge variant="outline" className="border-blue-500 text-blue-700">Active</Badge>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                      <div>
                        <p className="font-medium">Lisinopril 10mg</p>
                        <p className="text-sm text-gray-600">Once daily in the morning</p>
                      </div>
                      <Badge variant="outline" className="border-blue-500 text-blue-700">Active</Badge>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <div>
                        <p className="font-medium">Cetirizine 10mg</p>
                        <p className="text-sm text-gray-600">As needed for allergies</p>
                      </div>
                      <Badge variant="outline" className="border-gray-500 text-gray-700">PRN</Badge>
                    </div>
                  </div>
                </div>

                {/* Previous Visits */}
                <div>
                  <h3 className="text-lg font-semibold mb-3">Previous Visits</h3>
                  <div className="space-y-3">
                    <div className="border-l-4 border-blue-500 pl-4 py-2">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="font-medium">Routine Diabetes Follow-up</p>
                          <p className="text-sm text-gray-600">December 1, 2024</p>
                        </div>
                        <Badge variant="outline">Completed</Badge>
                      </div>
                      <p className="text-sm text-gray-700 mt-1">HbA1c slightly elevated. Discussed dietary modifications.</p>
                    </div>
                    
                    <div className="border-l-4 border-green-500 pl-4 py-2">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="font-medium">Annual Physical Exam</p>
                          <p className="text-sm text-gray-600">September 15, 2024</p>
                        </div>
                        <Badge variant="outline">Completed</Badge>
                      </div>
                      <p className="text-sm text-gray-700 mt-1">Overall health stable. Blood pressure well controlled.</p>
                    </div>

                    <div className="border-l-4 border-orange-500 pl-4 py-2">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="font-medium">Allergy Consultation</p>
                          <p className="text-sm text-gray-600">June 20, 2024</p>
                        </div>
                        <Badge variant="outline">Completed</Badge>
                      </div>
                      <p className="text-sm text-gray-700 mt-1">Seasonal allergies confirmed. Prescribed antihistamine regimen.</p>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3 pt-4 border-t">
                  <Button 
                    onClick={() => {
                      setPatientDetailsDialog(false);
                      setPrescriptionDialog(true);
                    }}
                    className="flex items-center gap-2"
                  >
                    <Pill className="w-4 h-4" />
                    Write Prescription
                  </Button>
                  <Button 
                    variant="outline"
                    onClick={() => {
                      setPatientDetailsDialog(false);
                      setLabOrderDialog(true);
                    }}
                    className="flex items-center gap-2"
                  >
                    <TestTube className="w-4 h-4" />
                    Order Lab Tests
                  </Button>
                  <Button variant="outline" className="flex items-center gap-2">
                    <FileText className="w-4 h-4" />
                    Add Notes
                  </Button>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}