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
  Users, 
  Calendar, 
  Clock, 
  Phone, 
  DollarSign, 
  CreditCard, 
  CheckCircle, 
  AlertCircle, 
  UserPlus, 
  Search, 
  Bell,
  MapPin,
  Shield,
  FileText,
  MessageCircle,
  Printer,
  QrCode,
  Smartphone,
  Clipboard
} from "lucide-react";

export default function ReceptionistDashboard() {
  const { user } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Fetch queue data
  const { data: queueData = [], isLoading: queueLoading } = useQuery({
    queryKey: ["/api/receptionist/queue"],
  });

  // Fetch today's appointments
  const { data: appointments = [], isLoading: appointmentsLoading } = useQuery({
    queryKey: ["/api/receptionist/appointments"],
  });

  // Fetch check-in data
  const { data: checkInData = [], isLoading: checkInLoading } = useQuery({
    queryKey: ["/api/receptionist/check-ins"],
  });

  // Fetch billing data
  const { data: billingData = [], isLoading: billingLoading } = useQuery({
    queryKey: ["/api/receptionist/billing"],
  });

  const waitingPatients = queueData.filter((p: any) => p.status === 'waiting');
  const todayAppointments = appointments.filter((a: any) => 
    new Date(a.appointmentDate).toDateString() === new Date().toDateString()
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 via-white to-purple-50">
      {/* Header */}
      <div className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Welcome, {user?.firstName} {user?.lastName}
              </h1>
              <p className="text-gray-600">Reception & Patient Flow Management</p>
            </div>
            <div className="flex items-center gap-4">
              <Badge variant="outline" className="bg-pink-100 text-pink-800">
                <Users className="w-4 h-4 mr-1" />
                Front Desk
              </Badge>
              <Button onClick={() => window.location.href = "/api/logout"} variant="ghost">
                Logout
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Key Metrics Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Waiting Patients</p>
                  <p className="text-3xl font-bold text-gray-900">{waitingPatients.length}</p>
                  <p className="text-sm text-blue-600">
                    <Clock className="w-4 h-4 inline mr-1" />
                    Avg wait: {Math.round(waitingPatients.reduce((acc: number, p: any) => acc + (p.estimatedWaitTime || 0), 0) / waitingPatients.length) || 0} min
                  </p>
                </div>
                <Users className="w-8 h-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Today's Appointments</p>
                  <p className="text-3xl font-bold text-gray-900">{todayAppointments.length}</p>
                  <p className="text-sm text-green-600">
                    <CheckCircle className="w-4 h-4 inline mr-1" />
                    {todayAppointments.filter((a: any) => a.status === 'completed').length} completed
                  </p>
                </div>
                <Calendar className="w-8 h-8 text-green-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Pending Payments</p>
                  <p className="text-3xl font-bold text-gray-900">{billingData.filter((b: any) => b.status === 'pending').length}</p>
                  <p className="text-sm text-orange-600">
                    <DollarSign className="w-4 h-4 inline mr-1" />
                    ${billingData.reduce((acc: number, b: any) => acc + (b.amount || 0), 0).toLocaleString()}
                  </p>
                </div>
                <CreditCard className="w-8 h-8 text-orange-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Check-ins Today</p>
                  <p className="text-3xl font-bold text-gray-900">{checkInData.length}</p>
                  <p className="text-sm text-purple-600">
                    <UserPlus className="w-4 h-4 inline mr-1" />
                    New registrations: {checkInData.filter((c: any) => c.isNewPatient).length}
                  </p>
                </div>
                <Clipboard className="w-8 h-8 text-purple-600" />
              </div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="queue" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="queue">Live Queue</TabsTrigger>
            <TabsTrigger value="checkin">Check-in/Out</TabsTrigger>
            <TabsTrigger value="appointments">Appointments</TabsTrigger>
            <TabsTrigger value="billing">Billing</TabsTrigger>
            <TabsTrigger value="communication">Communications</TabsTrigger>
          </TabsList>

          {/* Live Queue Management Tab */}
          <TabsContent value="queue">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <HealthcareCard title="Live Queue Management" description="Real-time patient flow and waiting times">
                  <div className="space-y-3">
                    {queueLoading ? (
                      <div className="text-center py-8 text-gray-500">Loading queue data...</div>
                    ) : queueData.length === 0 ? (
                      <div className="text-center py-8 text-gray-500">No patients in queue</div>
                    ) : (
                      queueData.map((patient: any) => (
                        <div key={patient.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                          <div className="flex items-center gap-4">
                            <div className="w-12 h-12 bg-pink-100 rounded-full flex items-center justify-center">
                              <span className="text-lg font-bold text-pink-600">#{patient.queueNumber}</span>
                            </div>
                            <div>
                              <p className="font-semibold">{patient.patientName || `Patient #${patient.queueNumber}`}</p>
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
                                <Badge 
                                  variant="outline"
                                  className={
                                    patient.status === 'waiting' ? 'border-blue-500 text-blue-700' :
                                    patient.status === 'called' ? 'border-yellow-500 text-yellow-700' :
                                    'border-green-500 text-green-700'
                                  }
                                >
                                  {patient.status?.toUpperCase()}
                                </Badge>
                                <span className="text-xs text-gray-500">
                                  Waiting: {patient.estimatedWaitTime || 0} min
                                </span>
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Button size="sm" variant="outline">
                              <Phone className="w-4 h-4 mr-1" />
                              Call
                            </Button>
                            <Button size="sm">
                              <MapPin className="w-4 h-4 mr-1" />
                              Assign Room
                            </Button>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </HealthcareCard>
              </div>

              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Queue Actions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <Button className="w-full justify-start" variant="outline">
                        <UserPlus className="w-4 h-4 mr-2" />
                        Add Walk-in Patient
                      </Button>
                      <Button className="w-full justify-start" variant="outline">
                        <Bell className="w-4 h-4 mr-2" />
                        Call Next Patient
                      </Button>
                      <Button className="w-full justify-start" variant="outline">
                        <QrCode className="w-4 h-4 mr-2" />
                        Generate Queue Number
                      </Button>
                      <Button className="w-full justify-start" variant="outline">
                        <Printer className="w-4 h-4 mr-2" />
                        Print Queue Status
                      </Button>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Department Wait Times</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {[
                        { dept: 'General Medicine', wait: 15, patients: 5 },
                        { dept: 'Cardiology', wait: 25, patients: 3 },
                        { dept: 'Emergency', wait: 5, patients: 2 },
                        { dept: 'Pediatrics', wait: 20, patients: 4 }
                      ].map((dept) => (
                        <div key={dept.dept} className="flex items-center justify-between py-2">
                          <div>
                            <p className="text-sm font-medium">{dept.dept}</p>
                            <p className="text-xs text-gray-600">{dept.patients} patients</p>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-semibold">{dept.wait} min</p>
                            <div className={`w-12 h-2 rounded-full ${
                              dept.wait < 10 ? 'bg-green-500' :
                              dept.wait < 20 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Patient Check-in/Check-out Tab */}
          <TabsContent value="checkin">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <HealthcareCard title="Patient Check-in" description="Registration and appointment check-in">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <Button className="h-20 flex-col" variant="outline">
                      <UserPlus className="w-6 h-6 mb-2" />
                      <span>New Patient Registration</span>
                    </Button>
                    <Button className="h-20 flex-col" variant="outline">
                      <CheckCircle className="w-6 h-6 mb-2" />
                      <span>Appointment Check-in</span>
                    </Button>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h4 className="font-semibold mb-3">Recent Check-ins</h4>
                    <div className="space-y-3">
                      {checkInData.slice(0, 5).map((checkin: any) => (
                        <div key={checkin.id} className="flex items-center justify-between py-2">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                              <CheckCircle className="w-4 h-4 text-green-600" />
                            </div>
                            <div>
                              <p className="font-medium">{checkin.patientName}</p>
                              <p className="text-sm text-gray-600">
                                {checkin.isNewPatient ? 'New Patient' : 'Return Visit'}
                              </p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="text-sm">{new Date(checkin.checkedInAt).toLocaleTimeString()}</p>
                            <Badge variant="outline" className="text-xs">
                              {checkin.department}
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </HealthcareCard>

              <HealthcareCard title="Insurance & Verification" description="Insurance verification and eligibility">
                <div className="space-y-4">
                  <div className="grid gap-3">
                    <Button className="w-full justify-start" variant="outline">
                      <Shield className="w-4 h-4 mr-2" />
                      Verify Insurance Coverage
                    </Button>
                    <Button className="w-full justify-start" variant="outline">
                      <FileText className="w-4 h-4 mr-2" />
                      Check Eligibility
                    </Button>
                    <Button className="w-full justify-start" variant="outline">
                      <CreditCard className="w-4 h-4 mr-2" />
                      Process Co-pay
                    </Button>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h4 className="font-semibold mb-3">Insurance Alerts</h4>
                    <div className="space-y-2">
                      {[
                        { patient: 'John Smith', issue: 'Insurance expired', type: 'error' },
                        { patient: 'Sarah Johnson', issue: 'Pre-auth required', type: 'warning' },
                        { patient: 'Mike Davis', issue: 'High deductible', type: 'info' }
                      ].map((alert, index) => (
                        <div key={index} className="flex items-center justify-between py-2">
                          <div className="flex items-center gap-2">
                            <AlertCircle className={`w-4 h-4 ${
                              alert.type === 'error' ? 'text-red-500' :
                              alert.type === 'warning' ? 'text-yellow-500' : 'text-blue-500'
                            }`} />
                            <div>
                              <p className="text-sm font-medium">{alert.patient}</p>
                              <p className="text-xs text-gray-600">{alert.issue}</p>
                            </div>
                          </div>
                          <Button size="sm" variant="outline">Resolve</Button>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </HealthcareCard>
            </div>
          </TabsContent>

          {/* Appointment Scheduler Tab */}
          <TabsContent value="appointments">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <HealthcareCard title="Appointment Scheduler" description="Calendar management and doctor availability">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h4 className="font-semibold">Today's Schedule</h4>
                      <Button size="sm">
                        <Calendar className="w-4 h-4 mr-1" />
                        New Appointment
                      </Button>
                    </div>
                    
                    <div className="space-y-3">
                      {appointmentsLoading ? (
                        <div className="text-center py-4 text-gray-500">Loading appointments...</div>
                      ) : todayAppointments.length === 0 ? (
                        <div className="text-center py-4 text-gray-500">No appointments scheduled</div>
                      ) : (
                        todayAppointments.map((appointment: any) => (
                          <div key={appointment.id} className="flex items-center justify-between p-3 border rounded-lg">
                            <div className="flex items-center gap-3">
                              <div className="text-center">
                                <p className="text-sm font-bold">{new Date(appointment.scheduledTime).toLocaleTimeString()}</p>
                                <p className="text-xs text-gray-600">{appointment.duration} min</p>
                              </div>
                              <div>
                                <p className="font-medium">{appointment.patientName}</p>
                                <p className="text-sm text-gray-600">{appointment.appointmentType}</p>
                                <p className="text-xs text-gray-500">Dr. {appointment.doctorName}</p>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge 
                                variant="outline"
                                className={
                                  appointment.status === 'completed' ? 'border-green-500 text-green-700' :
                                  appointment.status === 'in_progress' ? 'border-yellow-500 text-yellow-700' :
                                  appointment.status === 'cancelled' ? 'border-red-500 text-red-700' :
                                  'border-blue-500 text-blue-700'
                                }
                              >
                                {appointment.status}
                              </Badge>
                              <Button size="sm" variant="outline">
                                <Phone className="w-4 h-4 mr-1" />
                                Call
                              </Button>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </HealthcareCard>
              </div>

              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Doctor Availability</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {[
                        { doctor: 'Dr. Smith', status: 'Available', nextSlot: '2:30 PM' },
                        { doctor: 'Dr. Johnson', status: 'Busy', nextSlot: '4:00 PM' },
                        { doctor: 'Dr. Williams', status: 'Available', nextSlot: '1:45 PM' },
                        { doctor: 'Dr. Brown', status: 'On Break', nextSlot: '3:15 PM' }
                      ].map((doctor) => (
                        <div key={doctor.doctor} className="flex items-center justify-between py-2">
                          <div>
                            <p className="text-sm font-medium">{doctor.doctor}</p>
                            <p className="text-xs text-gray-600">Next: {doctor.nextSlot}</p>
                          </div>
                          <Badge 
                            variant="outline"
                            className={
                              doctor.status === 'Available' ? 'border-green-500 text-green-700' :
                              doctor.status === 'Busy' ? 'border-red-500 text-red-700' :
                              'border-yellow-500 text-yellow-700'
                            }
                          >
                            {doctor.status}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Quick Scheduler</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <Input placeholder="Patient name" />
                      <Input type="date" />
                      <Input type="time" />
                      <Button className="w-full">
                        <Calendar className="w-4 h-4 mr-2" />
                        Schedule Appointment
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Billing Tab */}
          <TabsContent value="billing">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <HealthcareCard title="Payment Processing" description="Billing and payment management">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-green-50 rounded-lg text-center">
                      <DollarSign className="w-6 h-6 text-green-600 mx-auto mb-2" />
                      <p className="text-lg font-bold text-green-600">
                        ${billingData.filter((b: any) => b.status === 'paid').reduce((acc: number, b: any) => acc + b.amount, 0).toLocaleString()}
                      </p>
                      <p className="text-xs text-gray-600">Collected Today</p>
                    </div>
                    <div className="p-4 bg-orange-50 rounded-lg text-center">
                      <AlertCircle className="w-6 h-6 text-orange-600 mx-auto mb-2" />
                      <p className="text-lg font-bold text-orange-600">
                        ${billingData.filter((b: any) => b.status === 'pending').reduce((acc: number, b: any) => acc + b.amount, 0).toLocaleString()}
                      </p>
                      <p className="text-xs text-gray-600">Outstanding</p>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h4 className="font-semibold mb-3">Pending Payments</h4>
                    <div className="space-y-3">
                      {billingData.filter((b: any) => b.status === 'pending').slice(0, 5).map((bill: any) => (
                        <div key={bill.id} className="flex items-center justify-between py-2">
                          <div>
                            <p className="font-medium">{bill.patientName}</p>
                            <p className="text-sm text-gray-600">Invoice #{bill.invoiceNumber}</p>
                          </div>
                          <div className="text-right">
                            <p className="font-bold text-orange-600">${bill.amount}</p>
                            <Button size="sm" variant="outline">Process</Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </HealthcareCard>

              <HealthcareCard title="Insurance Claims" description="Claims processing and management">
                <div className="space-y-4">
                  <div className="grid gap-3">
                    <Button className="w-full justify-start" variant="outline">
                      <FileText className="w-4 h-4 mr-2" />
                      Submit New Claim
                    </Button>
                    <Button className="w-full justify-start" variant="outline">
                      <Search className="w-4 h-4 mr-2" />
                      Check Claim Status
                    </Button>
                    <Button className="w-full justify-start" variant="outline">
                      <Printer className="w-4 h-4 mr-2" />
                      Print Claim Forms
                    </Button>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h4 className="font-semibold mb-3">Recent Claims</h4>
                    <div className="space-y-3">
                      {[
                        { patient: 'Alice Cooper', claim: 'CLM-2024-001', status: 'approved', amount: 250 },
                        { patient: 'Bob Wilson', claim: 'CLM-2024-002', status: 'pending', amount: 180 },
                        { patient: 'Carol Davis', claim: 'CLM-2024-003', status: 'rejected', amount: 320 }
                      ].map((claim) => (
                        <div key={claim.claim} className="flex items-center justify-between py-2">
                          <div>
                            <p className="font-medium">{claim.patient}</p>
                            <p className="text-sm text-gray-600">{claim.claim}</p>
                          </div>
                          <div className="text-right">
                            <p className="font-bold">${claim.amount}</p>
                            <Badge 
                              variant="outline"
                              className={
                                claim.status === 'approved' ? 'border-green-500 text-green-700' :
                                claim.status === 'rejected' ? 'border-red-500 text-red-700' :
                                'border-yellow-500 text-yellow-700'
                              }
                            >
                              {claim.status}
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </HealthcareCard>
            </div>
          </TabsContent>

          {/* Communication Hub Tab */}
          <TabsContent value="communication">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <HealthcareCard title="Communication Hub" description="Patient notifications and general inquiries">
                <div className="space-y-4">
                  <div className="grid gap-3">
                    <Button className="w-full justify-start" variant="outline">
                      <MessageCircle className="w-4 h-4 mr-2" />
                      Send Appointment Reminders
                    </Button>
                    <Button className="w-full justify-start" variant="outline">
                      <Smartphone className="w-4 h-4 mr-2" />
                      SMS Notifications
                    </Button>
                    <Button className="w-full justify-start" variant="outline">
                      <Bell className="w-4 h-4 mr-2" />
                      Emergency Announcements
                    </Button>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h4 className="font-semibold mb-3">Recent Messages</h4>
                    <div className="space-y-3">
                      {[
                        { patient: 'Emma Thompson', type: 'Appointment Reminder', time: '10:30 AM', status: 'sent' },
                        { patient: 'David Lee', type: 'Lab Results Ready', time: '11:15 AM', status: 'delivered' },
                        { patient: 'Maria Garcia', type: 'Payment Due', time: '9:45 AM', status: 'read' }
                      ].map((message, index) => (
                        <div key={index} className="flex items-center justify-between py-2">
                          <div>
                            <p className="font-medium">{message.patient}</p>
                            <p className="text-sm text-gray-600">{message.type}</p>
                          </div>
                          <div className="text-right">
                            <p className="text-sm">{message.time}</p>
                            <Badge variant="outline" className="text-xs">
                              {message.status}
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </HealthcareCard>

              <HealthcareCard title="Patient Information" description="Quick patient lookup and information">
                <div className="space-y-4">
                  <div className="space-y-3">
                    <Input placeholder="Search patient by name or ID" />
                    <Button className="w-full">
                      <Search className="w-4 h-4 mr-2" />
                      Search Patient Records
                    </Button>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h4 className="font-semibold mb-3">Frequently Requested Info</h4>
                    <div className="space-y-2">
                      {[
                        'Office hours and contact information',
                        'Insurance acceptance and verification',
                        'Prescription refill procedures',
                        'Lab result availability',
                        'Appointment cancellation policy'
                      ].map((info, index) => (
                        <div key={index} className="p-2 bg-gray-50 rounded text-sm">
                          {info}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </HealthcareCard>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}