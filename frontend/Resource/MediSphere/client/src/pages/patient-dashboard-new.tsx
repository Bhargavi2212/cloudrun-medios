import { useState, useEffect } from "react";
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
import { Checkbox } from "@/components/ui/checkbox";
import { HealthcareCard } from "@/components/ui/healthcare-card";
import { 
  Activity, 
  Bell, 
  Calendar, 
  Clock, 
  Download, 
  FileText, 
  Heart, 
  MapPin, 
  MessageCircle, 
  Pill, 
  Shield, 
  Smartphone, 
  User, 
  UserCheck,
  AlertTriangle,
  CheckCircle,
  FlaskConical,
  Stethoscope,
  QrCode,
  Wifi,
  WifiOff,
  RefreshCw
} from "lucide-react";

export default function PatientDashboard() {
  const { user } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isOffline, setIsOffline] = useState(!navigator.onLine);
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);

  // Monitor online/offline status
  useEffect(() => {
    const handleOnline = () => setIsOffline(false);
    const handleOffline = () => setIsOffline(true);
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Fetch patient's current visit status
  const { data: visitStatus, isLoading: visitLoading } = useQuery({
    queryKey: ["/api/patient/visit-status"],
    refetchInterval: 30000, // Refresh every 30 seconds for real-time updates
  });

  // Fetch health history
  const { data: healthHistory = [], isLoading: historyLoading } = useQuery({
    queryKey: ["/api/patient/health-history"],
  });

  // Fetch notifications
  const { data: notifications = [], isLoading: notificationsLoading } = useQuery({
    queryKey: ["/api/patient/notifications"],
    refetchInterval: 60000, // Check for new notifications every minute
  });

  // Fetch lab results
  const { data: labResults = [], isLoading: labLoading } = useQuery({
    queryKey: ["/api/patient/lab-results"],
  });

  // Self check-in mutation
  const checkInMutation = useMutation({
    mutationFn: async (data: any) => apiRequest('/api/patient/check-in', 'POST', data),
    onSuccess: () => {
      toast({ title: "Success", description: "Checked in successfully!" });
      queryClient.invalidateQueries({ queryKey: ["/api/patient/visit-status"] });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to check in", variant: "destructive" });
    }
  });

  const symptomOptions = [
    "Fever", "Headache", "Nausea", "Fatigue", "Chest Pain", 
    "Shortness of Breath", "Abdominal Pain", "Dizziness", "Cough", "Sore Throat"
  ];

  const handleSymptomToggle = (symptom: string) => {
    setSelectedSymptoms(prev => 
      prev.includes(symptom) 
        ? prev.filter(s => s !== symptom)
        : [...prev, symptom]
    );
  };

  const handleCheckIn = () => {
    checkInMutation.mutate({
      symptoms: selectedSymptoms,
      preferences: {
        preferredLanguage: 'English',
        communicationMethod: 'SMS'
      }
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <div className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Welcome, {user?.firstName} {user?.lastName}
              </h1>
              <p className="text-gray-600">Your Health Dashboard</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                {isOffline ? (
                  <Badge variant="outline" className="bg-red-100 text-red-800">
                    <WifiOff className="w-4 h-4 mr-1" />
                    Offline Mode
                  </Badge>
                ) : (
                  <Badge variant="outline" className="bg-green-100 text-green-800">
                    <Wifi className="w-4 h-4 mr-1" />
                    Connected
                  </Badge>
                )}
              </div>
              <Button onClick={() => window.location.href = "/api/logout"} variant="ghost">
                Logout
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Live Visit Tracker */}
        {visitStatus && (
          <div className="mb-6">
            <HealthcareCard 
              title="Live Visit Tracker" 
              description="Real-time updates on your current visit"
              priority={visitStatus.priority}
              status={visitStatus.status}
            >
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                    <Clock className="w-6 h-6 text-blue-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Current Wait Time</p>
                    <p className="text-2xl font-bold text-blue-600">
                      {visitStatus.estimatedWaitTime || 0} min
                    </p>
                    <p className="text-xs text-gray-500">Queue position: #{visitStatus.queuePosition || 1}</p>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                    <Stethoscope className="w-6 h-6 text-green-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Assigned Doctor</p>
                    <p className="text-lg font-semibold">Dr. {visitStatus.doctorName || 'TBD'}</p>
                    <p className="text-xs text-gray-500">{visitStatus.department || 'General Medicine'}</p>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
                    <MapPin className="w-6 h-6 text-purple-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Location</p>
                    <p className="text-lg font-semibold">Room {visitStatus.roomNumber || 'TBD'}</p>
                    <p className="text-xs text-gray-500">Floor {visitStatus.floor || '1'}</p>
                  </div>
                </div>
              </div>

              {(visitStatus.labUpdates || visitStatus.prescriptionUpdates) && (
                <div className="mt-6 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                  <h4 className="font-semibold text-yellow-800 mb-2">Recent Updates</h4>
                  <div className="space-y-2">
                    {visitStatus.labUpdates?.map((update: any, index: number) => (
                      <div key={index} className="flex items-center gap-2 text-sm">
                        <FlaskConical className="w-4 h-4 text-yellow-600" />
                        <span>{update.message}</span>
                        <Badge variant="outline" className="text-xs">{update.time}</Badge>
                      </div>
                    ))}
                    {visitStatus.prescriptionUpdates?.map((update: any, index: number) => (
                      <div key={index} className="flex items-center gap-2 text-sm">
                        <Pill className="w-4 h-4 text-yellow-600" />
                        <span>{update.message}</span>
                        <Badge variant="outline" className="text-xs">{update.time}</Badge>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </HealthcareCard>
          </div>
        )}

        <Tabs defaultValue="checkin" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="checkin">Self Check-In</TabsTrigger>
            <TabsTrigger value="history">Health History</TabsTrigger>
            <TabsTrigger value="notifications">Notifications</TabsTrigger>
            <TabsTrigger value="consent">Consent & Privacy</TabsTrigger>
          </TabsList>

          {/* Self Check-In Tab */}
          <TabsContent value="checkin">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <HealthcareCard title="Quick Check-In" description="Tap your card or enter symptoms">
                <div className="space-y-6">
                  <div className="text-center p-6 border-2 border-dashed border-blue-300 rounded-lg bg-blue-50">
                    <QrCode className="w-12 h-12 text-blue-600 mx-auto mb-4" />
                    <p className="text-lg font-semibold text-blue-800 mb-2">Tap Your Card</p>
                    <p className="text-sm text-blue-600">Use your patient card to quick check-in</p>
                    <Button className="mt-4" variant="outline">
                      <Smartphone className="w-4 h-4 mr-2" />
                      Scan QR Code
                    </Button>
                  </div>

                  <div className="space-y-4">
                    <h4 className="font-semibold">Current Symptoms</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {symptomOptions.map((symptom) => (
                        <div key={symptom} className="flex items-center space-x-2">
                          <Checkbox
                            id={symptom}
                            checked={selectedSymptoms.includes(symptom)}
                            onCheckedChange={() => handleSymptomToggle(symptom)}
                          />
                          <label htmlFor={symptom} className="text-sm font-medium">
                            {symptom}
                          </label>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-3">
                    <label className="text-sm font-medium">Additional Notes</label>
                    <Textarea placeholder="Describe any other symptoms or concerns..." />
                  </div>

                  <Button 
                    onClick={handleCheckIn} 
                    className="w-full"
                    disabled={checkInMutation.isPending}
                  >
                    {checkInMutation.isPending ? (
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <UserCheck className="w-4 h-4 mr-2" />
                    )}
                    Complete Check-In
                  </Button>
                </div>
              </HealthcareCard>

              <HealthcareCard title="Preferences" description="Communication and visit preferences">
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium">Preferred Communication</label>
                    <div className="mt-2 space-y-2">
                      <div className="flex items-center space-x-2">
                        <Checkbox id="sms" />
                        <label htmlFor="sms" className="text-sm">SMS Text Messages</label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Checkbox id="email" />
                        <label htmlFor="email" className="text-sm">Email Updates</label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Checkbox id="app" defaultChecked />
                        <label htmlFor="app" className="text-sm">In-App Notifications</label>
                      </div>
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium">Language Preference</label>
                    <Input className="mt-2" defaultValue="English" />
                  </div>

                  <div>
                    <label className="text-sm font-medium">Emergency Contact</label>
                    <Input className="mt-2" placeholder="Phone number" />
                  </div>

                  <Button variant="outline" className="w-full">
                    <Shield className="w-4 h-4 mr-2" />
                    Update Preferences
                  </Button>
                </div>
              </HealthcareCard>
            </div>
          </TabsContent>

          {/* Health History Tab */}
          <TabsContent value="history">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <HealthcareCard title="Health History" description="Past visits and medical summaries">
                  <div className="space-y-4">
                    {historyLoading ? (
                      <div className="text-center py-8 text-gray-500">Loading health history...</div>
                    ) : healthHistory.length === 0 ? (
                      <div className="text-center py-8 text-gray-500">No health history available</div>
                    ) : (
                      healthHistory.map((record: any) => (
                        <div key={record.id} className="border rounded-lg p-4 hover:bg-gray-50">
                          <div className="flex items-center justify-between mb-3">
                            <div>
                              <h4 className="font-semibold">{record.visitType || 'General Consultation'}</h4>
                              <p className="text-sm text-gray-600">
                                {new Date(record.date).toLocaleDateString()} • Dr. {record.doctorName}
                              </p>
                            </div>
                            <Badge variant="outline">{record.status || 'Completed'}</Badge>
                          </div>
                          
                          <p className="text-sm text-gray-700 mb-3">
                            {record.summary || 'Routine checkup with vital signs monitoring and general health assessment.'}
                          </p>
                          
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4 text-xs text-gray-500">
                              <span>Hospital: {record.hospital || 'General Hospital'}</span>
                              <span>Department: {record.department || 'General Medicine'}</span>
                            </div>
                            <Button size="sm" variant="outline">
                              <FileText className="w-4 h-4 mr-1" />
                              View Details
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
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Download className="w-5 h-5" />
                      Offline Access
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <p className="text-sm text-gray-600">
                        Download your health summary for offline access
                      </p>
                      <Button className="w-full" variant="outline">
                        <Download className="w-4 h-4 mr-2" />
                        Download Summary
                      </Button>
                      <div className="text-xs text-gray-500 space-y-1">
                        <p>• Last 12 months of visits</p>
                        <p>• Current medications</p>
                        <p>• Emergency contact info</p>
                        <p>• Insurance details</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Recent Lab Results</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {labResults.slice(0, 3).map((result: any) => (
                        <div key={result.id} className="flex items-center justify-between py-2">
                          <div>
                            <p className="font-medium text-sm">{result.testName || 'Blood Test'}</p>
                            <p className="text-xs text-gray-600">{new Date(result.date).toLocaleDateString()}</p>
                          </div>
                          <Badge 
                            variant="outline"
                            className={
                              result.status === 'normal' ? 'border-green-500 text-green-700' :
                              result.status === 'abnormal' ? 'border-red-500 text-red-700' :
                              'border-yellow-500 text-yellow-700'
                            }
                          >
                            {result.status || 'Normal'}
                          </Badge>
                        </div>
                      ))}
                      <Button size="sm" variant="outline" className="w-full">
                        View All Results
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Notifications Tab */}
          <TabsContent value="notifications">
            <HealthcareCard title="Notifications" description="Alerts for results, appointments, and updates">
              <div className="space-y-4">
                {notificationsLoading ? (
                  <div className="text-center py-8 text-gray-500">Loading notifications...</div>
                ) : notifications.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">No new notifications</div>
                ) : (
                  notifications.map((notification: any) => (
                    <div key={notification.id} className="flex items-start gap-4 p-4 border rounded-lg hover:bg-gray-50">
                      <div className="w-10 h-10 rounded-full flex items-center justify-center bg-blue-100">
                        {notification.type === 'lab_result' ? (
                          <FlaskConical className="w-5 h-5 text-blue-600" />
                        ) : notification.type === 'appointment' ? (
                          <Calendar className="w-5 h-5 text-blue-600" />
                        ) : notification.type === 'prescription' ? (
                          <Pill className="w-5 h-5 text-blue-600" />
                        ) : (
                          <Bell className="w-5 h-5 text-blue-600" />
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <h4 className="font-semibold">{notification.title}</h4>
                          <span className="text-xs text-gray-500">
                            {new Date(notification.createdAt).toLocaleTimeString()}
                          </span>
                        </div>
                        <p className="text-sm text-gray-700">{notification.message}</p>
                        {notification.actionRequired && (
                          <Button size="sm" className="mt-2">
                            {notification.actionText || 'Take Action'}
                          </Button>
                        )}
                      </div>
                      {!notification.isRead && (
                        <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </HealthcareCard>
          </TabsContent>

          {/* Consent & Privacy Tab */}
          <TabsContent value="consent">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <HealthcareCard title="Consent Management" description="Manage your data sharing preferences">
                <div className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <p className="font-medium">Share data with specialists</p>
                        <p className="text-sm text-gray-600">Allow referral doctors to access your records</p>
                      </div>
                      <Checkbox defaultChecked />
                    </div>
                    
                    <div className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <p className="font-medium">Emergency access</p>
                        <p className="text-sm text-gray-600">Allow emergency rooms to access critical information</p>
                      </div>
                      <Checkbox defaultChecked />
                    </div>
                    
                    <div className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <p className="font-medium">Research participation</p>
                        <p className="text-sm text-gray-600">Anonymous data for medical research</p>
                      </div>
                      <Checkbox />
                    </div>
                    
                    <div className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <p className="font-medium">Family access</p>
                        <p className="text-sm text-gray-600">Allow designated family members to view summaries</p>
                      </div>
                      <Checkbox />
                    </div>
                  </div>
                  
                  <Button className="w-full">
                    <Shield className="w-4 h-4 mr-2" />
                    Update Consent Preferences
                  </Button>
                </div>
              </HealthcareCard>

              <HealthcareCard title="Privacy & Security" description="Your data protection settings">
                <div className="space-y-4">
                  <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                    <div className="flex items-center gap-2 mb-2">
                      <CheckCircle className="w-5 h-5 text-green-600" />
                      <span className="font-semibold text-green-800">HIPAA Compliant</span>
                    </div>
                    <p className="text-sm text-green-700">
                      Your health information is protected under HIPAA regulations and encrypted end-to-end.
                    </p>
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Data encryption</span>
                      <Badge variant="outline" className="bg-green-100 text-green-800">Active</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Audit logging</span>
                      <Badge variant="outline" className="bg-green-100 text-green-800">Enabled</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Access monitoring</span>
                      <Badge variant="outline" className="bg-green-100 text-green-800">Active</Badge>
                    </div>
                  </div>

                  <div className="pt-4 border-t">
                    <Button variant="outline" className="w-full">
                      <FileText className="w-4 h-4 mr-2" />
                      Download Privacy Report
                    </Button>
                    <p className="text-xs text-gray-500 text-center mt-2">
                      See who has accessed your data in the last 90 days
                    </p>
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