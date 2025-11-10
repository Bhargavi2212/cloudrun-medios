import { useEffect } from "react";
import { useAuth } from "@/hooks/useAuth";
import { useToast } from "@/hooks/use-toast";
import { useQuery } from "@tanstack/react-query";
import Sidebar from "@/components/layout/sidebar";
import Header from "@/components/layout/header";
import StatsCard from "@/components/dashboard/stats-card";

export default function PatientDashboard() {
  const { toast } = useToast();
  const { isAuthenticated, isLoading, user } = useAuth();

  // Redirect if not authenticated or not patient
  useEffect(() => {
    if (!isLoading && (!isAuthenticated || user?.role !== 'patient')) {
      toast({
        title: "Unauthorized",
        description: "You need patient access to view this page.",
        variant: "destructive",
      });
      setTimeout(() => {
        window.location.href = "/api/login";
      }, 500);
      return;
    }
  }, [isAuthenticated, isLoading, user, toast]);

  const { data: patientQueue, isLoading: queueLoading } = useQuery({
    queryKey: ["/api/queue", { patientId: user?.roleData?.id }],
    enabled: isAuthenticated && user?.role === 'patient' && user?.roleData?.id,
  });

  const { data: appointments, isLoading: appointmentsLoading } = useQuery({
    queryKey: ["/api/appointments", { patientId: user?.roleData?.id }],
    enabled: isAuthenticated && user?.role === 'patient' && user?.roleData?.id,
  });

  const { data: vitals, isLoading: vitalsLoading } = useQuery({
    queryKey: ["/api/patients", user?.roleData?.id, "vitals"],
    enabled: isAuthenticated && user?.role === 'patient' && user?.roleData?.id,
  });

  if (isLoading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-medical-bg">
        <div className="text-center">
          <div className="w-16 h-16 bg-medical-blue rounded-xl flex items-center justify-center mx-auto mb-4 animate-pulse">
            <i className="fas fa-user text-white text-2xl"></i>
          </div>
          <p className="text-medical-gray">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  const currentVisit = patientQueue?.[0];
  const isInQueue = currentVisit && currentVisit.status === 'waiting';

  return (
    <div className="min-h-screen bg-medical-bg">
      <Header 
        title={`Welcome, ${user.firstName} ${user.lastName}`}
        subtitle={`Patient Portal • ID: ${user.roleData?.patientId || 'N/A'}`}
        user={user}
      />
      
      <div className="flex">
        <Sidebar userRole="patient" />
        
        <main className="flex-1 p-6">
          {/* Current Status & QR Check-in */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            {/* Current Visit Status */}
            <div className="lg:col-span-2 bg-white rounded-xl shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">Current Visit Status</h3>
              
              {isInQueue ? (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center">
                        <i className="fas fa-clock"></i>
                      </div>
                      <div>
                        <p className="text-lg font-semibold text-gray-900">Currently in Queue</p>
                        <p className="text-blue-600 font-medium">{currentVisit.category} - {currentVisit.reasonForVisit}</p>
                      </div>
                    </div>
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                      Position #{currentVisit.queuePosition}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <p className="text-2xl font-bold text-gray-900">
                        {currentVisit.estimatedWaitTime || 'N/A'} min
                      </p>
                      <p className="text-sm text-medical-gray">Estimated Wait</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-gray-900">
                        {currentVisit.checkinTime ? new Date(currentVisit.checkinTime).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : 'N/A'}
                      </p>
                      <p className="text-sm text-medical-gray">Check-in Time</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-gray-900">
                        {currentVisit.roomAssigned || 'TBD'}
                      </p>
                      <p className="text-sm text-medical-gray">Assigned Room</p>
                    </div>
                  </div>

                  {/* Progress Steps */}
                  <div className="mt-6">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex flex-col items-center">
                        <div className="w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center mb-2">
                          <i className="fas fa-check text-xs"></i>
                        </div>
                        <span className="text-green-600 font-medium">Checked In</span>
                      </div>
                      <div className="flex-1 h-1 bg-green-500 mx-2"></div>
                      <div className="flex flex-col items-center">
                        <div className="w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center mb-2">
                          <i className="fas fa-check text-xs"></i>
                        </div>
                        <span className="text-green-600 font-medium">Vitals Taken</span>
                      </div>
                      <div className="flex-1 h-1 bg-blue-500 mx-2"></div>
                      <div className="flex flex-col items-center">
                        <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center mb-2 animate-pulse">
                          <span className="text-xs font-bold">{currentVisit.queuePosition}</span>
                        </div>
                        <span className="text-blue-600 font-medium">In Queue</span>
                      </div>
                      <div className="flex-1 h-1 bg-gray-300 mx-2"></div>
                      <div className="flex flex-col items-center">
                        <div className="w-8 h-8 bg-gray-300 text-gray-600 rounded-full flex items-center justify-center mb-2">
                          <i className="fas fa-user-md text-xs"></i>
                        </div>
                        <span className="text-gray-600">Consultation</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 mb-6">
                  <div className="text-center">
                    <div className="w-16 h-16 bg-gray-300 rounded-full flex items-center justify-center mx-auto mb-4">
                      <i className="fas fa-clipboard-check text-gray-600 text-2xl"></i>
                    </div>
                    <h4 className="text-lg font-semibold text-gray-900 mb-2">No Active Visit</h4>
                    <p className="text-gray-600 mb-4">You are not currently checked in for a visit.</p>
                    <button className="bg-medical-blue text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors">
                      Check In Now
                    </button>
                  </div>
                </div>
              )}

              {/* Quick Actions */}
              <div className="grid grid-cols-3 gap-4">
                <button className="flex flex-col items-center p-4 bg-green-50 text-green-700 rounded-lg hover:bg-green-100 transition-colors">
                  <i className="fas fa-notes-medical text-xl mb-2"></i>
                  <span className="text-sm font-medium">Update Symptoms</span>
                </button>
                
                <button className="flex flex-col items-center p-4 bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 transition-colors">
                  <i className="fas fa-list text-xl mb-2"></i>
                  <span className="text-sm font-medium">View Queue</span>
                </button>
                
                <button className="flex flex-col items-center p-4 bg-yellow-50 text-yellow-700 rounded-lg hover:bg-yellow-100 transition-colors">
                  <i className="fas fa-phone text-xl mb-2"></i>
                  <span className="text-sm font-medium">Contact Reception</span>
                </button>
              </div>
            </div>

            {/* QR Check-in Card */}
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Check-in</h3>
              
              <div className="text-center mb-6">
                <div className="w-32 h-32 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <i className="fas fa-qrcode text-4xl text-gray-400"></i>
                </div>
                <p className="text-sm text-gray-600 mb-4">
                  Show this QR code at reception for quick check-in
                </p>
                <button className="w-full bg-medical-blue text-white py-2 px-4 rounded-lg font-medium hover:bg-blue-700 transition-colors">
                  Generate QR Code
                </button>
              </div>

              <div className="border-t pt-4">
                <h4 className="font-medium text-gray-900 mb-2">Patient Information</h4>
                <div className="space-y-1 text-sm">
                  <p><span className="text-gray-600">ID:</span> {user.roleData?.patientId}</p>
                  <p><span className="text-gray-600">DOB:</span> {user.roleData?.dateOfBirth ? new Date(user.roleData.dateOfBirth).toLocaleDateString() : 'Not set'}</p>
                  <p><span className="text-gray-600">Blood Type:</span> {user.roleData?.bloodType || 'Not set'}</p>
                  <p><span className="text-gray-600">Emergency Contact:</span> {user.roleData?.emergencyContact || 'Not set'}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Health Overview */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* Recent Vitals */}
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">Recent Vitals</h3>
              
              {vitalsLoading ? (
                <div className="animate-pulse space-y-4">
                  {[1, 2].map((i) => (
                    <div key={i} className="grid grid-cols-2 gap-4">
                      <div className="h-16 bg-gray-200 rounded-lg"></div>
                      <div className="h-16 bg-gray-200 rounded-lg"></div>
                    </div>
                  ))}
                </div>
              ) : vitals && vitals.length > 0 ? (
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 p-3 rounded-lg">
                    <p className="text-xs text-blue-600 font-medium">BLOOD PRESSURE</p>
                    <p className="text-lg font-bold text-blue-700">
                      {vitals[0].bloodPressureSystolic}/{vitals[0].bloodPressureDiastolic}
                    </p>
                    <p className="text-xs text-blue-600">
                      {new Date(vitals[0].recordedAt).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="bg-green-50 p-3 rounded-lg">
                    <p className="text-xs text-green-600 font-medium">HEART RATE</p>
                    <p className="text-lg font-bold text-green-700">{vitals[0].heartRate} bpm</p>
                    <p className="text-xs text-green-600">
                      {new Date(vitals[0].recordedAt).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="bg-red-50 p-3 rounded-lg">
                    <p className="text-xs text-red-600 font-medium">TEMPERATURE</p>
                    <p className="text-lg font-bold text-red-700">
                      {vitals[0].temperature ? (vitals[0].temperature / 10).toFixed(1) : 'N/A'}°F
                    </p>
                    <p className="text-xs text-red-600">
                      {new Date(vitals[0].recordedAt).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="bg-purple-50 p-3 rounded-lg">
                    <p className="text-xs text-purple-600 font-medium">OXYGEN SAT</p>
                    <p className="text-lg font-bold text-purple-700">{vitals[0].oxygenSaturation}%</p>
                    <p className="text-xs text-purple-600">
                      {new Date(vitals[0].recordedAt).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <i className="fas fa-heartbeat text-4xl text-gray-300 mb-4"></i>
                  <p className="text-gray-500">No recent vitals recorded</p>
                </div>
              )}
            </div>

            {/* Upcoming Appointments */}
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">Upcoming Appointments</h3>
              
              {appointmentsLoading ? (
                <div className="animate-pulse space-y-4">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="p-4 bg-gray-50 rounded-lg">
                      <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                      <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                    </div>
                  ))}
                </div>
              ) : appointments && appointments.length > 0 ? (
                <div className="space-y-4">
                  {appointments.slice(0, 3).map((appointment: any) => (
                    <div key={appointment.id} className="p-4 border border-gray-200 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-gray-900">{appointment.appointmentType}</h4>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          appointment.status === 'scheduled' ? 'bg-green-100 text-green-800' :
                          appointment.status === 'cancelled' ? 'bg-red-100 text-red-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {appointment.status}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600">
                        {new Date(appointment.scheduledTime).toLocaleDateString()} at{' '}
                        {new Date(appointment.scheduledTime).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                      </p>
                      {appointment.roomNumber && (
                        <p className="text-sm text-gray-600">Room: {appointment.roomNumber}</p>
                      )}
                    </div>
                  ))}
                  
                  {appointments.length > 3 && (
                    <button className="w-full text-medical-blue font-medium text-sm hover:bg-blue-50 py-2 rounded-lg transition-colors">
                      View All Appointments ({appointments.length} total)
                    </button>
                  )}
                </div>
              ) : (
                <div className="text-center py-8">
                  <i className="fas fa-calendar-alt text-4xl text-gray-300 mb-4"></i>
                  <p className="text-gray-500">No upcoming appointments</p>
                  <button className="mt-3 bg-medical-blue text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors">
                    Schedule Appointment
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Health Summary */}
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-6">Health Summary</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Allergies</h4>
                <div className="text-sm">
                  {user.roleData?.allergies ? (
                    <span className="px-2 py-1 bg-red-100 text-red-800 rounded-full text-xs">
                      {user.roleData.allergies}
                    </span>
                  ) : (
                    <p className="text-gray-500">No known allergies</p>
                  )}
                </div>
              </div>
              
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Emergency Contact</h4>
                <div className="text-sm text-gray-600">
                  <p>{user.roleData?.emergencyContact || 'Not set'}</p>
                  <p>{user.roleData?.emergencyPhone || 'No phone number'}</p>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Medical Info</h4>
                <div className="text-sm text-gray-600">
                  <p>Blood Type: {user.roleData?.bloodType || 'Not set'}</p>
                  <p>Gender: {user.roleData?.gender || 'Not specified'}</p>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
