import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/hooks/useAuth";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { HealthcareCard } from "@/components/ui/healthcare-card";
import { 
  Building, 
  Users, 
  DollarSign, 
  TrendingUp, 
  Activity, 
  Calendar, 
  Bed,
  UserCheck,
  AlertTriangle,
  BarChart3,
  Settings,
  PieChart,
  Clock,
  Hospital,
  Stethoscope,
  BrainCircuit,
  Shield
} from "lucide-react";

export default function AdminDashboard() {
  const { user } = useAuth();

  // Fetch hospital analytics
  const { data: hospitalStats = {}, isLoading: statsLoading } = useQuery({
    queryKey: ["/api/admin/hospital-stats"],
  });

  // Fetch staff data
  const { data: staffData = [], isLoading: staffLoading } = useQuery({
    queryKey: ["/api/admin/staff"],
  });

  // Fetch department metrics
  const { data: departments = [], isLoading: deptLoading } = useQuery({
    queryKey: ["/api/admin/departments"],
  });

  // Fetch financial data
  const { data: financialData = {}, isLoading: financeLoading } = useQuery({
    queryKey: ["/api/admin/finances"],
  });

  // Fetch bed occupancy
  const { data: bedOccupancy = {}, isLoading: bedLoading } = useQuery({
    queryKey: ["/api/admin/bed-occupancy"],
  });

  const totalStaff = staffData.length;
  const activeStaff = staffData.filter((s: any) => s.isActive).length;
  const doctorsCount = staffData.filter((s: any) => s.role === 'doctor').length;
  const nursesCount = staffData.filter((s: any) => s.role === 'nurse').length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50">
      {/* Header */}
      <div className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Hospital Administrator
              </h1>
              <p className="text-gray-600">Operations Management Dashboard</p>
            </div>
            <div className="flex items-center gap-4">
              <Badge variant="outline" className="bg-purple-100 text-purple-800">
                <Shield className="w-4 h-4 mr-1" />
                Admin Access
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
                  <p className="text-sm font-medium text-gray-600">Total Staff</p>
                  <p className="text-3xl font-bold text-gray-900">{totalStaff}</p>
                  <p className="text-sm text-green-600">
                    <TrendingUp className="w-4 h-4 inline mr-1" />
                    {activeStaff} active
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
                  <p className="text-sm font-medium text-gray-600">Bed Occupancy</p>
                  <p className="text-3xl font-bold text-gray-900">{bedOccupancy.occupancyRate || 0}%</p>
                  <p className="text-sm text-gray-600">
                    {bedOccupancy.occupiedBeds || 0}/{bedOccupancy.totalBeds || 0} beds
                  </p>
                </div>
                <Bed className="w-8 h-8 text-green-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Monthly Revenue</p>
                  <p className="text-3xl font-bold text-gray-900">
                    ${financialData.monthlyRevenue?.toLocaleString() || '0'}
                  </p>
                  <p className="text-sm text-green-600">
                    <TrendingUp className="w-4 h-4 inline mr-1" />
                    +{financialData.growthRate || 0}%
                  </p>
                </div>
                <DollarSign className="w-8 h-8 text-emerald-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Patient Satisfaction</p>
                  <p className="text-3xl font-bold text-gray-900">{hospitalStats.satisfaction || 0}%</p>
                  <p className="text-sm text-blue-600">
                    <Activity className="w-4 h-4 inline mr-1" />
                    {hospitalStats.feedbackCount || 0} reviews
                  </p>
                </div>
                <BarChart3 className="w-8 h-8 text-purple-600" />
              </div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="operations" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="operations">Operations</TabsTrigger>
            <TabsTrigger value="staff">Staff Management</TabsTrigger>
            <TabsTrigger value="departments">Departments</TabsTrigger>
            <TabsTrigger value="financial">Financial</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
          </TabsList>

          {/* Operations Tab */}
          <TabsContent value="operations">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Hospital Operations Center */}
              <HealthcareCard title="Hospital Operations Center" description="Real-time hospital status">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <Hospital className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                      <p className="text-2xl font-bold text-blue-600">{departments.length}</p>
                      <p className="text-sm text-gray-600">Active Departments</p>
                    </div>
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <Stethoscope className="w-8 h-8 text-green-600 mx-auto mb-2" />
                      <p className="text-2xl font-bold text-green-600">{hospitalStats.todayConsultations || 0}</p>
                      <p className="text-sm text-gray-600">Today's Consultations</p>
                    </div>
                  </div>
                  
                  <div className="border rounded-lg p-4">
                    <h4 className="font-semibold mb-3">Department Status</h4>
                    <div className="space-y-2">
                      {departments.slice(0, 5).map((dept: any) => (
                        <div key={dept.id} className="flex items-center justify-between py-2">
                          <span className="text-sm font-medium">{dept.name}</span>
                          <Badge 
                            variant="outline"
                            className={dept.status === 'operational' ? 'border-green-500 text-green-700' : 'border-yellow-500 text-yellow-700'}
                          >
                            {dept.status || 'Operational'}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </HealthcareCard>

              {/* Patient Flow Management */}
              <HealthcareCard title="Patient Flow Management" description="Admission and discharge tracking">
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-3 text-center">
                    <div className="p-3 bg-blue-50 rounded-lg">
                      <p className="text-lg font-bold text-blue-600">{hospitalStats.admissions || 0}</p>
                      <p className="text-xs text-gray-600">Today's Admissions</p>
                    </div>
                    <div className="p-3 bg-green-50 rounded-lg">
                      <p className="text-lg font-bold text-green-600">{hospitalStats.discharges || 0}</p>
                      <p className="text-xs text-gray-600">Discharges</p>
                    </div>
                    <div className="p-3 bg-orange-50 rounded-lg">
                      <p className="text-lg font-bold text-orange-600">{hospitalStats.pendingDischarges || 0}</p>
                      <p className="text-xs text-gray-600">Pending</p>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h4 className="font-semibold mb-3">Bed Availability by Type</h4>
                    <div className="space-y-2">
                      {[
                        { type: 'ICU', available: 2, total: 10 },
                        { type: 'General', available: 15, total: 50 },
                        { type: 'Private', available: 5, total: 20 },
                        { type: 'Emergency', available: 3, total: 8 }
                      ].map((room) => (
                        <div key={room.type} className="flex items-center justify-between">
                          <span className="text-sm">{room.type}</span>
                          <div className="flex items-center gap-2">
                            <div className="w-20 bg-gray-200 rounded-full h-2">
                              <div 
                                className="bg-green-500 h-2 rounded-full"
                                style={{ width: `${(room.available / room.total) * 100}%` }}
                              ></div>
                            </div>
                            <span className="text-xs text-gray-600">{room.available}/{room.total}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </HealthcareCard>
            </div>
          </TabsContent>

          {/* Staff Management Tab */}
          <TabsContent value="staff">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <HealthcareCard title="Staff Overview" description="Employee management and scheduling">
                  <div className="space-y-4">
                    <div className="grid grid-cols-3 gap-4">
                      <div className="text-center p-4 bg-blue-50 rounded-lg">
                        <Users className="w-6 h-6 text-blue-600 mx-auto mb-2" />
                        <p className="text-xl font-bold text-blue-600">{doctorsCount}</p>
                        <p className="text-sm text-gray-600">Doctors</p>
                      </div>
                      <div className="text-center p-4 bg-green-50 rounded-lg">
                        <UserCheck className="w-6 h-6 text-green-600 mx-auto mb-2" />
                        <p className="text-xl font-bold text-green-600">{nursesCount}</p>
                        <p className="text-sm text-gray-600">Nurses</p>
                      </div>
                      <div className="text-center p-4 bg-purple-50 rounded-lg">
                        <Settings className="w-6 h-6 text-purple-600 mx-auto mb-2" />
                        <p className="text-xl font-bold text-purple-600">{staffData.filter((s: any) => s.role === 'admin').length}</p>
                        <p className="text-sm text-gray-600">Admin Staff</p>
                      </div>
                    </div>

                    <div className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-semibold">Recent Staff Activities</h4>
                        <Button size="sm" variant="outline">Manage Staff</Button>
                      </div>
                      <div className="space-y-3">
                        {staffData.slice(0, 5).map((staff: any) => (
                          <div key={staff.id} className="flex items-center justify-between py-2">
                            <div className="flex items-center gap-3">
                              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                                <Users className="w-4 h-4 text-blue-600" />
                              </div>
                              <div>
                                <p className="font-medium">{staff.firstName} {staff.lastName}</p>
                                <p className="text-sm text-gray-600">{staff.role?.toUpperCase()} • {staff.department}</p>
                              </div>
                            </div>
                            <Badge 
                              variant="outline"
                              className={staff.isActive ? 'border-green-500 text-green-700' : 'border-gray-500 text-gray-700'}
                            >
                              {staff.isActive ? 'Active' : 'Inactive'}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </HealthcareCard>
              </div>

              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Staff Actions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <Button className="w-full justify-start" variant="outline">
                        <Users className="w-4 h-4 mr-2" />
                        Add New Staff
                      </Button>
                      <Button className="w-full justify-start" variant="outline">
                        <Calendar className="w-4 h-4 mr-2" />
                        Schedule Management
                      </Button>
                      <Button className="w-full justify-start" variant="outline">
                        <DollarSign className="w-4 h-4 mr-2" />
                        Payroll Overview
                      </Button>
                      <Button className="w-full justify-start" variant="outline">
                        <BarChart3 className="w-4 h-4 mr-2" />
                        Performance Reports
                      </Button>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Shift Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {[
                        { shift: 'Morning', count: 12, color: 'bg-blue-500' },
                        { shift: 'Evening', count: 10, color: 'bg-green-500' },
                        { shift: 'Night', count: 8, color: 'bg-purple-500' }
                      ].map((shift) => (
                        <div key={shift.shift} className="flex items-center justify-between">
                          <span className="text-sm font-medium">{shift.shift}</span>
                          <div className="flex items-center gap-2">
                            <div className="flex items-center gap-1">
                              <div className={`w-3 h-3 rounded-full ${shift.color}`}></div>
                              <span className="text-sm">{shift.count}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Departments Tab */}
          <TabsContent value="departments">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {departments.map((dept: any) => (
                <HealthcareCard 
                  key={dept.id} 
                  title={dept.name} 
                  description={dept.description}
                  className="hover:shadow-lg transition-shadow"
                >
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-3 bg-blue-50 rounded-lg">
                        <p className="text-lg font-bold text-blue-600">{dept.staffCount || 0}</p>
                        <p className="text-xs text-gray-600">Staff Members</p>
                      </div>
                      <div className="text-center p-3 bg-green-50 rounded-lg">
                        <p className="text-lg font-bold text-green-600">{dept.todayPatients || 0}</p>
                        <p className="text-xs text-gray-600">Today's Patients</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between pt-2">
                      <Badge variant="outline" className="text-xs">
                        Head: Dr. {dept.headDoctorName || 'TBD'}
                      </Badge>
                      <Button size="sm" variant="outline">
                        View Details
                      </Button>
                    </div>
                  </div>
                </HealthcareCard>
              ))}
            </div>
          </TabsContent>

          {/* Financial Tab */}
          <TabsContent value="financial">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <HealthcareCard title="Financial Dashboard" description="Revenue and expense tracking">
                  <div className="space-y-6">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 bg-green-50 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <TrendingUp className="w-5 h-5 text-green-600" />
                          <span className="font-semibold text-green-800">Revenue</span>
                        </div>
                        <p className="text-2xl font-bold text-green-600">
                          ${financialData.totalRevenue?.toLocaleString() || '0'}
                        </p>
                        <p className="text-sm text-green-600">This Month</p>
                      </div>
                      <div className="p-4 bg-red-50 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <TrendingUp className="w-5 h-5 text-red-600 rotate-180" />
                          <span className="font-semibold text-red-800">Expenses</span>
                        </div>
                        <p className="text-2xl font-bold text-red-600">
                          ${financialData.totalExpenses?.toLocaleString() || '0'}
                        </p>
                        <p className="text-sm text-red-600">This Month</p>
                      </div>
                    </div>

                    <div className="border rounded-lg p-4">
                      <h4 className="font-semibold mb-3">Department Revenue</h4>
                      <div className="space-y-3">
                        {departments.slice(0, 5).map((dept: any) => (
                          <div key={dept.id} className="flex items-center justify-between">
                            <span className="text-sm font-medium">{dept.name}</span>
                            <span className="text-sm font-bold text-green-600">
                              ${dept.revenue?.toLocaleString() || '0'}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </HealthcareCard>
              </div>

              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Financial Actions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <Button className="w-full justify-start" variant="outline">
                        <DollarSign className="w-4 h-4 mr-2" />
                        Generate Reports
                      </Button>
                      <Button className="w-full justify-start" variant="outline">
                        <PieChart className="w-4 h-4 mr-2" />
                        Budget Planning
                      </Button>
                      <Button className="w-full justify-start" variant="outline">
                        <BarChart3 className="w-4 h-4 mr-2" />
                        Cost Analysis
                      </Button>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Key Metrics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm">Profit Margin</span>
                          <span className="text-sm font-semibold">{financialData.profitMargin || 0}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-green-500 h-2 rounded-full"
                            style={{ width: `${financialData.profitMargin || 0}%` }}
                          ></div>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm">Collection Rate</span>
                          <span className="text-sm font-semibold">{financialData.collectionRate || 0}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-500 h-2 rounded-full"
                            style={{ width: `${financialData.collectionRate || 0}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <HealthcareCard title="ML Insights Center" description="AI-powered analytics and predictions">
                <div className="space-y-4">
                  <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border">
                    <div className="flex items-center gap-2 mb-3">
                      <BrainCircuit className="w-6 h-6 text-purple-600" />
                      <span className="font-semibold text-purple-800">AI Recommendations</span>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm">• Optimize ICU bed allocation for 15% efficiency gain</p>
                      <p className="text-sm">• Schedule additional nurses for expected patient surge</p>
                      <p className="text-sm">• Consider emergency equipment maintenance this week</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-yellow-50 rounded-lg text-center">
                      <AlertTriangle className="w-6 h-6 text-yellow-600 mx-auto mb-1" />
                      <p className="text-lg font-bold text-yellow-600">3</p>
                      <p className="text-xs text-gray-600">Risk Alerts</p>
                    </div>
                    <div className="p-3 bg-green-50 rounded-lg text-center">
                      <TrendingUp className="w-6 h-6 text-green-600 mx-auto mb-1" />
                      <p className="text-lg font-bold text-green-600">98%</p>
                      <p className="text-xs text-gray-600">Prediction Accuracy</p>
                    </div>
                  </div>
                </div>
              </HealthcareCard>

              <HealthcareCard title="System Performance" description="Infrastructure and operational metrics">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-blue-50 rounded-lg text-center">
                      <Activity className="w-6 h-6 text-blue-600 mx-auto mb-1" />
                      <p className="text-lg font-bold text-blue-600">99.8%</p>
                      <p className="text-xs text-gray-600">System Uptime</p>
                    </div>
                    <div className="p-3 bg-green-50 rounded-lg text-center">
                      <Clock className="w-6 h-6 text-green-600 mx-auto mb-1" />
                      <p className="text-lg font-bold text-green-600">2.1s</p>
                      <p className="text-xs text-gray-600">Avg Response Time</p>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4">
                    <h4 className="font-semibold mb-3">System Health</h4>
                    <div className="space-y-3">
                      {[
                        { component: 'Database', status: 'Healthy', color: 'text-green-600' },
                        { component: 'API Services', status: 'Healthy', color: 'text-green-600' },
                        { component: 'Queue System', status: 'Warning', color: 'text-yellow-600' },
                        { component: 'Notifications', status: 'Healthy', color: 'text-green-600' }
                      ].map((item) => (
                        <div key={item.component} className="flex items-center justify-between">
                          <span className="text-sm font-medium">{item.component}</span>
                          <span className={`text-sm font-semibold ${item.color}`}>
                            {item.status}
                          </span>
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