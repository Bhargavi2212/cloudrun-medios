import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { 
  TrendingUp, 
  TrendingDown, 
  Building2, 
  Users, 
  DollarSign, 
  Activity, 
  Globe, 
  Brain, 
  AlertTriangle, 
  CheckCircle,
  Plus,
  Settings,
  Download,
  BarChart3,
  MapPin,
  Shield,
  Zap,
  FileText,
  Clock,
  Target
} from "lucide-react";

export default function SuperAdminDashboard() {
  const [selectedHospital, setSelectedHospital] = useState<string | null>(null);

  // Fetch global analytics data
  const { data: globalStats, isLoading: statsLoading } = useQuery({
    queryKey: ["/api/super-admin/global-stats"],
  });

  // Fetch hospital data
  const { data: hospitals = [], isLoading: hospitalsLoading } = useQuery({
    queryKey: ["/api/super-admin/hospitals"],
  });

  // Fetch ML insights
  const { data: mlInsights = [], isLoading: mlLoading } = useQuery({
    queryKey: ["/api/super-admin/ml-insights"],
  });

  // Fetch system users
  const { data: systemUsers = [], isLoading: usersLoading } = useQuery({
    queryKey: ["/api/super-admin/users"],
  });

  // Fetch financial overview
  const { data: financialData, isLoading: financialLoading } = useQuery({
    queryKey: ["/api/super-admin/finances"],
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      {/* Header */}
      <div className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                <Globe className="w-8 h-8 text-indigo-600" />
                Super Admin Dashboard
              </h1>
              <p className="text-gray-600">Multi-Hospital System Oversight & Management</p>
            </div>
            <div className="flex items-center gap-4">
              <Dialog>
                <DialogTrigger asChild>
                  <Button>
                    <Plus className="w-4 h-4 mr-2" />
                    Add Hospital
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Add New Hospital</DialogTitle>
                    <DialogDescription>
                      Add a new hospital to the network
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 pt-4">
                    <Input placeholder="Hospital Name" />
                    <Input placeholder="Location" />
                    <Input placeholder="Contact Email" />
                    <Button className="w-full">Add Hospital</Button>
                  </div>
                </DialogContent>
              </Dialog>
              <Button variant="outline">
                <Settings className="w-4 h-4 mr-2" />
                System Settings
              </Button>
              <Button onClick={() => window.location.href = "/"} variant="ghost">
                Switch Role
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Global Analytics Hub */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-indigo-600" />
            Global Analytics Hub
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="border-blue-200 bg-blue-50">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-blue-700">Total Hospitals</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-2xl font-bold text-blue-800">{hospitals.length || 12}</p>
                    <p className="text-xs text-blue-600 flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" />
                      +2 this month
                    </p>
                  </div>
                  <Building2 className="w-8 h-8 text-blue-600" />
                </div>
              </CardContent>
            </Card>

            <Card className="border-green-200 bg-green-50">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-green-700">Total Patients Today</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-2xl font-bold text-green-800">{globalStats?.totalPatients || 1247}</p>
                    <p className="text-xs text-green-600 flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" />
                      +8.2% vs yesterday
                    </p>
                  </div>
                  <Users className="w-8 h-8 text-green-600" />
                </div>
              </CardContent>
            </Card>

            <Card className="border-purple-200 bg-purple-50">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-purple-700">System Revenue</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-2xl font-bold text-purple-800">${globalStats?.totalRevenue || '2.1M'}</p>
                    <p className="text-xs text-purple-600 flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" />
                      +12.5% this month
                    </p>
                  </div>
                  <DollarSign className="w-8 h-8 text-purple-600" />
                </div>
              </CardContent>
            </Card>

            <Card className="border-orange-200 bg-orange-50">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-orange-700">System Efficiency</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-2xl font-bold text-orange-800">{globalStats?.efficiency || 94}%</p>
                    <p className="text-xs text-orange-600 flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" />
                      +3.1% optimized
                    </p>
                  </div>
                  <Zap className="w-8 h-8 text-orange-600" />
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        <Tabs defaultValue="hospitals" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="hospitals">Hospital Management</TabsTrigger>
            <TabsTrigger value="users">System Users</TabsTrigger>
            <TabsTrigger value="finances">Revenue & Finance</TabsTrigger>
            <TabsTrigger value="ml-insights">ML Insights Center</TabsTrigger>
            <TabsTrigger value="system">System Config</TabsTrigger>
          </TabsList>

          {/* Hospital Management Panel */}
          <TabsContent value="hospitals">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Building2 className="w-5 h-5" />
                      Hospital Network Overview
                    </CardTitle>
                    <CardDescription>
                      Monitor performance across all hospitals in the network
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {hospitalsLoading ? (
                        <div className="text-center py-8 text-gray-500">Loading hospitals...</div>
                      ) : (
                        hospitals.map((hospital: any) => (
                          <div key={hospital.id} className="border rounded-lg p-4 hover:bg-gray-50">
                            <div className="flex items-center justify-between mb-3">
                              <div className="flex items-center gap-3">
                                <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center">
                                  <Building2 className="w-5 h-5 text-indigo-600" />
                                </div>
                                <div>
                                  <h4 className="font-semibold">{hospital.name || `Hospital ${hospital.id}`}</h4>
                                  <p className="text-sm text-gray-600 flex items-center gap-1">
                                    <MapPin className="w-3 h-3" />
                                    {hospital.location || 'Metro City'}
                                  </p>
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <Badge 
                                  variant="outline"
                                  className={
                                    hospital.status === 'active' ? 'border-green-500 text-green-700' :
                                    hospital.status === 'maintenance' ? 'border-yellow-500 text-yellow-700' :
                                    'border-red-500 text-red-700'
                                  }
                                >
                                  {hospital.status || 'Active'}
                                </Badge>
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-4 gap-4 text-sm">
                              <div>
                                <p className="text-gray-600">Patients Today</p>
                                <p className="font-semibold">{hospital.todayPatients || 156}</p>
                              </div>
                              <div>
                                <p className="text-gray-600">Avg Wait Time</p>
                                <p className="font-semibold">{hospital.avgWaitTime || 18} min</p>
                              </div>
                              <div>
                                <p className="text-gray-600">Bed Occupancy</p>
                                <p className="font-semibold">{hospital.bedOccupancy || 82}%</p>
                              </div>
                              <div>
                                <p className="text-gray-600">Staff Count</p>
                                <p className="font-semibold">{hospital.staffCount || 45}</p>
                              </div>
                            </div>
                            
                            <div className="mt-3 flex justify-between items-center">
                              <div className="text-xs text-gray-500">
                                Performance Score: {hospital.performanceScore || 94}/100
                              </div>
                              <Button size="sm" variant="outline" onClick={() => setSelectedHospital(hospital.id)}>
                                View Details
                              </Button>
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Network KPIs</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Average Patient Satisfaction</span>
                        <span>92%</span>
                      </div>
                      <Progress value={92} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>System Uptime</span>
                        <span>99.8%</span>
                      </div>
                      <Progress value={99.8} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Rural Coverage</span>
                        <span>78%</span>
                      </div>
                      <Progress value={78} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Emergency Response</span>
                        <span>96%</span>
                      </div>
                      <Progress value={96} className="h-2" />
                    </div>
                  </div>
                  
                  <div className="mt-6 p-3 bg-blue-50 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="w-4 h-4 text-blue-600" />
                      <span className="font-medium text-blue-800">Key Insight</span>
                    </div>
                    <p className="text-sm text-blue-700">
                      Rural Hospital 3 achieved 30% wait time reduction through AI triage optimization
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* System Users Management */}
          <TabsContent value="users">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Shield className="w-5 h-5" />
                      System User Management
                    </CardTitle>
                    <CardDescription>
                      Manage administrators and users across all hospitals
                    </CardDescription>
                  </div>
                  <Button>
                    <Plus className="w-4 h-4 mr-2" />
                    Add User
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Role</TableHead>
                      <TableHead>Hospital</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Last Active</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {usersLoading ? (
                      <TableRow>
                        <TableCell colSpan={6} className="text-center py-8 text-gray-500">
                          Loading users...
                        </TableCell>
                      </TableRow>
                    ) : (
                      systemUsers.map((user: any) => (
                        <TableRow key={user.id}>
                          <TableCell className="font-medium">{user.name || 'Dr. Smith'}</TableCell>
                          <TableCell>
                            <Badge variant="outline">{user.role || 'Admin'}</Badge>
                          </TableCell>
                          <TableCell>{user.hospital || 'General Hospital'}</TableCell>
                          <TableCell>
                            <Badge 
                              variant="outline"
                              className={
                                user.status === 'active' ? 'border-green-500 text-green-700' :
                                'border-gray-500 text-gray-700'
                              }
                            >
                              {user.status || 'Active'}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-sm text-gray-600">
                            {user.lastActive || '2 hours ago'}
                          </TableCell>
                          <TableCell>
                            <Button size="sm" variant="outline">Edit</Button>
                          </TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Revenue & Finance */}
          <TabsContent value="finances">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <DollarSign className="w-5 h-5" />
                    Revenue Overview
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 bg-green-50 rounded-lg">
                        <p className="text-sm text-green-600">Monthly Revenue</p>
                        <p className="text-2xl font-bold text-green-800">
                          ${financialData?.monthlyRevenue?.toLocaleString() || '850,000'}
                        </p>
                        <p className="text-xs text-green-600">+12.5% vs last month</p>
                      </div>
                      <div className="p-4 bg-blue-50 rounded-lg">
                        <p className="text-sm text-blue-600">Annual Revenue</p>
                        <p className="text-2xl font-bold text-blue-800">
                          ${financialData?.annualRevenue?.toLocaleString() || '9.2M'}
                        </p>
                        <p className="text-xs text-blue-600">+8.7% vs last year</p>
                      </div>
                    </div>
                    
                    <div className="space-y-3">
                      <h4 className="font-semibold">Revenue by Hospital Type</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Urban Hospitals</span>
                          <span className="font-medium">$5.2M (56%)</span>
                        </div>
                        <Progress value={56} className="h-2" />
                        
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Rural Hospitals</span>
                          <span className="font-medium">$2.8M (31%)</span>
                        </div>
                        <Progress value={31} className="h-2" />
                        
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Specialty Centers</span>
                          <span className="font-medium">$1.2M (13%)</span>
                        </div>
                        <Progress value={13} className="h-2" />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Cost Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 bg-orange-50 rounded-lg">
                        <p className="text-sm text-orange-600">Total Expenses</p>
                        <p className="text-2xl font-bold text-orange-800">
                          ${financialData?.totalExpenses?.toLocaleString() || '6.1M'}
                        </p>
                        <p className="text-xs text-orange-600">66% of revenue</p>
                      </div>
                      <div className="p-4 bg-purple-50 rounded-lg">
                        <p className="text-sm text-purple-600">Net Profit</p>
                        <p className="text-2xl font-bold text-purple-800">
                          ${financialData?.netProfit?.toLocaleString() || '3.1M'}
                        </p>
                        <p className="text-xs text-purple-600">34% margin</p>
                      </div>
                    </div>

                    <div className="space-y-3">
                      <h4 className="font-semibold">Expense Breakdown</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span>Staff Salaries</span>
                          <span>$3.2M (52%)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Equipment & Supplies</span>
                          <span>$1.8M (29%)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Infrastructure</span>
                          <span>$0.7M (11%)</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Technology & AI</span>
                          <span>$0.4M (8%)</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* ML Insights Center */}
          <TabsContent value="ml-insights">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    AI Predictions & Alerts
                  </CardTitle>
                  <CardDescription>
                    Real-time insights from the ML pipeline
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {mlLoading ? (
                      <div className="text-center py-8 text-gray-500">Loading ML insights...</div>
                    ) : (
                      mlInsights.map((insight: any) => (
                        <div key={insight.id} className="border rounded-lg p-4">
                          <div className="flex items-start gap-3">
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                              insight.type === 'alert' ? 'bg-red-100' :
                              insight.type === 'prediction' ? 'bg-blue-100' :
                              'bg-green-100'
                            }`}>
                              {insight.type === 'alert' ? 
                                <AlertTriangle className="w-4 h-4 text-red-600" /> :
                                insight.type === 'prediction' ?
                                <Brain className="w-4 h-4 text-blue-600" /> :
                                <CheckCircle className="w-4 h-4 text-green-600" />
                              }
                            </div>
                            <div className="flex-1">
                              <h4 className="font-semibold">{insight.title || 'Outbreak Detection Alert'}</h4>
                              <p className="text-sm text-gray-600 mt-1">
                                {insight.description || 'Dengue fever cases trending up 23% in Rural Hospital 2. Recommend increased testing and prevention measures.'}
                              </p>
                              <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                                <span>Confidence: {insight.confidence || 94}%</span>
                                <span>Impact: {insight.impact || 'High'}</span>
                                <span>{insight.timeAgo || '2 hours ago'}</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                    
                    {/* Sample insights when no data */}
                    {mlInsights.length === 0 && !mlLoading && (
                      <>
                        <div className="border rounded-lg p-4">
                          <div className="flex items-start gap-3">
                            <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                              <AlertTriangle className="w-4 h-4 text-red-600" />
                            </div>
                            <div className="flex-1">
                              <h4 className="font-semibold">Outbreak Detection Alert</h4>
                              <p className="text-sm text-gray-600 mt-1">
                                Dengue fever cases trending up 23% in Rural Hospital 2. Recommend increased testing and prevention measures.
                              </p>
                              <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                                <span>Confidence: 94%</span>
                                <span>Impact: High</span>
                                <span>2 hours ago</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="border rounded-lg p-4">
                          <div className="flex items-start gap-3">
                            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                              <Brain className="w-4 h-4 text-blue-600" />
                            </div>
                            <div className="flex-1">
                              <h4 className="font-semibold">Resource Optimization</h4>
                              <p className="text-sm text-gray-600 mt-1">
                                AI recommends reallocating 3 nurses from Metro Hospital to Rural Hospital 1 for optimal coverage.
                              </p>
                              <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                                <span>Confidence: 89%</span>
                                <span>Impact: Medium</span>
                                <span>4 hours ago</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>System Analytics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold mb-3">Real-time Metrics</h4>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm">AI Processing Load</span>
                          <span className="text-sm font-medium">73%</span>
                        </div>
                        <Progress value={73} className="h-2" />
                        
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Data Sync Status</span>
                          <span className="text-sm font-medium">98%</span>
                        </div>
                        <Progress value={98} className="h-2" />
                        
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Emergency Alerts</span>
                          <span className="text-sm font-medium">3 Active</span>
                        </div>
                      </div>
                    </div>

                    <div className="pt-4 border-t">
                      <h4 className="font-semibold mb-3">Recent Activities</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                          <span>ML model updated - 3% accuracy improvement</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                          <span>New hospital onboarded successfully</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
                          <span>System maintenance scheduled</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* System Configuration */}
          <TabsContent value="system">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings className="w-5 h-5" />
                    System Configuration
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <div>
                      <h4 className="font-semibold mb-3">AI & ML Settings</h4>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Federated Learning</span>
                          <Badge className="bg-green-100 text-green-800">Enabled</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Real-time Analytics</span>
                          <Badge className="bg-green-100 text-green-800">Active</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Auto-scaling</span>
                          <Badge className="bg-blue-100 text-blue-800">Configured</Badge>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-3">Security & Compliance</h4>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm">HIPAA Compliance</span>
                          <Badge className="bg-green-100 text-green-800">Verified</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Data Encryption</span>
                          <Badge className="bg-green-100 text-green-800">AES-256</Badge>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm">Audit Logging</span>
                          <Badge className="bg-green-100 text-green-800">Active</Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>System Health</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-semibold mb-3">Performance Metrics</h4>
                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span>CPU Usage</span>
                            <span>67%</span>
                          </div>
                          <Progress value={67} className="h-2" />
                        </div>
                        
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span>Memory Usage</span>
                            <span>54%</span>
                          </div>
                          <Progress value={54} className="h-2" />
                        </div>
                        
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span>Network Load</span>
                            <span>82%</span>
                          </div>
                          <Progress value={82} className="h-2" />
                        </div>
                      </div>
                    </div>

                    <div className="pt-4 border-t">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-semibold">System Reports</h4>
                        <Button size="sm" variant="outline">
                          <Download className="w-4 h-4 mr-2" />
                          Export
                        </Button>
                      </div>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span>Daily Performance Report</span>
                          <Button size="sm" variant="ghost">Download</Button>
                        </div>
                        <div className="flex justify-between">
                          <span>Security Audit Log</span>
                          <Button size="sm" variant="ghost">Download</Button>
                        </div>
                        <div className="flex justify-between">
                          <span>Financial Summary</span>
                          <Button size="sm" variant="ghost">Download</Button>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}