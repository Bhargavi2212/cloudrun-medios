import { useState } from 'react';
import DashboardLayout from '../components/layout/dashboard-layout';
import { usePatientStore, Patient } from '../store/patientStore';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { UserPlus, Clock, AlertTriangle, CheckCircle } from 'lucide-react';

const statusColors = {
  AWAITING_VITALS: 'bg-yellow-100 text-yellow-800',
  AWAITING_DOCTOR_ASSIGNMENT: 'bg-orange-100 text-orange-800',
  AWAITING_DOCTOR: 'bg-blue-100 text-blue-800',
  IN_CONSULTATION: 'bg-green-100 text-green-800',
  AWAITING_DISCHARGE: 'bg-purple-100 text-purple-800',
  COMPLETED: 'bg-gray-100 text-gray-800',
};

export default function ReceptionistDashboard() {
  const { patients, addPatient } = usePatientStore();
  const [checkinDialogOpen, setCheckinDialogOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  
  // New patient form state
  const [newPatient, setNewPatient] = useState({
    name: '',
    age: '',
    phone: '',
    chiefComplaint: '',
  });

  const handleCheckin = () => {
    if (newPatient.name && newPatient.age && newPatient.phone && newPatient.chiefComplaint) {
      addPatient({
        name: newPatient.name,
        age: parseInt(newPatient.age),
        phone: newPatient.phone,
        chiefComplaint: newPatient.chiefComplaint,
      });
      
      // Reset form
      setNewPatient({ name: '', age: '', phone: '', chiefComplaint: '' });
      setCheckinDialogOpen(false);
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getWaitTime = (checkinTime: Date) => {
    const now = new Date();
    const diff = Math.floor((now.getTime() - checkinTime.getTime()) / (1000 * 60));
    return `${diff} min`;
  };

  const filteredPatients = patients.filter(patient =>
    patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    patient.phone.includes(searchTerm)
  );

  return (
    <DashboardLayout>
      <div className="p-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Live Patient Queue</h1>
            <p className="text-gray-600 mt-1">Real-time patient status and management</p>
          </div>
          
          <Dialog open={checkinDialogOpen} onOpenChange={setCheckinDialogOpen}>
            <DialogTrigger asChild>
              <Button size="lg" data-testid="button-checkin-patient">
                <UserPlus className="w-5 h-5 mr-2" />
                Check-In Patient
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md">
              <DialogHeader>
                <DialogTitle>Check-In New Patient</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Patient Name</label>
                  <Input
                    value={newPatient.name}
                    onChange={(e) => setNewPatient({...newPatient, name: e.target.value})}
                    placeholder="Enter patient name"
                    data-testid="input-patient-name"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Age</label>
                  <Input
                    type="number"
                    value={newPatient.age}
                    onChange={(e) => setNewPatient({...newPatient, age: e.target.value})}
                    placeholder="Enter age"
                    data-testid="input-patient-age"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Phone Number</label>
                  <Input
                    value={newPatient.phone}
                    onChange={(e) => setNewPatient({...newPatient, phone: e.target.value})}
                    placeholder="Enter phone number"
                    data-testid="input-patient-phone"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Chief Complaint</label>
                  <Textarea
                    value={newPatient.chiefComplaint}
                    onChange={(e) => setNewPatient({...newPatient, chiefComplaint: e.target.value})}
                    placeholder="Describe the main reason for visit"
                    data-testid="input-chief-complaint"
                  />
                </div>
                <Button onClick={handleCheckin} className="w-full" data-testid="button-submit-checkin">
                  Check-In Patient
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-yellow-100 rounded-lg">
                  <Clock className="w-6 h-6 text-yellow-600" />
                </div>
                <div className="ml-4">
                  <p className="text-2xl font-bold">{patients.filter(p => p.status === 'AWAITING_VITALS').length}</p>
                  <p className="text-sm text-gray-600">Awaiting Vitals</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <AlertTriangle className="w-6 h-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-2xl font-bold">{patients.filter(p => p.status === 'AWAITING_DOCTOR').length}</p>
                  <p className="text-sm text-gray-600">Awaiting Doctor</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-green-100 rounded-lg">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-2xl font-bold">{patients.filter(p => p.status === 'IN_CONSULTATION').length}</p>
                  <p className="text-sm text-gray-600">In Consultation</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-gray-100 rounded-lg">
                  <UserPlus className="w-6 h-6 text-gray-600" />
                </div>
                <div className="ml-4">
                  <p className="text-2xl font-bold">{patients.length}</p>
                  <p className="text-sm text-gray-600">Total Patients</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Search */}
        <div className="mb-6">
          <Input
            placeholder="Search patients by name or phone..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="max-w-md"
            data-testid="input-search-patients"
          />
        </div>

        {/* Patient Queue Table */}
        <Card>
          <CardHeader>
            <CardTitle>Patient Queue</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-4">Patient</th>
                    <th className="text-left p-4">Chief Complaint</th>
                    <th className="text-left p-4">Status</th>
                    <th className="text-left p-4">Assigned Doctor</th>
                    <th className="text-left p-4">Check-in Time</th>
                    <th className="text-left p-4">Wait Time</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredPatients.map((patient) => (
                    <tr key={patient.id} className="border-b hover:bg-gray-50" data-testid={`patient-row-${patient.id}`}>
                      <td className="p-4">
                        <div>
                          <p className="font-medium">{patient.name}</p>
                          <p className="text-sm text-gray-500">Age {patient.age} • {patient.phone}</p>
                        </div>
                      </td>
                      <td className="p-4">
                        <p className="text-sm">{patient.chiefComplaint}</p>
                      </td>
                      <td className="p-4">
                        <Badge className={statusColors[patient.status]}>
                          {patient.status.replace(/_/g, ' ')}
                        </Badge>
                      </td>
                      <td className="p-4">
                        <p className="text-sm">
                          {patient.assignedDoctorName || 'Not assigned'}
                        </p>
                      </td>
                      <td className="p-4">
                        <p className="text-sm">{formatTime(patient.checkinTime)}</p>
                      </td>
                      <td className="p-4">
                        <p className="text-sm">{getWaitTime(patient.checkinTime)}</p>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {filteredPatients.length === 0 && (
              <div className="text-center py-8">
                <p className="text-gray-500">No patients found</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}