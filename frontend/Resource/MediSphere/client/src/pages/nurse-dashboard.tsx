import { useState } from 'react';
import DashboardLayout from '../components/layout/dashboard-layout';
import { usePatientStore } from '../store/patientStore';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Activity, Heart, Thermometer, Weight, Droplet } from 'lucide-react';

export default function NurseDashboard() {
  const { patients, updatePatientVitals } = usePatientStore();
  const [vitalsDialogOpen, setVitalsDialogOpen] = useState(false);
  const [selectedPatient, setSelectedPatient] = useState<any>(null);
  
  const [vitals, setVitals] = useState({
    heartRate: '',
    bloodPressure: '',
    temperature: '',
    weight: '',
    oxygenSaturation: '',
  });

  const triageQueue = patients.filter(patient => patient.status === 'AWAITING_VITALS');

  const handleTakeVitals = (patient: any) => {
    setSelectedPatient(patient);
    setVitalsDialogOpen(true);
    // Clear previous values
    setVitals({
      heartRate: '',
      bloodPressure: '',
      temperature: '',
      weight: '',
      oxygenSaturation: '',
    });
  };

  const handleSubmitVitals = () => {
    if (selectedPatient && vitals.heartRate && vitals.bloodPressure && vitals.temperature && vitals.weight && vitals.oxygenSaturation) {
      updatePatientVitals(selectedPatient.id, {
        heartRate: parseInt(vitals.heartRate),
        bloodPressure: vitals.bloodPressure,
        temperature: parseFloat(vitals.temperature),
        weight: parseFloat(vitals.weight),
        oxygenSaturation: parseInt(vitals.oxygenSaturation),
      });
      
      setVitalsDialogOpen(false);
      setSelectedPatient(null);
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

  return (
    <DashboardLayout>
      <div className="p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Triage Queue</h1>
          <p className="text-gray-600 mt-1">Patients awaiting vital signs assessment</p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <Activity className="w-6 h-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-2xl font-bold">{triageQueue.length}</p>
                  <p className="text-sm text-gray-600">Awaiting Vitals</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-green-100 rounded-lg">
                  <Heart className="w-6 h-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-2xl font-bold">{patients.filter(p => p.vitals).length}</p>
                  <p className="text-sm text-gray-600">Vitals Completed</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center">
                <div className="p-2 bg-yellow-100 rounded-lg">
                  <Thermometer className="w-6 h-6 text-yellow-600" />
                </div>
                <div className="ml-4">
                  <p className="text-2xl font-bold">{triageQueue.length > 0 ? getWaitTime(triageQueue[0].checkinTime) : '0 min'}</p>
                  <p className="text-sm text-gray-600">Longest Wait</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Triage Queue */}
        <Card>
          <CardHeader>
            <CardTitle>Patients Awaiting Vitals</CardTitle>
          </CardHeader>
          <CardContent>
            {triageQueue.length === 0 ? (
              <div className="text-center py-8">
                <Activity className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">No patients awaiting vitals</p>
              </div>
            ) : (
              <div className="space-y-4">
                {triageQueue
                  .sort((a, b) => a.checkinTime.getTime() - b.checkinTime.getTime())
                  .map((patient) => (
                  <div key={patient.id} className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50" data-testid={`triage-patient-${patient.id}`}>
                    <div className="flex-1">
                      <div className="flex items-center space-x-4">
                        <div>
                          <h3 className="font-semibold text-lg">{patient.name}</h3>
                          <p className="text-sm text-gray-600">Age {patient.age} • {patient.phone}</p>
                        </div>
                        <Badge variant="outline" className="bg-yellow-50 text-yellow-700">
                          Waiting {getWaitTime(patient.checkinTime)}
                        </Badge>
                      </div>
                      <div className="mt-2">
                        <p className="text-sm"><strong>Chief Complaint:</strong> {patient.chiefComplaint}</p>
                        <p className="text-xs text-gray-500">Checked in at {formatTime(patient.checkinTime)}</p>
                      </div>
                    </div>
                    <Button onClick={() => handleTakeVitals(patient)} data-testid={`button-take-vitals-${patient.id}`}>
                      Take Vitals
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Vitals Dialog */}
        <Dialog open={vitalsDialogOpen} onOpenChange={setVitalsDialogOpen}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Record Vitals - {selectedPatient?.name}</DialogTitle>
            </DialogHeader>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <label className="text-sm font-medium flex items-center mb-2">
                  <Heart className="w-4 h-4 mr-2 text-red-500" />
                  Heart Rate (BPM)
                </label>
                <Input
                  type="number"
                  value={vitals.heartRate}
                  onChange={(e) => setVitals({...vitals, heartRate: e.target.value})}
                  placeholder="Enter heart rate"
                  data-testid="input-heart-rate"
                />
              </div>
              
              <div>
                <label className="text-sm font-medium flex items-center mb-2">
                  <Activity className="w-4 h-4 mr-2 text-blue-500" />
                  Blood Pressure
                </label>
                <Input
                  value={vitals.bloodPressure}
                  onChange={(e) => setVitals({...vitals, bloodPressure: e.target.value})}
                  placeholder="e.g., 120/80"
                  data-testid="input-blood-pressure"
                />
              </div>
              
              <div>
                <label className="text-sm font-medium flex items-center mb-2">
                  <Thermometer className="w-4 h-4 mr-2 text-orange-500" />
                  Temperature (°F)
                </label>
                <Input
                  type="number"
                  step="0.1"
                  value={vitals.temperature}
                  onChange={(e) => setVitals({...vitals, temperature: e.target.value})}
                  placeholder="Enter temperature"
                  data-testid="input-temperature"
                />
              </div>
              
              <div>
                <label className="text-sm font-medium flex items-center mb-2">
                  <Weight className="w-4 h-4 mr-2 text-gray-500" />
                  Weight (lbs)
                </label>
                <Input
                  type="number"
                  value={vitals.weight}
                  onChange={(e) => setVitals({...vitals, weight: e.target.value})}
                  placeholder="Enter weight"
                  data-testid="input-weight"
                />
              </div>
              
              <div className="col-span-2">
                <label className="text-sm font-medium flex items-center mb-2">
                  <Droplet className="w-4 h-4 mr-2 text-blue-400" />
                  Oxygen Saturation (%)
                </label>
                <Input
                  type="number"
                  value={vitals.oxygenSaturation}
                  onChange={(e) => setVitals({...vitals, oxygenSaturation: e.target.value})}
                  placeholder="Enter oxygen saturation"
                  data-testid="input-oxygen-saturation"
                />
              </div>
            </div>
            
            <div className="flex justify-end space-x-2 mt-6">
              <Button variant="outline" onClick={() => setVitalsDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleSubmitVitals} data-testid="button-submit-vitals">
                Submit Vitals
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </DashboardLayout>
  );
}