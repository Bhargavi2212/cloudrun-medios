import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";
import { apiRequest } from "@/lib/queryClient";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface VitalsModalProps {
  isOpen: boolean;
  onClose: () => void;
  patientId?: string;
  queueId?: string;
}

export default function VitalsModal({ isOpen, onClose, patientId, queueId }: VitalsModalProps) {
  const [vitalsData, setVitalsData] = useState({
    patientId: patientId || "",
    queueId: queueId || "",
    bloodPressureSystolic: "",
    bloodPressureDiastolic: "",
    heartRate: "",
    temperature: "",
    oxygenSaturation: "",
    weight: "",
    height: "",
  });

  const { toast } = useToast();
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const vitalsMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await apiRequest("POST", "/api/vitals", data);
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Vitals Recorded",
        description: "Patient vitals have been recorded successfully.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/patients"] });
      onClose();
      resetForm();
    },
    onError: (error) => {
      toast({
        title: "Recording Failed",
        description: error.message || "Failed to record vitals. Please try again.",
        variant: "destructive",
      });
    },
  });

  const resetForm = () => {
    setVitalsData({
      patientId: patientId || "",
      queueId: queueId || "",
      bloodPressureSystolic: "",
      bloodPressureDiastolic: "",
      heartRate: "",
      temperature: "",
      oxygenSaturation: "",
      weight: "",
      height: "",
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!vitalsData.patientId) {
      toast({
        title: "Missing Patient",
        description: "Please select a patient first.",
        variant: "destructive",
      });
      return;
    }

    // Convert string values to numbers and multiply temperature by 10 for storage
    const processedData = {
      patientId: vitalsData.patientId,
      queueId: vitalsData.queueId || undefined,
      bloodPressureSystolic: vitalsData.bloodPressureSystolic ? parseInt(vitalsData.bloodPressureSystolic) : undefined,
      bloodPressureDiastolic: vitalsData.bloodPressureDiastolic ? parseInt(vitalsData.bloodPressureDiastolic) : undefined,
      heartRate: vitalsData.heartRate ? parseInt(vitalsData.heartRate) : undefined,
      temperature: vitalsData.temperature ? Math.round(parseFloat(vitalsData.temperature) * 10) : undefined,
      oxygenSaturation: vitalsData.oxygenSaturation ? parseInt(vitalsData.oxygenSaturation) : undefined,
      weight: vitalsData.weight ? Math.round(parseFloat(vitalsData.weight) * 10) : undefined,
      height: vitalsData.height ? parseInt(vitalsData.height) : undefined,
    };

    vitalsMutation.mutate(processedData);
  };

  const getVitalStatus = (type: string, value: string) => {
    if (!value) return null;
    
    const numValue = parseFloat(value);
    
    switch (type) {
      case 'systolic':
        if (numValue < 90) return { color: 'text-blue-600', label: 'Low' };
        if (numValue > 140) return { color: 'text-red-600', label: 'High' };
        return { color: 'text-green-600', label: 'Normal' };
      case 'diastolic':
        if (numValue < 60) return { color: 'text-blue-600', label: 'Low' };
        if (numValue > 90) return { color: 'text-red-600', label: 'High' };
        return { color: 'text-green-600', label: 'Normal' };
      case 'heartRate':
        if (numValue < 60) return { color: 'text-blue-600', label: 'Low' };
        if (numValue > 100) return { color: 'text-red-600', label: 'High' };
        return { color: 'text-green-600', label: 'Normal' };
      case 'temperature':
        if (numValue < 97) return { color: 'text-blue-600', label: 'Low' };
        if (numValue > 99.5) return { color: 'text-red-600', label: 'Fever' };
        return { color: 'text-green-600', label: 'Normal' };
      case 'oxygen':
        if (numValue < 95) return { color: 'text-red-600', label: 'Low' };
        return { color: 'text-green-600', label: 'Normal' };
      default:
        return null;
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-auto">
        <DialogHeader>
          <DialogTitle>Record Patient Vitals</DialogTitle>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Patient Selection */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="patientId">Patient ID *</Label>
              <Input
                id="patientId"
                type="text"
                placeholder="Enter patient ID"
                value={vitalsData.patientId}
                onChange={(e) => setVitalsData({ ...vitalsData, patientId: e.target.value })}
                required
              />
            </div>
            <div>
              <Label htmlFor="queueId">Queue ID (Optional)</Label>
              <Input
                id="queueId"
                type="text"
                placeholder="Associated queue item"
                value={vitalsData.queueId}
                onChange={(e) => setVitalsData({ ...vitalsData, queueId: e.target.value })}
              />
            </div>
          </div>

          {/* Vital Signs */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Blood Pressure */}
            <div className="space-y-4">
              <h3 className="font-medium text-gray-900 flex items-center">
                <i className="fas fa-heart text-red-500 mr-2"></i>
                Blood Pressure
              </h3>
              <div className="space-y-2">
                <div>
                  <Label htmlFor="systolic">Systolic (mmHg)</Label>
                  <Input
                    id="systolic"
                    type="number"
                    placeholder="120"
                    value={vitalsData.bloodPressureSystolic}
                    onChange={(e) => setVitalsData({ ...vitalsData, bloodPressureSystolic: e.target.value })}
                  />
                  {vitalsData.bloodPressureSystolic && (
                    <p className={`text-xs ${getVitalStatus('systolic', vitalsData.bloodPressureSystolic)?.color}`}>
                      {getVitalStatus('systolic', vitalsData.bloodPressureSystolic)?.label}
                    </p>
                  )}
                </div>
                <div>
                  <Label htmlFor="diastolic">Diastolic (mmHg)</Label>
                  <Input
                    id="diastolic"
                    type="number"
                    placeholder="80"
                    value={vitalsData.bloodPressureDiastolic}
                    onChange={(e) => setVitalsData({ ...vitalsData, bloodPressureDiastolic: e.target.value })}
                  />
                  {vitalsData.bloodPressureDiastolic && (
                    <p className={`text-xs ${getVitalStatus('diastolic', vitalsData.bloodPressureDiastolic)?.color}`}>
                      {getVitalStatus('diastolic', vitalsData.bloodPressureDiastolic)?.label}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Heart Rate & Temperature */}
            <div className="space-y-4">
              <h3 className="font-medium text-gray-900 flex items-center">
                <i className="fas fa-heartbeat text-blue-500 mr-2"></i>
                Cardiac & Temperature
              </h3>
              <div className="space-y-2">
                <div>
                  <Label htmlFor="heartRate">Heart Rate (bpm)</Label>
                  <Input
                    id="heartRate"
                    type="number"
                    placeholder="72"
                    value={vitalsData.heartRate}
                    onChange={(e) => setVitalsData({ ...vitalsData, heartRate: e.target.value })}
                  />
                  {vitalsData.heartRate && (
                    <p className={`text-xs ${getVitalStatus('heartRate', vitalsData.heartRate)?.color}`}>
                      {getVitalStatus('heartRate', vitalsData.heartRate)?.label}
                    </p>
                  )}
                </div>
                <div>
                  <Label htmlFor="temperature">Temperature (°F)</Label>
                  <Input
                    id="temperature"
                    type="number"
                    step="0.1"
                    placeholder="98.6"
                    value={vitalsData.temperature}
                    onChange={(e) => setVitalsData({ ...vitalsData, temperature: e.target.value })}
                  />
                  {vitalsData.temperature && (
                    <p className={`text-xs ${getVitalStatus('temperature', vitalsData.temperature)?.color}`}>
                      {getVitalStatus('temperature', vitalsData.temperature)?.label}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Other Measurements */}
            <div className="space-y-4">
              <h3 className="font-medium text-gray-900 flex items-center">
                <i className="fas fa-ruler text-green-500 mr-2"></i>
                Other Measurements
              </h3>
              <div className="space-y-2">
                <div>
                  <Label htmlFor="oxygenSaturation">Oxygen Saturation (%)</Label>
                  <Input
                    id="oxygenSaturation"
                    type="number"
                    placeholder="98"
                    min="0"
                    max="100"
                    value={vitalsData.oxygenSaturation}
                    onChange={(e) => setVitalsData({ ...vitalsData, oxygenSaturation: e.target.value })}
                  />
                  {vitalsData.oxygenSaturation && (
                    <p className={`text-xs ${getVitalStatus('oxygen', vitalsData.oxygenSaturation)?.color}`}>
                      {getVitalStatus('oxygen', vitalsData.oxygenSaturation)?.label}
                    </p>
                  )}
                </div>
                <div>
                  <Label htmlFor="weight">Weight (lbs)</Label>
                  <Input
                    id="weight"
                    type="number"
                    step="0.1"
                    placeholder="150.0"
                    value={vitalsData.weight}
                    onChange={(e) => setVitalsData({ ...vitalsData, weight: e.target.value })}
                  />
                </div>
                <div>
                  <Label htmlFor="height">Height (inches)</Label>
                  <Input
                    id="height"
                    type="number"
                    placeholder="70"
                    value={vitalsData.height}
                    onChange={(e) => setVitalsData({ ...vitalsData, height: e.target.value })}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="border-t pt-4">
            <h4 className="font-medium text-gray-900 mb-3">Quick Actions</h4>
            <div className="flex space-x-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setVitalsData({
                  ...vitalsData,
                  bloodPressureSystolic: "120",
                  bloodPressureDiastolic: "80",
                  heartRate: "72",
                  temperature: "98.6",
                  oxygenSaturation: "98"
                })}
              >
                <i className="fas fa-magic mr-1"></i>
                Normal Values
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={resetForm}
              >
                <i className="fas fa-eraser mr-1"></i>
                Clear All
              </Button>
            </div>
          </div>

          {/* Submit Actions */}
          <div className="flex items-center justify-end space-x-4 pt-6 border-t border-gray-200">
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button 
              type="submit" 
              disabled={vitalsMutation.isPending}
              className="bg-medical-teal hover:bg-teal-700"
            >
              {vitalsMutation.isPending ? (
                <>
                  <i className="fas fa-spinner fa-spin mr-2"></i>
                  Recording...
                </>
              ) : (
                <>
                  <i className="fas fa-save mr-2"></i>
                  Record Vitals
                </>
              )}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
