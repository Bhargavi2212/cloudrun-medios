import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { insertQueueSchema } from "@shared/schema";

interface PatientCheckinModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function PatientCheckinModal({ isOpen, onClose }: PatientCheckinModalProps) {
  const [formData, setFormData] = useState({
    patientName: "",
    patientId: "",
    reasonForVisit: "",
    category: "",
    priority: "normal" as const,
  });
  const [qrScanning, setQrScanning] = useState(false);
  
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const checkinMutation = useMutation({
    mutationFn: async (data: any) => {
      const response = await apiRequest("POST", "/api/queue", data);
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Check-in Successful",
        description: "Patient has been added to the queue successfully.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/queue"] });
      queryClient.invalidateQueries({ queryKey: ["/api/stats"] });
      onClose();
      resetForm();
    },
    onError: (error) => {
      toast({
        title: "Check-in Failed",
        description: error.message || "Failed to check in patient. Please try again.",
        variant: "destructive",
      });
    },
  });

  const resetForm = () => {
    setFormData({
      patientName: "",
      patientId: "",
      reasonForVisit: "",
      category: "",
      priority: "normal",
    });
    setQrScanning(false);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.patientId || !formData.reasonForVisit || !formData.category) {
      toast({
        title: "Missing Information",
        description: "Please fill in all required fields.",
        variant: "destructive",
      });
      return;
    }

    checkinMutation.mutate({
      patientId: formData.patientId,
      reasonForVisit: formData.reasonForVisit,
      category: formData.category,
      priority: formData.priority,
    });
  };

  const handleQRScan = () => {
    setQrScanning(true);
    // Simulate QR code scan - in real implementation, this would use camera
    setTimeout(() => {
      setFormData({
        ...formData,
        patientName: "John Doe",
        patientId: "JD-2024-" + Math.floor(Math.random() * 1000),
      });
      setQrScanning(false);
      toast({
        title: "QR Code Scanned",
        description: "Patient information loaded successfully.",
      });
    }, 2000);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-auto">
        <DialogHeader>
          <DialogTitle>Patient Check-in</DialogTitle>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* QR Code Scanner Section */}
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
            <i className="fas fa-qrcode text-4xl text-gray-400 mb-3"></i>
            <p className="text-gray-600 mb-3">Scan Patient QR Code or Enter Details Manually</p>
            <Button 
              type="button" 
              onClick={handleQRScan}
              disabled={qrScanning}
              className="bg-medical-blue hover:bg-blue-700"
            >
              <i className="fas fa-camera mr-2"></i>
              {qrScanning ? "Scanning..." : "Activate Scanner"}
            </Button>
          </div>

          {/* Patient Details Form */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="patientName">Patient Name</Label>
              <Input
                id="patientName"
                type="text"
                placeholder="Enter patient name"
                value={formData.patientName}
                onChange={(e) => setFormData({ ...formData, patientName: e.target.value })}
              />
            </div>
            <div>
              <Label htmlFor="patientId">Patient ID *</Label>
              <Input
                id="patientId"
                type="text"
                placeholder="Patient ID or generate new"
                value={formData.patientId}
                onChange={(e) => setFormData({ ...formData, patientId: e.target.value })}
                required
              />
            </div>
          </div>

          {/* Reason for Visit */}
          <div>
            <Label htmlFor="reasonForVisit">Reason for Visit *</Label>
            <Input
              id="reasonForVisit"
              type="text"
              placeholder="Describe the reason for visit"
              value={formData.reasonForVisit}
              onChange={(e) => setFormData({ ...formData, reasonForVisit: e.target.value })}
              required
            />
          </div>

          {/* Medical Category */}
          <div>
            <Label htmlFor="category">Medical Category *</Label>
            <Select value={formData.category} onValueChange={(value) => setFormData({ ...formData, category: value })}>
              <SelectTrigger>
                <SelectValue placeholder="Select medical category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="General">General Medicine</SelectItem>
                <SelectItem value="Cardiology">Cardiology</SelectItem>
                <SelectItem value="Pediatrics">Pediatrics</SelectItem>
                <SelectItem value="Neurology">Neurology</SelectItem>
                <SelectItem value="Orthopedics">Orthopedics</SelectItem>
                <SelectItem value="Dermatology">Dermatology</SelectItem>
                <SelectItem value="Emergency">Emergency</SelectItem>
                <SelectItem value="Follow-up">Follow-up</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Priority Level */}
          <div>
            <Label>Priority Level *</Label>
            <RadioGroup
              value={formData.priority}
              onValueChange={(value) => setFormData({ ...formData, priority: value as any })}
              className="flex space-x-4 mt-2"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="low" id="low" />
                <Label htmlFor="low" className="text-sm">Low</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="normal" id="normal" />
                <Label htmlFor="normal" className="text-sm">Normal</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="high" id="high" />
                <Label htmlFor="high" className="text-sm">High</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="urgent" id="urgent" />
                <Label htmlFor="urgent" className="text-sm">Urgent</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="emergency" id="emergency" />
                <Label htmlFor="emergency" className="text-sm">Emergency</Label>
              </div>
            </RadioGroup>
          </div>

          {/* Submit Actions */}
          <div className="flex items-center justify-end space-x-4 pt-6 border-t border-gray-200">
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button 
              type="submit" 
              disabled={checkinMutation.isPending}
              className="bg-medical-blue hover:bg-blue-700"
            >
              {checkinMutation.isPending ? (
                <>
                  <i className="fas fa-spinner fa-spin mr-2"></i>
                  Processing...
                </>
              ) : (
                "Complete Check-in"
              )}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
