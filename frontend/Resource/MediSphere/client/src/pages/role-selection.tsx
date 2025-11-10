import { useState } from "react";
import { useAuth } from "@/hooks/useAuth";
import { useToast } from "@/hooks/use-toast";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { isUnauthorizedError } from "@/lib/authUtils";

export default function RoleSelection() {
  const { user } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [selectedRole, setSelectedRole] = useState<string>("");

  const roleMutation = useMutation({
    mutationFn: async (role: string) => {
      await apiRequest("POST", "/api/auth/select-role", { role });
    },
    onSuccess: () => {
      toast({
        title: "Role Selected",
        description: "Welcome to Medi OS! Redirecting to your dashboard...",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/auth/user"] });
      setTimeout(() => {
        window.location.reload();
      }, 1000);
    },
    onError: (error) => {
      if (isUnauthorizedError(error)) {
        toast({
          title: "Unauthorized",
          description: "You are logged out. Logging in again...",
          variant: "destructive",
        });
        setTimeout(() => {
          window.location.href = "/api/login";
        }, 500);
        return;
      }
      toast({
        title: "Error",
        description: "Failed to select role. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleRoleSelect = (role: string) => {
    setSelectedRole(role);
    roleMutation.mutate(role);
  };

  const roles = [
    {
      id: "super-admin",
      title: "Super Admin",
      description: "Multi-hospital system oversight",
      icon: "fas fa-globe",
      color: "bg-indigo-600 hover:bg-indigo-700",
      features: ["Global analytics", "Multi-hospital management", "System administration", "ML insights center"]
    },
    {
      id: "admin",
      title: "Hospital Admin",
      description: "Single hospital operations & management",
      icon: "fas fa-user-shield",
      color: "bg-purple-600 hover:bg-purple-700",
      features: ["Hospital operations", "Staff & inventory", "Financial dashboard", "Patient flow"]
    },
    {
      id: "doctor",
      title: "Doctor",
      description: "Patient consultations and medical care",
      icon: "fas fa-user-md",
      color: "bg-blue-600 hover:bg-blue-700",
      features: ["Patient queue", "Medical records", "Prescriptions", "Consultation tools"]
    },
    {
      id: "nurse",
      title: "Nurse",
      description: "Patient care and vital monitoring",
      icon: "fas fa-user-nurse",
      color: "bg-teal-600 hover:bg-teal-700",
      features: ["Vitals recording", "Patient monitoring", "Task management", "Doctor coordination"]
    },
    {
      id: "receptionist",
      title: "Receptionist",
      description: "Patient check-in and appointment management",
      icon: "fas fa-clipboard-check",
      color: "bg-green-600 hover:bg-green-700",
      features: ["Patient check-in", "Queue management", "Room status", "Appointment scheduling"]
    },
    {
      id: "patient",
      title: "Patient",
      description: "View your health information and appointments",
      icon: "fas fa-user-injured",
      color: "bg-gray-600 hover:bg-gray-700",
      features: ["Appointment tracking", "Medical history", "Visit status", "Health summaries"]
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl w-full space-y-8">
        <div className="text-center">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center">
              <i className="fas fa-heartbeat text-white text-2xl"></i>
            </div>
            <h1 className="text-3xl font-bold text-gray-900">Medi OS</h1>
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Welcome, {user?.firstName}!</h2>
          <p className="text-gray-600 mb-8">Select your role to access your personalized dashboard</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {roles.map((role) => (
            <div key={role.id} className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow">
              <div className="p-6">
                <div className="flex items-center space-x-4 mb-4">
                  <div className={`w-12 h-12 ${role.color.split(' ')[0]} rounded-lg flex items-center justify-center`}>
                    <i className={`${role.icon} text-white text-xl`}></i>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{role.title}</h3>
                    <p className="text-sm text-gray-600">{role.description}</p>
                  </div>
                </div>

                <div className="space-y-2 mb-6">
                  {role.features.map((feature, index) => (
                    <div key={index} className="flex items-center space-x-2 text-sm text-gray-600">
                      <i className="fas fa-check text-green-500 text-xs"></i>
                      <span>{feature}</span>
                    </div>
                  ))}
                </div>

                <button
                  onClick={() => window.location.href = `/${role.id}`}
                  className={`w-full ${role.color} text-white py-3 px-4 rounded-lg font-medium transition-colors hover:scale-105 transform transition-all duration-200 flex items-center justify-center space-x-2`}
                >
                  <span>Enter {role.title} Dashboard</span>
                  <i className="fas fa-arrow-right"></i>
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="text-center">
          <button
            onClick={() => window.location.href = "/api/logout"}
            className="text-gray-500 hover:text-gray-700 text-sm flex items-center space-x-2 mx-auto"
          >
            <i className="fas fa-sign-out-alt"></i>
            <span>Sign out</span>
          </button>
        </div>
      </div>
    </div>
  );
}