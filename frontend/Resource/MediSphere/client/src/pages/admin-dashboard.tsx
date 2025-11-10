import { useEffect } from "react";
import { useAuth } from "@/hooks/useAuth";
import { useToast } from "@/hooks/use-toast";
import { useQuery } from "@tanstack/react-query";
import Sidebar from "@/components/layout/sidebar";
import Header from "@/components/layout/header";
import StatsCard from "@/components/dashboard/stats-card";
import QueueManagement from "@/components/dashboard/queue-management";
import RoomStatus from "@/components/dashboard/room-status";
import DoctorStatus from "@/components/dashboard/doctor-status";

export default function AdminDashboard() {
  const { toast } = useToast();
  const { isAuthenticated, isLoading, user } = useAuth();

  // Redirect if not authenticated or not admin
  useEffect(() => {
    if (!isLoading && (!isAuthenticated || user?.role !== 'admin')) {
      toast({
        title: "Unauthorized",
        description: "You need admin access to view this page.",
        variant: "destructive",
      });
      setTimeout(() => {
        window.location.href = "/api/login";
      }, 500);
      return;
    }
  }, [isAuthenticated, isLoading, user, toast]);

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["/api/stats"],
    enabled: isAuthenticated && user?.role === 'admin',
  });

  const { data: queue, isLoading: queueLoading } = useQuery({
    queryKey: ["/api/queue"],
    enabled: isAuthenticated && user?.role === 'admin',
  });

  const { data: rooms, isLoading: roomsLoading } = useQuery({
    queryKey: ["/api/rooms"],
    enabled: isAuthenticated && user?.role === 'admin',
  });

  const { data: doctors, isLoading: doctorsLoading } = useQuery({
    queryKey: ["/api/doctors"],
    enabled: isAuthenticated && user?.role === 'admin',
  });

  if (isLoading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-medical-bg">
        <div className="text-center">
          <div className="w-16 h-16 bg-medical-blue rounded-xl flex items-center justify-center mx-auto mb-4 animate-pulse">
            <i className="fas fa-heartbeat text-white text-2xl"></i>
          </div>
          <p className="text-medical-gray">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-medical-bg">
      <Header 
        title="Admin Dashboard" 
        subtitle="Hospital Management Overview"
        user={user}
      />
      
      <div className="flex">
        <Sidebar userRole="admin" />
        
        <main className="flex-1 p-6">
          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <StatsCard
              title="Total Patients Today"
              value={stats?.currentPatients?.toString() || "0"}
              icon="fas fa-user-injured"
              color="blue"
              subtitle="Currently in system"
              isLoading={statsLoading}
            />
            <StatsCard
              title="Available Beds"
              value={`${stats?.availableRooms || 0}/${stats?.totalRooms || 0}`}
              icon="fas fa-bed"
              color="green"
              subtitle="Bed occupancy"
              isLoading={statsLoading}
            />
            <StatsCard
              title="Active Staff"
              value={`${stats?.activeStaff || 0}/${stats?.totalStaff || 0}`}
              icon="fas fa-users"
              color="purple"
              subtitle="Currently on duty"
              isLoading={statsLoading}
            />
            <StatsCard
              title="Queue Length"
              value={stats?.waitingQueue?.toString() || "0"}
              icon="fas fa-clock"
              color="yellow"
              subtitle="Patients waiting"
              isLoading={statsLoading}
            />
          </div>

          {/* Main Dashboard Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <QueueManagement 
              queue={queue || []} 
              isLoading={queueLoading}
              userRole="admin"
            />
            <RoomStatus 
              rooms={rooms || []} 
              isLoading={roomsLoading}
            />
          </div>

          {/* Doctor Status & Recent Activity */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <DoctorStatus 
                doctors={doctors || []} 
                isLoading={doctorsLoading}
              />
            </div>
            
            {/* Quick Actions */}
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
              <div className="space-y-3">
                <button className="w-full flex items-center space-x-3 p-3 text-left bg-medical-blue text-white rounded-lg hover:bg-blue-700 transition-colors">
                  <i className="fas fa-plus-circle"></i>
                  <span className="font-medium">Add New Staff</span>
                </button>
                <button className="w-full flex items-center space-x-3 p-3 text-left bg-medical-teal text-white rounded-lg hover:bg-teal-700 transition-colors">
                  <i className="fas fa-bed"></i>
                  <span className="font-medium">Room Assignment</span>
                </button>
                <button className="w-full flex items-center space-x-3 p-3 text-left bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors">
                  <i className="fas fa-download"></i>
                  <span className="font-medium">Export Reports</span>
                </button>
                <button className="w-full flex items-center space-x-3 p-3 text-left bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors">
                  <i className="fas fa-bell"></i>
                  <span className="font-medium">Send Broadcast</span>
                </button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
