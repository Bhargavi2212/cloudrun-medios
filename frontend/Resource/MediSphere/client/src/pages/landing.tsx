import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function Landing() {
  const handleLogin = () => {
    window.location.href = "/api/login";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-medical-blue to-medical-teal flex items-center justify-center p-4">
      <Card className="w-full max-w-md shadow-2xl">
        <CardContent className="p-8">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-medical-blue rounded-xl flex items-center justify-center mx-auto mb-4">
              <i className="fas fa-heartbeat text-white text-2xl"></i>
            </div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Medi OS</h1>
            <p className="text-medical-gray">Hospital Management System</p>
          </div>
          
          <div className="space-y-6">
            <div className="text-center">
              <p className="text-gray-600 mb-6">
                Streamline hospital operations with intelligent patient management, 
                real-time queues, and automated staff coordination.
              </p>
            </div>
            
            <Button 
              onClick={handleLogin}
              className="w-full bg-medical-blue text-white py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              Sign In to Continue
            </Button>
            
            <div className="mt-6 pt-6 border-t border-gray-200">
              <p className="text-center text-sm text-medical-gray mb-3">
                Roles supported:
              </p>
              <div className="flex flex-wrap gap-2 justify-center">
                <span className="px-3 py-1 text-xs bg-red-100 text-red-700 rounded-full">Admin</span>
                <span className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded-full">Doctor</span>
                <span className="px-3 py-1 text-xs bg-green-100 text-green-700 rounded-full">Nurse</span>
                <span className="px-3 py-1 text-xs bg-purple-100 text-purple-700 rounded-full">Receptionist</span>
                <span className="px-3 py-1 text-xs bg-yellow-100 text-yellow-700 rounded-full">Patient</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
