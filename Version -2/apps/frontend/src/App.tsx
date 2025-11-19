import { Navigate, Route, Routes } from "react-router-dom";
import { useAuthStore } from "./store/authStore";
import { ProtectedRoute } from "./components/auth/ProtectedRoute";
import { LoginPage } from "./pages/LoginPage";
import { ReceptionistDashboard } from "./pages/receptionist/ReceptionistDashboard";
import { NurseDashboard } from "./pages/nurse/NurseDashboard";
import { DoctorDashboard } from "./pages/doctor/DoctorDashboard";
import { AdminDashboard } from "./pages/admin/AdminDashboard";

const App = () => {
  console.log("App component rendering...");
  const { user, token, getDefaultRoute } = useAuthStore();
  const isAuthenticated = Boolean(user && token);
  console.log("Auth store accessed, isAuthenticated:", isAuthenticated);

  return (
    <Routes>
      <Route path="/login" element={isAuthenticated ? <Navigate to={getDefaultRoute()} replace /> : <LoginPage />} />
      
      <Route
        path="/receptionist/dashboard"
        element={
          <ProtectedRoute requiredRole="RECEPTIONIST">
            <ReceptionistDashboard />
          </ProtectedRoute>
        }
      />
      
      <Route
        path="/nurse/dashboard"
        element={
          <ProtectedRoute requiredRole="NURSE">
            <NurseDashboard />
          </ProtectedRoute>
        }
      />
      
      <Route
        path="/doctor/dashboard"
        element={
          <ProtectedRoute requiredRole="DOCTOR">
            <DoctorDashboard />
          </ProtectedRoute>
        }
      />
      
      <Route
        path="/admin/dashboard"
        element={
          <ProtectedRoute requiredRole="ADMIN">
            <AdminDashboard />
          </ProtectedRoute>
        }
      />
      
      <Route
        path="/"
        element={
          isAuthenticated ? (
            <Navigate to={getDefaultRoute()} replace />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
      
      <Route path="*" element={<Navigate to={isAuthenticated ? getDefaultRoute() : "/login"} replace />} />
    </Routes>
  );
};

export default App;

