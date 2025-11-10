import { Route, Switch, Redirect } from "wouter";
import { useAuthStore } from "./store/authStore";
import { Toaster } from "@/components/ui/toaster";
import LoginPage from "./pages/login";
import ReceptionistDashboard from "./pages/receptionist-dashboard";
import NurseDashboard from "./pages/nurse-dashboard";
import DoctorDashboard from "./pages/doctor-dashboard";
import NotFound from "./pages/not-found";

function ProtectedRoute({ component: Component }: { component: React.ComponentType }) {
  const isAuthenticated = useAuthStore(state => state.isAuthenticated);
  
  console.log('ProtectedRoute - isAuthenticated:', isAuthenticated);
  
  if (!isAuthenticated) {
    return <Redirect to="/login" />;
  }
  
  return <Component />;
}

function RoleBasedDashboard() {
  const user = useAuthStore(state => state.user);
  
  console.log('RoleBasedDashboard - user:', user);
  
  if (!user) return <Redirect to="/login" />;
  
  switch (user.role) {
    case 'RECEPTIONIST':
      return <ReceptionistDashboard />;
    case 'NURSE':
      return <NurseDashboard />;
    case 'DOCTOR':
      return <DoctorDashboard />;
    case 'ADMIN':
      return <ReceptionistDashboard />; // Default to receptionist view for admin
    default:
      return <NotFound />;
  }
}

export default function App() {
  const isAuthenticated = useAuthStore(state => state.isAuthenticated);

  return (
    <>
      <Switch>
        <Route path="/login" component={LoginPage} />
        <Route path="/dashboard">
          <ProtectedRoute component={RoleBasedDashboard} />
        </Route>
        <Route path="/">
          {isAuthenticated ? <Redirect to="/dashboard" /> : <Redirect to="/login" />}
        </Route>
        <Route component={NotFound} />
      </Switch>
      <Toaster />
    </>
  );
}
