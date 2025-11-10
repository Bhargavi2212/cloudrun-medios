import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from '@/components/ui/toaster';
import { queryClient } from '@/lib/queryClient';
import { ProtectedRoute } from '@/components/auth/ProtectedRoute';
import { AppLayout } from '@/components/layout/AppLayout';
import { ThemeProvider } from '@/contexts/ThemeContext';

// Pages
import LoginPage from '@/pages/LoginPage';
import RegisterPage from '@/pages/RegisterPage';
import ForgotPasswordPage from '@/pages/ForgotPasswordPage';
import ResetPasswordPage from '@/pages/ResetPasswordPage';
import AccountSettingsPage from '@/pages/AccountSettingsPage';
import ReceptionistDashboard from '@/pages/ReceptionistDashboard';
import CheckInView from '@/pages/receptionist/CheckInView';
import NurseDashboard from '@/pages/NurseDashboard';
import DoctorWorkflow from '@/pages/doctor/DoctorWorkflow';
import AdminDashboard from '@/pages/AdminDashboard';
import NotFoundPage from '@/pages/NotFoundPage';
import Root from '@/pages/Root';

function App() {
  return (
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <Router>
        <Routes>
          {/* Public Routes */}
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/forgot-password" element={<ForgotPasswordPage />} />
          <Route path="/reset-password" element={<ResetPasswordPage />} />
          
          {/* Protected Routes with Layout */}
          <Route
            path="/"
            element={<Root />}
          />
          
          <Route
            path="/account"
            element={
              <ProtectedRoute>
                <AppLayout>
                  <AccountSettingsPage />
                </AppLayout>
              </ProtectedRoute>
            }
          />

          <Route
            path="/receptionist"
            element={
              <ProtectedRoute allowedRoles={['RECEPTIONIST']}>
                <AppLayout>
                  <ReceptionistDashboard />
                </AppLayout>
              </ProtectedRoute>
            }
          />
          
          <Route
            path="/check-in"
            element={
              <ProtectedRoute allowedRoles={['RECEPTIONIST', 'ADMIN']}>
                <AppLayout>
                  <CheckInView />
                </AppLayout>
              </ProtectedRoute>
            }
          />
          
          <Route
            path="/nurse"
            element={
              <ProtectedRoute allowedRoles={['NURSE']}>
                <AppLayout>
                  <NurseDashboard />
                </AppLayout>
              </ProtectedRoute>
            }
          />
          
          <Route
            path="/doctor"
            element={
              <ProtectedRoute allowedRoles={['DOCTOR']}>
                <AppLayout>
                  <DoctorWorkflow />
                </AppLayout>
              </ProtectedRoute>
            }
          />
          
          <Route
            path="/admin"
            element={
              <ProtectedRoute allowedRoles={['ADMIN']}>
                <AppLayout>
                  <AdminDashboard />
                </AppLayout>
              </ProtectedRoute>
            }
          />

          {/* 404 Page */}
          <Route path="/404" element={<NotFoundPage />} />
          
          {/* Catch all route */}
          <Route path="*" element={<Navigate to="/404" replace />} />
        </Routes>
        <Toaster />
      </Router>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

export default App;
