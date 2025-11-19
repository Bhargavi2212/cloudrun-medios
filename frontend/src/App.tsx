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
import ScribeLivePage from '@/pages/scribe/ScribeLive';
import ScribeReviewPage from '@/pages/scribe/ScribeReview';
import { DocumentReviewDashboard } from '@/pages/nurse/DocumentReviewDashboard';

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

          <Route path="/receptionist" element={<Navigate to="/receptionist/dashboard" replace />} />
          <Route
            path="/receptionist/dashboard"
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
          
          <Route path="/nurse" element={<Navigate to="/nurse/dashboard" replace />} />
          <Route
            path="/nurse/dashboard"
            element={
              <ProtectedRoute allowedRoles={['NURSE']}>
                <AppLayout>
                  <NurseDashboard />
                </AppLayout>
              </ProtectedRoute>
            }
          />

          <Route path="/doctor" element={<Navigate to="/doctor/dashboard" replace />} />
          <Route
            path="/doctor/dashboard"
            element={
              <ProtectedRoute allowedRoles={['DOCTOR']}>
                <AppLayout>
                  <DoctorWorkflow />
                </AppLayout>
              </ProtectedRoute>
            }
          />

          <Route
            path="/scribe/live"
            element={
              <ProtectedRoute allowedRoles={['NURSE', 'DOCTOR', 'ADMIN']}>
                <AppLayout>
                  <ScribeLivePage />
                </AppLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/scribe/review"
            element={
              <ProtectedRoute allowedRoles={['NURSE', 'DOCTOR', 'ADMIN']}>
                <AppLayout>
                  <ScribeReviewPage />
                </AppLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/scribe/review/:sessionId"
            element={
              <ProtectedRoute allowedRoles={['NURSE', 'DOCTOR', 'ADMIN']}>
                <AppLayout>
                  <ScribeReviewPage />
                </AppLayout>
              </ProtectedRoute>
            }
          />

          <Route
            path="/nurse/documents/review"
            element={
              <ProtectedRoute allowedRoles={['NURSE', 'DOCTOR', 'ADMIN']}>
                <AppLayout>
                  <DocumentReviewDashboard />
                </AppLayout>
              </ProtectedRoute>
            }
          />

          <Route path="/admin" element={<Navigate to="/admin/dashboard" replace />} />
          <Route
            path="/admin/dashboard"
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
