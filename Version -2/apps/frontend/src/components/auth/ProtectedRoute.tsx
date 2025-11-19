import { ReactNode } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuthStore } from '../../store/authStore';
import { UserRole } from '../../shared/types/auth';

interface ProtectedRouteProps {
  children: ReactNode;
  requiredRole?: UserRole | UserRole[];
  requiredPermission?: string;
}

export const ProtectedRoute = ({ children, requiredRole, requiredPermission }: ProtectedRouteProps) => {
  const { user, token, hasRole, hasPermission, getDefaultRoute } = useAuthStore();
  const isAuthenticated = Boolean(user && token);

  if (!isAuthenticated || !user) {
    return <Navigate to="/login" replace />;
  }

  if (requiredRole && !hasRole(requiredRole)) {
    const defaultRoute = getDefaultRoute();
    return <Navigate to={defaultRoute} replace />;
  }

  if (requiredPermission && !hasPermission(requiredPermission)) {
    const defaultRoute = getDefaultRoute();
    return <Navigate to={defaultRoute} replace />;
  }

  return <>{children}</>;
};

