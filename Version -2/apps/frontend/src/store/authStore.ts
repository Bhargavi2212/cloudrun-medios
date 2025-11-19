import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { User, UserRole } from '../shared/types/auth';
import { authAPI } from '../shared/services/api';

export const ROLE_CONFIG = {
  RECEPTIONIST: {
    displayName: 'Receptionist',
    color: 'pink',
    route: '/receptionist/dashboard',
    permissions: ['patient.checkin', 'queue.view', 'triage.perform'],
  },
  NURSE: {
    displayName: 'Nurse',
    color: 'blue',
    route: '/nurse/dashboard',
    permissions: ['patient.vitals', 'queue.view', 'patient.view', 'document.upload'],
  },
  DOCTOR: {
    displayName: 'Doctor',
    color: 'green',
    route: '/doctor/dashboard',
    permissions: ['consultation.perform', 'notes.create', 'scribe.use', 'patient.view'],
  },
  ADMIN: {
    displayName: 'Administrator',
    color: 'purple',
    route: '/admin/dashboard',
    permissions: ['*'],
  },
} as const;

interface AuthState {
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  clearError: () => void;
  hasRole: (role: UserRole | UserRole[]) => boolean;
  hasPermission: (permission: string) => boolean;
  getDefaultRoute: () => string;
}

const getPermissionsForRole = (role: UserRole): string[] => {
  const permissions = ROLE_CONFIG[role]?.permissions ?? [];
  return [...permissions];
};

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          // Try real API first
          const response = await authAPI.login(email, password);
          if (response && typeof response === 'object' && 'user' in response && 'access_token' in response) {
            const { user, access_token, refresh_token } = response;
            set({
              user,
              token: access_token,
              refreshToken: refresh_token || null,
              isAuthenticated: true,
              isLoading: false,
              error: null,
            });
            return;
          }
          throw new Error('Invalid response format from server');
        } catch (apiError: unknown) {
          // If API fails, fall back to mock login for demo
          const emailLower = email.toLowerCase().trim();
          const mockUsers: Record<string, { user: User; token: string }> = {
            'receptionist@hospital.com': {
              user: {
                id: '1',
                email: 'receptionist@hospital.com',
                full_name: 'Sarah Johnson',
                role: 'RECEPTIONIST',
                first_name: 'Sarah',
                last_name: 'Johnson',
              },
              token: 'mock-token-receptionist',
            },
            'nurse@hospital.com': {
              user: {
                id: '2',
                email: 'nurse@hospital.com',
                full_name: 'Mike Chen',
                role: 'NURSE',
                first_name: 'Mike',
                last_name: 'Chen',
              },
              token: 'mock-token-nurse',
            },
            'doctor@hospital.com': {
              user: {
                id: '3',
                email: 'doctor@hospital.com',
                full_name: 'Dr. Emily Rodriguez',
                role: 'DOCTOR',
                first_name: 'Emily',
                last_name: 'Rodriguez',
              },
              token: 'mock-token-doctor',
            },
            'admin@hospital.com': {
              user: {
                id: '4',
                email: 'admin@hospital.com',
                full_name: 'Admin User',
                role: 'ADMIN',
                first_name: 'Admin',
                last_name: 'User',
              },
              token: 'mock-token-admin',
            },
          };

          if (password === 'demo123' && mockUsers[emailLower]) {
            const { user, token } = mockUsers[emailLower];
            set({
              user,
              token,
              refreshToken: null,
              isAuthenticated: true,
              isLoading: false,
              error: null,
            });
            return;
          }

          // If not a demo credential, throw the original error
          const errorObj = apiError as { response?: { data?: { detail?: string } }; message?: string };
          const errorMessage = errorObj?.response?.data?.detail || errorObj?.message || 'Login failed';
          set({
            user: null,
            token: null,
            refreshToken: null,
            isAuthenticated: false,
            isLoading: false,
            error: errorMessage,
          });
          throw new Error(errorMessage);
        }
      },

      logout: () => {
        set({
          user: null,
          token: null,
          refreshToken: null,
          isAuthenticated: false,
          error: null,
        });
      },

      clearError: () => set({ error: null }),

      hasRole: (role: UserRole | UserRole[]) => {
        const { user } = get();
        if (!user) return false;
        if (Array.isArray(role)) {
          return role.includes(user.role);
        }
        return user.role === role;
      },

      hasPermission: (permission: string) => {
        const { user } = get();
        if (!user) return false;
        const permissions = getPermissionsForRole(user.role);
        return permissions.includes('*') || permissions.includes(permission);
      },

      getDefaultRoute: () => {
        const { user } = get();
        if (!user) return '/login';
        return ROLE_CONFIG[user.role]?.route || '/';
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        refreshToken: state.refreshToken,
        isAuthenticated: state.isAuthenticated,
      }),
      onRehydrateStorage: () => (state, error) => {
        if (error) {
          console.error('Error rehydrating auth store:', error);
          // Clear corrupted storage
          try {
            localStorage.removeItem('auth-storage');
          } catch (e) {
            console.error('Error clearing corrupted storage:', e);
          }
        }
      },
    }
  )
);

