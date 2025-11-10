import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type UserRole = 'RECEPTIONIST' | 'NURSE' | 'DOCTOR' | 'ADMIN';

export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
}

interface AuthStore {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  setUser: (user: User, token: string) => void;
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,

      login: async (email: string, password: string) => {
        try {
          // Simulate login for demo - replace with real API call
          const mockUsers = [
            { id: '1', name: 'Sarah Johnson', email: 'receptionist@hospital.com', role: 'RECEPTIONIST' as UserRole },
            { id: '2', name: 'Mike Chen', email: 'nurse@hospital.com', role: 'NURSE' as UserRole },
            { id: '3', name: 'Dr. Emily Rodriguez', email: 'doctor@hospital.com', role: 'DOCTOR' as UserRole },
            { id: '4', name: 'Admin User', email: 'admin@hospital.com', role: 'ADMIN' as UserRole },
          ];

          // Small delay to simulate API call
          await new Promise(resolve => setTimeout(resolve, 500));

          const user = mockUsers.find(u => u.email === email);
          if (user && password === 'demo123') {
            const token = 'mock-jwt-token-' + user.id;
            const newState = { user, token, isAuthenticated: true };
            set(newState);
            console.log('Auth state updated:', newState);
            return; // Successful login
          } else {
            throw new Error('Invalid credentials');
          }
        } catch (error) {
          console.error('Login error:', error);
          throw error;
        }
      },

      logout: () => {
        set({ user: null, token: null, isAuthenticated: false });
      },

      setUser: (user: User, token: string) => {
        set({ user, token, isAuthenticated: true });
      },
    }),
    {
      name: 'auth-storage',
    }
  )
);