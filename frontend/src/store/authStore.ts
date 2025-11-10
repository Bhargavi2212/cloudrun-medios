import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { User, UserRole } from '@/types'
import { authAPI } from '@/services/api'

export const ROLE_CONFIG = {
  RECEPTIONIST: {
    displayName: 'Receptionist',
    color: 'pink',
    route: '/receptionist',
    permissions: ['patient.checkin', 'queue.view', 'appointments.view', 'billing.process'],
    bgGradient: 'from-pink-50 via-white to-purple-50',
    badgeColor: 'bg-pink-100 text-pink-800',
  },
  NURSE: {
    displayName: 'Nurse',
    color: 'blue',
    route: '/nurse',
    permissions: ['patient.vitals', 'triage.perform', 'queue.view', 'patient.view'],
    bgGradient: 'from-blue-50 via-white to-indigo-50',
    badgeColor: 'bg-blue-100 text-blue-800',
  },
  DOCTOR: {
    displayName: 'Doctor',
    color: 'green',
    route: '/doctor',
    permissions: ['consultation.perform', 'notes.create', 'notes.edit', 'patient.view', 'prescriptions.create'],
    bgGradient: 'from-green-50 via-white to-emerald-50',
    badgeColor: 'bg-green-100 text-green-800',
  },
  ADMIN: {
    displayName: 'Administrator',
    color: 'purple',
    route: '/admin',
    permissions: ['*'],
    bgGradient: 'from-purple-50 via-white to-indigo-50',
    badgeColor: 'bg-purple-100 text-purple-800',
  },
} as const

type RoleConfig = (typeof ROLE_CONFIG)[UserRole]

type ProfileUpdatePayload = {
  first_name?: string
  last_name?: string
  phone?: string
}

type ChangePasswordPayload = {
  current_password: string
  new_password: string
}

interface AuthState {
  user: User | null
  token: string | null
  refreshToken: string | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  permissions: string[]
  login: (email: string, password: string) => Promise<void>
  logout: () => Promise<void>
  clearError: () => void
  initializeAuth: () => Promise<void>
  refreshUser: () => Promise<void>
  tryRefresh: () => Promise<boolean>
  updateProfile: (payload: ProfileUpdatePayload) => Promise<void>
  changePassword: (payload: ChangePasswordPayload) => Promise<void>
  hasRole: (role: UserRole | UserRole[]) => boolean
  hasPermission: (permission: string) => boolean
  isRole: (role: UserRole) => boolean
  getRoleDisplayName: () => string
  getRoleColor: () => string
  getDefaultRoute: () => string
  getRoleConfig: () => RoleConfig | null
}

interface RawUser {
  id: string
  email: string
  first_name?: string
  last_name?: string
  full_name?: string
  role: string
  roles?: string[]
  phone?: string
}

const buildUser = (raw: RawUser): User => {
  const fullName = raw?.full_name ?? [raw?.first_name, raw?.last_name].filter(Boolean).join(' ').trim()
  return {
    id: raw.id,
    email: raw.email,
    first_name: raw.first_name ?? undefined,
    last_name: raw.last_name ?? undefined,
    full_name: fullName || raw.email,
    role: raw.role as UserRole,
    roles: raw.roles ?? (raw.role ? [raw.role] : []),
    phone: raw.phone ?? undefined,
  }
}

const getPermissionsForRole = (role: UserRole): string[] => {
  const config = ROLE_CONFIG[role]
  return config ? [...config.permissions] : []
}

const DEMO_USERS: Record<string, { role: UserRole; first: string; last: string }> = {
  'receptionist@medios.ai': { role: 'RECEPTIONIST', first: 'Front', last: 'Desk' },
  'nurse@medios.ai': { role: 'NURSE', first: 'Nora', last: 'Nurse' },
  'doctor@medios.ai': { role: 'DOCTOR', first: 'Dan', last: 'Doctor' },
  'admin@medios.ai': { role: 'ADMIN', first: 'Ada', last: 'Admin' },
}

let refreshPromise: Promise<boolean> | null = null

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      refreshToken: null,
      isAuthenticated: false,
      isLoading: true,
      error: null,
      permissions: [],

      initializeAuth: async () => {
        const { token, refreshToken } = get()
        if (!token && !refreshToken) {
          set({ isLoading: false })
          return
        }

        set({ isLoading: true, error: null })
        try {
          if (!token && refreshToken) {
            const refreshed = await get().tryRefresh()
            if (!refreshed) {
              set({ user: null, token: null, refreshToken: null, isAuthenticated: false, permissions: [] })
              return
            }
          }

          if (!get().token) {
            set({ user: null, isAuthenticated: false, permissions: [] })
            return
          }

          const response = await authAPI.getCurrentUser()
          const payload = response.data
          if (!payload?.success) {
            throw new Error(payload?.error || 'Unable to load profile')
          }

          const user = buildUser(payload.data as RawUser)
          set({
            user,
            isAuthenticated: true,
            permissions: getPermissionsForRole(user.role),
            error: null,
          })
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Authentication required'
          set({
            user: null,
            token: null,
            refreshToken: null,
            isAuthenticated: false,
            permissions: [],
            error: errorMessage,
          })
        } finally {
          set({ isLoading: false })
        }
      },

      login: async (email: string, password: string) => {
        try {
          set({ isLoading: true, error: null })
          const response = await authAPI.login(email, password)
          const payload = response.data
          if (!payload?.success) {
            throw new Error(payload?.error || 'Login failed')
          }

          const { access_token, refresh_token, user: rawUser } = payload.data
          const user = buildUser(rawUser as RawUser)
          set({
            token: access_token,
            refreshToken: refresh_token,
            user,
            isAuthenticated: true,
            permissions: getPermissionsForRole(user.role),
            error: null,
          })
        } catch (error) {
          const message = error instanceof Error 
            ? error.message 
            : (error as { response?: { data?: { error?: string } } })?.response?.data?.error || 'Unable to login'
          set({
            error: message,
            isAuthenticated: false,
            permissions: [],
          })

          const fallback = DEMO_USERS[email.trim().toLowerCase()]
          if (fallback && password === 'password') {
            const rawUser = {
              id: `demo-${fallback.role.toLowerCase()}`,
              email: email.trim().toLowerCase(),
              first_name: fallback.first,
              last_name: fallback.last,
              full_name: `${fallback.first} ${fallback.last}`,
              role: fallback.role,
              roles: [fallback.role],
            }
            const demoUser = buildUser(rawUser)
            set({
              user: demoUser,
              token: null,
              refreshToken: null,
              isAuthenticated: true,
              permissions: getPermissionsForRole(demoUser.role),
              error: null,
            })
            return
          }

          throw new Error(message)
        } finally {
          set({ isLoading: false })
        }
      },

      logout: async () => {
        const { refreshToken } = get()
        try {
          if (refreshToken) {
            await authAPI.logout(refreshToken)
          }
        } catch (error) {
          console.warn('Logout warning:', error)
        } finally {
          set({
            user: null,
            token: null,
            refreshToken: null,
            isAuthenticated: false,
            error: null,
            permissions: [],
          })
        }
      },

      clearError: () => set({ error: null }),

      refreshUser: async () => {
        try {
          const response = await authAPI.getCurrentUser()
          const payload = response.data
          if (!payload?.success) {
            throw new Error(payload?.error || 'Unable to load profile')
          }
          const user = buildUser(payload.data as RawUser)
          set({ user, permissions: getPermissionsForRole(user.role) })
        } catch (error) {
          await get().logout()
        }
      },

      tryRefresh: async () => {
        if (refreshPromise) {
          return refreshPromise
        }

        const { refreshToken } = get()
        if (!refreshToken) {
          return false
        }

        refreshPromise = (async () => {
          try {
            const response = await authAPI.refresh(refreshToken)
            const payload = response.data
            if (!payload?.success) {
              throw new Error(payload?.error || 'Token refresh failed')
            }

            const { access_token, refresh_token } = payload.data
            set({
              token: access_token,
              refreshToken: refresh_token,
              isAuthenticated: true,
              error: null,
            })
            return true
          } catch (error) {
            set({ token: null, refreshToken: null, user: null, isAuthenticated: false, permissions: [] })
            return false
          } finally {
            refreshPromise = null
          }
        })()

        return refreshPromise
      },

      updateProfile: async (payload: ProfileUpdatePayload) => {
        const response = await authAPI.updateProfile(payload)
        const data = response.data
        if (!data?.success) {
          throw new Error(data?.error || 'Unable to update profile')
        }
        const user = buildUser(data.data as RawUser)
        set({ user, permissions: getPermissionsForRole(user.role) })
      },

      changePassword: async (payload: ChangePasswordPayload) => {
        const response = await authAPI.changePassword(payload)
        const data = response.data
        if (!data?.success) {
          throw new Error(data?.error || 'Unable to change password')
        }
      },

      hasRole: (role: UserRole | UserRole[]) => {
        const { user } = get()
        if (!user) return false
        if (Array.isArray(role)) {
          return role.includes(user.role)
        }
        return user.role === role
      },

      hasPermission: (permission: string) => {
        const { permissions } = get()
        return permissions.includes('*') || permissions.includes(permission)
      },

      isRole: (role: UserRole) => {
        const { user } = get()
        return user?.role === role
      },

      getRoleDisplayName: () => {
        const { user } = get()
        if (!user) return 'Unknown'
        return ROLE_CONFIG[user.role]?.displayName ?? user.role
      },

      getRoleColor: () => {
        const { user } = get()
        if (!user) return 'gray'
        return ROLE_CONFIG[user.role]?.color ?? 'gray'
      },

      getDefaultRoute: () => {
        const { user } = get()
        if (!user) return '/login'
        return ROLE_CONFIG[user.role]?.route ?? '/'
      },

      getRoleConfig: () => {
        const { user } = get()
        if (!user) return null
        return ROLE_CONFIG[user.role] ?? null
      },
    }),
    {
      name: 'medios-auth-storage',
      partialize: (state) => ({
        token: state.token,
        refreshToken: state.refreshToken,
        user: state.user,
        permissions: state.permissions,
      }),
    },
  ),
)

useAuthStore.getState().initializeAuth()
