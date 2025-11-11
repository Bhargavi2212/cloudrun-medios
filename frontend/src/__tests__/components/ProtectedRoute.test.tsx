import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '../utils'
import { ProtectedRoute } from '@/components/auth/ProtectedRoute'
import { useAuthStore } from '@/store/authStore'

// Mock the auth store
vi.mock('@/store/authStore', () => ({
  useAuthStore: vi.fn(),
  ROLE_CONFIG: {},
}))

interface MockAuthStoreReturn {
  isAuthenticated: boolean
  user: { id: string; email: string; role: string } | null
  isLoading: boolean
}

const mockUseAuthStore = vi.mocked(useAuthStore)

describe('ProtectedRoute', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows loading state when isLoading is true', () => {
    // Mock loading state
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: false,
      user: null,
      isLoading: true,
    } as MockAuthStoreReturn)

    render(
      <ProtectedRoute>
        <div>Protected Content</div>
      </ProtectedRoute>
    )

    expect(screen.getByText('Loading...')).toBeInTheDocument()
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument()
  })

  it('redirects to login when not authenticated', () => {
    // Mock unauthenticated state
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: false,
      user: null,
      isLoading: false,
    } as MockAuthStoreReturn)

    render(
      <ProtectedRoute>
        <div>Protected Content</div>
      </ProtectedRoute>
    )

    // Navigate component should be rendered (redirect happens)
    // Protected content should not be visible
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument()
  })

  it('renders children when authenticated', () => {
    // Mock authenticated state
    mockUseAuthStore.mockReturnValue({
      isAuthenticated: true,
      user: {
        id: '1',
        email: 'test@test.com',
        role: 'doctor',
      },
      isLoading: false,
    } as MockAuthStoreReturn)

    render(
      <ProtectedRoute>
        <div>Protected Content</div>
      </ProtectedRoute>
    )

    expect(screen.getByText('Protected Content')).toBeInTheDocument()
  })
})

