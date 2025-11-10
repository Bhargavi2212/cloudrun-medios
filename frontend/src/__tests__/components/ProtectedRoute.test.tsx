import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '../utils'
import { ProtectedRoute } from '@/components/auth/ProtectedRoute'
import { useAuthStore } from '@/store/authStore'

// Mock the auth store
vi.mock('@/store/authStore', () => ({
  useAuthStore: vi.fn(),
  ROLE_CONFIG: {},
}))

describe('ProtectedRoute', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows loading state when isLoading is true', () => {
    // Mock loading state
    ;(useAuthStore as any).mockReturnValue({
      isAuthenticated: false,
      user: null,
      isLoading: true,
    })

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
    ;(useAuthStore as any).mockReturnValue({
      isAuthenticated: false,
      user: null,
      isLoading: false,
    })

    const { container } = render(
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
    ;(useAuthStore as any).mockReturnValue({
      isAuthenticated: true,
      user: {
        id: '1',
        email: 'test@test.com',
        role: 'doctor',
      },
      isLoading: false,
    })

    render(
      <ProtectedRoute>
        <div>Protected Content</div>
      </ProtectedRoute>
    )

    expect(screen.getByText('Protected Content')).toBeInTheDocument()
  })
})

