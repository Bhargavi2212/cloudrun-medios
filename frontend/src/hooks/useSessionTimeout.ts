import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'
import { useToast } from '@/components/ui/use-toast'

interface JWTPayload {
  exp: number
  iat: number
  sub: string
  [key: string]: string | number | boolean | undefined
}

/**
 * Decode JWT token to extract payload.
 * JWT tokens are base64url-encoded JSON, so we can decode them manually.
 */
function decodeJWT(token: string): JWTPayload | null {
  try {
    const parts = token.split('.')
    if (parts.length !== 3) {
      return null
    }
    // Decode the payload (second part)
    const payload = parts[1]
    // Replace URL-safe base64 characters
    const base64 = payload.replace(/-/g, '+').replace(/_/g, '/')
    // Add padding if needed
    const padded = base64 + '='.repeat((4 - (base64.length % 4)) % 4)
    // Decode base64
    const decoded = atob(padded)
    // Parse JSON
    return JSON.parse(decoded) as JWTPayload
  } catch {
    return null
  }
}

const ACCESS_TOKEN_EXPIRY_WARNING_MINUTES = 2 // Warn 2 minutes before expiry
const ACCESS_TOKEN_REFRESH_THRESHOLD_MINUTES = 5 // Refresh 5 minutes before expiry
const CHECK_INTERVAL_MS = 60000 // Check every minute

export function useSessionTimeout() {
  const { token, refreshToken, isAuthenticated, tryRefresh, logout } = useAuthStore()
  const { toast } = useToast()
  const navigate = useNavigate()
  const [timeUntilExpiry, setTimeUntilExpiry] = useState<number | null>(null)
  const [showWarning, setShowWarning] = useState(false)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const warningShownRef = useRef(false)

  const getTokenExpiry = (tokenString: string | null): number | null => {
    if (!tokenString) return null
    const decoded = decodeJWT(tokenString)
    if (!decoded || !decoded.exp) {
      return null
    }
    return decoded.exp * 1000 // Convert to milliseconds
  }

  const checkTokenExpiry = useCallback(async () => {
    if (!isAuthenticated || !token) {
      setTimeUntilExpiry(null)
      setShowWarning(false)
      return
    }

    const expiryTime = getTokenExpiry(token)
    if (!expiryTime) {
      return
    }

    const now = Date.now()
    const timeUntilExpiryMs = expiryTime - now
    const timeUntilExpiryMinutes = timeUntilExpiryMs / (1000 * 60)

    setTimeUntilExpiry(timeUntilExpiryMinutes)

    // If token has expired
    if (timeUntilExpiryMs <= 0) {
      // Try to refresh
      if (refreshToken) {
        const refreshed = await tryRefresh()
        if (!refreshed) {
          // Refresh failed, logout
          toast({
            title: 'Session Expired',
            description: 'Your session has expired. Please sign in again.',
            variant: 'destructive',
          })
          await logout()
          navigate('/login', { replace: true })
        }
      } else {
        // No refresh token, logout
        toast({
          title: 'Session Expired',
          description: 'Your session has expired. Please sign in again.',
          variant: 'destructive',
        })
        await logout()
        navigate('/login', { replace: true })
      }
      return
    }

    // Show warning if token is about to expire
    if (timeUntilExpiryMinutes <= ACCESS_TOKEN_EXPIRY_WARNING_MINUTES && !warningShownRef.current) {
      setShowWarning(true)
      warningShownRef.current = true
      toast({
        title: 'Session Expiring Soon',
        description: `Your session will expire in ${Math.ceil(timeUntilExpiryMinutes)} minute(s). Your session will be extended automatically.`,
      })
    }

    // Auto-refresh token if it's close to expiring
    if (timeUntilExpiryMinutes <= ACCESS_TOKEN_REFRESH_THRESHOLD_MINUTES && refreshToken) {
      const refreshed = await tryRefresh()
      if (refreshed) {
        warningShownRef.current = false
        setShowWarning(false)
        toast({
          title: 'Session Extended',
          description: 'Your session has been extended automatically.',
        })
      }
    }
  }, [isAuthenticated, token, refreshToken, tryRefresh, logout, navigate, toast])

  useEffect(() => {
    if (!isAuthenticated) {
      setTimeUntilExpiry(null)
      setShowWarning(false)
      warningShownRef.current = false
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
      return
    }

    // Initial check
    checkTokenExpiry()

    // Set up interval to check token expiry
    intervalRef.current = setInterval(checkTokenExpiry, CHECK_INTERVAL_MS)

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }, [isAuthenticated, checkTokenExpiry])

  // Reset warning when token changes (after refresh)
  useEffect(() => {
    if (token) {
      warningShownRef.current = false
      setShowWarning(false)
    }
  }, [token])

  return {
    timeUntilExpiry,
    showWarning,
    extendSession: async () => {
      if (refreshToken) {
        const refreshed = await tryRefresh()
        if (refreshed) {
          warningShownRef.current = false
          setShowWarning(false)
          toast({
            title: 'Session Extended',
            description: 'Your session has been extended successfully.',
          })
        } else {
          toast({
            title: 'Unable to Extend Session',
            description: 'Please sign in again.',
            variant: 'destructive',
          })
          await logout()
          navigate('/login', { replace: true })
        }
      }
    },
  }
}

