import { useSessionTimeout } from '@/hooks/useSessionTimeout'

/**
 * SessionTimeoutWarning component that monitors token expiry and shows warnings.
 * This component uses the useSessionTimeout hook which handles automatic token refresh
 * and displays toast notifications.
 */
export function SessionTimeoutWarning() {
  // This hook handles all session timeout logic including warnings via toast
  useSessionTimeout()
  
  // Component doesn't render anything - all warnings are handled via toast in the hook
  return null
}

