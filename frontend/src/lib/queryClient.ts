import { QueryClient } from '@tanstack/react-query';

interface ErrorWithResponse {
  response?: {
    status?: number;
  };
}

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: (failureCount, error: unknown) => {
        const err = error as ErrorWithResponse;
        // Don't retry on 4xx errors except 408, 429
        if (err?.response?.status && err.response.status >= 400 && err.response.status < 500) {
          if (err.response.status === 408 || err.response.status === 429) {
            return failureCount < 2;
          }
          return false;
        }
        // Retry on network errors and 5xx errors
        return failureCount < 3;
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
    mutations: {
      retry: (failureCount, error: unknown) => {
        const err = error as ErrorWithResponse;
        // Don't retry mutations on 4xx errors
        if (err?.response?.status && err.response.status >= 400 && err.response.status < 500) {
          return false;
        }
        // Retry on network errors and 5xx errors
        return failureCount < 2;
      },
    },
  },
});