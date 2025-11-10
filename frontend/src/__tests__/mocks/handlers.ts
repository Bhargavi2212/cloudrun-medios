import { http, HttpResponse } from 'msw'

// Mock Service Worker handlers for API mocking
export const handlers = [
  // Auth endpoints
  http.post('/api/v1/auth/login', () => {
    return HttpResponse.json({
      success: true,
      data: {
        access_token: 'mock-access-token',
        refresh_token: 'mock-refresh-token',
        user: {
          id: '1',
          email: 'doctor@test.com',
          full_name: 'Test Doctor',
          role: 'doctor',
        },
      },
    })
  }),

  http.post('/api/v1/auth/register', () => {
    return HttpResponse.json({
      success: true,
      data: {
        access_token: 'mock-access-token',
        refresh_token: 'mock-refresh-token',
        user: {
          id: '1',
          email: 'newuser@test.com',
          full_name: 'New User',
          role: 'receptionist',
        },
      },
    })
  }),

  http.get('/api/v1/auth/me', () => {
    return HttpResponse.json({
      success: true,
      data: {
        id: '1',
        email: 'doctor@test.com',
        full_name: 'Test Doctor',
        role: 'doctor',
      },
    })
  }),

  // Queue endpoints
  http.get('/api/v1/queue', () => {
    return HttpResponse.json({
      success: true,
      data: {
        states: [],
        totals_by_stage: {
          waiting: 0,
          triage: 0,
          scribe: 0,
          discharge: 0,
        },
        average_wait_seconds: 0,
      },
    })
  }),

  // Patients endpoints
  http.get('/api/v1/patients', () => {
    return HttpResponse.json({
      success: true,
      data: [],
    })
  }),
]

