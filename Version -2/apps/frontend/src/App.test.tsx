import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider, CssBaseline } from "@mui/material";

import App from "./App";
import { appTheme } from "./theme";
import { useAuthStore } from "./store/authStore";

vi.mock("./store/authStore", () => ({
  useAuthStore: vi.fn(),
}));

vi.mock("./shared/services/manageService", () => {
  const mockPatients = [
    {
      id: "patient-1",
      mrn: "MRN-123",
      first_name: "Jane",
      last_name: "Doe",
      dob: null,
      sex: null,
      contact_info: null,
      created_at: "2025-01-01T00:00:00.000Z",
      updated_at: "2025-01-01T00:00:00.000Z",
    },
  ];

  const mockPortableProfile = {
    patient: mockPatients[0],
    timeline: [],
    summaries: [],
    sources: ["local"],
  };

  return {
    fetchPatients: vi.fn(async () => mockPatients),
    createPatient: vi.fn(),
    classifyTriage: vi.fn(),
    checkInPatient: vi.fn(async () => mockPortableProfile),
  };
});

describe("App", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders login page when not authenticated", () => {
    (useAuthStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      user: null,
      token: null,
      isAuthenticated: false,
      getDefaultRoute: () => "/receptionist/dashboard",
    });

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    render(
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={appTheme}>
          <CssBaseline />
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </ThemeProvider>
      </QueryClientProvider>,
    );

    expect(screen.getByText(/Medi OS Portal/i)).toBeInTheDocument();
    expect(screen.getByText(/Sign in to your account/i)).toBeInTheDocument();
  });
});
