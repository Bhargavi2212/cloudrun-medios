/**
 * Hospital configuration system for multi-hospital support.
 * Provides hospital definitions and context for switching between hospitals.
 */

export interface HospitalConfig {
  id: string;
  name: string;
  apiUrls: {
    manage: string;
    scribe: string;
    summarizer: string;
    dol: string;
    federation: string;
  };
}

/**
 * Get API URLs from environment variables with fallback to localhost for development.
 */
const getApiUrl = (envVar: string | undefined, defaultPort: number): string => {
  if (envVar) {
    return envVar;
  }
  // Fallback to localhost for development
  if (typeof window !== "undefined" && window.location.hostname === "localhost") {
    return `http://localhost:${defaultPort}`;
  }
  return `http://localhost:${defaultPort}`;
};

/**
 * Available hospital configurations.
 * Uses environment variables for production, falls back to localhost for development.
 */
export const HOSPITALS: Record<string, HospitalConfig> = {
  "hospital-a": {
    id: "hospital-a",
    name: "City Hospital",
    apiUrls: {
      manage: getApiUrl(import.meta.env.VITE_MANAGE_API_URL_HOSPITAL_A, 8001),
      scribe: getApiUrl(import.meta.env.VITE_SCRIBE_API_URL_HOSPITAL_A, 8002),
      summarizer: getApiUrl(import.meta.env.VITE_SUMMARIZER_API_URL_HOSPITAL_A, 8003),
      dol: getApiUrl(import.meta.env.VITE_DOL_API_URL, 8004),
      federation: getApiUrl(import.meta.env.VITE_FEDERATION_API_URL, 8010),
    },
  },
  "hospital-b": {
    id: "hospital-b",
    name: "County Hospital",
    apiUrls: {
      manage: getApiUrl(import.meta.env.VITE_MANAGE_API_URL_HOSPITAL_B, 8011),
      scribe: getApiUrl(import.meta.env.VITE_SCRIBE_API_URL_HOSPITAL_B, 8012),
      summarizer: getApiUrl(import.meta.env.VITE_SUMMARIZER_API_URL_HOSPITAL_B, 8013),
      dol: getApiUrl(import.meta.env.VITE_DOL_API_URL, 8004), // Shared DOL orchestrator
      federation: getApiUrl(import.meta.env.VITE_FEDERATION_API_URL, 8010), // Shared federation aggregator
    },
  },
};

/**
 * Default hospital ID (Hospital A).
 */
export const DEFAULT_HOSPITAL_ID = "hospital-a";

/**
 * LocalStorage key for storing selected hospital ID.
 */
const STORAGE_KEY = "selectedHospitalId";

/**
 * Get the currently selected hospital ID from localStorage.
 * Returns default if not set or invalid.
 */
export function getSelectedHospitalId(): string {
  if (typeof window === "undefined") {
    return DEFAULT_HOSPITAL_ID;
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && stored in HOSPITALS) {
      return stored;
    }
  } catch (error) {
    console.warn("Failed to read hospital selection from localStorage:", error);
  }

  // Default to Hospital A if not set or invalid
  return DEFAULT_HOSPITAL_ID;
}

/**
 * Set the selected hospital ID in localStorage.
 */
export function setSelectedHospitalId(hospitalId: string): void {
  if (typeof window === "undefined") {
    return;
  }

  if (!(hospitalId in HOSPITALS)) {
    console.warn(`Invalid hospital ID: ${hospitalId}. Defaulting to ${DEFAULT_HOSPITAL_ID}`);
    hospitalId = DEFAULT_HOSPITAL_ID;
  }

  try {
    localStorage.setItem(STORAGE_KEY, hospitalId);
  } catch (error) {
    console.error("Failed to save hospital selection to localStorage:", error);
  }
}

/**
 * Get the hospital configuration for the currently selected hospital.
 */
export function getCurrentHospital(): HospitalConfig {
  const hospitalId = getSelectedHospitalId();
  return HOSPITALS[hospitalId] || HOSPITALS[DEFAULT_HOSPITAL_ID];
}

/**
 * Get hospital configuration by ID.
 */
export function getHospitalById(hospitalId: string): HospitalConfig | null {
  return HOSPITALS[hospitalId] || null;
}

/**
 * Initialize localStorage with default hospital if empty.
 * Should be called on app startup.
 */
export function initializeHospitalSelection(): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored || !(stored in HOSPITALS)) {
      // Initialize with default
      localStorage.setItem(STORAGE_KEY, DEFAULT_HOSPITAL_ID);
    }
  } catch (error) {
    console.error("Failed to initialize hospital selection:", error);
  }
}

/**
 * Get all available hospitals as an array.
 */
export function getAllHospitals(): HospitalConfig[] {
  return Object.values(HOSPITALS);
}

