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
 * Available hospital configurations.
 */
export const HOSPITALS: Record<string, HospitalConfig> = {
  "hospital-a": {
    id: "hospital-a",
    name: "City Hospital",
    apiUrls: {
      manage: "http://localhost:8001",
      scribe: "http://localhost:8002",
      summarizer: "http://localhost:8003",
      dol: "http://localhost:8004",
      federation: "http://localhost:8010",
    },
  },
  "hospital-b": {
    id: "hospital-b",
    name: "County Hospital",
    apiUrls: {
      manage: "http://localhost:8011",
      scribe: "http://localhost:8012",
      summarizer: "http://localhost:8013",
      dol: "http://localhost:8004", // Shared DOL orchestrator
      federation: "http://localhost:8010", // Shared federation aggregator
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

