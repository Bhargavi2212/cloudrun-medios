/**
 * React context for hospital selection and management.
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import {
  HospitalConfig,
  setSelectedHospitalId,
  getCurrentHospital,
  getAllHospitals,
  initializeHospitalSelection,
  HOSPITALS,
} from "../config/hospitalConfig";
import { useQueryClient } from "@tanstack/react-query";

interface HospitalContextType {
  currentHospital: HospitalConfig;
  availableHospitals: HospitalConfig[];
  switchHospital: (hospitalId: string) => void;
  refreshHospital: () => void;
}

const HospitalContext = createContext<HospitalContextType | undefined>(undefined);

interface HospitalProviderProps {
  children: React.ReactNode;
}

/**
 * Provider component for hospital context.
 * Manages hospital selection state and provides it to child components.
 */
export function HospitalProvider({ children }: HospitalProviderProps) {
  const queryClient = useQueryClient();

  // Initialize hospital selection on mount
  useEffect(() => {
    initializeHospitalSelection();
  }, []);

  const [currentHospital, setCurrentHospital] = useState<HospitalConfig>(getCurrentHospital());

  const switchHospital = useCallback(
    (hospitalId: string) => {
      if (!(hospitalId in HOSPITALS)) {
        console.warn(`Invalid hospital ID: ${hospitalId}`);
        return;
      }

      setSelectedHospitalId(hospitalId);
      setCurrentHospital(getCurrentHospital());

      // Invalidate all React Query caches when hospital switches
      // This ensures fresh data is always shown for the new hospital
      queryClient.clear();
    },
    [queryClient]
  );

  const refreshHospital = useCallback(() => {
    setCurrentHospital(getCurrentHospital());
  }, []);

  const value: HospitalContextType = {
    currentHospital,
    availableHospitals: getAllHospitals(),
    switchHospital,
    refreshHospital,
  };

  return <HospitalContext.Provider value={value}>{children}</HospitalContext.Provider>;
}

/**
 * Hook to access hospital context.
 * Throws error if used outside HospitalProvider.
 */
export function useHospital(): HospitalContextType {
  const context = useContext(HospitalContext);
  if (context === undefined) {
    throw new Error("useHospital must be used within a HospitalProvider");
  }
  return context;
}

