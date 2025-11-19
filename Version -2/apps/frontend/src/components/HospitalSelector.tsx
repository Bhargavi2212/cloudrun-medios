/**
 * Hospital selector component for switching between hospitals.
 * Includes ARIA labels for accessibility.
 */

import React from "react";
import { FormControl, InputLabel, Select, MenuItem, SelectChangeEvent, Box, Chip } from "@mui/material";
import { useHospital } from "../shared/contexts/HospitalContext";

export function HospitalSelector() {
  const { currentHospital, availableHospitals, switchHospital } = useHospital();

  const handleChange = (event: SelectChangeEvent<string>) => {
    const newHospitalId = event.target.value;
    switchHospital(newHospitalId);
  };

  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
      <FormControl
        size="small"
        sx={{ minWidth: 200 }}
        aria-label="Hospital selection"
        aria-describedby="hospital-selector-description"
      >
        <InputLabel id="hospital-selector-label">Hospital</InputLabel>
        <Select
          labelId="hospital-selector-label"
          id="hospital-selector"
          value={currentHospital.id}
          label="Hospital"
          onChange={handleChange}
          aria-label="Select hospital"
          aria-describedby="hospital-selector-description"
        >
          {availableHospitals.map((hospital) => (
            <MenuItem key={hospital.id} value={hospital.id}>
              {hospital.name}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <Chip
        label={currentHospital.name}
        color="primary"
        size="small"
        aria-label={`Current hospital: ${currentHospital.name}`}
      />
      <Box
        id="hospital-selector-description"
        sx={{ display: "none" }}
        aria-live="polite"
        aria-atomic="true"
      >
        Currently viewing {currentHospital.name}. Use the dropdown to switch hospitals.
      </Box>
    </Box>
  );
}

