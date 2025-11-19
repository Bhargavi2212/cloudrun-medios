import { useState } from "react";
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  FormGroup,
  Grid,
  MenuItem,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import { AppLayout } from "../../layout/AppLayout";
import { useQueueData } from "../../hooks/useQueueData";
import { usePatientSearch } from "../../hooks/usePatientSearch";
import { usePatientCreate } from "../../hooks/usePatientCreate";
import { manageAPI, type Patient, type QueuePatient } from "../../shared/services/api";

const getTriageLabel = (level?: number | null) => {
  if (!level) return "Unknown";
  return `ESI ${level}`;
};

const formatMinutes = (minutes: number | null | undefined) =>
  typeof minutes === "number" ? `${Math.round(minutes)} min` : "—";

export const ReceptionistDashboard = () => {
  const { patients, metrics, isLoading, isError, refetch } = useQueueData();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [chiefComplaint, setChiefComplaint] = useState("");
  const [isCheckInPending, setIsCheckInPending] = useState(false);
  const [checkInFlags, setCheckInFlags] = useState({
    injury: false,
    ambulance_arrival: false,
    seen_72h: false,
  });
  const [isNewPatientOpen, setIsNewPatientOpen] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const { data: searchResults, isFetching: isSearching } = usePatientSearch(searchQuery);
  const createPatientMutation = usePatientCreate();

  const [newPatientForm, setNewPatientForm] = useState({
    mrn: "",
    first_name: "",
    last_name: "",
    date_of_birth: "",
    sex: "M" as "M" | "F" | "Other",
    contact_phone: "",
    contact_email: "",
  });

  const handlePatientSelect = (patient: Patient) => {
    setSelectedPatient(patient);
    setSearchQuery("");
  };

  const handleCheckIn = async () => {
    if (!selectedPatient || !chiefComplaint.trim()) {
      setErrorMessage("Check-in requires a patient and chief complaint.");
      return;
    }

    try {
      setIsCheckInPending(true);
      setErrorMessage(null);
      const response = await manageAPI.checkInPatient({
        patient_id: selectedPatient.id,
        chief_complaint: chiefComplaint.trim(),
        injury: checkInFlags.injury,
        ambulance_arrival: checkInFlags.ambulance_arrival,
        seen_72h: checkInFlags.seen_72h,
      });
      
      // Show success message with DOL profile status
      let successMsg = `${selectedPatient.first_name} ${selectedPatient.last_name} checked in successfully.`;
      if (response.dol_profile_found === true) {
        successMsg += " Retrieved history from network.";
      } else if (response.dol_profile_found === false) {
        // DOL query was attempted but patient not found
        successMsg += " (Partial network history - external source unavailable)";
      }
      setSuccessMessage(successMsg);
      setSelectedPatient(null);
      setChiefComplaint("");
      setCheckInFlags({
        injury: false,
        ambulance_arrival: false,
        seen_72h: false,
      });
      refetch();
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (error: unknown) {
      const message =
        error instanceof Error
          ? error.message
          : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || "Unable to check in patient";
      setErrorMessage(message);
    } finally {
      setIsCheckInPending(false);
    }
  };

  const handleCreatePatient = async () => {
    if (!newPatientForm.first_name || !newPatientForm.last_name) {
      setErrorMessage("First name and last name are required");
      return;
    }

    try {
      setErrorMessage(null);
      const payload = {
        mrn: newPatientForm.mrn.trim() || undefined, // Let backend auto-generate if empty
        first_name: newPatientForm.first_name.trim(),
        last_name: newPatientForm.last_name.trim(),
        dob: newPatientForm.date_of_birth || undefined,
        sex: newPatientForm.sex,
        contact_info: {
          phone: newPatientForm.contact_phone || undefined,
          email: newPatientForm.contact_email || undefined,
        },
      };
      const patient = await createPatientMutation.mutateAsync(payload);
      setSuccessMessage(`Patient ${patient.first_name} ${patient.last_name} created successfully.`);
      setSelectedPatient(patient);
      setIsNewPatientOpen(false);
      setNewPatientForm({
        mrn: "",
        first_name: "",
        last_name: "",
        date_of_birth: "",
        sex: "M",
        contact_phone: "",
        contact_email: "",
      });
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (error: unknown) {
      const message =
        error instanceof Error
          ? error.message
          : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || "Unable to create patient";
      setErrorMessage(message);
    }
  };

  return (
    <AppLayout>
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Receptionist Dashboard
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Real-time patient queue management
        </Typography>

        {errorMessage && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setErrorMessage(null)}>
            {errorMessage}
          </Alert>
        )}

        {successMessage && (
          <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccessMessage(null)}>
            {successMessage}
          </Alert>
        )}

        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="primary">
                  {metrics.total}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total in Queue
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="warning.main">
                  {metrics.awaitingVitals}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Awaiting Vitals
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="success.main">
                  {metrics.inConsultation}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  In Consultation
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="secondary">
                  {Math.round(metrics.averageWait)} min
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Average Wait
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Patient Check-In
                </Typography>

                <Box sx={{ mb: 2 }}>
                  <Autocomplete
                    options={searchResults || []}
                    getOptionLabel={(option) =>
                      `${option.first_name} ${option.last_name}`
                    }
                    loading={isSearching}
                    inputValue={searchQuery}
                    onInputChange={(_, newValue) => setSearchQuery(newValue)}
                    onChange={(_, newValue) => {
                      if (newValue) handlePatientSelect(newValue);
                    }}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Search existing patient"
                        placeholder="Type at least 2 characters..."
                        fullWidth
                      />
                    )}
                    renderOption={(props, option) => (
                      <Box component="li" {...props}>
                        <Box>
                          <Typography variant="body1">
                            {option.first_name} {option.last_name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            DOB:{" "}
                            {option.dob
                              ? new Date(option.dob).toLocaleDateString()
                              : "Unknown"}
                          </Typography>
                        </Box>
                      </Box>
                    )}
                  />
                </Box>

                <Button
                  variant="outlined"
                  fullWidth
                  onClick={() => setIsNewPatientOpen(true)}
                  sx={{ mb: 2 }}
                >
                  Register New Patient
                </Button>

                {selectedPatient && (
                  <Box
                    sx={{
                      p: 2,
                      bgcolor: "primary.50",
                      borderRadius: 1,
                      border: "1px solid",
                      borderColor: "primary.200",
                    }}
                  >
                    <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2 }}>
                      <Box>
                        <Typography variant="subtitle1" fontWeight="bold">
                          {selectedPatient.first_name} {selectedPatient.last_name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          DOB:{" "}
                          {selectedPatient.dob
                            ? new Date(selectedPatient.dob).toLocaleDateString()
                            : "Unknown"}
                        </Typography>
                      </Box>
                      <Button
                        size="small"
                        onClick={() => setSelectedPatient(null)}
                      >
                        Clear
                      </Button>
                    </Box>

                    <TextField
                      fullWidth
                      label="Chief Complaint"
                      multiline
                      rows={3}
                      value={chiefComplaint}
                      onChange={(e) => setChiefComplaint(e.target.value)}
                      placeholder="Describe the primary reason for the visit"
                      sx={{ mb: 2 }}
                    />
                    <FormGroup sx={{ mb: 2 }}>
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={checkInFlags.injury}
                            onChange={(event) =>
                              setCheckInFlags((prev) => ({
                                ...prev,
                                injury: event.target.checked,
                              }))
                            }
                          />
                        }
                        label="Visit is related to an injury or accident"
                      />
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={checkInFlags.ambulance_arrival}
                            onChange={(event) =>
                              setCheckInFlags((prev) => ({
                                ...prev,
                                ambulance_arrival: event.target.checked,
                              }))
                            }
                          />
                        }
                        label="Patient arrived via ambulance"
                      />
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={checkInFlags.seen_72h}
                            onChange={(event) =>
                              setCheckInFlags((prev) => ({
                                ...prev,
                                seen_72h: event.target.checked,
                              }))
                            }
                          />
                        }
                        label="Patient was seen here in the past 72 hours"
                      />
                    </FormGroup>

                    <Button
                      variant="contained"
                      fullWidth
                      onClick={handleCheckIn}
                      disabled={!chiefComplaint.trim() || isCheckInPending}
                    >
                      {isCheckInPending ? "Checking in..." : "Complete Check-In"}
                    </Button>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2 }}>
                  <Typography variant="h6">Live Patient Queue</Typography>
                  <Button size="small" onClick={() => refetch()}>
                    Refresh
                  </Button>
                </Box>

                {isError && (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    Unable to load queue data. Please try again shortly.
                  </Alert>
                )}

                {isLoading ? (
                  <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
                    <CircularProgress />
                  </Box>
                ) : (
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Patient</TableCell>
                          <TableCell>Chief Complaint</TableCell>
                          <TableCell>Status</TableCell>
                          <TableCell>Triage</TableCell>
                          <TableCell>Wait</TableCell>
                          <TableCell>Est. Wait</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {patients.length === 0 ? (
                          <TableRow>
                            <TableCell colSpan={6} align="center">
                              No patients currently in queue.
                            </TableCell>
                          </TableRow>
                        ) : (
                          patients.map((patient: QueuePatient) => (
                            <TableRow key={patient.consultation_id}>
                              <TableCell>{patient.patient_name}</TableCell>
                              <TableCell
                                sx={{
                                  maxWidth: 200,
                                  overflow: "hidden",
                                  textOverflow: "ellipsis",
                                  whiteSpace: "nowrap",
                                }}
                                title={patient.chief_complaint || undefined}
                              >
                                {patient.chief_complaint ?? "—"}
                              </TableCell>
                              <TableCell>
                                <Chip
                                  label={patient.status}
                                  size="small"
                                  color={
                                    patient.status === "waiting"
                                      ? "warning"
                                      : patient.status === "triage"
                                      ? "info"
                                      : patient.status === "scribe"
                                      ? "success"
                                      : "default"
                                  }
                                />
                              </TableCell>
                              <TableCell>
                                <Chip
                                  label={getTriageLabel(patient.triage_level)}
                                  size="small"
                                  color="primary"
                                />
                              </TableCell>
                              <TableCell>{formatMinutes(patient.wait_time_minutes)}</TableCell>
                              <TableCell>
                                {formatMinutes(patient.estimated_wait_minutes)}
                              </TableCell>
                            </TableRow>
                          ))
                        )}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Dialog open={isNewPatientOpen} onClose={() => setIsNewPatientOpen(false)} maxWidth="sm" fullWidth>
          <DialogTitle>Register New Patient</DialogTitle>
          <DialogContent>
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="First Name"
                  value={newPatientForm.first_name}
                  onChange={(e) =>
                    setNewPatientForm({ ...newPatientForm, first_name: e.target.value })
                  }
                  required
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Last Name"
                  value={newPatientForm.last_name}
                  onChange={(e) =>
                    setNewPatientForm({ ...newPatientForm, last_name: e.target.value })
                  }
                  required
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="MRN (Medical Record Number)"
                  value={newPatientForm.mrn}
                  onChange={(e) =>
                    setNewPatientForm({ ...newPatientForm, mrn: e.target.value })
                  }
                  placeholder="Leave empty to auto-generate"
                  helperText="Enter existing MRN for cross-hospital patient matching"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  type="date"
                  label="Date of Birth"
                  value={newPatientForm.date_of_birth}
                  onChange={(e) =>
                    setNewPatientForm({ ...newPatientForm, date_of_birth: e.target.value })
                  }
                  InputLabelProps={{ shrink: true }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  select
                  label="Sex"
                  value={newPatientForm.sex}
                  onChange={(e) =>
                    setNewPatientForm({
                      ...newPatientForm,
                      sex: e.target.value as "M" | "F" | "Other",
                    })
                  }
                >
                  <MenuItem value="M">Male</MenuItem>
                  <MenuItem value="F">Female</MenuItem>
                  <MenuItem value="Other">Other</MenuItem>
                </TextField>
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Phone"
                  value={newPatientForm.contact_phone}
                  onChange={(e) =>
                    setNewPatientForm({ ...newPatientForm, contact_phone: e.target.value })
                  }
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  type="email"
                  label="Email"
                  value={newPatientForm.contact_email}
                  onChange={(e) =>
                    setNewPatientForm({ ...newPatientForm, contact_email: e.target.value })
                  }
                />
              </Grid>
            </Grid>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setIsNewPatientOpen(false)}>Cancel</Button>
            <Button
              onClick={handleCreatePatient}
              variant="contained"
              disabled={createPatientMutation.isPending}
            >
              {createPatientMutation.isPending ? "Creating..." : "Create"}
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </AppLayout>
  );
};
