import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Grid,
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
import UploadFileIcon from "@mui/icons-material/UploadFile";
import { AppLayout } from "../../layout/AppLayout";
import { useQueueData } from "../../hooks/useQueueData";
import { manageAPI, type NurseVitalsResponse, type QueuePatient } from "../../shared/services/api";
import { uploadDocument, listDocuments, deleteDocument, type FileAsset } from "../../shared/services/documentService";
import { generateSummary, fetchSummaryHistory, updateSummary, deleteSummary } from "../../shared/services/summarizerService";
import { PatientTimeline } from "../../shared/components/PatientTimeline";
import EditIcon from "@mui/icons-material/Edit";
import DeleteIcon from "@mui/icons-material/Delete";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import IconButton from "@mui/material/IconButton";
import { LoadingButton } from "@mui/lab";

type VitalsFormField = "hr" | "rr" | "sbp" | "dbp" | "temp_c" | "spo2" | "pain" | "notes";

type VitalsFormState = Record<VitalsFormField, string>;

const defaultVitalsForm: VitalsFormState = {
  hr: "",
  rr: "",
  sbp: "",
  dbp: "",
  temp_c: "",
  spo2: "",
  pain: "",
  notes: "",
};

const vitalFieldMeta: { key: VitalsFormField; label: string; type: "number" | "text"; helper?: string }[] = [
  { key: "hr", label: "Heart Rate (bpm)", type: "number" },
  { key: "rr", label: "Respiratory Rate (breaths/min)", type: "number" },
  { key: "sbp", label: "Systolic BP (mmHg)", type: "number" },
  { key: "dbp", label: "Diastolic BP (mmHg)", type: "number" },
  { key: "temp_c", label: "Temperature (°C)", type: "number" },
  { key: "spo2", label: "SpO₂ (%)", type: "number" },
  { key: "pain", label: "Pain Score (0-10)", type: "number" },
  { key: "notes", label: "Nurse Notes", type: "text", helper: "Optional observations or interventions" },
];

const vitalKeys = ["hr", "rr", "sbp", "dbp", "temp_c", "spo2", "pain"];

const hasFullVitals = (patient: QueuePatient): boolean => {
  if (!patient.vitals) return false;
  return vitalKeys.every((key) => typeof patient.vitals?.[key] === "number");
};

export const NurseDashboard = () => {
  const { patients, metrics, isLoading, isError, refetch } = useQueueData();
  const [selectedPatient, setSelectedPatient] = useState<QueuePatient | null>(null);
  const [vitalsForm, setVitalsForm] = useState<VitalsFormState>(defaultVitalsForm);
  const [isSubmittingVitals, setIsSubmittingVitals] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [triageResult, setTriageResult] = useState<NurseVitalsResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [uploadMessage, setUploadMessage] = useState<{ text: string; severity: "info" | "success" | "error" } | null>(null);
  const [editSummaryDialog, setEditSummaryDialog] = useState<{ open: boolean; summaryId: string; text: string }>({ open: false, summaryId: "", text: "" });
  const [deleteConfirmDialog, setDeleteConfirmDialog] = useState<{ open: boolean; type: "summary" | "document" | null; id: string; name: string }>({ open: false, type: null, id: "", name: "" });
  const fileInputRef = useRef<HTMLInputElement>(null);

  const awaitingVitals = useMemo(
    () => patients.filter((patient) => !hasFullVitals(patient)),
    [patients]
  );
  const triagedPatients = useMemo(
    () => patients.filter((patient) => hasFullVitals(patient)),
    [patients]
  );

  const {
    data: documents = [],
    isFetching: isFetchingDocuments,
    refetch: refetchDocuments,
  } = useQuery<FileAsset[]>({
    queryKey: ["patient-documents", selectedPatient?.patient_id],
    queryFn: () =>
      selectedPatient ? listDocuments({ patient_id: selectedPatient.patient_id }) : Promise.resolve([]),
    enabled: Boolean(selectedPatient),
  });

  const {
    data: summaryHistory = [],
    refetch: refetchSummaries,
  } = useQuery({
    queryKey: ["nurse-summary", selectedPatient?.patient_id],
    queryFn: () => (selectedPatient ? fetchSummaryHistory(selectedPatient.patient_id) : Promise.resolve([])),
    enabled: Boolean(selectedPatient),
  });

  const latestSummary = summaryHistory && summaryHistory.length > 0 ? summaryHistory[0] : null;

  useEffect(() => {
    if (!selectedPatient) return;
    const updated = patients.find((p) => p.consultation_id === selectedPatient.consultation_id);
    if (updated) {
      setSelectedPatient(updated);
      if (updated.vitals) {
      setVitalsForm({
        hr: updated.vitals.hr?.toString() ?? "",
        rr: updated.vitals.rr?.toString() ?? "",
        sbp: updated.vitals.sbp?.toString() ?? "",
        dbp: updated.vitals.dbp?.toString() ?? "",
        temp_c: updated.vitals.temp_c?.toString() ?? "",
        spo2: updated.vitals.spo2?.toString() ?? "",
        pain: updated.vitals.pain?.toString() ?? "",
        notes: typeof updated.vitals.notes === "string" ? updated.vitals.notes : "",
      });
      }
    }
  }, [patients, selectedPatient]);

  const handleSelectPatient = (patient: QueuePatient) => {
    setSelectedPatient(patient);
    setTriageResult(null);
    setSuccessMessage(null);
    setErrorMessage(null);
    if (patient.vitals) {
      setVitalsForm({
        hr: patient.vitals.hr?.toString() ?? "",
        rr: patient.vitals.rr?.toString() ?? "",
        sbp: patient.vitals.sbp?.toString() ?? "",
        dbp: patient.vitals.dbp?.toString() ?? "",
        temp_c: patient.vitals.temp_c?.toString() ?? "",
        spo2: patient.vitals.spo2?.toString() ?? "",
        pain: patient.vitals.pain?.toString() ?? "",
        notes: typeof patient.vitals.notes === "string" ? patient.vitals.notes : "",
      });
    } else {
      setVitalsForm(defaultVitalsForm);
    }
  };

  const handleVitalsFieldChange =
    (field: VitalsFormField) =>
    (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setVitalsForm((prev) => ({
        ...prev,
        [field]: event.target.value,
      }));
    };

  const prefillVitalsFromSelected = () => {
    if (!selectedPatient?.vitals) return;
    setVitalsForm({
      hr: selectedPatient.vitals.hr?.toString() ?? "",
      rr: selectedPatient.vitals.rr?.toString() ?? "",
      sbp: selectedPatient.vitals.sbp?.toString() ?? "",
      dbp: selectedPatient.vitals.dbp?.toString() ?? "",
      temp_c: selectedPatient.vitals.temp_c?.toString() ?? "",
      spo2: selectedPatient.vitals.spo2?.toString() ?? "",
      pain: selectedPatient.vitals.pain?.toString() ?? "",
      notes: typeof selectedPatient.vitals.notes === "string" ? selectedPatient.vitals.notes : "",
    });
  };

  const validateVitals = (): boolean => {
    for (const key of vitalKeys) {
      if (!vitalsForm[key as VitalsFormField] || vitalsForm[key as VitalsFormField].trim() === "") {
        setErrorMessage("All vital fields are required before running nurse triage.");
        return false;
      }
    }
    const painScore = Number(vitalsForm.pain);
    if (Number.isNaN(painScore) || painScore < 0 || painScore > 10) {
      setErrorMessage("Pain score must be a number between 0 and 10.");
      return false;
    }
    return true;
  };

  const handleVitalsSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedPatient) {
      setErrorMessage("Select a patient before recording vitals.");
      return;
    }
    if (!validateVitals()) return;

    setIsSubmittingVitals(true);
    setErrorMessage(null);
    try {
      const payload = {
        hr: Number(vitalsForm.hr),
        rr: Number(vitalsForm.rr),
        sbp: Number(vitalsForm.sbp),
        dbp: Number(vitalsForm.dbp),
        temp_c: Number(vitalsForm.temp_c),
        spo2: Number(vitalsForm.spo2),
        pain: Number(vitalsForm.pain),
        notes: vitalsForm.notes || undefined,
      };
      const response = await manageAPI.recordVitals(selectedPatient.consultation_id, payload);
      setTriageResult(response);
      setSuccessMessage(`Vitals saved. Updated triage level: ESI ${response.triage_level}`);
      setSelectedPatient((prev) =>
        prev
          ? {
              ...prev,
              triage_level: response.triage_level,
              vitals: response.vitals,
            }
          : prev
      );
      refetch();
      refetchDocuments();
      if (selectedPatient) {
        const highlight = `Vitals updated: HR ${payload.hr} bpm, RR ${payload.rr} breaths/min, BP ${payload.sbp}/${payload.dbp} mmHg, Temp ${payload.temp_c}°C, SpO₂ ${payload.spo2}%, Pain ${payload.pain}/10.`;
        generateSummary({
          patient_id: selectedPatient.patient_id,
          encounter_ids: [selectedPatient.consultation_id],
          highlights: [highlight],
        }).catch((summaryError) => {
          console.warn("Unable to generate summary from vitals update", summaryError);
        });
      }
      setTimeout(() => setSuccessMessage(null), 4000);
    } catch (error: unknown) {
      const message =
        error instanceof Error
          ? error.message
          : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || "Unable to record vitals. Please try again.";
      setErrorMessage(message);
    } finally {
      setIsSubmittingVitals(false);
    }
  };

  const handleDocumentUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    console.log("[UPLOAD] handleDocumentUpload called", event.target.files, selectedPatient);
    const file = event.target.files?.[0];
    if (!file || !selectedPatient) {
      console.log("[UPLOAD] Early return - file:", !!file, "selectedPatient:", !!selectedPatient);
      return;
    }
    console.log("[UPLOAD] Starting upload for file:", file.name, "patient:", selectedPatient.patient_id);
    setIsUploading(true);
    setUploadMessage(null);
    try {
      console.log("[UPLOAD] Calling uploadDocument API...");
      const result = await uploadDocument({
        file,
        patient_id: selectedPatient.patient_id,
        encounter_id: selectedPatient.consultation_id,
        upload_method: "nurse_dashboard",
      });
      console.log("[UPLOAD] Upload successful:", result);
      setUploadMessage({
        text: `${result.original_filename ?? "Document"} uploaded successfully.`,
        severity: "success",
      });
      console.log("[UPLOAD] Triggering summary generation...");
      generateSummary({
        patient_id: selectedPatient.patient_id,
        encounter_ids: [selectedPatient.consultation_id],
        highlights: [
          `Document uploaded: ${result.original_filename ?? "Attachment"} (${result.content_type ?? "file"})`,
        ],
      })
        .then(() => {
          console.log("[UPLOAD] Summary generation completed, refreshing summaries...");
          // Wait a moment for the backend to finish processing, then refetch
          setTimeout(() => {
            refetchSummaries();
          }, 2000);
        })
        .catch((summaryError) => {
          console.warn("[UPLOAD] Unable to generate summary after document upload", summaryError);
        });
      setTimeout(() => setUploadMessage(null), 4000);
      refetchDocuments();
      event.target.value = "";
    } catch (error: unknown) {
      console.error("[UPLOAD] Upload failed:", error);
      const errorObj = error as { message?: string; response?: { data?: { detail?: string }; status?: number } };
      console.error("[UPLOAD] Error details:", {
        message: errorObj?.message,
        response: errorObj?.response?.data,
        status: errorObj?.response?.status,
        fullError: error,
      });
      // Log the full error response for debugging
      if (errorObj?.response?.data) {
        console.error("[UPLOAD] Full error response:", JSON.stringify(errorObj.response.data, null, 2));
      }
      const message =
        error instanceof Error
          ? error.message
          : errorObj?.response?.data?.detail || "Unable to upload document.";
      setUploadMessage({ text: message, severity: "error" });
    } finally {
      setIsUploading(false);
    }
  };

  const renderQueueTable = (data: QueuePatient[], emptyLabel: string) => (
    <TableContainer component={Paper} variant="outlined">
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Patient</TableCell>
            <TableCell>Chief Complaint</TableCell>
            <TableCell>Triage</TableCell>
            <TableCell>Wait (min)</TableCell>
            <TableCell align="right">Action</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.length === 0 ? (
            <TableRow>
              <TableCell colSpan={4} align="center">
                {emptyLabel}
              </TableCell>
            </TableRow>
          ) : (
            data.map((patient) => (
              <TableRow key={patient.consultation_id} selected={selectedPatient?.consultation_id === patient.consultation_id}>
                <TableCell>
                  <Typography fontWeight={600}>{patient.patient_name}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Age: {patient.age ?? "—"}
                  </Typography>
                </TableCell>
                <TableCell>{patient.chief_complaint ?? "—"}</TableCell>
                <TableCell>
                  <Chip
                    size="small"
                    color={
                      patient.triage_level && patient.triage_level <= 2
                        ? "error"
                        : patient.triage_level === 3
                        ? "warning"
                        : "default"
                    }
                    label={patient.triage_level ? `ESI ${patient.triage_level}` : "Pending"}
                  />
                </TableCell>
                <TableCell>{patient.wait_time_minutes}</TableCell>
                <TableCell align="right">
                  <Button size="small" variant="outlined" onClick={() => handleSelectPatient(patient)}>
                    {selectedPatient?.consultation_id === patient.consultation_id ? "Selected" : "Select"}
                  </Button>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );

  return (
    <AppLayout>
      <Box>
        <Typography variant="h4" gutterBottom>
          Nurse Dashboard
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Prep station for capturing vitals, running nurse triage, and attaching historical records.
        </Typography>

        <Grid container spacing={2} sx={{ mb: 2 }}>
          {[
            { label: "Total in Queue", value: metrics.total },
            { label: "Awaiting Vitals", value: awaitingVitals.length },
            { label: "In Consultation", value: metrics.inConsultation },
            { label: "High Priority", value: metrics.highPriority },
          ].map((stat) => (
            <Grid item xs={6} md={3} key={stat.label}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h5" fontWeight={600}>
                    {stat.value}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {stat.label}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

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
        {uploadMessage && (
          <Alert severity={uploadMessage.severity} sx={{ mb: 2 }} onClose={() => setUploadMessage(null)}>
            {uploadMessage.text}
          </Alert>
        )}

        <Grid container spacing={3}>
          <Grid item xs={12} lg={6}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Patients Awaiting Vitals
                </Typography>
                {isError ? (
                  <Alert severity="error">Unable to load queue data.</Alert>
                ) : isLoading ? (
                  <Box sx={{ display: "flex", justifyContent: "center", p: 2 }}>
                    <CircularProgress size={32} />
                  </Box>
                ) : (
                  renderQueueTable(awaitingVitals, "All checked-in patients have vitals recorded.")
                )}
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recently Triaged Patients
                </Typography>
                {isLoading ? (
                  <Box sx={{ display: "flex", justifyContent: "center", p: 2 }}>
                    <CircularProgress size={32} />
                  </Box>
                ) : (
                  renderQueueTable(triagedPatients, "No triaged patients yet.")
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} lg={6}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Vitals Capture & Nurse Triage
                </Typography>
                {selectedPatient ? (
                  <Box component="form" onSubmit={handleVitalsSubmit}>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle1" fontWeight={600}>
                        {selectedPatient.patient_name}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {selectedPatient.chief_complaint || "No complaint provided"}
                      </Typography>
                      <Chip
                        size="small"
                        color={
                          selectedPatient.triage_level && selectedPatient.triage_level <= 2
                            ? "error"
                            : selectedPatient.triage_level === 3
                            ? "warning"
                            : "default"
                        }
                        sx={{ mt: 1 }}
                        label={
                          selectedPatient.triage_level
                            ? `Current triage: ESI ${selectedPatient.triage_level}`
                            : "Triage pending"
                        }
                      />
                      <Box sx={{ mt: 1, display: "flex", gap: 1, flexWrap: "wrap" }}>
                        {selectedPatient.vitals && typeof selectedPatient.vitals.injury === "boolean" && selectedPatient.vitals.injury && <Chip size="small" color="warning" label="Injury/Accident" />}
                        {selectedPatient.vitals && typeof selectedPatient.vitals.ambulance_arrival === "boolean" && selectedPatient.vitals.ambulance_arrival && (
                          <Chip size="small" color="info" label="Arrived via Ambulance" />
                        )}
                        {selectedPatient.vitals && typeof selectedPatient.vitals.seen_72h === "boolean" && selectedPatient.vitals.seen_72h && <Chip size="small" color="default" label="Seen in Last 72h" />}
                        <Chip size="small" label={`Wait: ${selectedPatient.wait_time_minutes} min`} />
                      </Box>
                    </Box>
                    <Grid container spacing={2}>
                      {vitalFieldMeta.map((field) => (
                        <Grid item xs={12} sm={field.key === "notes" ? 12 : 6} key={field.key}>
                          <TextField
                            fullWidth
                            label={field.label}
                            type={field.type === "number" ? "number" : "text"}
                            multiline={field.key === "notes"}
                            minRows={field.key === "notes" ? 2 : undefined}
                            value={vitalsForm[field.key]}
                            onChange={handleVitalsFieldChange(field.key)}
                            inputProps={field.key === "pain" ? { min: 0, max: 10 } : undefined}
                            helperText={field.helper}
                          />
                        </Grid>
                      ))}
                    </Grid>
                    <Box sx={{ display: "flex", gap: 1, mt: 2, flexWrap: "wrap" }}>
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={prefillVitalsFromSelected}
                        disabled={!selectedPatient?.vitals}
                      >
                        Load current vitals
                      </Button>
                      <Button
                        variant="text"
                        size="small"
                        onClick={() => setVitalsForm(defaultVitalsForm)}
                        disabled={Object.values(vitalsForm).every((value) => value === "")}
                      >
                        Clear form
                      </Button>
                    </Box>
                    <Button
                      type="submit"
                      variant="contained"
                      color="primary"
                      fullWidth
                      sx={{ mt: 2 }}
                      disabled={isSubmittingVitals}
                    >
                      {isSubmittingVitals ? "Saving..." : "Save Vitals & Run Triage"}
                    </Button>
                  </Box>
                ) : (
                  <Typography color="text.secondary">Select a patient from the queue to begin vitals.</Typography>
                )}
              </CardContent>
            </Card>

            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Upload Past Records
                </Typography>
                {selectedPatient ? (
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                    <Button
                      variant="outlined"
                      startIcon={<UploadFileIcon />}
                      disabled={isUploading}
                      onClick={() => {
                        console.log("[UPLOAD] Button clicked, triggering file input");
                        fileInputRef.current?.click();
                      }}
                    >
                      {isUploading ? "Uploading..." : "Attach File"}
                    </Button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf,.doc,.docx,.jpg,.jpeg,.png"
                      style={{ display: "none" }}
                      onChange={handleDocumentUpload}
                    />
                    <Typography variant="caption" color="text.secondary">
                      Supported formats: PDF, Word, or images. Files attach to the active encounter.
                    </Typography>
                  </Box>
                ) : (
                  <Typography color="text.secondary">Select a patient to upload supporting records.</Typography>
                )}
              </CardContent>
            </Card>

            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Uploaded Documents
                </Typography>
                {selectedPatient ? (
                  isFetchingDocuments ? (
                    <Box sx={{ display: "flex", justifyContent: "center", p: 2 }}>
                      <CircularProgress size={24} />
                    </Box>
                  ) : documents.length === 0 ? (
                    <Typography color="text.secondary">No records uploaded yet.</Typography>
                  ) : (
                    <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>File</TableCell>
                            <TableCell>Status</TableCell>
                            <TableCell>Uploaded</TableCell>
                            <TableCell align="right">Actions</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {documents.map((doc) => (
                            <TableRow key={doc.id}>
                              <TableCell>{doc.original_filename || "Untitled file"}</TableCell>
                              <TableCell>
                                <Chip
                                  size="small"
                                  label={doc.status}
                                  color={doc.status === "uploaded" ? "success" : doc.status === "completed" ? "primary" : "default"}
                                />
                              </TableCell>
                              <TableCell>{doc.created_at ? new Date(doc.created_at).toLocaleString() : "—"}</TableCell>
                              <TableCell align="right">
                                <IconButton
                                  size="small"
                                  onClick={() => setDeleteConfirmDialog({ open: true, type: "document", id: doc.id, name: doc.original_filename || "Document" })}
                                >
                                  <DeleteIcon fontSize="small" />
                                </IconButton>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  )
                ) : (
                  <Typography color="text.secondary">Select a patient to view uploaded records.</Typography>
                )}
              </CardContent>
            </Card>

            {triageResult && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Latest Triage Result
                  </Typography>
                  <Typography variant="subtitle1" fontWeight={600}>
                    ESI {triageResult.triage_level}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    {triageResult.explanation}
                  </Typography>
                </CardContent>
              </Card>
            )}

            {latestSummary && (
              <Card sx={{ mb: 3 }}>
                <CardContent>
                  <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
                    <Typography variant="h6">AI Summary</Typography>
                    <Box>
                      <IconButton
                        size="small"
                        onClick={() => setEditSummaryDialog({ open: true, summaryId: latestSummary.id, text: latestSummary.summary_text })}
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => setDeleteConfirmDialog({ open: true, type: "summary", id: latestSummary.id, name: "AI Summary" })}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  </Box>
                  {latestSummary.structured_data ? (
                    <PatientTimeline
                      data={latestSummary.structured_data}
                      onViewOriginal={(entry) => {
                        if (entry.source.original_file) {
                          // TODO: Implement view original document
                          console.log("View original:", entry.source.original_file);
                        }
                      }}
                    />
                  ) : (
                    <Typography variant="body2" sx={{ whiteSpace: "pre-line" }}>
                      {latestSummary.summary_text}
                    </Typography>
                  )}
                </CardContent>
              </Card>
            )}
          </Grid>
        </Grid>
      </Box>

      {/* Edit Summary Dialog */}
      <Dialog open={editSummaryDialog.open} onClose={() => setEditSummaryDialog({ open: false, summaryId: "", text: "" })} maxWidth="md" fullWidth>
        <DialogTitle>Edit AI Summary</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            multiline
            rows={8}
            value={editSummaryDialog.text}
            onChange={(e) => setEditSummaryDialog({ ...editSummaryDialog, text: e.target.value })}
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditSummaryDialog({ open: false, summaryId: "", text: "" })}>Cancel</Button>
          <LoadingButton
            variant="contained"
            onClick={async () => {
              try {
                await updateSummary(editSummaryDialog.summaryId, { summary_text: editSummaryDialog.text });
                refetchSummaries();
                setEditSummaryDialog({ open: false, summaryId: "", text: "" });
              } catch {
                setErrorMessage("Failed to update summary");
              }
            }}
          >
            Save
          </LoadingButton>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteConfirmDialog.open} onClose={() => setDeleteConfirmDialog({ open: false, type: null, id: "", name: "" })}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>Are you sure you want to delete &quot;{deleteConfirmDialog.name}&quot;? This action cannot be undone.</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmDialog({ open: false, type: null, id: "", name: "" })}>Cancel</Button>
          <LoadingButton
            variant="contained"
            color="error"
            onClick={async () => {
              try {
                if (deleteConfirmDialog.type === "summary") {
                  await deleteSummary(deleteConfirmDialog.id);
                  refetchSummaries();
                } else if (deleteConfirmDialog.type === "document") {
                  await deleteDocument(deleteConfirmDialog.id);
                  refetchDocuments();
                }
                setDeleteConfirmDialog({ open: false, type: null, id: "", name: "" });
              } catch {
                setErrorMessage(`Failed to delete ${deleteConfirmDialog.name}`);
              }
            }}
          >
            Delete
          </LoadingButton>
        </DialogActions>
      </Dialog>
    </AppLayout>
  );
};

