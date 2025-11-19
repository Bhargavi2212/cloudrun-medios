import { useEffect, useMemo, useRef, useState } from "react";
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
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import { LoadingButton } from "@mui/lab";

import { AppLayout } from "../../layout/AppLayout";
import { useQueueData } from "../../hooks/useQueueData";
import { type QueuePatient } from "../../shared/services/api";
import { listDocuments, deleteDocument, type FileAsset } from "../../shared/services/documentService";
import { fetchSummaryHistory, updateSummary, deleteSummary } from "../../shared/services/summarizerService";
import { createTranscript, generateSoap, updateSoapNote, deleteSoapNote } from "../../shared/services/scribeService";
import { generateSummary } from "../../shared/services/summarizerService";
import type { SoapNoteResponse } from "../../shared/types/api";
import { PatientTimeline } from "../../shared/components/PatientTimeline";
import EditIcon from "@mui/icons-material/Edit";
import DeleteIcon from "@mui/icons-material/Delete";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import IconButton from "@mui/material/IconButton";

// Speech Recognition API types (browser-specific)
interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  abort: () => void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
}

interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
}

interface SpeechRecognitionResultList {
  length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
  isFinal: boolean;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

const formatVitals = (vitals?: Record<string, unknown> | null): Array<{ label: string; value: string | number }> => {
  if (!vitals) return [];
  const displayOrder: Array<{ key: string; label: string; unit?: string }> = [
    { key: "hr", label: "Heart Rate", unit: "bpm" },
    { key: "rr", label: "Respiratory Rate", unit: "breaths/min" },
    { key: "sbp", label: "Systolic BP", unit: "mmHg" },
    { key: "dbp", label: "Diastolic BP", unit: "mmHg" },
    { key: "temp_c", label: "Temperature", unit: "°C" },
    { key: "spo2", label: "SpO₂", unit: "%" },
    { key: "pain", label: "Pain Score" },
  ];
  return displayOrder
    .filter(({ key }) => typeof vitals[key] === "number")
    .map(({ key, label, unit }) => {
      const val = vitals[key] as number;
      return {
        label,
        value: unit ? `${val} ${unit}` : val,
      };
    });
};

export const DoctorDashboard = () => {
  const { patients, metrics, isLoading, isError, refetch } = useQueueData();
  const [selectedPatient, setSelectedPatient] = useState<QueuePatient | null>(null);
  const [transcript, setTranscript] = useState("");
  const [soapNote, setSoapNote] = useState<SoapNoteResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [isGeneratingSoap, setIsGeneratingSoap] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const [isListening, setIsListening] = useState(false);
  const manuallyStoppedRef = useRef<boolean>(false);
  const shouldKeepListeningRef = useRef<boolean>(false);
  const finalTranscriptRef = useRef<string>("");
  const [editSummaryDialog, setEditSummaryDialog] = useState<{ open: boolean; summaryId: string; text: string }>({ open: false, summaryId: "", text: "" });
  const [editSoapDialog, setEditSoapDialog] = useState<{ open: boolean; note: SoapNoteResponse | null }>({ open: false, note: null });
  const [deleteConfirmDialog, setDeleteConfirmDialog] = useState<{ open: boolean; type: "summary" | "soap" | "document" | null; id: string; name: string }>({ open: false, type: null, id: "", name: "" });

  useEffect(() => {
    if (!selectedPatient && patients.length > 0) {
      setSelectedPatient(patients[0]);
    }
  }, [patients, selectedPatient]);

  const queueForDoctors = useMemo(() => patients.filter((patient) => patient.status !== "discharge"), [patients]);

  const {
    data: documents = [],
    isFetching: isFetchingDocuments,
    refetch: refetchDocuments,
  } = useQuery<FileAsset[]>({
    queryKey: ["doctor-documents", selectedPatient?.patient_id, selectedPatient?.consultation_id],
    queryFn: () =>
      selectedPatient
        ? listDocuments({
            patient_id: selectedPatient.patient_id,
            encounter_id: selectedPatient.consultation_id,
          })
        : Promise.resolve([]),
    enabled: Boolean(selectedPatient),
  });

  const {
    data: summaryHistory = [],
    isFetching: isFetchingSummaries,
    refetch: refetchSummaries,
  } = useQuery({
    queryKey: ["doctor-summary", selectedPatient?.patient_id],
    queryFn: () => (selectedPatient ? fetchSummaryHistory(selectedPatient.patient_id) : Promise.resolve([])),
    enabled: Boolean(selectedPatient),
    // Removed refetchInterval - only fetch when explicitly needed (consultation complete, SOAP note generated, etc.)
    staleTime: 5 * 60 * 1000, // Consider stale after 5 minutes (summary doesn't change frequently)
  });

  const latestSummary = summaryHistory && summaryHistory.length > 0 ? summaryHistory[0] : null;

  const handleSelectPatient = (patient: QueuePatient) => {
    setSelectedPatient(patient);
    setTranscript("");
    setSoapNote(null);
    setErrorMessage(null);
    setInfoMessage(null);
  };

  const handleStartConsultation = () => {
    if (!selectedPatient) return;
    setTranscript("");
    finalTranscriptRef.current = "";
    setSoapNote(null);
    stopListening();
    setInfoMessage("Consultation session started. Click 'Start Listening' to begin AI Scribe.");
  };

  const handleGenerateSoap = async () => {
    if (!selectedPatient || !transcript.trim()) {
      setErrorMessage("Enter a transcript before generating a note.");
      return;
    }

    setErrorMessage(null);
    setInfoMessage(null);
    setIsGeneratingSoap(true);
    try {
      await createTranscript({ encounter_id: selectedPatient.consultation_id, transcript });
      const soap = await generateSoap(selectedPatient.consultation_id, transcript);
      setSoapNote(soap);
      setInfoMessage("SOAP note generated successfully with AI Scribe.");
      const highlights = [soap.assessment ?? "", soap.plan ?? ""].filter((text) => text && text.trim()) as string[];
      if (highlights.length > 0) {
        await generateSummary({
          patient_id: selectedPatient.patient_id,
          encounter_ids: [selectedPatient.consultation_id],
          highlights,
        });
        refetchSummaries();
      }
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail || "Failed to generate SOAP note.";
      setErrorMessage(message);
    } finally {
      setIsGeneratingSoap(false);
    }
  };

  const startListening = () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setErrorMessage("Browser does not support live speech recognition. Please type the transcript manually.");
      return;
    }
    if (isListening) return;
    
    manuallyStoppedRef.current = false;
    shouldKeepListeningRef.current = true;
    // Preserve existing transcript - don't reset it on restart
    if (!finalTranscriptRef.current && transcript.trim()) {
      finalTranscriptRef.current = transcript;
    } else if (!transcript.trim() && finalTranscriptRef.current) {
      // If transcript state is empty but ref has content, restore it
      setTranscript(finalTranscriptRef.current);
    } else {
      // Sync ref with current state
      finalTranscriptRef.current = transcript;
    }
    
    try {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";
      recognitionRef.current = recognition;
      
      recognition.onresult = (event: SpeechRecognitionEvent) => {
        let interimTranscript = "";
        // Always start from the current transcript state to preserve previous content
        let finalTranscript = transcript || finalTranscriptRef.current || "";
        
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const transcriptText = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += (finalTranscript ? " " : "") + transcriptText;
            finalTranscriptRef.current = finalTranscript;
          } else {
            interimTranscript += transcriptText;
          }
        }
        
        // Update transcript with final + current interim
        const displayText = finalTranscript + (interimTranscript ? ` ${interimTranscript}` : "");
        if (displayText.trim()) {
          setTranscript(displayText.trim());
        }
      };
      
      recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error("Speech recognition error", event);
        // Don't stop on recoverable errors - let it auto-restart
        if (event.error === "no-speech" || event.error === "audio-capture" || event.error === "aborted") {
          // These are common and we can continue - onend will handle restart
          return;
        }
        // For other errors, log but don't stop - let it auto-restart
        console.warn("Speech recognition error (will auto-restart):", event.error);
      };
      
      recognition.onend = () => {
        recognitionRef.current = null;
        setIsListening(false);
        
        // Preserve current transcript before restarting
        // Sync finalTranscriptRef with current transcript state to preserve it
        if (transcript.trim()) {
          finalTranscriptRef.current = transcript;
        }
        
        // Always auto-restart unless doctor manually stopped it
        if (shouldKeepListeningRef.current && !manuallyStoppedRef.current) {
          console.log("Speech recognition ended (silence/timeout), auto-restarting...");
          setTimeout(() => {
            // Double-check we should still be listening
            if (shouldKeepListeningRef.current && !manuallyStoppedRef.current) {
              // Preserve transcript by syncing ref before restart
              finalTranscriptRef.current = transcript;
              startListening();
            }
          }, 100);
        } else {
          console.log("Speech recognition stopped (manually stopped)");
        }
      };
      
      recognition.start();
      setInfoMessage("AI Scribe microphone is listening... (will continue until you click Stop)");
      setIsListening(true);
    } catch (error) {
      console.error("Unable to start speech recognition", error);
      setErrorMessage("Unable to start speech recognition.");
    }
  };

  const stopListening = () => {
    manuallyStoppedRef.current = true;
    shouldKeepListeningRef.current = false;
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {
        // Ignore errors when stopping
      }
      recognitionRef.current = null;
    }
    setIsListening(false);
    setInfoMessage("AI Scribe stopped.");
  };

  useEffect(
    () => () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    },
    []
  );

  const renderQueueTable = (data: QueuePatient[], emptyMessage: string) => (
    <TableContainer component={Paper} variant="outlined">
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Patient</TableCell>
            <TableCell>Chief Complaint</TableCell>
            <TableCell>Triage</TableCell>
            <TableCell>Wait</TableCell>
            <TableCell align="right">Action</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {data.length === 0 ? (
            <TableRow>
              <TableCell colSpan={5} align="center">
                {emptyMessage}
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
                <TableCell>{patient.wait_time_minutes} min</TableCell>
                <TableCell align="right">
                  <Button size="small" onClick={() => handleSelectPatient(patient)}>
                    {selectedPatient?.consultation_id === patient.consultation_id ? "Selected" : "View"}
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
          Doctor Dashboard
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Review vitals, past records, AI summaries, and capture consultation notes with AI Scribe.
        </Typography>

        {errorMessage && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setErrorMessage(null)}>
            {errorMessage}
          </Alert>
        )}
        {infoMessage && (
          <Alert severity="info" sx={{ mb: 2 }} onClose={() => setInfoMessage(null)}>
            {infoMessage}
          </Alert>
        )}

        <Grid container spacing={3}>
          <Grid item xs={12} lg={5}>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" color="primary">
                      {metrics.total}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      In Queue
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" color="warning.main">
                      {metrics.highPriority}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      High Priority
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
                      Avg Wait
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2 }}>
                  <Typography variant="h6">Patients</Typography>
                  <Button size="small" onClick={() => refetch()}>
                    Refresh
                  </Button>
                </Box>
                {isError ? (
                  <Alert severity="error">Unable to load queue data.</Alert>
                ) : isLoading ? (
                  <Box sx={{ display: "flex", justifyContent: "center", p: 2 }}>
                    <CircularProgress size={32} />
                  </Box>
                ) : (
                  renderQueueTable(queueForDoctors, "No patients waiting for consultation.")
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} lg={7}>
            {selectedPatient ? (
              <Stack spacing={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <Box>
                        <Typography variant="h6">{selectedPatient.patient_name}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Chief complaint: {selectedPatient.chief_complaint || "—"}
                        </Typography>
                      </Box>
                      <Button variant="outlined" onClick={handleStartConsultation}>
                        Start Consultation
                      </Button>
                    </Box>
                    <Box sx={{ mt: 1, display: "flex", gap: 1, flexWrap: "wrap" }}>
                      <Chip
                        label={selectedPatient.triage_level ? `ESI ${selectedPatient.triage_level}` : "ESI pending"}
                        color={
                          selectedPatient.triage_level && selectedPatient.triage_level <= 2
                            ? "error"
                            : selectedPatient.triage_level === 3
                            ? "warning"
                            : "default"
                        }
                      />
                      {selectedPatient.vitals && typeof selectedPatient.vitals.ambulance_arrival === "boolean" && selectedPatient.vitals.ambulance_arrival && <Chip size="small" label="Arrived via ambulance" color="info" />}
                      {selectedPatient.vitals && typeof selectedPatient.vitals.injury === "boolean" && selectedPatient.vitals.injury && <Chip size="small" label="Injury case" color="warning" />}
                    </Box>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Latest Vitals
                    </Typography>
                    {selectedPatient.vitals ? (
                      <Table size="small">
                        <TableBody>
                          {formatVitals(selectedPatient.vitals).map((item) => (
                            <TableRow key={item.label}>
                              <TableCell sx={{ border: 0, width: "40%" }}>{item.label}</TableCell>
                              <TableCell sx={{ border: 0 }}>{item.value}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    ) : (
                      <Typography color="text.secondary">Vitals not captured yet.</Typography>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Uploaded Records
                    </Typography>
                    {isFetchingDocuments ? (
                      <Box sx={{ display: "flex", justifyContent: "center", p: 2 }}>
                        <CircularProgress size={24} />
                      </Box>
                    ) : documents.length === 0 ? (
                      <Typography color="text.secondary">No past records for this encounter yet.</Typography>
                    ) : (
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
                                <Chip size="small" label={doc.status} color={doc.status === "uploaded" ? "success" : "default"} />
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
                    )}
                  </CardContent>
                </Card>

                <Card
                  sx={{
                    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    color: "white",
                    boxShadow: 3,
                  }}
                >
                  <CardContent>
                    <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2, pb: 1.5, borderBottom: "1px solid rgba(255, 255, 255, 0.2)" }}>
                      <Typography variant="h6" sx={{ color: "white", fontWeight: 600 }}>
                        AI Summary
                      </Typography>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                        {latestSummary && (
                          <Typography variant="caption" sx={{ color: "rgba(255, 255, 255, 0.8)", fontSize: "0.75rem" }}>
                            {latestSummary.created_at
                              ? `Updated ${new Date(latestSummary.created_at).toLocaleString()}`
                              : "Just now"}
                          </Typography>
                        )}
                        <Button
                          size="small"
                          onClick={() => refetchSummaries()}
                          disabled={isFetchingSummaries}
                          sx={{
                            minWidth: "auto",
                            color: "white",
                            "&:hover": { bgcolor: "rgba(255, 255, 255, 0.2)" },
                          }}
                        >
                          {isFetchingSummaries ? <CircularProgress size={16} sx={{ color: "white" }} /> : "🔄"}
                        </Button>
                      </Box>
                    </Box>
                    {isFetchingSummaries ? (
                      <Box sx={{ display: "flex", justifyContent: "center", p: 3 }}>
                        <CircularProgress size={32} sx={{ color: "white" }} />
                      </Box>
                    ) : latestSummary ? (
                      <Box>
                        {latestSummary.structured_data ? (
                          <Box sx={{ bgcolor: "rgba(255, 255, 255, 0.05)", borderRadius: 1, p: 1 }}>
                            <PatientTimeline
                              data={latestSummary.structured_data}
                              onViewOriginal={(entry) => {
                                if (entry.source.original_file) {
                                  // TODO: Implement view original document
                                  console.log("View original:", entry.source.original_file);
                                }
                              }}
                            />
                          </Box>
                        ) : (
                          <>
                            {latestSummary.encounter_ids && latestSummary.encounter_ids.length > 0 && (
                              <Box
                                sx={{
                                  mb: 2,
                                  p: 1.5,
                                  bgcolor: "rgba(255, 255, 255, 0.1)",
                                  borderRadius: 1,
                                  backdropFilter: "blur(10px)",
                                }}
                              >
                                <Typography variant="subtitle2" sx={{ color: "rgba(255, 255, 255, 0.9)", mb: 0.5, fontWeight: 600 }}>
                                  Recent Encounters
                                </Typography>
                                <Typography variant="body2" sx={{ color: "rgba(255, 255, 255, 0.85)", fontSize: "0.875rem" }}>
                                  {latestSummary.encounter_ids.length} encounter{latestSummary.encounter_ids.length !== 1 ? "s" : ""} included
                                </Typography>
                              </Box>
                            )}
                            <Box
                              sx={{
                                mb: 2,
                                p: 1.5,
                                bgcolor: "rgba(255, 255, 255, 0.1)",
                                borderRadius: 1,
                                backdropFilter: "blur(10px)",
                              }}
                            >
                              <Typography variant="subtitle2" sx={{ color: "rgba(255, 255, 255, 0.9)", mb: 1, fontWeight: 600 }}>
                                Clinical Summary
                              </Typography>
                              <Typography
                                variant="body2"
                                sx={{
                                  color: "rgba(255, 255, 255, 0.9)",
                                  whiteSpace: "pre-line",
                                  lineHeight: 1.6,
                                  fontSize: "0.875rem",
                                }}
                              >
                                {latestSummary.summary_text || "Summary available but empty."}
                              </Typography>
                            </Box>
                            {latestSummary.model_version && (
                              <Box sx={{ mt: 2, pt: 1.5, borderTop: "1px solid rgba(255, 255, 255, 0.2)" }}>
                                <Typography variant="caption" sx={{ color: "rgba(255, 255, 255, 0.7)", fontSize: "0.7rem" }}>
                                  Model: {latestSummary.model_version} • Confidence:{" "}
                                  {latestSummary.confidence_score ? `${Math.round(latestSummary.confidence_score * 100)}%` : "N/A"}
                                </Typography>
                              </Box>
                            )}
                          </>
                        )}
                      </Box>
                    ) : (
                      <Box
                        sx={{
                          p: 3,
                          textAlign: "center",
                          bgcolor: "rgba(255, 255, 255, 0.1)",
                          borderRadius: 1,
                        }}
                      >
                        <Typography variant="body2" sx={{ color: "rgba(255, 255, 255, 0.8)", mb: 1 }}>
                          No AI summary generated yet for this patient.
                        </Typography>
                        <Typography variant="caption" sx={{ color: "rgba(255, 255, 255, 0.6)", fontSize: "0.75rem" }}>
                          Summary will appear after vitals, documents, or SOAP notes are added.
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      AI Scribe
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      Capture or paste the consultation transcript and let the AI generate a SOAP note.
                    </Typography>
                    <Stack spacing={2}>
                      <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", alignItems: "center" }}>
                        <Button
                          variant="outlined"
                          color={isListening ? "error" : "primary"}
                          onClick={isListening ? stopListening : startListening}
                        >
                          {isListening ? "Stop Listening" : "Start Listening"}
                        </Button>
                        {isListening && <Chip size="small" color="info" label="Listening..." />}
                      </Box>
                      <TextField
                        label="Encounter ID"
                        value={selectedPatient.consultation_id}
                        disabled
                        helperText="Encounter selected from queue"
                      />
                      <TextField
                        label="Transcript"
                        value={transcript}
                        onChange={(event) => setTranscript(event.target.value)}
                        multiline
                        minRows={4}
                        placeholder="Doctor: Tell me about your symptoms..."
                      />
                      {soapNote && (
                        <Box>
                          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
                            <Typography variant="subtitle2">
                              Latest SOAP Note (model {soapNote.model_version})
                            </Typography>
                            <Box>
                              <IconButton
                                size="small"
                                onClick={() => setEditSoapDialog({ open: true, note: soapNote })}
                                sx={{ mr: 0.5 }}
                              >
                                <EditIcon fontSize="small" />
                              </IconButton>
                              <IconButton
                                size="small"
                                onClick={() => setDeleteConfirmDialog({ open: true, type: "soap", id: soapNote.id, name: "SOAP Note" })}
                              >
                                <DeleteIcon fontSize="small" />
                              </IconButton>
                            </Box>
                          </Box>
                          <Typography variant="body2" sx={{ whiteSpace: "pre-line" }}>
                            <strong>Subjective:</strong> {soapNote.subjective || "—"}
                            {"\n"}
                            <strong>Objective:</strong> {soapNote.objective || "—"}
                            {"\n"}
                            <strong>Assessment:</strong> {soapNote.assessment || "—"}
                            {"\n"}
                            <strong>Plan:</strong> {soapNote.plan || "—"}
                          </Typography>
                        </Box>
                      )}
                      <LoadingButton
                        variant="contained"
                        onClick={handleGenerateSoap}
                        loading={isGeneratingSoap}
                        disabled={!transcript.trim()}
                      >
                        Generate SOAP Note
                      </LoadingButton>
                    </Stack>
                  </CardContent>
                </Card>
              </Stack>
            ) : (
              <Card>
                <CardContent>
                  <Typography color="text.secondary">Select a patient from the queue to view consultation tools.</Typography>
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

      {/* Edit SOAP Note Dialog */}
      <Dialog open={editSoapDialog.open} onClose={() => setEditSoapDialog({ open: false, note: null })} maxWidth="md" fullWidth>
        <DialogTitle>Edit SOAP Note</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField
              fullWidth
              label="Subjective"
              multiline
              rows={3}
              value={editSoapDialog.note?.subjective || ""}
              onChange={(e) => setEditSoapDialog({ ...editSoapDialog, note: editSoapDialog.note ? { ...editSoapDialog.note, subjective: e.target.value } : null })}
            />
            <TextField
              fullWidth
              label="Objective"
              multiline
              rows={3}
              value={editSoapDialog.note?.objective || ""}
              onChange={(e) => setEditSoapDialog({ ...editSoapDialog, note: editSoapDialog.note ? { ...editSoapDialog.note, objective: e.target.value } : null })}
            />
            <TextField
              fullWidth
              label="Assessment"
              multiline
              rows={3}
              value={editSoapDialog.note?.assessment || ""}
              onChange={(e) => setEditSoapDialog({ ...editSoapDialog, note: editSoapDialog.note ? { ...editSoapDialog.note, assessment: e.target.value } : null })}
            />
            <TextField
              fullWidth
              label="Plan"
              multiline
              rows={3}
              value={editSoapDialog.note?.plan || ""}
              onChange={(e) => setEditSoapDialog({ ...editSoapDialog, note: editSoapDialog.note ? { ...editSoapDialog.note, plan: e.target.value } : null })}
            />
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditSoapDialog({ open: false, note: null })}>Cancel</Button>
          <LoadingButton
            variant="contained"
            onClick={async () => {
              if (!editSoapDialog.note) return;
              try {
                const updated = await updateSoapNote(editSoapDialog.note.id, {
                  subjective: editSoapDialog.note.subjective,
                  objective: editSoapDialog.note.objective,
                  assessment: editSoapDialog.note.assessment,
                  plan: editSoapDialog.note.plan,
                });
                setSoapNote(updated);
                setEditSoapDialog({ open: false, note: null });
              } catch {
                setErrorMessage("Failed to update SOAP note");
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
                } else if (deleteConfirmDialog.type === "soap") {
                  await deleteSoapNote(deleteConfirmDialog.id);
                  setSoapNote(null);
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

