import React, { useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Grid,
  Divider,
  IconButton,
  Tooltip,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import VisibilityIcon from "@mui/icons-material/Visibility";
import type { StructuredTimelineData, TimelineEntry } from "../types/api";

interface PatientTimelineProps {
  data: StructuredTimelineData;
  onViewOriginal?: (entry: TimelineEntry) => void;
}

export const PatientTimeline: React.FC<PatientTimelineProps> = ({ data, onViewOriginal }) => {
  const [expandedEntries, setExpandedEntries] = useState<Set<string>>(new Set());

  const toggleEntry = (entryId: string) => {
    setExpandedEntries((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(entryId)) {
        newSet.delete(entryId);
      } else {
        newSet.add(entryId);
      }
      return newSet;
    });
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  };

  const formatSource = (sourceType: string) => {
    const sourceMap: Record<string, string> = {
      ai_scribe: "AI Scribe (Live recording)",
      uploaded_pdf: "Uploaded PDF",
      uploaded_image: "Uploaded Image",
      manual_entry: "Manual Entry",
    };
    return sourceMap[sourceType] || sourceType;
  };

  const getStatusIcon = (confidence: number) => {
    if (confidence >= 95) return "✅";
    if (confidence >= 85) return "✓";
    return "⚠️";
  };

  const hasAlerts = data.alerts.allergies.length > 0 || data.alerts.chronic_conditions.length > 0 || data.alerts.recent_events.length > 0 || data.alerts.warnings.length > 0;

  return (
    <Box>
      {/* Patient Header */}
      <Card sx={{ mb: 3, bgcolor: "primary.main", color: "white" }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            👤 PATIENT: {data.patient.name}
          </Typography>
          <Typography variant="body1">
            Age {data.patient.age ?? "N/A"} | {data.patient.patient_id}
          </Typography>
          <Typography variant="body2" sx={{ mt: 1, opacity: 0.9 }}>
            📋 Medical History: {data.years_of_history.toFixed(1)} YEARS OF RECORDS
          </Typography>
        </CardContent>
      </Card>

      {/* Critical Alerts */}
      {hasAlerts && (
        <Card sx={{ mb: 3, bgcolor: "error.light", color: "error.contrastText" }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              ⚠️ CRITICAL ALERTS (Updated Real-Time)
            </Typography>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              {data.alerts.allergies.length > 0 && (
                <Box>
                  <Typography variant="subtitle2">🚨 Allergies:</Typography>
                  <Typography variant="body2">{data.alerts.allergies.join(", ")}</Typography>
                </Box>
              )}
              {data.alerts.chronic_conditions.length > 0 && (
                <Box>
                  <Typography variant="subtitle2">🚨 Chronic Conditions:</Typography>
                  <Typography variant="body2">{data.alerts.chronic_conditions.join(", ")}</Typography>
                </Box>
              )}
              {data.alerts.recent_events.length > 0 && (
                <Box>
                  <Typography variant="subtitle2">🚨 Recent Events:</Typography>
                  {data.alerts.recent_events.map((event, i) => (
                    <Typography key={i} variant="body2">
                      🚨 {event}
                    </Typography>
                  ))}
                </Box>
              )}
              {data.alerts.warnings.length > 0 && (
                <Box>
                  <Typography variant="subtitle2">🚨 Warnings:</Typography>
                  {data.alerts.warnings.map((warning, i) => (
                    <Typography key={i} variant="body2">
                      🚨 {warning}
                    </Typography>
                  ))}
                </Box>
              )}
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Timeline */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            📅 TIMELINE (Newest First, {data.total_entries} Entries)
          </Typography>
          <Divider sx={{ my: 2 }} />
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {data.timeline.map((entry) => {
              const isExpanded = expandedEntries.has(entry.id);
              return (
                <Card key={entry.id} variant="outlined" sx={{ bgcolor: "background.paper" }}>
                  <CardContent>
                    {/* Entry Header */}
                    <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", mb: 1 }}>
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="subtitle1" fontWeight={600}>
                          [{formatDate(entry.date)}] {getStatusIcon(entry.source.confidence)} {entry.title}
                        </Typography>
                        <Box sx={{ display: "flex", gap: 1, mt: 1, flexWrap: "wrap" }}>
                          <Chip size="small" label={`Source: ${formatSource(entry.source.type)}`} variant="outlined" />
                          <Chip
                            size="small"
                            label={`Confidence: ${entry.source.confidence}%`}
                            color={entry.source.confidence >= 95 ? "success" : entry.source.confidence >= 85 ? "default" : "warning"}
                          />
                          {entry.source.reviewed_by && (
                            <Chip size="small" label={`Reviewed by ${entry.source.reviewed_by}`} color="info" />
                          )}
                        </Box>
                      </Box>
                      <Box sx={{ display: "flex", gap: 0.5 }}>
                        {entry.can_view_original && onViewOriginal && (
                          <Tooltip title="View Original">
                            <IconButton size="small" onClick={() => onViewOriginal(entry)}>
                              <VisibilityIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                        <IconButton size="small" onClick={() => toggleEntry(entry.id)}>
                          {isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                        </IconButton>
                      </Box>
                    </Box>

                    {/* Collapsed Preview */}
                    {!isExpanded && (
                      <Box sx={{ mt: 1 }}>
                        {entry.data.chief_complaint && (
                          <Typography variant="body2" color="text.secondary">
                            <strong>Chief:</strong> {entry.data.chief_complaint}
                          </Typography>
                        )}
                        {entry.data.diagnosis && (
                          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                            <strong>Diagnosis:</strong> {entry.data.diagnosis}
                          </Typography>
                        )}
                      </Box>
                    )}

                    {/* Expanded Details */}
                    {isExpanded && (
                      <Box sx={{ mt: 2, pt: 2, borderTop: "1px solid", borderColor: "divider" }}>
                        <Grid container spacing={2}>
                          {entry.data.chief_complaint && (
                            <Grid item xs={12}>
                              <Typography variant="body2">
                                <strong>Chief Complaint:</strong> {entry.data.chief_complaint}
                              </Typography>
                            </Grid>
                          )}

                          {entry.data.vitals && (
                            <Grid item xs={12}>
                              <Typography variant="body2">
                                <strong>Vitals:</strong>{" "}
                                {[
                                  entry.data.vitals.hr && `HR ${entry.data.vitals.hr}`,
                                  entry.data.vitals.bp && `BP ${entry.data.vitals.bp}`,
                                  entry.data.vitals.temp && `Temp ${entry.data.vitals.temp}°C`,
                                  entry.data.vitals.rr && `RR ${entry.data.vitals.rr}`,
                                  entry.data.vitals.o2 && `O2 ${entry.data.vitals.o2}%`,
                                ]
                                  .filter(Boolean)
                                  .join(", ")}
                              </Typography>
                            </Grid>
                          )}

                          {entry.data.rfv && (
                            <Grid item xs={12}>
                              <Typography variant="body2">
                                <strong>RFV:</strong> {entry.data.rfv}
                              </Typography>
                            </Grid>
                          )}

                          {entry.data.tests && entry.data.tests.length > 0 && (
                            <Grid item xs={12}>
                              <Typography variant="body2">
                                <strong>Tests:</strong> {entry.data.tests.join(", ")}
                              </Typography>
                            </Grid>
                          )}

                          {entry.data.subjective && (
                            <Grid item xs={12}>
                              <Typography variant="body2">
                                <strong>Subjective:</strong> {entry.data.subjective}
                              </Typography>
                            </Grid>
                          )}

                          {entry.data.objective && (
                            <Grid item xs={12}>
                              <Typography variant="body2">
                                <strong>Objective:</strong> {entry.data.objective}
                              </Typography>
                            </Grid>
                          )}

                          {entry.data.diagnosis && (
                            <Grid item xs={12}>
                              <Typography variant="body2">
                                <strong>Assessment:</strong> {entry.data.diagnosis}
                              </Typography>
                            </Grid>
                          )}

                          {entry.data.medications && entry.data.medications.length > 0 && (
                            <Grid item xs={12}>
                              <Typography variant="body2">
                                <strong>Medications:</strong> {entry.data.medications.join(", ")}
                              </Typography>
                            </Grid>
                          )}

                          {entry.data.plan && (
                            <Grid item xs={12}>
                              <Typography variant="body2">
                                <strong>Plan:</strong> {entry.data.plan}
                              </Typography>
                            </Grid>
                          )}

                          {entry.data.disposition && (
                            <Grid item xs={12}>
                              <Typography variant="body2">
                                <strong>Disposition:</strong> {entry.data.disposition}
                              </Typography>
                            </Grid>
                          )}

                          {entry.data.file_name && (
                            <Grid item xs={12}>
                              <Typography variant="body2">
                                <strong>File:</strong> {entry.data.file_name}
                              </Typography>
                            </Grid>
                          )}
                        </Grid>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

