import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  Grid,
  Stack,
  Typography,
  Tooltip,
} from "@mui/material";
import LocalHospitalIcon from "@mui/icons-material/LocalHospital";
import CloudIcon from "@mui/icons-material/Cloud";
import { useState } from "react";

import { LoadingState } from "../../components/LoadingState";
import { ErrorState } from "../../components/ErrorState";
import { PatientList } from "./components/PatientList";
import { usePatients } from "./hooks/usePatients";
import { usePortableProfile } from "./hooks/usePortableProfile";

export const PatientProfilePage = () => {
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null);
  const patientsQuery = usePatients();
  const portableProfileQuery = usePortableProfile(selectedPatientId);

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h4">Portable Patient Profiles</Typography>
        <Typography variant="body1" color="text.secondary">
          Inspect a patient&apos;s unified timeline assembled from local and federated data sources.
        </Typography>
      </Grid>
      <Grid item xs={12} md={4}>
        {patientsQuery.isLoading ? (
          <LoadingState label="Loading patients..." />
        ) : patientsQuery.isError ? (
          <ErrorState message="Unable to load patients." />
        ) : (
          <PatientList
            patients={patientsQuery.data ?? []}
            selectedPatientId={selectedPatientId}
            onSelect={setSelectedPatientId}
          />
        )}
      </Grid>
      <Grid item xs={12} md={8}>
        {!selectedPatientId && <Alert severity="info">Select a patient to view their portable profile.</Alert>}
        {selectedPatientId && portableProfileQuery.isLoading && <LoadingState label="Fetching portable profile..." />}
        {selectedPatientId && portableProfileQuery.isError && (
          <ErrorState message="Unable to fetch portable profile. Ensure DOL is reachable." />
        )}
        {portableProfileQuery.data && (
          <Stack spacing={3}>
            <Card>
              <CardContent>
                <Typography variant="h6">Patient Overview</Typography>
                <Typography>
                  {portableProfileQuery.data.patient.first_name} {portableProfileQuery.data.patient.last_name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  MRN: {portableProfileQuery.data.patient.mrn}
                </Typography>
                {portableProfileQuery.data.sources.length > 0 && (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Sources
                    </Typography>
                    <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                      {portableProfileQuery.data.sources.map((source) => (
                        <Chip key={source} label={source} size="small" color="primary" variant="outlined" />
                      ))}
                    </Stack>
                  </Box>
                )}
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <Typography variant="h6">Timeline</Typography>
                <Divider sx={{ my: 2 }} />
                <Stack spacing={2}>
                  {portableProfileQuery.data.timeline.map((event, index) => {
                    const isLocal = event.source === "local";
                    const isFederated = event.source === "federated";
                    return (
                      <Box key={`${event.event_type}-${event.timestamp}-${index}`}>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                          <Typography variant="subtitle2">{event.event_type.toUpperCase()}</Typography>
                          {isLocal && (
                            <Tooltip title="From your hospital">
                              <Chip
                                icon={<LocalHospitalIcon />}
                                label="Local Record"
                                size="small"
                                color="primary"
                                variant="outlined"
                                aria-label="Local record from your hospital"
                              />
                            </Tooltip>
                          )}
                          {isFederated && (
                            <Tooltip title="From network - originating site hidden for privacy">
                              <Chip
                                icon={<CloudIcon />}
                                label="Network History"
                                size="small"
                                color="secondary"
                                variant="outlined"
                                aria-label="Network history from federated source"
                              />
                            </Tooltip>
                          )}
                        </Box>
                        <Typography variant="body2" color="text.secondary">
                          {new Date(event.timestamp).toLocaleString()}
                        </Typography>
                        <Typography variant="body1">
                          {JSON.stringify(event.content, null, 2).replace(/[{}"]/g, "").trim()}
                        </Typography>
                      </Box>
                    );
                  })}
                </Stack>
              </CardContent>
            </Card>
            <Card>
              <CardContent>
                <Typography variant="h6">Summaries</Typography>
                {portableProfileQuery.data.summaries.length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    No summaries found.
                  </Typography>
                ) : (
                  <Stack spacing={2} sx={{ mt: 1 }}>
                    {portableProfileQuery.data.summaries.map((summary) => (
                      <Box key={summary.id}>
                        <Typography variant="subtitle2">Round {summary.model_version ?? "N/A"}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          {new Date(summary.created_at).toLocaleString()}
                        </Typography>
                        <Typography variant="body1">{summary.summary_text}</Typography>
                      </Box>
                    ))}
                  </Stack>
                )}
              </CardContent>
            </Card>
          </Stack>
        )}
      </Grid>
    </Grid>
  );
};

