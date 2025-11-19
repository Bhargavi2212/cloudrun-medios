import { Grid, Typography } from "@mui/material";
import { TriageForm } from "./components/TriageForm";

export const TriagePage = () => {
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h4">Triage Simulation</Typography>
        <Typography variant="body1" color="text.secondary">
          Enter vital signs to simulate triage acuity scoring. The backend will be replaced with a federated model.
        </Typography>
      </Grid>
      <Grid item xs={12}>
        <TriageForm />
      </Grid>
    </Grid>
  );
};

