import { LoadingButton } from "@mui/lab";
import { Card, CardContent, Grid, Stack, TextField, Typography } from "@mui/material";
import { useState } from "react";
import { useTriageClassification } from "../hooks/useTriage";

export const TriageForm = () => {
  const [values, setValues] = useState({
    hr: 90,
    rr: 18,
    sbp: 120,
    dbp: 75,
    temp_c: 37.0,
    spo2: 98,
    pain: 2,
  });
  const triageMutation = useTriageClassification();

  const handleChange = (field: keyof typeof values) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setValues((prev) => ({ ...prev, [field]: Number(event.target.value) }));
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await triageMutation.mutateAsync(values);
  };

  return (
    <Stack spacing={3}>
      <Card component="form" onSubmit={handleSubmit}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Triage Classification
          </Typography>
          <Grid container spacing={2}>
            {Object.entries(values).map(([field, value]) => (
              <Grid item xs={6} md={3} key={field}>
                <TextField
                  fullWidth
                  type="number"
                  label={field.toUpperCase()}
                  value={value}
                  onChange={handleChange(field as keyof typeof values)}
                />
              </Grid>
            ))}
          </Grid>
          <LoadingButton type="submit" variant="contained" sx={{ mt: 3 }} loading={triageMutation.isPending}>
            Classify
          </LoadingButton>
        </CardContent>
      </Card>
      {triageMutation.data && (
        <Card>
          <CardContent>
            <Typography variant="h6">Result</Typography>
            <Typography variant="body1" sx={{ mt: 1 }}>
              Acuity Level: {triageMutation.data.acuity_level}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Model Version: {triageMutation.data.model_version}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {triageMutation.data.explanation}
            </Typography>
          </CardContent>
        </Card>
      )}
    </Stack>
  );
};

