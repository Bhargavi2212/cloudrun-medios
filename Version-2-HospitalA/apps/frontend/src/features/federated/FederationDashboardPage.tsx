import { LoadingButton } from "@mui/lab";
import {
  Card,
  CardContent,
  Grid,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { fetchGlobalModel, submitModelUpdate } from "../../shared/services/federationService";
import { ErrorState } from "../../components/ErrorState";
import { LoadingState } from "../../components/LoadingState";

const MODEL_NAME = "triage";

export const FederationDashboardPage = () => {
  const [hospitalId, setHospitalId] = useState("hospital-a");
  const [weights, setWeights] = useState("0.1,0.2,0.3");
  const globalModelQuery = useQuery({
    queryKey: ["global-model", MODEL_NAME],
    queryFn: () => fetchGlobalModel(MODEL_NAME),
  });

  const submitMutation = useMutation({
    mutationFn: () =>
      submitModelUpdate({
        model_name: MODEL_NAME,
        round_id: (globalModelQuery.data?.round_id ?? 0) + 1,
        hospital_id: hospitalId,
        weights: {
          dense: weights.split(",").map((value) => Number(value.trim())),
        },
      }),
    onSuccess: () => {
      globalModelQuery.refetch();
    },
  });

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h4">Federated Learning Dashboard</Typography>
        <Typography variant="body1" color="text.secondary">
          Track the latest global model and simulate a model update submission from this hospital.
        </Typography>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6">Current Global Model</Typography>
            {globalModelQuery.isLoading ? (
              <LoadingState label="Loading global model..." />
            ) : globalModelQuery.isError ? (
              <ErrorState message="Global model not available yet." />
            ) : (
              <>
                <Typography variant="subtitle2" sx={{ mt: 2 }}>
                  Round {globalModelQuery.data.round_id}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Contributors: {globalModelQuery.data.contributor_count}
                </Typography>
                <pre style={{ fontSize: 12, marginTop: 16 }}>
                  {JSON.stringify(globalModelQuery.data.weights, null, 2)}
                </pre>
              </>
            )}
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6">Simulate Model Update</Typography>
            <Stack spacing={2} sx={{ mt: 2 }}>
              <TextField
                label="Hospital ID"
                value={hospitalId}
                onChange={(event) => setHospitalId(event.target.value)}
              />
              <TextField
                label="Weights (comma separated)"
                value={weights}
                onChange={(event) => setWeights(event.target.value)}
              />
              <LoadingButton
                variant="contained"
                onClick={() => submitMutation.mutate()}
                loading={submitMutation.isPending}
              >
                Submit Update
              </LoadingButton>
              {submitMutation.data && (
                <Typography variant="body2" color="success.main">
                  Update accepted. Contributors this round: {submitMutation.data.contributor_count}
                </Typography>
              )}
            </Stack>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

