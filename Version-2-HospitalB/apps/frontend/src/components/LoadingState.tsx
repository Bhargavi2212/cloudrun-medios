import { Box, CircularProgress, Typography } from "@mui/material";

export const LoadingState = ({ label = "Loading..." }: { label?: string }) => (
  <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
    <CircularProgress size={24} />
    <Typography>{label}</Typography>
  </Box>
);

