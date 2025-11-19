import { Alert } from "@mui/material";

interface ErrorStateProps {
  message?: string;
}

export const ErrorState = ({ message = "Something went wrong." }: ErrorStateProps) => (
  <Alert severity="error">{message}</Alert>
);

