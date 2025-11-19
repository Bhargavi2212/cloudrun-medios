import { createTheme } from "@mui/material/styles";

export const appTheme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#1f7a8c",
    },
    secondary: {
      main: "#ff7f50",
    },
    background: {
      default: "#f5f7fb",
    },
  },
  typography: {
    fontFamily: "Inter, Roboto, Helvetica, Arial, sans-serif",
    h1: {
      fontWeight: 600,
      fontSize: "2.4rem",
    },
    h2: {
      fontWeight: 600,
      fontSize: "2rem",
    },
    button: {
      textTransform: "none",
      fontWeight: 600,
    },
  },
});

