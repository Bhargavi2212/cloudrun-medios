import {
  AppBar,
  Box,
  Button,
  Toolbar,
  Typography,
} from "@mui/material";
import React, { ReactNode, useCallback, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuthStore } from "../store/authStore";
import { HospitalSelector } from "../components/HospitalSelector";
import { useHospital } from "../shared/contexts/HospitalContext";

interface AppLayoutProps {
  children: ReactNode;
}

export const AppLayout = ({ children }: AppLayoutProps) => {
  const navigate = useNavigate();
  const { user, logout } = useAuthStore();
  const { currentHospital } = useHospital();
  const previousHospitalRef = React.useRef<string | null>(null);
  const isInitialMount = React.useRef<boolean>(true);

  // Clear auth state when hospital switches (but not on initial mount)
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      previousHospitalRef.current = currentHospital.id;
      return;
    }

    if (previousHospitalRef.current !== null && previousHospitalRef.current !== currentHospital.id) {
      // Hospital has changed - clear auth to prevent cross-context leakage
      logout();
      previousHospitalRef.current = currentHospital.id;
      // Navigate to login after hospital switch
      navigate("/login");
    } else {
      previousHospitalRef.current = currentHospital.id;
    }
  }, [currentHospital.id, logout, navigate]);

  const handleLogout = useCallback(() => {
    logout();
    navigate("/login");
  }, [logout, navigate]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Medi OS Portal - {user?.role || "User"}
          </Typography>
          <Box sx={{ mr: 2, display: "flex", alignItems: "center", gap: 2 }}>
            <HospitalSelector />
            {user && (
              <>
                <Typography variant="body2">
            {user?.full_name || user?.email}
          </Typography>
                <Button color="inherit" onClick={handleLogout} aria-label="Logout">
            Logout
          </Button>
              </>
            )}
          </Box>
        </Toolbar>
      </AppBar>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          bgcolor: "background.default",
        }}
      >
        {children}
      </Box>
    </Box>
  );
};
