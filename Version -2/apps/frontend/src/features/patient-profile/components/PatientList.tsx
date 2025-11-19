import {
  Avatar,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  List,
  ListItem,
  ListItemAvatar,
  ListItemButton,
  ListItemText,
  Stack,
  TextField,
} from "@mui/material";
import PersonIcon from "@mui/icons-material/Person";
import { useState } from "react";
import { Patient } from "../../../shared/types/api";
import { useCreatePatient } from "../hooks/usePatients";
import { LoadingButton } from "@mui/lab";

interface PatientListProps {
  patients: Patient[];
  selectedPatientId: string | null;
  onSelect: (patientId: string) => void;
}

export const PatientList = ({ patients, selectedPatientId, onSelect }: PatientListProps) => {
  const [open, setOpen] = useState(false);
  const [mrn, setMrn] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const createPatient = useCreatePatient();

  const handleCreate = async () => {
    const patient = await createPatient.mutateAsync({
      mrn,
      first_name: firstName,
      last_name: lastName,
    });
    onSelect(patient.id);
    setOpen(false);
    setMrn("");
    setFirstName("");
    setLastName("");
  };

  return (
    <Box>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
        <Button variant="contained" color="primary" onClick={() => setOpen(true)}>
          Add Patient
        </Button>
      </Stack>
      <List dense sx={{ maxHeight: 420, overflowY: "auto" }}>
        {patients.map((patient) => (
          <ListItem key={patient.id} disablePadding>
            <ListItemButton selected={patient.id === selectedPatientId} onClick={() => onSelect(patient.id)}>
              <ListItemAvatar>
                <Avatar>
                  <PersonIcon />
                </Avatar>
              </ListItemAvatar>
              <ListItemText primary={`${patient.first_name} ${patient.last_name}`} secondary={patient.mrn} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Dialog open={open} onClose={() => setOpen(false)} fullWidth maxWidth="sm">
        <DialogTitle>Create Patient</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField label="MRN" value={mrn} onChange={(event) => setMrn(event.target.value)} required />
            <Stack direction="row" spacing={2}>
              <TextField
                fullWidth
                label="First Name"
                value={firstName}
                onChange={(event) => setFirstName(event.target.value)}
              />
              <TextField
                fullWidth
                label="Last Name"
                value={lastName}
                onChange={(event) => setLastName(event.target.value)}
              />
            </Stack>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>Cancel</Button>
          <LoadingButton
            onClick={handleCreate}
            variant="contained"
            loading={createPatient.isPending}
            disabled={!mrn || !firstName || !lastName}
          >
            Save
          </LoadingButton>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

