import { Navigate, Route, Routes } from "react-router-dom";

import { AppLayout } from "./layout/AppLayout";
import { PatientProfilePage } from "./features/patient-profile/PatientProfilePage";
import { TriagePage } from "./features/triage/TriagePage";
import { ScribePage } from "./features/scribe/ScribePage";
import { FederationDashboardPage } from "./features/federated/FederationDashboardPage";
import { DocumentsPage } from "./features/documents/DocumentsPage";

const App = () => {
  return (
    <AppLayout>
      <Routes>
        <Route path="/" element={<PatientProfilePage />} />
        <Route path="/triage" element={<TriagePage />} />
        <Route path="/scribe" element={<ScribePage />} />
        <Route path="/federation" element={<FederationDashboardPage />} />
        <Route path="/documents" element={<DocumentsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppLayout>
  );
};

export default App;

