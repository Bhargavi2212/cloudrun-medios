import { Box, Grid, Tab, Tabs, Typography } from "@mui/material";
import { DocumentUpload } from "./components/DocumentUpload";
import { DocumentReviewDashboard } from "./components/DocumentReviewDashboard";
import { useState } from "react";

export const DocumentsPage = () => {
  const [activeTab, setActiveTab] = useState<"upload" | "review">("upload");

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h4">Document Management</Typography>
        <Typography variant="body1" color="text.secondary">
          Upload and review patient documents.
        </Typography>
      </Grid>
      <Grid item xs={12}>
        <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 3 }}>
          <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
            <Tab label="Upload Documents" value="upload" />
            <Tab label="Review Documents" value="review" />
          </Tabs>
        </Box>
        {activeTab === "upload" ? (
          <DocumentUpload />
        ) : (
          <DocumentReviewDashboard />
        )}
      </Grid>
    </Grid>
  );
};

