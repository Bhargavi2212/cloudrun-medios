import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Grid,
  Typography,
  TextField,
  CircularProgress,
} from "@mui/material";
import { CheckCircle, Cancel, Visibility, Refresh } from "@mui/icons-material";
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listPendingDocuments,
  getDocument,
  processDocument,
  confirmDocument,
  rejectDocument,
  type FileAsset,
} from "../../../shared/services/documentService";

const formatDate = (dateString: string): string => {
  try {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "just now";
    if (diffMins < 60) return `${diffMins} minute${diffMins !== 1 ? "s" : ""} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours !== 1 ? "s" : ""} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays !== 1 ? "s" : ""} ago`;
    return date.toLocaleDateString();
  } catch {
    return dateString;
  }
};

const getStatusChip = (status: string) => {
  const statusMap: Record<string, { label: string; color: "default" | "primary" | "secondary" | "error" | "info" | "success" | "warning" }> = {
    uploaded: { label: "Uploaded", color: "info" },
    processing: { label: "Processing", color: "primary" },
    processed: { label: "Processed", color: "success" },
    needs_review: { label: "Needs Review", color: "warning" },
    failed: { label: "Failed", color: "error" },
  };
  const config = statusMap[status] || { label: status, color: "default" };
  return <Chip label={config.label} color={config.color} size="small" />;
};

const getConfidenceChip = (confidence: number | null | undefined) => {
  if (confidence === null || confidence === undefined) {
    return <Chip label="N/A" size="small" />;
  }
  const percent = Math.round(confidence * 100);
  if (confidence >= 0.8) {
    return <Chip label={`High (${percent}%)`} color="success" size="small" />;
  }
  if (confidence >= 0.6) {
    return <Chip label={`Medium (${percent}%)`} color="warning" size="small" />;
  }
  return <Chip label={`Low (${percent}%)`} color="error" size="small" />;
};

export const DocumentReviewDashboard = () => {
  const [selectedDocument, setSelectedDocument] = useState<FileAsset | null>(null);
  const [isViewingDetails, setIsViewingDetails] = useState(false);
  const [reviewNotes, setReviewNotes] = useState("");
  const queryClient = useQueryClient();

  const { data: pendingDocuments, isLoading, refetch } = useQuery({
    queryKey: ["pending-documents"],
    queryFn: () => listPendingDocuments(),
  });

  const processMutation = useMutation({
    mutationFn: (fileId: string) => processDocument(fileId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pending-documents"] });
      setSelectedDocument(null);
      setIsViewingDetails(false);
    },
  });

  const confirmMutation = useMutation({
    mutationFn: ({ fileId, notes }: { fileId: string; notes?: string }) => confirmDocument(fileId, notes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pending-documents"] });
      setSelectedDocument(null);
      setIsViewingDetails(false);
      setReviewNotes("");
    },
  });

  const rejectMutation = useMutation({
    mutationFn: ({ fileId, reason }: { fileId: string; reason?: string }) => rejectDocument(fileId, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pending-documents"] });
      setSelectedDocument(null);
      setIsViewingDetails(false);
      setReviewNotes("");
    },
  });

  const handleViewDetails = async (document: FileAsset) => {
    setSelectedDocument(document);
    setIsViewingDetails(true);
    // If not processed yet, try to process it
    if (document.status === "uploaded" && !document.extraction_status) {
      try {
        await processMutation.mutateAsync(document.id);
        // Refetch the document to get updated data
        const updated = await getDocument(document.id);
        setSelectedDocument(updated);
      } catch (error) {
        console.error("Processing error:", error);
      }
    }
  };

  const handleConfirm = () => {
    if (!selectedDocument) return;
    confirmMutation.mutate({ fileId: selectedDocument.id, notes: reviewNotes || undefined });
  };

  const handleReject = () => {
    if (!selectedDocument) return;
    rejectMutation.mutate({ fileId: selectedDocument.id, reason: reviewNotes || undefined });
  };

  if (isLoading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
        <Typography variant="h4">Document Review Dashboard</Typography>
        <Button startIcon={<Refresh />} onClick={() => refetch()}>
          Refresh
        </Button>
      </Box>

      {pendingDocuments && pendingDocuments.length > 0 ? (
        <Grid container spacing={2}>
          {pendingDocuments.map((doc) => (
            <Grid item xs={12} sm={6} md={4} key={doc.id}>
              <Card>
                <CardContent>
                  <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "start", mb: 2 }}>
                    <Box>
                      <Typography variant="h6" noWrap>
                        {doc.original_filename || "Untitled Document"}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {formatDate(doc.created_at)} • {doc.document_type || "Unknown type"}
                        {doc.size_bytes && ` • ${(doc.size_bytes / 1024).toFixed(1)} KB`}
                      </Typography>
                    </Box>
                  </Box>
                  <Box sx={{ display: "flex", gap: 1, mb: 2, flexWrap: "wrap" }}>
                    {getStatusChip(doc.status)}
                    {getConfidenceChip(doc.extraction_confidence)}
                  </Box>
                  {doc.processing_notes && (
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {doc.processing_notes}
                    </Typography>
                  )}
                  <Box sx={{ display: "flex", gap: 1 }}>
                    <Button
                      size="small"
                      startIcon={<Visibility />}
                      onClick={() => handleViewDetails(doc)}
                    >
                      View Details
                    </Button>
                    <Button
                      size="small"
                      color="success"
                      startIcon={<CheckCircle />}
                      onClick={handleConfirm}
                    >
                      Approve
                    </Button>
                    <Button
                      size="small"
                      color="error"
                      startIcon={<Cancel />}
                      onClick={handleReject}
                    >
                      Reject
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : (
        <Card>
          <CardContent>
            <Typography variant="body1" color="text.secondary" align="center">
              No documents currently require review.
            </Typography>
          </CardContent>
        </Card>
      )}

      {/* Document Details Dialog */}
      <Dialog open={isViewingDetails} onClose={() => setIsViewingDetails(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Document Details: {selectedDocument?.original_filename || "Unknown"}
        </DialogTitle>
        <DialogContent>
          {selectedDocument && (
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Extraction Data:
              </Typography>
              <Box
                sx={{
                  p: 2,
                  bgcolor: "grey.100",
                  borderRadius: 1,
                  mb: 2,
                  maxHeight: 400,
                  overflow: "auto",
                }}
              >
                <pre style={{ margin: 0, fontSize: "0.875rem" }}>
                  {JSON.stringify(selectedDocument.extraction_data || {}, null, 2)}
                </pre>
              </Box>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Review Notes"
                value={reviewNotes}
                onChange={(e) => setReviewNotes(e.target.value)}
                placeholder="Add notes for this review..."
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsViewingDetails(false)}>Close</Button>
          <Button onClick={handleConfirm} color="success" startIcon={<CheckCircle />}>
            Approve
          </Button>
          <Button onClick={handleReject} color="error" startIcon={<Cancel />}>
            Reject
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

