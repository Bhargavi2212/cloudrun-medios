import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  LinearProgress,
  Typography,
} from "@mui/material";
import { CloudUpload, Camera, FolderOpen } from "@mui/icons-material";
import { useState, useCallback, useRef } from "react";
import { uploadDocument, processDocument, type UploadDocumentPayload } from "../../../shared/services/documentService";

interface DocumentUploadProps {
  patientId?: string;
  encounterId?: string;
  onUploadComplete?: (fileId: string) => void;
}

interface UploadedFile {
  file: File;
  id: string;
  status: "pending" | "uploading" | "processing" | "completed" | "error";
  progress: number;
  fileId?: string;
  error?: string;
}

export const DocumentUpload = ({ patientId, encounterId, onUploadComplete }: DocumentUploadProps) => {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMethod, setUploadMethod] = useState<"file_picker" | "drag_and_drop" | "camera">("file_picker");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((selectedFiles: FileList | null, method: typeof uploadMethod) => {
    if (!selectedFiles || selectedFiles.length === 0) return;

    const newFiles: UploadedFile[] = Array.from(selectedFiles).map((file) => ({
      file,
      id: URL.createObjectURL(file),
      status: "pending",
      progress: 0,
    }));

    setFiles((prev) => [...prev, ...newFiles]);
    setUploadMethod(method);
  }, []);

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFileSelect(e.target.files, "file_picker");
  };

  const handleCameraInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFileSelect(e.target.files, "camera");
  };

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      handleFileSelect(e.dataTransfer.files, "drag_and_drop");
    },
    [handleFileSelect]
  );

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleRemoveFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const handleUpload = useCallback(async () => {
    if (files.length === 0) return;

    setIsUploading(true);

    for (const fileState of files) {
      if (fileState.status !== "pending") continue;

      try {
        // Update to uploading
        setFiles((prev) =>
          prev.map((f) => (f.id === fileState.id ? { ...f, status: "uploading", progress: 10 } : f))
        );

        // Upload file
        const payload: UploadDocumentPayload = {
          file: fileState.file,
          patient_id: patientId,
          encounter_id: encounterId,
          upload_method: uploadMethod,
        };

        const uploadResult = await uploadDocument(payload);

        // Update to processing
        setFiles((prev) =>
          prev.map((f) =>
            f.id === fileState.id
              ? { ...f, status: "processing", progress: 50, fileId: uploadResult.file_id }
              : f
          )
        );

        // Process document
        try {
          await processDocument(uploadResult.file_id);
          setFiles((prev) =>
            prev.map((f) => (f.id === fileState.id ? { ...f, status: "completed", progress: 100 } : f))
          );
          if (onUploadComplete && uploadResult.file_id) {
            onUploadComplete(uploadResult.file_id);
          }
        } catch (error) {
          console.error("Processing error:", error);
          setFiles((prev) =>
            prev.map((f) =>
              f.id === fileState.id
                ? { ...f, status: "error", progress: 0, error: "Processing failed" }
                : f
            )
          );
        }
      } catch (error) {
        console.error("Upload error:", error);
        setFiles((prev) =>
          prev.map((f) =>
            f.id === fileState.id
              ? { ...f, status: "error", progress: 0, error: error instanceof Error ? error.message : "Upload failed" }
              : f
          )
        );
      }
    }

    setIsUploading(false);
  }, [files, patientId, encounterId, uploadMethod, onUploadComplete]);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Upload Document
        </Typography>

        {/* Upload Methods */}
        <Box sx={{ display: "flex", gap: 2, mb: 3 }}>
          <Button
            variant="outlined"
            startIcon={<FolderOpen />}
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
          >
            Choose File
          </Button>
          <Button
            variant="outlined"
            startIcon={<Camera />}
            onClick={() => cameraInputRef.current?.click()}
            disabled={isUploading}
          >
            Camera
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            style={{ display: "none" }}
            onChange={handleFileInputChange}
            accept=".pdf,.jpg,.jpeg,.png,.gif,.bmp,.txt,.doc,.docx"
          />
          <input
            ref={cameraInputRef}
            type="file"
            capture="environment"
            accept="image/*"
            style={{ display: "none" }}
            onChange={handleCameraInputChange}
          />
        </Box>

        {/* Drag and Drop Area */}
        <Box
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          sx={{
            border: "2px dashed",
            borderColor: "primary.main",
            borderRadius: 2,
            p: 4,
            textAlign: "center",
            mb: 3,
            cursor: "pointer",
            "&:hover": {
              backgroundColor: "action.hover",
            },
          }}
        >
          <CloudUpload sx={{ fontSize: 48, color: "primary.main", mb: 2 }} />
          <Typography variant="body1" color="text.secondary">
            Drag and drop files here, or click to select
          </Typography>
        </Box>

        {/* File List */}
        {files.length > 0 && (
          <Box sx={{ mb: 2 }}>
            {files.map((fileState) => (
              <Card key={fileState.id} sx={{ mb: 2 }}>
                <CardContent>
                  <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
                    <Typography variant="body2">{fileState.file.name}</Typography>
                    <Button size="small" onClick={() => handleRemoveFile(fileState.id)} disabled={isUploading}>
                      Remove
                    </Button>
                  </Box>
                  {fileState.status !== "pending" && (
                    <Box>
                      <LinearProgress
                        variant="determinate"
                        value={fileState.progress}
                        sx={{ mb: 1 }}
                      />
                      <Typography variant="caption" color="text.secondary">
                        {fileState.status === "uploading" && "Uploading..."}
                        {fileState.status === "processing" && "Processing..."}
                        {fileState.status === "completed" && "Completed"}
                        {fileState.status === "error" && `Error: ${fileState.error}`}
                      </Typography>
                    </Box>
                  )}
                </CardContent>
              </Card>
            ))}
          </Box>
        )}

        {/* Upload Button */}
        <Button
          variant="contained"
          startIcon={isUploading ? <CircularProgress size={20} /> : <CloudUpload />}
          onClick={handleUpload}
          disabled={files.length === 0 || isUploading}
          fullWidth
        >
          {isUploading ? "Uploading..." : "Upload Documents"}
        </Button>
      </CardContent>
    </Card>
  );
};

