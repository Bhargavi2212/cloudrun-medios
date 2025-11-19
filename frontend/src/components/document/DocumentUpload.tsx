import { useState, useRef, useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { manageAPI } from '@/services/api'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { useToast } from '@/components/ui/use-toast'
import { Loader2, Upload, Camera, X, FileText, CheckCircle2, AlertCircle } from 'lucide-react'
import type { ConsultationRecord } from '@/types'

interface DocumentUploadProps {
  consultationId: string
  onUploadComplete?: (records: ConsultationRecord[]) => void
  uploadMethod?: 'camera' | 'drag_and_drop' | 'file_picker' | 'scan' | 'manual'
}

interface UploadFile {
  file: File
  id: string
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error'
  progress: number
  record?: ConsultationRecord
  error?: string
}

export function DocumentUpload({ consultationId, onUploadComplete, uploadMethod = 'file_picker' }: DocumentUploadProps) {
  const [files, setFiles] = useState<UploadFile[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [notes, setNotes] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const cameraInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()
  const queryClient = useQueryClient()

  const addFiles = useCallback((newFiles: FileList | File[]) => {
    const fileArray = Array.from(newFiles)
    const uploadFiles: UploadFile[] = fileArray.map((file) => ({
      file,
      id: `${Date.now()}-${Math.random()}`,
      status: 'pending',
      progress: 0,
    }))
    setFiles((prev) => [...prev, ...uploadFiles])
  }, [])

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
        addFiles(e.target.files)
      }
    },
    [addFiles],
  )

  const handleCameraCapture = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
        addFiles(e.target.files)
      }
      // Reset camera input
      if (cameraInputRef.current) {
        cameraInputRef.current.value = ''
      }
    },
    [addFiles],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      if (e.dataTransfer.files) {
        addFiles(e.dataTransfer.files)
      }
    },
    [addFiles],
  )

  const removeFile = useCallback((id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id))
  }, [])

  const pollProcessingStatus = useCallback(
    async (fileId: string, fileIndex: number) => {
      const maxAttempts = 30
      let attempts = 0

      const poll = async () => {
        try {
          const status = await manageAPI.getRecordStatus(fileId)
          attempts++

          if (status.status === 'completed') {
            setFiles((prev) =>
              prev.map((f, idx) =>
                idx === fileIndex
                  ? {
                      ...f,
                      status: 'completed',
                      progress: 100,
                    }
                  : f,
              ),
            )
            return
          }

          if (status.status === 'failed' || status.status === 'needs_review') {
            setFiles((prev) =>
              prev.map((f, idx) =>
                idx === fileIndex
                  ? {
                      ...f,
                      status: status.status === 'failed' ? 'error' : 'completed',
                      progress: status.status === 'failed' ? 0 : 100,
                      error: status.status === 'failed' ? 'Processing failed' : undefined,
                    }
                  : f,
              ),
            )
            return
          }

          // Still processing
          if (attempts < maxAttempts) {
            setFiles((prev) =>
              prev.map((f, idx) =>
                idx === fileIndex
                  ? {
                      ...f,
                      status: 'processing',
                      progress: Math.min(50 + (attempts / maxAttempts) * 40, 90),
                    }
                  : f,
              ),
            )
            setTimeout(poll, 2000)
          } else {
            setFiles((prev) =>
              prev.map((f, idx) =>
                idx === fileIndex
                  ? {
                      ...f,
                      status: 'error',
                      progress: 0,
                      error: 'Processing timeout',
                    }
                  : f,
              ),
            )
          }
        } catch (error) {
          console.error('Status poll error:', error)
          if (attempts < maxAttempts) {
            setTimeout(poll, 2000)
          }
        }
      }

      setTimeout(poll, 2000)
    },
    [],
  )

  const handleUpload = useCallback(async () => {
    if (files.length === 0) {
      toast({
        title: 'No files selected',
        description: 'Please select at least one file to upload.',
        variant: 'destructive',
      })
      return
    }

    setIsUploading(true)
    const fileList = files.map((f) => f.file)

    // Update all files to uploading status
    setFiles((prev) =>
      prev.map((f) => ({
        ...f,
        status: 'uploading',
        progress: 10,
      })),
    )

    try {
      const result = await manageAPI.uploadConsultationRecordsWithMethod(consultationId, fileList, {
        notes: notes || undefined,
        uploadMethod,
      })

      // Update files with completed status
      setFiles((prev) =>
        prev.map((f, idx) => {
          const record = result.records[idx]
          return {
            ...f,
            status: record ? 'processing' : 'error',
            progress: record ? 50 : 0,
            record,
            error: record ? undefined : 'Upload failed',
          }
        }),
      )

      // Poll for processing status
      result.records.forEach((record: ConsultationRecord, idx: number) => {
        if (record) {
          pollProcessingStatus(record.id, idx)
        }
      })

      toast({
        title: 'Upload successful',
        description: `${result.records.length} file(s) uploaded and queued for processing.`,
      })

      if (onUploadComplete) {
        onUploadComplete(result.records)
      }

      // Invalidate queries to refresh data
      queryClient.invalidateQueries({ queryKey: ['consultation-records', consultationId] })
      queryClient.invalidateQueries({ queryKey: ['pending-documents'] })

      // Clear notes after successful upload
      setNotes('')
    } catch (error) {
      console.error('Upload error:', error)
      setFiles((prev) =>
        prev.map((f) => ({
          ...f,
          status: 'error',
          progress: 0,
          error: error instanceof Error ? error.message : 'Upload failed',
        })),
      )
      toast({
        title: 'Upload failed',
        description: error instanceof Error ? error.message : 'Failed to upload files. Please try again.',
        variant: 'destructive',
      })
    } finally {
      setIsUploading(false)
    }
  }, [files, consultationId, notes, uploadMethod, onUploadComplete, toast, queryClient, pollProcessingStatus])

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const getStatusIcon = (status: UploadFile['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      case 'uploading':
      case 'processing':
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
      default:
        return <FileText className="h-4 w-4 text-gray-400" />
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Upload Documents</CardTitle>
        <CardDescription>
          Upload patient records, lab reports, or other medical documents. Supports PDF, images, and text files.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Upload Methods */}
        <div className="flex gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
          >
            <Upload className="mr-2 h-4 w-4" />
            Choose Files
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={() => cameraInputRef.current?.click()}
            disabled={isUploading}
          >
            <Camera className="mr-2 h-4 w-4" />
            Camera
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.jpg,.jpeg,.png,.heic,.tiff,.txt"
            onChange={handleFileSelect}
            className="hidden"
          />
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleCameraCapture}
            className="hidden"
          />
        </div>

        {/* Drag and Drop Area */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            isDragging
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
              : 'border-gray-300 dark:border-gray-700'
          }`}
        >
          <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Drag and drop files here, or click to select
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
            PDF, Images (JPG, PNG, HEIC, TIFF), or Text files
          </p>
        </div>

        {/* Notes Input */}
        <div>
          <label className="block text-sm font-medium mb-2">Notes (optional)</label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Add any notes about these documents..."
            className="w-full min-h-[80px] px-3 py-2 border rounded-md"
            disabled={isUploading}
          />
        </div>

        {/* File List */}
        {files.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">Selected Files ({files.length})</h4>
            {files.map((uploadFile) => (
              <div
                key={uploadFile.id}
                className="flex items-center gap-3 p-3 border rounded-lg bg-gray-50 dark:bg-gray-900"
              >
                {getStatusIcon(uploadFile.status)}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{uploadFile.file.name}</p>
                  <p className="text-xs text-gray-500">
                    {formatFileSize(uploadFile.file.size)} • {uploadFile.file.type || 'Unknown type'}
                  </p>
                  {uploadFile.status === 'uploading' || uploadFile.status === 'processing' ? (
                    <Progress value={uploadFile.progress} className="mt-2 h-1" />
                  ) : null}
                  {uploadFile.error && (
                    <p className="text-xs text-red-500 mt-1">{uploadFile.error}</p>
                  )}
                </div>
                {uploadFile.status !== 'uploading' && uploadFile.status !== 'processing' && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => removeFile(uploadFile.id)}
                    disabled={isUploading}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Upload Button */}
        {files.length > 0 && (
          <Button
            onClick={handleUpload}
            disabled={isUploading}
            className="w-full"
          >
            {isUploading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Uploading...
              </>
            ) : (
              <>
                <Upload className="mr-2 h-4 w-4" />
                Upload {files.length} File{files.length !== 1 ? 's' : ''}
              </>
            )}
          </Button>
        )}
      </CardContent>
    </Card>
  )
}

