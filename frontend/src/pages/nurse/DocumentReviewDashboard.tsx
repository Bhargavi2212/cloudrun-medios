import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { manageAPI } from '@/services/api'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Textarea } from '@/components/ui/textarea'
import { useToast } from '@/hooks/use-toast'
import {
  CheckCircle2,
  XCircle,
  Eye,
  FileText,
  AlertTriangle,
  Loader2,
  RefreshCw,
  Download,
} from 'lucide-react'
import type { ConsultationRecord } from '@/types'
// Date formatting utility
const formatDate = (dateString: string): string => {
  try {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'just now'
    if (diffMins < 60) return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`
    if (diffHours < 24) return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`
    if (diffDays < 7) return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`
    return date.toLocaleDateString()
  } catch {
    return dateString
  }
}

export function DocumentReviewDashboard() {
  const [selectedDocument, setSelectedDocument] = useState<ConsultationRecord | null>(null)
  const [extractionData, setExtractionData] = useState<Record<string, unknown> | null>(null)
  const [isViewingExtraction, setIsViewingExtraction] = useState(false)
  const [confirmNotes, setConfirmNotes] = useState('')
  const [rejectReason, setRejectReason] = useState('')
  const { toast } = useToast()
  const queryClient = useQueryClient()

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['pending-documents'],
    queryFn: () => manageAPI.listPendingDocuments(),
  })

  const confirmMutation = useMutation({
    mutationFn: (fileId: string) => manageAPI.confirmDocument(fileId, confirmNotes || undefined),
    onSuccess: () => {
      toast({
        title: 'Document confirmed',
        description: 'The document has been approved and added to the timeline.',
      })
      setSelectedDocument(null)
      setConfirmNotes('')
      queryClient.invalidateQueries({ queryKey: ['pending-documents'] })
      queryClient.invalidateQueries({ queryKey: ['consultation-records'] })
    },
    onError: (error) => {
      toast({
        title: 'Confirmation failed',
        description: error instanceof Error ? error.message : 'Failed to confirm document.',
        variant: 'destructive',
      })
    },
  })

  const rejectMutation = useMutation({
    mutationFn: (fileId: string) => manageAPI.rejectDocument(fileId, rejectReason || undefined),
    onSuccess: () => {
      toast({
        title: 'Document rejected',
        description: 'The document has been rejected. Please request a re-upload.',
      })
      setSelectedDocument(null)
      setRejectReason('')
      queryClient.invalidateQueries({ queryKey: ['pending-documents'] })
    },
    onError: (error) => {
      toast({
        title: 'Rejection failed',
        description: error instanceof Error ? error.message : 'Failed to reject document.',
        variant: 'destructive',
      })
    },
  })

  const handleViewExtraction = async (record: ConsultationRecord) => {
    try {
      const data = await manageAPI.getDocumentExtraction(record.id)
      setExtractionData(data)
      setSelectedDocument(record)
      setIsViewingExtraction(true)
    } catch (error) {
      toast({
        title: 'Failed to load extraction',
        description: error instanceof Error ? error.message : 'Could not load document extraction data.',
        variant: 'destructive',
      })
    }
  }

  const handleConfirm = () => {
    if (selectedDocument) {
      confirmMutation.mutate(selectedDocument.id)
    }
  }

  const handleReject = () => {
    if (selectedDocument) {
      rejectMutation.mutate(selectedDocument.id)
    }
  }

  const getStatusBadge = (record: ConsultationRecord) => {
    if (record.status === 'needs_review') {
      return <Badge variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-300">Needs Review</Badge>
    }
    if (record.status === 'failed') {
      return <Badge variant="destructive">Failed</Badge>
    }
    return <Badge variant="secondary">{record.status}</Badge>
  }

  const getConfidenceBadge = (confidence?: number | null) => {
    if (!confidence) return null
    if (confidence >= 0.9) {
      return <Badge className="bg-green-500">High ({Math.round(confidence * 100)}%)</Badge>
    }
    if (confidence >= 0.75) {
      return <Badge className="bg-yellow-500">Medium ({Math.round(confidence * 100)}%)</Badge>
    }
    return <Badge className="bg-red-500">Low ({Math.round(confidence * 100)}%)</Badge>
  }


  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
      </div>
    )
  }

  const records = data?.records || []
  const count = data?.count || 0

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Document Review</h1>
          <p className="text-muted-foreground mt-1">
            Review and approve documents that need manual verification
          </p>
        </div>
        <Button onClick={() => refetch()} variant="outline" size="sm">
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh
        </Button>
      </div>

      {count === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <CheckCircle2 className="h-12 w-12 text-green-500 mb-4" />
            <h3 className="text-lg font-semibold mb-2">All caught up!</h3>
            <p className="text-muted-foreground text-center">
              There are no documents pending review at this time.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4">
          {records.map((record) => (
            <Card key={record.id} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <FileText className="h-5 w-5 text-gray-400" />
                      <CardTitle className="text-lg">{record.original_filename || 'Untitled Document'}</CardTitle>
                    </div>
                    <CardDescription>
                      Uploaded {record.uploaded_at ? formatDate(record.uploaded_at) : 'Unknown'} • {record.document_type || 'Unknown type'}
                      {record.size_bytes && ` • ${(record.size_bytes / 1024).toFixed(1)} KB`}
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    {getStatusBadge(record)}
                    {getConfidenceBadge(record.confidence)}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {record.processing_notes && (
                  <div className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-950 rounded-md">
                    <div className="flex items-start gap-2">
                      <AlertTriangle className="h-4 w-4 text-yellow-600 mt-0.5" />
                      <p className="text-sm text-yellow-800 dark:text-yellow-200">{record.processing_notes}</p>
                    </div>
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleViewExtraction(record)}
                  >
                    <Eye className="mr-2 h-4 w-4" />
                    View Extraction
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => window.open(record.download_url, '_blank')}
                  >
                    <Download className="mr-2 h-4 w-4" />
                    Download
                  </Button>
                  <Button
                    variant="default"
                    size="sm"
                    onClick={() => {
                      setSelectedDocument(record)
                      setIsViewingExtraction(false)
                    }}
                    className="ml-auto"
                  >
                    <CheckCircle2 className="mr-2 h-4 w-4" />
                    Review
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Review Dialog */}
      <Dialog open={selectedDocument !== null && !isViewingExtraction} onOpenChange={(open) => {
        if (!open) {
          setSelectedDocument(null)
          setConfirmNotes('')
          setRejectReason('')
        }
      }}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Review Document</DialogTitle>
            <DialogDescription>
              {selectedDocument?.original_filename || 'Untitled Document'}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <h4 className="font-medium mb-2">Status</h4>
              <div className="flex items-center gap-2">
                {getStatusBadge(selectedDocument!)}
                {getConfidenceBadge(selectedDocument?.confidence)}
              </div>
            </div>
            {selectedDocument?.processing_notes && (
              <div>
                <h4 className="font-medium mb-2">Processing Notes</h4>
                <p className="text-sm text-muted-foreground">{selectedDocument.processing_notes}</p>
              </div>
            )}
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => handleViewExtraction(selectedDocument!)}
              >
                <Eye className="mr-2 h-4 w-4" />
                View Extraction Data
              </Button>
              <Button
                variant="outline"
                onClick={() => window.open(selectedDocument?.download_url, '_blank')}
              >
                <Download className="mr-2 h-4 w-4" />
                Download Original
              </Button>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Confirmation Notes (optional)</label>
              <Textarea
                value={confirmNotes}
                onChange={(e) => setConfirmNotes(e.target.value)}
                placeholder="Add any notes about this confirmation..."
                className="min-h-[80px]"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Rejection Reason (if rejecting)</label>
              <Textarea
                value={rejectReason}
                onChange={(e) => setRejectReason(e.target.value)}
                placeholder="Explain why this document should be rejected..."
                className="min-h-[80px]"
              />
            </div>
            <div className="flex justify-end gap-2 pt-4 border-t">
              <Button
                variant="outline"
                onClick={() => {
                  setSelectedDocument(null)
                  setConfirmNotes('')
                  setRejectReason('')
                }}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleReject}
                disabled={rejectMutation.isPending}
              >
                {rejectMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <XCircle className="mr-2 h-4 w-4" />
                )}
                Reject
              </Button>
              <Button
                onClick={handleConfirm}
                disabled={confirmMutation.isPending}
              >
                {confirmMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <CheckCircle2 className="mr-2 h-4 w-4" />
                )}
                Confirm
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Extraction Data Dialog */}
      <Dialog open={isViewingExtraction} onOpenChange={(open) => {
        if (!open) {
          setIsViewingExtraction(false)
          setExtractionData(null)
        }
      }}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Extraction Data</DialogTitle>
            <DialogDescription>
              View the extracted data from {selectedDocument?.original_filename || 'the document'}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            {extractionData && (
              <>
                <div>
                  <h4 className="font-medium mb-2">Confidence</h4>
                  {getConfidenceBadge(extractionData.confidence as number)}
                </div>
                <div>
                  <h4 className="font-medium mb-2">Extracted Data</h4>
                  <pre className="p-4 bg-gray-50 dark:bg-gray-900 rounded-md overflow-x-auto text-xs">
                    {JSON.stringify(extractionData.extraction_data, null, 2)}
                  </pre>
                </div>
                {extractionData.raw_text_preview && (
                  <div>
                    <h4 className="font-medium mb-2">Raw Text Preview</h4>
                    <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-md text-sm max-h-64 overflow-y-auto">
                      {extractionData.raw_text_preview}
                    </div>
                  </div>
                )}
              </>
            )}
            <div className="flex justify-end gap-2 pt-4 border-t">
              <Button
                variant="outline"
                onClick={() => {
                  setIsViewingExtraction(false)
                  setExtractionData(null)
                }}
              >
                Close
              </Button>
              <Button
                onClick={() => {
                  setIsViewingExtraction(false)
                  setExtractionData(null)
                  setSelectedDocument(selectedDocument)
                }}
              >
                Review Document
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

