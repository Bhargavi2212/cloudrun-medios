import React, { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { CheckCircle2, XCircle, Send, AlertCircle } from 'lucide-react'
import { useToast } from '@/components/ui/use-toast'

export type NoteStatus = 'draft' | 'pending_approval' | 'approved' | 'rejected'

interface NoteApprovalWorkflowProps {
  noteStatus: NoteStatus
  consultationId?: string
  onStatusChange?: (newStatus: NoteStatus) => void
  onSubmit?: () => Promise<void>
  onApprove?: () => Promise<void>
  onReject?: (reason: string) => Promise<void>
  canApprove?: boolean
  canReject?: boolean
  canSubmit?: boolean
}

export const NoteApprovalWorkflow: React.FC<NoteApprovalWorkflowProps> = ({
  noteStatus,
  consultationId,
  onStatusChange,
  onSubmit,
  onApprove,
  onReject,
  canApprove = false,
  canReject = false,
  canSubmit = true,
}) => {
  const { toast } = useToast()
  const [isRejectDialogOpen, setIsRejectDialogOpen] = useState(false)
  const [rejectionReason, setRejectionReason] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const getStatusBadge = (status: NoteStatus) => {
    switch (status) {
      case 'draft':
        return <Badge variant="secondary">Draft</Badge>
      case 'pending_approval':
        return <Badge variant="outline" className="bg-yellow-100 text-yellow-700">Pending Approval</Badge>
      case 'approved':
        return <Badge variant="default" className="bg-green-100 text-green-700">Approved</Badge>
      case 'rejected':
        return <Badge variant="destructive">Rejected</Badge>
      default:
        return <Badge variant="secondary">{status}</Badge>
    }
  }

  const handleSubmit = async () => {
    if (!onSubmit) return
    try {
      setIsSubmitting(true)
      await onSubmit()
      onStatusChange?.('pending_approval')
      toast({
        title: 'Note submitted',
        description: 'Note has been submitted for approval.',
      })
    } catch (error: any) {
      toast({
        title: 'Submission failed',
        description: error?.message || 'Failed to submit note for approval',
        variant: 'destructive',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleApprove = async () => {
    if (!onApprove) return
    try {
      setIsSubmitting(true)
      await onApprove()
      onStatusChange?.('approved')
      toast({
        title: 'Note approved',
        description: 'Note has been approved.',
      })
    } catch (error: any) {
      toast({
        title: 'Approval failed',
        description: error?.message || 'Failed to approve note',
        variant: 'destructive',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleReject = async () => {
    if (!onReject) return
    try {
      setIsSubmitting(true)
      await onReject(rejectionReason)
      onStatusChange?.('rejected')
      setIsRejectDialogOpen(false)
      setRejectionReason('')
      toast({
        title: 'Note rejected',
        description: 'Note has been rejected.',
      })
    } catch (error: any) {
      toast({
        title: 'Rejection failed',
        description: error?.message || 'Failed to reject note',
        variant: 'destructive',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Note Approval</span>
            {getStatusBadge(noteStatus)}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="text-sm text-gray-600">
            {noteStatus === 'draft' && (
              <p>This note is in draft status. Submit it for approval when ready.</p>
            )}
            {noteStatus === 'pending_approval' && (
              <p className="flex items-center gap-2">
                <AlertCircle className="h-4 w-4 text-yellow-600" />
                This note is pending approval. Waiting for review.
              </p>
            )}
            {noteStatus === 'approved' && (
              <p className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                This note has been approved and is finalized.
              </p>
            )}
            {noteStatus === 'rejected' && (
              <p className="flex items-center gap-2">
                <XCircle className="h-4 w-4 text-red-600" />
                This note has been rejected. Please review and make necessary changes.
              </p>
            )}
          </div>

          <div className="flex flex-wrap gap-2">
            {noteStatus === 'draft' && canSubmit && (
              <Button
                onClick={handleSubmit}
                disabled={isSubmitting}
                className="gap-2"
              >
                <Send className="h-4 w-4" />
                {isSubmitting ? 'Submitting...' : 'Submit for Approval'}
              </Button>
            )}

            {noteStatus === 'pending_approval' && (
              <>
                {canApprove && (
                  <Button
                    onClick={handleApprove}
                    disabled={isSubmitting}
                    variant="default"
                    className="gap-2 bg-green-600 hover:bg-green-700"
                  >
                    <CheckCircle2 className="h-4 w-4" />
                    {isSubmitting ? 'Approving...' : 'Approve Note'}
                  </Button>
                )}
                {canReject && (
                  <Button
                    onClick={() => setIsRejectDialogOpen(true)}
                    disabled={isSubmitting}
                    variant="destructive"
                    className="gap-2"
                  >
                    <XCircle className="h-4 w-4" />
                    Reject Note
                  </Button>
                )}
              </>
            )}

            {noteStatus === 'rejected' && canSubmit && (
              <Button
                onClick={handleSubmit}
                disabled={isSubmitting}
                variant="outline"
                className="gap-2"
              >
                <Send className="h-4 w-4" />
                {isSubmitting ? 'Resubmitting...' : 'Resubmit for Approval'}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      <Dialog open={isRejectDialogOpen} onOpenChange={setIsRejectDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reject Note</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <p className="text-sm text-gray-600">
              Please provide a reason for rejecting this note. This will help the author make necessary corrections.
            </p>
            <Textarea
              placeholder="Enter rejection reason (optional)"
              value={rejectionReason}
              onChange={(e) => setRejectionReason(e.target.value)}
              rows={4}
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsRejectDialogOpen(false)
                setRejectionReason('')
              }}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleReject}
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Rejecting...' : 'Reject Note'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

