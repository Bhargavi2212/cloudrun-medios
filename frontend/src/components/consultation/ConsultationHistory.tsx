import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
// Using native overflow instead of ScrollArea component
import { FileText, Clock, User, Download } from 'lucide-react'
import { formatDate, formatTime, formatRelativeTime } from '@/lib/date-utils'
import { downloadNote, downloadTranscription } from '@/lib/export-utils'

export interface ConsultationHistoryItem {
  id: string
  type: 'note' | 'transcription' | 'vitals' | 'document' | 'triage'
  title: string
  description?: string
  timestamp: string
  status: 'completed' | 'pending' | 'failed'
  content?: string
  metadata?: Record<string, unknown>
  patientName?: string
  consultationId?: string
}

interface ConsultationHistoryProps {
  items: ConsultationHistoryItem[]
  patientName?: string
  consultationId?: string
  onItemClick?: (item: ConsultationHistoryItem) => void
  title?: string
  emptyMessage?: string
}

export const ConsultationHistory: React.FC<ConsultationHistoryProps> = ({
  items,
  patientName,
  consultationId,
  onItemClick,
  title = 'Consultation History',
  emptyMessage = 'No history available for this consultation',
}) => {
  const getTypeIcon = (type: ConsultationHistoryItem['type']) => {
    switch (type) {
      case 'note':
        return <FileText className="h-4 w-4 text-blue-600" />
      case 'transcription':
        return <FileText className="h-4 w-4 text-green-600" />
      case 'vitals':
        return <User className="h-4 w-4 text-purple-600" />
      case 'document':
        return <FileText className="h-4 w-4 text-orange-600" />
      case 'triage':
        return <User className="h-4 w-4 text-red-600" />
      default:
        return <FileText className="h-4 w-4" />
    }
  }

  const getStatusBadge = (status: ConsultationHistoryItem['status']) => {
    switch (status) {
      case 'completed':
        return <Badge variant="default" className="bg-green-100 text-green-700">Completed</Badge>
      case 'pending':
        return <Badge variant="secondary">Pending</Badge>
      case 'failed':
        return <Badge variant="destructive">Failed</Badge>
      default:
        return null
    }
  }

  const handleDownload = (item: ConsultationHistoryItem) => {
    if (!item.content) return

    if (item.type === 'note') {
      downloadNote(item.content, patientName, consultationId)
    } else if (item.type === 'transcription') {
      downloadTranscription(item.content, patientName, consultationId)
    }
  }

  if (items.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-500 text-center py-8">{emptyMessage}</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[500px] overflow-y-auto">
          <div className="space-y-3">
            {items.map((item) => (
              <div
                key={item.id}
                className={`p-4 border rounded-lg hover:bg-gray-50 transition-colors ${
                  onItemClick ? 'cursor-pointer' : ''
                }`}
                onClick={() => onItemClick?.(item)}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3 flex-1 min-w-0">
                    <div className="mt-1">{getTypeIcon(item.type)}</div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-medium text-sm truncate">{item.title}</h4>
                        {getStatusBadge(item.status)}
                      </div>
                      {item.description && (
                        <p className="text-xs text-gray-500 truncate mb-2">{item.description}</p>
                      )}
                      <div className="flex items-center gap-2 text-xs text-gray-400">
                        <Clock className="h-3 w-3" />
                        <span>{formatRelativeTime(item.timestamp)}</span>
                        <span className="mx-1">•</span>
                        <span>
                          {formatDate(item.timestamp)} at {formatTime(item.timestamp)}
                        </span>
                      </div>
                      {item.content && (
                        <p className="text-xs text-gray-600 mt-2 line-clamp-2">{item.content}</p>
                      )}
                    </div>
                  </div>
                  {item.content && (item.type === 'note' || item.type === 'transcription') && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleDownload(item)
                      }}
                      title="Download"
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

