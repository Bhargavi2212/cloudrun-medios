import React, { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
// ScrollArea will be added when @radix-ui/react-scroll-area is installed
// import { ScrollArea } from '@/components/ui/scroll-area'
import { Download, FileText, Clock, User } from 'lucide-react'
import { formatDate, formatTime } from '@/lib/date-utils'

export interface HistoryItem {
  id: string
  title: string
  description?: string
  timestamp: string
  status: 'completed' | 'failed' | 'pending'
  type: 'note' | 'transcription' | 'triage' | 'summary'
  content?: string
  metadata?: Record<string, unknown>
}

interface HistoryViewProps {
  items: HistoryItem[]
  onDownload?: (item: HistoryItem) => void
  onView?: (item: HistoryItem) => void
  title?: string
  emptyMessage?: string
}

export const HistoryView: React.FC<HistoryViewProps> = ({
  items,
  onDownload,
  onView,
  title = 'History',
  emptyMessage = 'No history available',
}) => {
  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null)

  const getStatusBadge = (status: HistoryItem['status']) => {
    switch (status) {
      case 'completed':
        return <Badge variant="default" className="bg-green-100 text-green-700">Completed</Badge>
      case 'failed':
        return <Badge variant="destructive">Failed</Badge>
      case 'pending':
        return <Badge variant="secondary">Pending</Badge>
      default:
        return null
    }
  }

  const getTypeIcon = (type: HistoryItem['type']) => {
    switch (type) {
      case 'note':
        return <FileText className="h-4 w-4" />
      case 'transcription':
        return <FileText className="h-4 w-4" />
      case 'triage':
        return <User className="h-4 w-4" />
      case 'summary':
        return <FileText className="h-4 w-4" />
      default:
        return <FileText className="h-4 w-4" />
    }
  }

  const handleItemClick = (item: HistoryItem) => {
    setSelectedItem(item)
    if (onView) {
      onView(item)
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
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] overflow-y-auto">
            <div className="space-y-2">
              {items.map((item) => (
                <div
                  key={item.id}
                  className={`p-3 border rounded-lg cursor-pointer hover:bg-gray-50 transition-colors ${
                    selectedItem?.id === item.id ? 'bg-blue-50 border-blue-200' : ''
                  }`}
                  onClick={() => handleItemClick(item)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3 flex-1">
                      <div className="mt-1 text-gray-400">{getTypeIcon(item.type)}</div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h4 className="font-medium text-sm truncate">{item.title}</h4>
                          {getStatusBadge(item.status)}
                        </div>
                        {item.description && (
                          <p className="text-xs text-gray-500 truncate">{item.description}</p>
                        )}
                        <div className="flex items-center gap-2 mt-1 text-xs text-gray-400">
                          <Clock className="h-3 w-3" />
                          <span>
                            {formatDate(item.timestamp)} at {formatTime(item.timestamp)}
                          </span>
                        </div>
                      </div>
                    </div>
                    {onDownload && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation()
                          onDownload(item)
                        }}
                        className="ml-2"
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

      {selectedItem && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Details</span>
              {onDownload && (
                <Button variant="outline" size="sm" onClick={() => onDownload(selectedItem)}>
                  <Download className="h-4 w-4 mr-2" />
                  Download
                </Button>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">{selectedItem.title}</h4>
                {selectedItem.description && (
                  <p className="text-sm text-gray-600 mb-2">{selectedItem.description}</p>
                )}
                <div className="flex items-center gap-4 text-xs text-gray-500">
                  <span>{getStatusBadge(selectedItem.status)}</span>
                  <span>
                    {formatDate(selectedItem.timestamp)} at {formatTime(selectedItem.timestamp)}
                  </span>
                </div>
              </div>

              {selectedItem.content && (
                <div className="border rounded-lg p-4 bg-gray-50">
                  <h5 className="font-medium text-sm mb-2">Content</h5>
                  <pre className="text-xs whitespace-pre-wrap font-sans text-gray-700 max-h-64 overflow-y-auto">
                    {selectedItem.content}
                  </pre>
                </div>
              )}

              {selectedItem.metadata && Object.keys(selectedItem.metadata).length > 0 && (
                <div>
                  <h5 className="font-medium text-sm mb-2">Metadata</h5>
                  <div className="text-xs text-gray-600 space-y-1">
                    {Object.entries(selectedItem.metadata).map(([key, value]) => (
                      <div key={key}>
                        <span className="font-medium">{key}:</span> {String(value)}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

