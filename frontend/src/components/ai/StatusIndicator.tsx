import React from 'react'
import { Badge } from '@/components/ui/badge'
import { Loader2, CheckCircle2, XCircle, AlertCircle, Clock } from 'lucide-react'

export type ProcessingStatus = 'idle' | 'processing' | 'completed' | 'failed' | 'warning'

interface StatusIndicatorProps {
  status: ProcessingStatus
  message?: string
  size?: 'sm' | 'md' | 'lg'
}

const statusConfig: Record<ProcessingStatus, { icon: React.ReactNode; color: string; variant: 'default' | 'destructive' | 'secondary' | 'outline' }> = {
  idle: {
    icon: <Clock className="h-3 w-3" />,
    color: 'text-gray-500',
    variant: 'secondary',
  },
  processing: {
    icon: <Loader2 className="h-3 w-3 animate-spin" />,
    color: 'text-blue-500',
    variant: 'default',
  },
  completed: {
    icon: <CheckCircle2 className="h-3 w-3" />,
    color: 'text-green-500',
    variant: 'default',
  },
  failed: {
    icon: <XCircle className="h-3 w-3" />,
    color: 'text-red-500',
    variant: 'destructive',
  },
  warning: {
    icon: <AlertCircle className="h-3 w-3" />,
    color: 'text-yellow-500',
    variant: 'outline',
  },
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status, message, size = 'md' }) => {
  const config = statusConfig[status]
  const sizeClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  }

  return (
    <Badge variant={config.variant} className={`${config.color} ${sizeClasses[size]} gap-1`}>
      {config.icon}
      {message && <span>{message}</span>}
      {!message && (
        <span className="capitalize">
          {status === 'processing' ? 'Processing...' : status}
        </span>
      )}
    </Badge>
  )
}

