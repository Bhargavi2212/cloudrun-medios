import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface MetricCardProps {
  title: string
  value: string | number
  change?: number
  changeLabel?: string
  accent?: string
  icon?: React.ReactNode
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  changeLabel,
  accent = 'bg-blue-100 text-blue-700',
  icon,
}) => {
  const getTrendIcon = () => {
    if (change === undefined || change === null) return null
    if (change > 0) return <TrendingUp className="h-4 w-4 text-green-600" />
    if (change < 0) return <TrendingDown className="h-4 w-4 text-red-600" />
    return <Minus className="h-4 w-4 text-gray-400" />
  }

  const getTrendColor = () => {
    if (change === undefined || change === null) return 'text-gray-600'
    if (change > 0) return 'text-green-600'
    if (change < 0) return 'text-red-600'
    return 'text-gray-600'
  }

  // If accent is provided but no change/icon, use it as a simple badge style (backward compatibility)
  if (change === undefined && !icon) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-gray-500">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className={`inline-flex items-center px-3 py-2 rounded-md text-sm font-semibold ${accent}`}>
            {value}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon && <div className={accent}>{icon}</div>}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {change !== undefined && change !== null && (
          <div className={`text-xs flex items-center gap-1 mt-1 ${getTrendColor()}`}>
            {getTrendIcon()}
            <span>
              {Math.abs(change)}% {changeLabel || 'from last period'}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

