import React from 'react'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { X } from 'lucide-react'
import type { QueueStage } from '@/types'

interface QueueFiltersProps {
  searchQuery: string
  onSearchChange: (query: string) => void
  statusFilter: QueueStage | 'all'
  onStatusFilterChange: (status: QueueStage | 'all') => void
  triageFilter: number | 'all'
  onTriageFilterChange: (triage: number | 'all') => void
  activeFiltersCount: number
  onClearFilters: () => void
}

export const QueueFilters: React.FC<QueueFiltersProps> = ({
  searchQuery,
  onSearchChange,
  statusFilter,
  onStatusFilterChange,
  triageFilter,
  onTriageFilterChange,
  activeFiltersCount,
  onClearFilters,
}) => {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-4 items-end">
        <div className="flex-1 min-w-[200px]">
          <Label htmlFor="search">Search Patients</Label>
          <Input
            id="search"
            placeholder="Search by name or complaint..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
          />
        </div>

        <div className="w-[180px]">
          <Label htmlFor="status">Status</Label>
          <Select value={statusFilter} onValueChange={(value) => onStatusFilterChange(value as QueueStage | 'all')}>
            <SelectTrigger id="status">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Statuses</SelectItem>
              <SelectItem value="waiting">Waiting</SelectItem>
              <SelectItem value="triage">Triage</SelectItem>
              <SelectItem value="scribe">In Consultation</SelectItem>
              <SelectItem value="discharge">Discharged</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="w-[180px]">
          <Label htmlFor="triage">Triage Level</Label>
          <Select
            value={triageFilter.toString()}
            onValueChange={(value) => onTriageFilterChange(value === 'all' ? 'all' : parseInt(value, 10))}
          >
            <SelectTrigger id="triage">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Levels</SelectItem>
              <SelectItem value="1">ESI 1 (Critical)</SelectItem>
              <SelectItem value="2">ESI 2 (Emergent)</SelectItem>
              <SelectItem value="3">ESI 3 (Urgent)</SelectItem>
              <SelectItem value="4">ESI 4 (Less Urgent)</SelectItem>
              <SelectItem value="5">ESI 5 (Non-urgent)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {activeFiltersCount > 0 && (
          <Button variant="outline" size="sm" onClick={onClearFilters} className="gap-2">
            <X className="h-4 w-4" />
            Clear Filters ({activeFiltersCount})
          </Button>
        )}
      </div>

      {activeFiltersCount > 0 && (
        <div className="flex flex-wrap gap-2">
          {statusFilter !== 'all' && (
            <Badge variant="secondary" className="gap-1">
              Status: {statusFilter}
              <button
                onClick={() => onStatusFilterChange('all')}
                className="ml-1 hover:text-red-600"
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          )}
          {triageFilter !== 'all' && (
            <Badge variant="secondary" className="gap-1">
              Triage: ESI {triageFilter}
              <button
                onClick={() => onTriageFilterChange('all')}
                className="ml-1 hover:text-red-600"
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          )}
        </div>
      )}
    </div>
  )
}

