import { FormEvent, useState } from 'react'
import { aiScribeAPI } from '@/services/api'
import type { ScribeVital } from '@/types'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { useToast } from '@/components/ui/use-toast'

interface Props {
  sessionId: string
  onRecorded?: (vital: ScribeVital) => void
}

const INITIAL_STATE = {
  heart_rate: '',
  respiratory_rate: '',
  systolic_bp: '',
  diastolic_bp: '',
  temperature_c: '',
  oxygen_saturation: '',
  pain_score: '',
}

export const VitalsPanel = ({ sessionId, onRecorded }: Props) => {
  const [formState, setFormState] = useState(INITIAL_STATE)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const { toast } = useToast()

  const handleChange = (evt: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = evt.target
    setFormState((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setIsSubmitting(true)
    try {
      const payload = {
        heart_rate: parseNumber(formState.heart_rate),
        respiratory_rate: parseNumber(formState.respiratory_rate),
        systolic_bp: parseNumber(formState.systolic_bp),
        diastolic_bp: parseNumber(formState.diastolic_bp),
        temperature_c: parseNumber(formState.temperature_c),
        oxygen_saturation: parseNumber(formState.oxygen_saturation),
        pain_score: parseNumber(formState.pain_score),
        source: 'manual',
      }
      const response = await aiScribeAPI.recordVitals(sessionId, payload)
      const newVital = response.vitals as ScribeVital
      onRecorded?.(newVital)
      toast({ title: 'Vitals recorded', description: 'Measurements synced with the scribe session.' })
      setFormState(INITIAL_STATE)
    } catch (error) {
      console.error(error)
      toast({
        title: 'Failed to record vitals',
        description: 'Please verify the values and try again.',
        variant: 'destructive',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        {renderField('Heart Rate', 'heart_rate', formState.heart_rate, handleChange, 'bpm')}
        {renderField('Respiratory Rate', 'respiratory_rate', formState.respiratory_rate, handleChange, 'rpm')}
        {renderField('Systolic BP', 'systolic_bp', formState.systolic_bp, handleChange, 'mmHg')}
        {renderField('Diastolic BP', 'diastolic_bp', formState.diastolic_bp, handleChange, 'mmHg')}
        {renderField('Temperature (°C)', 'temperature_c', formState.temperature_c, handleChange, '°C')}
        {renderField('SpO₂', 'oxygen_saturation', formState.oxygen_saturation, handleChange, '%')}
        {renderField('Pain Score', 'pain_score', formState.pain_score, handleChange, '/10')}
      </div>
      <Button type="submit" disabled={isSubmitting || !sessionId}>
        {isSubmitting ? 'Recording…' : 'Record Vitals'}
      </Button>
    </form>
  )
}

const renderField = (
  label: string,
  name: string,
  value: string,
  onChange: (evt: React.ChangeEvent<HTMLInputElement>) => void,
  suffix?: string,
) => (
  <div className="flex flex-col space-y-1">
    <Label htmlFor={name}>{label}</Label>
    <div className="flex items-center space-x-2">
      <Input id={name} name={name} value={value} onChange={onChange} type="number" step="any" className="flex-1" />
      {suffix && <span className="text-sm text-gray-500">{suffix}</span>}
    </div>
  </div>
)

const parseNumber = (value: string) => {
  if (value === '' || value === undefined) {
    return undefined
  }
  return Number(value)
}

