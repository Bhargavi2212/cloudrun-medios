import { describe, it, expect } from 'vitest'
import { render, screen } from '../utils'
import { StatusIndicator, ProcessingStatus } from '@/components/ai/StatusIndicator'

describe('StatusIndicator', () => {
  const statuses: ProcessingStatus[] = ['idle', 'processing', 'completed', 'failed', 'warning']

  statuses.forEach((status) => {
    it(`renders ${status} status correctly`, () => {
      render(<StatusIndicator status={status} message={`Test ${status} message`} />)
      
      // Check that the message is displayed - this is the main functionality
      expect(screen.getByText(`Test ${status} message`)).toBeInTheDocument()
    })
  })

  it('displays custom message when provided', () => {
    const message = 'Custom status message'
    render(<StatusIndicator status="processing" message={message} />)
    
    expect(screen.getByText(message)).toBeInTheDocument()
  })

  it('renders with different sizes', () => {
    const { rerender } = render(<StatusIndicator status="completed" size="sm" />)
    expect(screen.getByText(/completed/i)).toBeInTheDocument()

    rerender(<StatusIndicator status="completed" size="lg" />)
    expect(screen.getByText(/completed/i)).toBeInTheDocument()
  })
})

