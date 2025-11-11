/**
 * Utility functions for exporting/downloading content
 */

/**
 * Download text content as a file
 */
export function downloadText(content: string, filename: string, mimeType: string = 'text/plain'): void {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

/**
 * Download content as JSON file
 */
export function downloadJSON(data: unknown, filename: string): void {
  const content = JSON.stringify(data, null, 2)
  downloadText(content, filename, 'application/json')
}

/**
 * Download note as text file
 */
export function downloadNote(note: string, patientName?: string, consultationId?: string): void {
  const timestamp = new Date().toISOString().split('T')[0]
  const filename = patientName
    ? `Clinical-Note-${patientName.replace(/\s+/g, '-')}-${timestamp}.txt`
    : consultationId
      ? `Clinical-Note-${consultationId}-${timestamp}.txt`
      : `Clinical-Note-${timestamp}.txt`
  downloadText(note, filename, 'text/plain')
}

/**
 * Download transcription as text file
 */
export function downloadTranscription(transcription: string, patientName?: string, consultationId?: string): void {
  const timestamp = new Date().toISOString().split('T')[0]
  const filename = patientName
    ? `Transcription-${patientName.replace(/\s+/g, '-')}-${timestamp}.txt`
    : consultationId
      ? `Transcription-${consultationId}-${timestamp}.txt`
      : `Transcription-${timestamp}.txt`
  downloadText(transcription, filename, 'text/plain')
}

/**
 * Download multiple items as a combined text file
 */
export function downloadCombined(
  items: Array<{ title: string; content: string }>,
  filename: string = 'export.txt',
): void {
  const content = items.map((item) => `=== ${item.title} ===\n\n${item.content}\n\n`).join('\n')
  downloadText(content, filename, 'text/plain')
}

/**
 * Copy text to clipboard
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch (error) {
    // Fallback for older browsers
    try {
      const textarea = document.createElement('textarea')
      textarea.value = text
      textarea.style.position = 'fixed'
      textarea.style.opacity = '0'
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
      return true
    } catch {
      return false
    }
  }
}

