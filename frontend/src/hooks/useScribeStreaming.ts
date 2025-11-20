import { useState, useRef, useCallback, useEffect } from 'react'
import { buildWebSocketUrl } from '@/services/api'
import type { ScribeSegment, ScribeVital, TriagePredictionSnapshot } from '@/types'

const PCM_CHUNK_SIZE = 4096

interface UseScribeStreamingOptions {
  sessionId?: string
}

export const useScribeStreaming = ({ sessionId }: UseScribeStreamingOptions) => {
  const audioWsRef = useRef<WebSocket | null>(null)
  const eventsWsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [segments, setSegments] = useState<ScribeSegment[]>([])
  const [vitals, setVitals] = useState<ScribeVital[]>([])
  const [triage, setTriage] = useState<TriagePredictionSnapshot | null>(null)
  const [events, setEvents] = useState<{ type: string; data: unknown }[]>([])
  const speakerRef = useRef<'doctor' | 'patient'>('doctor')

  const disconnectEvents = useCallback(() => {
    eventsWsRef.current?.close()
    eventsWsRef.current = null
  }, [])

  const connectEvents = useCallback(() => {
    if (!sessionId) return
    const url = buildWebSocketUrl(`/api/v1/scribe/sessions/${sessionId}/events`)
    const ws = new WebSocket(url)
    ws.onmessage = (message) => {
      try {
        const payload = JSON.parse(message.data)
        setEvents((prev) => [...prev.slice(-200), payload])
        switch (payload.type) {
          case 'scribe.transcript.delta':
            setSegments((prev) => [
              ...prev,
              {
                id: payload.data.id,
                session_id: sessionId,
                speaker_label: payload.data.speaker,
                text: payload.data.text,
                start_ms: payload.data.startMs,
                end_ms: payload.data.endMs,
                confidence: payload.data.confidence,
                created_at: new Date().toISOString(),
              },
            ])
            break
          case 'scribe.vitals.recorded':
            setVitals((prev) => [...prev, payload.data as ScribeVital])
            break
          case 'scribe.triage.updated':
            setTriage({
              id: payload.data?.prediction_id ?? Date.now(),
              esi_level: payload.data?.esi_level ?? 5,
              probability: payload.data?.probabilities?.[payload.data?.esi_level] ?? 0,
              probabilities: payload.data?.probabilities,
              flagged: payload.data?.flagged ?? false,
              created_at: new Date().toISOString(),
            })
            break
          default:
            break
        }
      } catch (error) {
        console.warn('Failed to parse scribe event', error)
      }
    }
    ws.onclose = () => {
      setTimeout(() => {
        if (isStreaming) {
          connectEvents()
        }
      }, 2000)
    }
    eventsWsRef.current = ws
  }, [sessionId, isStreaming])

  const startStreaming = useCallback(async () => {
    if (!sessionId || isStreaming) return
    const audioUrl = buildWebSocketUrl(`/api/v1/scribe/sessions/${sessionId}/audio`)
    const audioSocket = new WebSocket(audioUrl)
    audioWsRef.current = audioSocket

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const audioContext = new AudioContext({ sampleRate: 16000 })
    const sourceNode = audioContext.createMediaStreamSource(stream)
    const processorNode = audioContext.createScriptProcessor(PCM_CHUNK_SIZE, 1, 1)
    processorNode.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0)
      const pcmBuffer = convertFloat32ToInt16(input)
      if (audioSocket.readyState === WebSocket.OPEN) {
        // Convert ArrayBufferLike to ArrayBuffer
        const arrayBuffer = pcmBuffer.buffer instanceof ArrayBuffer 
          ? pcmBuffer.buffer 
          : new Uint8Array(pcmBuffer).buffer
        audioSocket.send(
          JSON.stringify({
            type: 'chunk',
            chunk: bufferToBase64(arrayBuffer),
            sampleRate: audioContext.sampleRate,
            speaker: speakerRef.current,
            timestampMs: Date.now(),
          }),
        )
      }
    }
    sourceNode.connect(processorNode)
    processorNode.connect(audioContext.destination)

    mediaStreamRef.current = stream
    audioContextRef.current = audioContext
    processorRef.current = processorNode
    setIsStreaming(true)
    connectEvents()
  }, [sessionId, isStreaming, connectEvents])

  const stopStreaming = useCallback(() => {
    setIsStreaming(false)
    disconnectEvents()
    audioWsRef.current?.close()
    audioWsRef.current = null
    processorRef.current?.disconnect()
    processorRef.current = null
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop())
    mediaStreamRef.current = null
    audioContextRef.current?.close()
    audioContextRef.current = null
  }, [disconnectEvents])

  useEffect(() => {
    setSegments([])
    setVitals([])
    setTriage(null)
  }, [sessionId])

  useEffect(() => {
    return () => {
      stopStreaming()
    }
  }, [stopStreaming])

  const setSpeaker = (speaker: 'doctor' | 'patient') => {
    speakerRef.current = speaker
  }

  return {
    isStreaming,
    startStreaming,
    stopStreaming,
    speaker: speakerRef.current,
    setSpeaker,
    segments,
    vitals,
    triage,
    events,
  }
}

const convertFloat32ToInt16 = (buffer: Float32Array): Int16Array => {
  const result = new Int16Array(buffer.length)
  for (let i = 0; i < buffer.length; i++) {
    const value = Math.max(-1, Math.min(1, buffer[i]))
    result[i] = value < 0 ? value * 0x8000 : value * 0x7fff
  }
  return result
}

const bufferToBase64 = (buffer: ArrayBuffer): string => {
  let binary = ''
  const bytes = new Uint8Array(buffer)
  const chunkSize = 0x8000
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize)
    binary += String.fromCharCode(...chunk)
  }
  return btoa(binary)
}

