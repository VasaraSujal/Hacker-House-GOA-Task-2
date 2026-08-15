import { useCallback, useEffect, useRef, useState } from 'react'

const MIME_CANDIDATES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/mp4',
  'audio/mpeg',
  'audio/wav',
]

function supportedMimeType(): string {
  if (typeof MediaRecorder === 'undefined') return ''
  return MIME_CANDIDATES.find((type) => MediaRecorder.isTypeSupported(type)) ?? ''
}

interface RecorderOptions {
  maxDurationSeconds?: number
  onRecordingReady: (blob: Blob) => void
}

export function useVoiceRecorder({
  maxDurationSeconds = 30,
  onRecordingReady,
}: RecorderOptions) {
  const [isRecording, setIsRecording] = useState(false)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)
  const cancelledRef = useRef(false)

  const cleanupStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (timerRef.current !== null) window.clearInterval(timerRef.current)
    timerRef.current = null
  }, [])

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current
    if (recorder?.state === 'recording') recorder.stop()
  }, [])

  const cancelRecording = useCallback(() => {
    cancelledRef.current = true
    stopRecording()
    cleanupStream()
    setIsRecording(false)
    setElapsedSeconds(0)
  }, [cleanupStream, stopRecording])

  const startRecording = useCallback(async () => {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('This browser does not support microphone recording.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mimeType = supportedMimeType()
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      streamRef.current = stream
      recorderRef.current = recorder
      chunksRef.current = []
      cancelledRef.current = false
      setElapsedSeconds(0)

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || mimeType || 'audio/webm' })
        cleanupStream()
        setIsRecording(false)
        setElapsedSeconds(0)
        if (!cancelledRef.current && blob.size > 0) onRecordingReady(blob)
      }
      recorder.onerror = () => {
        cleanupStream()
        setIsRecording(false)
        setError('Recording failed. Please try again.')
      }
      recorder.start()
      setIsRecording(true)
      timerRef.current = window.setInterval(() => {
        setElapsedSeconds((seconds) => {
          if (seconds + 1 >= maxDurationSeconds) stopRecording()
          return Math.min(seconds + 1, maxDurationSeconds)
        })
      }, 1000)
    } catch (cause) {
      cleanupStream()
      const denied =
        cause instanceof DOMException &&
        (cause.name === 'NotAllowedError' || cause.name === 'PermissionDeniedError')
      setError(
        denied
          ? 'Microphone access was denied. Please allow microphone access and try again.'
          : 'Unable to access the microphone. Please try again.',
      )
    }
  }, [cleanupStream, maxDurationSeconds, onRecordingReady, stopRecording])

  useEffect(
    () => () => {
      cancelledRef.current = true
      if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
      cleanupStream()
    },
    [cleanupStream],
  )

  return {
    isRecording,
    elapsedSeconds,
    error,
    setError,
    startRecording,
    stopRecording,
    cancelRecording,
    supported: typeof MediaRecorder !== 'undefined' && Boolean(navigator.mediaDevices?.getUserMedia),
  }
}
