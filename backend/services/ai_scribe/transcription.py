from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.services.model_manager import get_whisper_model

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    text: str
    confidence: float
    language: str
    duration_ms: float


class WhisperStreamTranscriber:
    """Thin wrapper responsible for running Whisper in thread executors."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._model_cached = False

    async def transcribe(self, payload: bytes, sample_rate: int) -> TranscriptResult:
        if not payload:
            return TranscriptResult(text="", confidence=0.0, language="en", duration_ms=0.0)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, payload, sample_rate)

    def _transcribe_sync(self, payload: bytes, sample_rate: int) -> TranscriptResult:
        model = get_whisper_model()
        audio = self._decode_pcm(payload, sample_rate)
        if audio.size == 0:
            return TranscriptResult(text="", confidence=0.0, language="en", duration_ms=0.0)
        result = model.transcribe(audio, fp16=False, condition_on_previous_text=False)
        text = (result or {}).get("text", "").strip()
        language = (result or {}).get("language", "en")
        segments = (result or {}).get("segments") or []
        confidence = self._estimate_confidence(segments)
        duration_ms = self._estimate_duration_ms(segments, len(audio))
        return TranscriptResult(text=text, confidence=confidence, language=language, duration_ms=duration_ms)

    def _decode_pcm(self, payload: bytes, sample_rate: int) -> np.ndarray:
        waveform = np.frombuffer(payload, dtype=np.int16).astype(np.float32)
        if waveform.size == 0:
            return waveform
        waveform /= 32768.0
        target_sr = 16000
        if sample_rate != target_sr and sample_rate > 0:
            waveform = self._resample(waveform, sample_rate, target_sr)
        return waveform

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        duration = audio.shape[0] / orig_sr
        target_length = max(int(duration * target_sr), 1)
        original_times = np.linspace(0, duration, num=audio.shape[0], endpoint=False)
        target_times = np.linspace(0, duration, num=target_length, endpoint=False)
        resampled = np.interp(target_times, original_times, audio)
        return resampled.astype(np.float32)

    @staticmethod
    def _estimate_confidence(segments: list[dict]) -> float:
        if not segments:
            return 0.0
        scores = []
        for seg in segments:
            if "avg_logprob" in seg and seg["avg_logprob"] is not None:
                scores.append(seg["avg_logprob"])
        if not scores:
            return 0.0
        # Whisper logprobs roughly between -1 and 0; map to 0..1
        normalized = [max(min((score + 1.0) / 1.0, 1.0), 0.0) for score in scores]
        return float(sum(normalized) / len(normalized))

    @staticmethod
    def _estimate_duration_ms(segments: list[dict], sample_count: int) -> float:
        if segments:
            start = segments[0].get("start", 0.0) or 0.0
            end = segments[-1].get("end") or start
            return max((end - start) * 1000.0, 0.0)
        return (sample_count / 16000.0) * 1000.0

