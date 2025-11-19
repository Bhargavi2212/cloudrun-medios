from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

from backend.database import crud
from backend.database.models import ScribeSessionStatus
from backend.database.session import get_session

from .transcription import TranscriptResult, WhisperStreamTranscriber

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    payload: bytes
    sample_rate: int
    speaker: str
    timestamp_ms: int


@dataclass
class StreamEvent:
    type: str
    payload: dict


@dataclass
class ScribeStreamState:
    session_id: str
    audio_queue: asyncio.Queue[Optional[AudioChunk]] = field(default_factory=lambda: asyncio.Queue(maxsize=32))
    event_queues: Set[asyncio.Queue[StreamEvent]] = field(default_factory=set)
    worker_task: Optional[asyncio.Task] = None
    closed: bool = False
    last_offset_ms: int = 0


class AudioGatewayService:
    """Manages streaming audio ingestion and Whisper transcription workers."""

    def __init__(self) -> None:
        self._states: Dict[str, ScribeStreamState] = {}
        self._lock = asyncio.Lock()
        self.transcriber = WhisperStreamTranscriber()

    # ------------------------------------------------------------------ #
    # Session lifecycle helpers
    # ------------------------------------------------------------------ #

    async def get_state(self, session_id: str) -> ScribeStreamState:
        async with self._lock:
            state = self._states.get(session_id)
            if state is None:
                state = ScribeStreamState(session_id=session_id)
                self._states[session_id] = state
            return state

    async def close_state(self, session_id: str) -> None:
        async with self._lock:
            state = self._states.get(session_id)
            if state is None:
                return
            state.closed = True
            await state.audio_queue.put(None)
            if state.worker_task:
                await asyncio.gather(state.worker_task, return_exceptions=True)
            for queue in list(state.event_queues):
                queue.put_nowait(
                    StreamEvent(
                        type="scribe.session.closed",
                        payload={"session_id": session_id},
                    )
                )
            self._states.pop(session_id, None)

    # ------------------------------------------------------------------ #
    # Event listeners
    # ------------------------------------------------------------------ #

    async def register_listener(self, session_id: str) -> asyncio.Queue[StreamEvent]:
        state = await self.get_state(session_id)
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=128)
        state.event_queues.add(queue)
        return queue

    async def unregister_listener(self, session_id: str, queue: asyncio.Queue[StreamEvent]) -> None:
        state = await self.get_state(session_id)
        state.event_queues.discard(queue)

    async def publish_event(self, session_id: str, event_type: str, payload: dict) -> None:
        state = await self.get_state(session_id)
        for queue in list(state.event_queues):
            try:
                queue.put_nowait(StreamEvent(type=event_type, payload=payload))
            except asyncio.QueueFull:
                logger.debug("Dropping scribe event for session %s due to slow consumer.", session_id)

    async def stream_events(self, session_id: str, websocket: WebSocket) -> None:
        queue = await self.register_listener(session_id)
        await websocket.accept()
        try:
            while True:
                event = await queue.get()
                await websocket.send_json({"type": event.type, "data": event.payload})
        except WebSocketDisconnect:
            logger.info("Event stream disconnected for session %s", session_id)
        finally:
            await self.unregister_listener(session_id, queue)
            await websocket.close()

    # ------------------------------------------------------------------ #
    # Audio ingestion
    # ------------------------------------------------------------------ #

    async def handle_audio_stream(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        logger.info("Audio stream connected for session %s", session_id)
        state = await self.get_state(session_id)
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                msg_type = data.get("type")
                if msg_type == "chunk":
                    chunk_b64 = data.get("chunk")
                    if not chunk_b64:
                        continue
                    chunk = AudioChunk(
                        payload=base64.b64decode(chunk_b64),
                        sample_rate=int(data.get("sampleRate") or 16000),
                        speaker=str(data.get("speaker") or "unknown"),
                        timestamp_ms=int(data.get("timestampMs") or 0),
                    )
                    await self._enqueue_chunk(state, chunk)
                elif msg_type == "stop":
                    break
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                else:
                    logger.debug("Unknown audio gateway message %s", msg_type)
        except WebSocketDisconnect:
            logger.info("Audio stream disconnected for session %s", session_id)
        except Exception:  # pragma: no cover
            logger.exception("Audio stream crashed for session %s", session_id)
        finally:
            await websocket.close()
            await self.close_state(session_id)

    async def _enqueue_chunk(self, state: ScribeStreamState, chunk: AudioChunk) -> None:
        if state.closed:
            return
        await state.audio_queue.put(chunk)
        await self._ensure_worker(state)
        await self._ensure_session_started(state.session_id)

    async def _ensure_worker(self, state: ScribeStreamState) -> None:
        if state.worker_task is None or state.worker_task.done():
            state.worker_task = asyncio.create_task(self._run_transcription_loop(state))

    async def _ensure_session_started(self, session_id: str) -> None:
        with get_session() as db:
            session_obj = crud.get_scribe_session(db, session_id)
            if session_obj and session_obj.status == ScribeSessionStatus.CREATED:
                crud.update_scribe_session(
                    db,
                    session_obj,
                    status=ScribeSessionStatus.STREAMING,
                )

    async def _run_transcription_loop(self, state: ScribeStreamState) -> None:
        logger.debug("Starting transcription loop for session %s", state.session_id)
        while True:
            chunk = await state.audio_queue.get()
            if chunk is None:
                break
            try:
                result = await self.transcriber.transcribe(chunk.payload, chunk.sample_rate)
            except RuntimeError as exc:
                logger.error("Whisper unavailable: %s", exc)
                await self.publish_event(
                    state.session_id,
                    "scribe.error",
                    {"message": str(exc)},
                )
                await asyncio.sleep(1)
                continue
            except Exception as exc:  # pragma: no cover
                logger.exception("Failed to transcribe chunk: %s", exc)
                await self.publish_event(
                    state.session_id,
                    "scribe.error",
                    {"message": "transcription_failed", "detail": str(exc)},
                )
                continue

            if result.text.strip():
                await self._persist_segment(state.session_id, chunk, result)
        logger.debug("Transcription loop completed for session %s", state.session_id)

    async def _persist_segment(self, session_id: str, chunk: AudioChunk, result: TranscriptResult) -> None:
        start_ms = chunk.timestamp_ms or 0
        end_ms = start_ms + int(result.duration_ms)
        metadata = {"language": result.language}
        with get_session() as db:
            session_obj = crud.get_scribe_session(db, session_id)
            segment = crud.add_transcript_segment(
                db,
                session_id=session_id,
                speaker_label=chunk.speaker,
                text=result.text.strip(),
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=result.confidence,
                metadata=metadata,
            )
            if session_obj:
                snapshot = " ".join(filter(None, [session_obj.transcript_snapshot, result.text.strip()]))
                crud.update_scribe_session(
                    db,
                    session_obj,
                    transcript_snapshot=snapshot.strip(),
                    language=session_obj.language or result.language,
                )
        await self.publish_event(
            session_id,
            "scribe.transcript.delta",
            {
                "id": segment.id,
                "text": segment.text,
                "speaker": segment.speaker_label,
                "startMs": start_ms,
                "endMs": end_ms,
                "confidence": result.confidence,
                "language": result.language,
            },
        )

