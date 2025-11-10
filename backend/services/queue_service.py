from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.orm import Session

from backend.database import crud
from backend.database.models import ConsultationStatus, QueueStage, QueueState
from backend.database.schemas import QueueStateRead
from backend.database.session import get_session

ALLOWED_TRANSITIONS: Dict[QueueStage, List[QueueStage]] = {
    QueueStage.WAITING: [QueueStage.TRIAGE],
    QueueStage.TRIAGE: [QueueStage.SCRIBE],
    QueueStage.SCRIBE: [QueueStage.DISCHARGE],
    QueueStage.DISCHARGE: [],
}


@dataclass
class QueueSnapshot:
    states: List[QueueStateRead]
    totals_by_stage: Dict[str, int]
    average_wait_seconds: Optional[int]


class QueueNotifier:
    """Tracks websocket and SSE connections and broadcasts queue updates."""

    def __init__(self) -> None:
        self._websocket_connections: Set[Any] = set()
        self._sse_connections: Set[Any] = set()  # SSE connections are async queues
        self._lock = asyncio.Lock()

    async def register_websocket(self, websocket) -> None:
        """Register a WebSocket connection."""
        async with self._lock:
            self._websocket_connections.add(websocket)

    async def unregister_websocket(self, websocket) -> None:
        """Unregister a WebSocket connection."""
        async with self._lock:
            self._websocket_connections.discard(websocket)

    async def register_sse(self, queue: asyncio.Queue) -> None:
        """Register an SSE connection (async queue for sending messages)."""
        async with self._lock:
            self._sse_connections.add(queue)

    async def unregister_sse(self, queue: asyncio.Queue) -> None:
        """Unregister an SSE connection."""
        async with self._lock:
            self._sse_connections.discard(queue)

    async def register(self, connection) -> None:
        """Register a connection (WebSocket or SSE queue)."""
        # Detect type: WebSocket has send_json method, SSE queue is asyncio.Queue
        if hasattr(connection, "send_json"):
            await self.register_websocket(connection)
        elif isinstance(connection, asyncio.Queue):
            await self.register_sse(connection)
        else:
            # Default to WebSocket for backward compatibility
            await self.register_websocket(connection)

    async def unregister(self, connection) -> None:
        """Unregister a connection (WebSocket or SSE queue)."""
        if hasattr(connection, "send_json"):
            await self.unregister_websocket(connection)
        elif isinstance(connection, asyncio.Queue):
            await self.unregister_sse(connection)
        else:
            await self.unregister_websocket(connection)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all registered connections (WebSocket and SSE)."""
        async with self._lock:
            coroutines = []
            # Broadcast to WebSocket connections
            for ws in list(self._websocket_connections):
                coroutines.append(self._send_websocket_safe(ws, message))
            # Broadcast to SSE connections
            for queue in list(self._sse_connections):
                coroutines.append(self._send_sse_safe(queue, message))
            if coroutines:
                await asyncio.gather(*coroutines, return_exceptions=True)

    @staticmethod
    async def _send_websocket_safe(websocket, message: Dict[str, Any]) -> None:
        """Safely send message to WebSocket connection."""
        try:
            await websocket.send_json(message)
        except Exception:
            # Fire and forget – the connection will be cleaned up when unregister is called.
            pass

    @staticmethod
    async def _send_sse_safe(queue: asyncio.Queue, message: Dict[str, Any]) -> None:
        """Safely send message to SSE connection queue."""
        try:
            await queue.put(message)
        except Exception:
            # Fire and forget – the connection will be cleaned up when unregister is called.
            pass


class QueueService:
    """Encapsulates state transitions and analytics for the patient queue."""

    def __init__(self, notifier: Optional[QueueNotifier] = None) -> None:
        self.notifier = notifier or QueueNotifier()

    # ------------------------------------------------------------------ #
    # CRUD / state machine helpers
    # ------------------------------------------------------------------ #

    def create_entry(
        self,
        *,
        patient_id: str,
        consultation_id: Optional[str],
        chief_complaint: Optional[str],
        priority_level: int,
        estimated_wait_seconds: Optional[int],
        assigned_to: Optional[str],
        created_by: Optional[str],
    ) -> QueueStateRead:
        with get_session() as session:
            consultation = self._ensure_consultation(
                session,
                patient_id=patient_id,
                consultation_id=consultation_id,
                chief_complaint=chief_complaint,
                priority_level=priority_level,
            )

            state = crud.create_queue_state(
                session,
                patient_id=patient_id,
                consultation_id=consultation.id if consultation else None,
                stage=QueueStage.WAITING,
                priority_level=priority_level,
                estimated_wait_seconds=estimated_wait_seconds,
                assigned_to=assigned_to,
            )
            crud.log_queue_event(
                session,
                queue_state_id=state.id,
                event_type="created",
                previous_stage=None,
                next_stage=QueueStage.WAITING,
                notes=None,
                created_by=created_by,
            )
            session.commit()
            state = crud.get_queue_state(session, state.id)  # reload with relationships

        queue_state_read = QueueStateRead.model_validate(state)
        self._schedule_broadcast(
            {"type": "queue.created", "data": queue_state_read.model_dump()}
        )
        return queue_state_read

    def transition_stage(
        self,
        queue_state_id: str,
        *,
        next_stage: QueueStage,
        notes: Optional[str],
        user_id: Optional[str],
        priority_level: Optional[int] = None,
    ) -> QueueStateRead:
        with get_session() as session:
            state = self._get_state_or_raise(session, queue_state_id)
            if next_stage not in ALLOWED_TRANSITIONS[state.current_stage]:
                raise ValueError(
                    f"Transition {state.current_stage} -> {next_stage} not allowed."
                )

            previous_stage = state.current_stage
            updates: Dict[str, Any] = {"current_stage": next_stage}
            if priority_level is not None:
                updates["priority_level"] = priority_level
            if next_stage == QueueStage.DISCHARGE:
                updates["estimated_wait_seconds"] = 0

            crud.update_queue_state(session, state, **updates)

            if state.consultation_id:
                consultation = state.consultation or crud.get_consultation(
                    session, state.consultation_id
                )
                if consultation:
                    consultation_updates: Dict[str, Any] = {}
                    new_status = self._derive_consultation_status(next_stage)
                    if new_status and consultation.status != new_status:
                        consultation_updates["status"] = new_status
                    if (
                        priority_level is not None
                        and consultation.triage_level != priority_level
                    ):
                        consultation_updates["triage_level"] = priority_level
                    if consultation_updates:
                        crud.update_consultation(
                            session, consultation, **consultation_updates
                        )

            crud.log_queue_event(
                session,
                queue_state_id=state.id,
                event_type="transition",
                previous_stage=previous_stage,
                next_stage=next_stage,
                notes=notes,
                created_by=user_id,
            )
            session.commit()
            state = crud.get_queue_state(session, state.id)

        payload = QueueStateRead.model_validate(state)
        self._schedule_broadcast(
            {"type": "queue.transition", "data": payload.model_dump()}
        )
        return payload

    def assign_member(
        self,
        queue_state_id: str,
        *,
        assigned_to: str,
        user_id: Optional[str],
    ) -> QueueStateRead:
        with get_session() as session:
            state = self._get_state_or_raise(session, queue_state_id)
            crud.update_queue_state(session, state, assigned_to=assigned_to)

            if state.consultation_id:
                consultation = state.consultation or crud.get_consultation(
                    session, state.consultation_id
                )
                if consultation and consultation.assigned_doctor_id != assigned_to:
                    crud.update_consultation(
                        session, consultation, assigned_doctor_id=assigned_to
                    )

            crud.log_queue_event(
                session,
                queue_state_id=state.id,
                event_type="assigned",
                previous_stage=state.current_stage,
                next_stage=state.current_stage,
                notes=f"Assigned to {assigned_to}",
                created_by=user_id,
            )
            session.commit()
            state = crud.get_queue_state(session, state.id)

        payload = QueueStateRead.model_validate(state)
        self._schedule_broadcast(
            {"type": "queue.assigned", "data": payload.model_dump()}
        )
        return payload

    def update_wait_time(
        self,
        queue_state_id: str,
        *,
        estimated_wait_seconds: int,
        user_id: Optional[str],
    ) -> QueueStateRead:
        with get_session() as session:
            state = self._get_state_or_raise(session, queue_state_id)
            crud.update_queue_state(
                session, state, estimated_wait_seconds=estimated_wait_seconds
            )
            crud.log_queue_event(
                session,
                queue_state_id=state.id,
                event_type="wait_update",
                previous_stage=state.current_stage,
                next_stage=state.current_stage,
                notes=f"Updated wait to {estimated_wait_seconds}s",
                created_by=user_id,
            )
            session.commit()
            state = crud.get_queue_state(session, state.id)

        payload = QueueStateRead.model_validate(state)
        self._schedule_broadcast(
            {"type": "queue.wait_update", "data": payload.model_dump()}
        )
        return payload

    # ------------------------------------------------------------------ #
    # Snapshot & analytics
    # ------------------------------------------------------------------ #

    def snapshot(self, *, stage: Optional[QueueStage] = None) -> QueueSnapshot:
        with get_session() as session:
            records = crud.list_queue_states(session, stage=stage)

        state_models = [QueueStateRead.model_validate(record) for record in records]
        totals: Dict[str, int] = {}
        wait_accumulator = 0
        wait_count = 0

        for state in state_models:
            totals[state.current_stage.value] = (
                totals.get(state.current_stage.value, 0) + 1
            )
            if state.estimated_wait_seconds:
                wait_accumulator += state.estimated_wait_seconds
                wait_count += 1

        average_wait = int(wait_accumulator / wait_count) if wait_count else None
        return QueueSnapshot(
            states=state_models,
            totals_by_stage=totals,
            average_wait_seconds=average_wait,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_state_or_raise(session: Session, queue_state_id: str) -> QueueState:
        normalized_id = str(queue_state_id)
        state = crud.get_queue_state(session, normalized_id)
        if state is None:
            raise ValueError(f"Queue state {normalized_id} not found.")
        return state

    @staticmethod
    def _derive_consultation_status(stage: QueueStage) -> Optional[ConsultationStatus]:
        if stage == QueueStage.WAITING:
            return ConsultationStatus.INTAKE
        if stage == QueueStage.TRIAGE:
            return ConsultationStatus.TRIAGE
        if stage == QueueStage.SCRIBE:
            return ConsultationStatus.IN_PROGRESS
        if stage == QueueStage.DISCHARGE:
            return ConsultationStatus.COMPLETED
        return None

    @staticmethod
    def _ensure_consultation(
        session: Session,
        *,
        patient_id: str,
        consultation_id: Optional[str],
        chief_complaint: Optional[str],
        priority_level: Optional[int],
    ):
        consultation = None
        if consultation_id:
            consultation = crud.get_consultation(session, consultation_id)
        if consultation is None:
            consultation = crud.get_active_consultation_for_patient(session, patient_id)

        if consultation is None:
            consultation = crud.create_consultation(
                session,
                patient_id=patient_id,
                status=ConsultationStatus.INTAKE,
                chief_complaint=chief_complaint,
                triage_level=priority_level,
            )
        else:
            updates: Dict[str, Any] = {}
            if chief_complaint and chief_complaint.strip():
                updates["chief_complaint"] = chief_complaint
            if priority_level is not None:
                updates["triage_level"] = priority_level
            if updates:
                crud.update_consultation(session, consultation, **updates)

        return consultation

    def _schedule_broadcast(self, message: Dict[str, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.notifier.broadcast(message))
        except RuntimeError:
            asyncio.run(self.notifier.broadcast(message))


queue_service = QueueService()


__all__ = ["QueueService", "queue_service", "QueueNotifier", "QueueSnapshot"]
