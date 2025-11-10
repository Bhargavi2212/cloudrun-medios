from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import json
import logging

from backend.database.models import QueueStage
from backend.security.dependencies import require_roles
from backend.security.permissions import UserRole
from backend.services.error_response import StandardResponse
from backend.services.queue_service import queue_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/queue", tags=["queue"])


class QueueCreateRequest(BaseModel):
    patient_id: str = Field(..., description="ID of the patient to enqueue.")
    chief_complaint: Optional[str] = Field(default=None)
    consultation_id: Optional[str] = Field(
        default=None, description="Existing consultation identifier if applicable."
    )
    priority_level: int = Field(default=3, ge=1, le=5)
    estimated_wait_seconds: Optional[int] = Field(default=None, ge=0)
    assigned_to: Optional[str] = Field(default=None)


class QueueTransitionRequest(BaseModel):
    next_stage: QueueStage
    priority_level: Optional[int] = Field(default=None, ge=1, le=5)
    notes: Optional[str] = None


class QueueAssignRequest(BaseModel):
    assigned_to: str = Field(..., description="User ID of the staff member.")


class QueueWaitUpdateRequest(BaseModel):
    estimated_wait_seconds: int = Field(..., ge=0)


@router.get("/", response_model=StandardResponse)
async def list_queue(
    stage: Optional[QueueStage] = Query(default=None),
) -> StandardResponse:
    snapshot = queue_service.snapshot(stage=stage)
    return StandardResponse(
        success=True,
        data={
            "states": [state.model_dump() for state in snapshot.states],
            "totals": snapshot.totals_by_stage,
            "average_wait_seconds": snapshot.average_wait_seconds,
        },
    )


@router.post(
    "/",
    response_model=StandardResponse,
    dependencies=[Depends(require_roles([UserRole.RECEPTIONIST, UserRole.NURSE]))],
    status_code=status.HTTP_201_CREATED,
)
async def enqueue_patient(payload: QueueCreateRequest) -> StandardResponse:
    try:
        state = queue_service.create_entry(
            patient_id=payload.patient_id,
            chief_complaint=payload.chief_complaint,
            consultation_id=payload.consultation_id,
            priority_level=payload.priority_level,
            estimated_wait_seconds=payload.estimated_wait_seconds,
            assigned_to=payload.assigned_to,
            created_by=None,
        )
        return StandardResponse(success=True, data=state.model_dump())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{queue_state_id}/advance",
    response_model=StandardResponse,
    dependencies=[Depends(require_roles([UserRole.NURSE, UserRole.DOCTOR]))],
)
async def advance_queue_state(
    queue_state_id: str,
    payload: QueueTransitionRequest,
) -> StandardResponse:
    try:
        state = queue_service.transition_stage(
            queue_state_id,
            next_stage=payload.next_stage,
            notes=payload.notes,
            user_id=None,
            priority_level=payload.priority_level,
        )
        return StandardResponse(success=True, data=state.model_dump())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{queue_state_id}/assign",
    response_model=StandardResponse,
    dependencies=[Depends(require_roles([UserRole.NURSE, UserRole.DOCTOR, UserRole.RECEPTIONIST]))],
)
async def assign_queue_state(
    queue_state_id: str,
    payload: QueueAssignRequest,
) -> StandardResponse:
    state = queue_service.assign_member(
        queue_state_id,
        assigned_to=payload.assigned_to,
        user_id=None,
    )
    return StandardResponse(success=True, data=state.model_dump())


@router.post(
    "/{queue_state_id}/wait",
    response_model=StandardResponse,
    dependencies=[Depends(require_roles([UserRole.RECEPTIONIST, UserRole.NURSE]))],
)
async def update_wait_time(
    queue_state_id: str,
    payload: QueueWaitUpdateRequest,
) -> StandardResponse:
    state = queue_service.update_wait_time(
        queue_state_id,
        estimated_wait_seconds=payload.estimated_wait_seconds,
        user_id=None,
    )
    return StandardResponse(success=True, data=state.model_dump())


@router.websocket("/ws")
async def queue_ws(websocket: WebSocket):
    """WebSocket endpoint for real-time queue updates."""
    await websocket.accept()
    await queue_service.notifier.register_websocket(websocket)
    try:
        snapshot = queue_service.snapshot()
        await websocket.send_json(
            {
                "type": "queue.snapshot",
                "data": {
                    "states": [state.model_dump() for state in snapshot.states],
                    "totals": snapshot.totals_by_stage,
                    "average_wait_seconds": snapshot.average_wait_seconds,
                },
            }
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await queue_service.notifier.unregister_websocket(websocket)


@router.get("/stream")
async def queue_stream(
    stage: Optional[QueueStage] = Query(default=None),
) -> StreamingResponse:
    """Server-Sent Events (SSE) endpoint for real-time queue updates.
    
    This endpoint streams queue updates as they happen, providing an alternative
    to WebSocket for clients that don't support WebSocket connections.
    
    The stream sends:
    - Initial snapshot on connection
    - Updates whenever the queue changes (new entries, state transitions, etc.)
    """
    async def generate_stream():
        """Generate SSE stream for queue updates."""
        # Create a queue for this SSE connection
        message_queue: asyncio.Queue = asyncio.Queue()
        
        try:
            # Register this SSE connection
            await queue_service.notifier.register_sse(message_queue)
            
            # Send initial snapshot
            snapshot = queue_service.snapshot(stage=stage)
            initial_message = {
                "type": "queue.snapshot",
                "data": {
                    "states": [state.model_dump() for state in snapshot.states],
                    "totals": snapshot.totals_by_stage,
                    "average_wait_seconds": snapshot.average_wait_seconds,
                },
            }
            yield f"data: {json.dumps(initial_message)}\n\n"
            
            # Listen for updates
            while True:
                try:
                    # Wait for message with timeout to allow periodic keep-alive
                    message = await asyncio.wait_for(message_queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Send keep-alive comment
                    yield ": keep-alive\n\n"
                except Exception as exc:
                    logger.error(f"Error in SSE stream: {exc}")
                    break
        except Exception as exc:
            logger.error(f"Error in queue SSE stream: {exc}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            # Unregister this SSE connection
            await queue_service.notifier.unregister_sse(message_queue)
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

