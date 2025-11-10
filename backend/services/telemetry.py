"""Utility helpers for recording telemetry events."""

from __future__ import annotations

from typing import Optional

from ..database import crud
from ..database.session import get_session


def record_llm_usage(
    *,
    request_id: Optional[str],
    user_id: Optional[str],
    model: str,
    tokens_prompt: int = 0,
    tokens_completion: int = 0,
    cost_cents: float = 0.0,
    status: str = "success",
) -> None:
    with get_session() as session:
        crud.log_llm_usage(
            session,
            request_id=request_id,
            user_id=user_id,
            model=model,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            cost_cents=cost_cents,
            status=status,
        )


def record_service_metric(
    *,
    service_name: str,
    metric_name: str,
    metric_value: float,
    metadata: Optional[dict] = None,
) -> None:
    with get_session() as session:
        crud.log_service_metric(
            session,
            service_name=service_name,
            metric_name=metric_name,
            metric_value=metric_value,
            metadata=metadata or {},
        )
