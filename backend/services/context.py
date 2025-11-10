from __future__ import annotations

import contextvars
from typing import Optional

_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


def set_request_id(request_id: Optional[str]) -> None:
    _request_id.set(request_id)


def get_request_id() -> Optional[str]:
    return _request_id.get()


__all__ = ["set_request_id", "get_request_id"]
