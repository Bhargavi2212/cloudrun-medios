"""Lightweight publish/subscribe utility for in-process notifications."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Dict, Set


class NotificationService:
    """In-memory async notification hub supporting simple channel subscriptions."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def subscribe(self, channel: str, *, max_queue_size: int = 32) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        async with self._lock:
            self._subscribers[channel].add(queue)
        return queue

    async def unsubscribe(self, channel: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            subscribers = self._subscribers.get(channel)
            if not subscribers:
                return
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(channel, None)

    def publish(self, channel: str, event: Dict[str, Any]) -> None:
        subscribers = list(self._subscribers.get(channel, ()))
        if not subscribers:
            return
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except Exception:
                    continue


# Shared singleton instance for application-wide usage
notification_service = NotificationService()
