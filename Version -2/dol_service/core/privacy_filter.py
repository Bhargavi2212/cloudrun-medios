"""
Privacy filtering utilities for portable profiles.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

REDACTED_FIELDS = {"location", "performed_by", "hospital_id"}


def redact_metadata(
    record: dict[str, Any], extra_fields: Iterable[str] | None = None
) -> dict[str, Any]:
    """
    Remove sensitive metadata fields from a record.
    """

    fields_to_strip = set(extra_fields or set())
    fields_to_strip.update(REDACTED_FIELDS)

    sanitized = deepcopy(record)
    for field in fields_to_strip:
        sanitized.pop(field, None)
    return sanitized


def sanitize_timeline(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Apply metadata redaction across a timeline.
    """

    return [redact_metadata(event) for event in events]
