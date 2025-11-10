"""
Medication tracking utilities.

This module analyses visit timelines to determine when medications were
started, adjusted, stopped, or remain ongoing. The results are attached to
each visit and summarised across the entire patient history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

from summarizer.timeline_builder import Timeline

DOSAGE_KEYWORDS = ("MG", "MCG", "TABLET", "CAPSULE", "PATCH", "PUFF", "UNIT")


@dataclass(slots=True)
class MedicationChange:
    """Represents a single medication change event."""

    change_type: str  # started, stopped, increased, decreased, adjusted, ongoing
    code: str
    description: Optional[str]
    visit_key: str
    recorded_time: Optional[str]
    previous_value: Optional[str] = None
    current_value: Optional[str] = None
    details: Optional[Dict] = None

    def to_dict(self) -> dict:
        payload = {
            "change_type": self.change_type,
            "code": self.code,
            "description": self.description,
            "visit_key": self.visit_key,
            "recorded_time": self.recorded_time,
            "previous_value": self.previous_value,
            "current_value": self.current_value,
        }
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass(slots=True)
class MedicationJourney:
    """Aggregated medication history summary."""

    per_medication: Dict[str, List[dict]]
    visits: List[dict]

    def to_dict(self) -> dict:
        return {
            "per_medication": self.per_medication,
            "visits": self.visits,
        }


def track_medication_changes(
    timeline: Timeline,
    *,
    stop_gap_days: int = 90,
) -> MedicationJourney:
    """
    Analyse a timeline and annotate visits with medication changes.

    Returns:
        MedicationJourney containing per-visit annotations and per-medication
        history suitable for downstream reasoning or presentation.
    """
    updated_visits: List[dict] = []
    medication_history: Dict[str, List[dict]] = {}

    state: Dict[str, Dict] = {}
    gap_threshold = timedelta(days=stop_gap_days)

    for visit in timeline.visits:
        visit_copy = dict(visit)
        visit_key = visit_copy["key"]
        visit_start = _parse_time(visit_copy.get("start_time"))
        current_meds = _collect_latest_medications(visit_copy.get("medications", []))
        changes: List[MedicationChange] = []

        for code, med_entry in current_meds.items():
            description = med_entry.get("description")
            signature = _build_signature(med_entry)
            current_value = _format_value(med_entry)
            recorded_time = med_entry.get("recorded_time")

            if code not in state:
                change = MedicationChange(
                    change_type="started",
                    code=code,
                    description=description,
                    visit_key=visit_key,
                    recorded_time=recorded_time or visit_copy.get("start_time"),
                    current_value=current_value,
                )
                changes.append(change)
                state[code] = {
                    "signature": signature,
                    "last_seen": _parse_time(recorded_time) or visit_start,
                    "last_value": current_value,
                    "description": description,
                }
            else:
                prev_state = state[code]
                prev_signature = prev_state["signature"]
                prev_value = prev_state["last_value"]
                last_seen = prev_state["last_seen"]

                new_last_seen = _parse_time(recorded_time) or visit_start or last_seen
                state[code]["last_seen"] = new_last_seen
                state[code]["description"] = description or prev_state["description"]

                if signature != prev_signature:
                    change_type, details = _classify_signature_change(
                        prev_value, current_value
                    )
                    change = MedicationChange(
                        change_type=change_type,
                        code=code,
                        description=description or prev_state["description"],
                        visit_key=visit_key,
                        recorded_time=recorded_time or visit_copy.get("start_time"),
                        previous_value=prev_value,
                        current_value=current_value,
                        details=details,
                    )
                    changes.append(change)
                    state[code]["signature"] = signature
                    state[code]["last_value"] = current_value

        # Detect potential stops based on gaps
        for code in list(state.keys()):
            if code in current_meds:
                continue
            last_seen = state[code]["last_seen"]
            if visit_start and last_seen and (visit_start - last_seen) >= gap_threshold:
                change = MedicationChange(
                    change_type="stopped",
                    code=code,
                    description=state[code]["description"],
                    visit_key=visit_key,
                    recorded_time=visit_copy.get("start_time"),
                    previous_value=state[code]["last_value"],
                    details={"last_seen": last_seen.isoformat()},
                )
                changes.append(change)
                medication_history.setdefault(code, []).append(change.to_dict())
                del state[code]

        for change in changes:
            medication_history.setdefault(change.code, []).append(change.to_dict())
        visit_copy["medication_changes"] = [change.to_dict() for change in changes]
        updated_visits.append(visit_copy)

    # Mark medications that remain active at the end of history
    for code, med_state in state.items():
        ongoing_change = MedicationChange(
            change_type="ongoing",
            code=code,
            description=med_state["description"],
            visit_key=updated_visits[-1]["key"] if updated_visits else "",
            recorded_time=updated_visits[-1]["end_time"] if updated_visits else None,
            current_value=med_state["last_value"],
            details={"status": "active"},
        )
        medication_history.setdefault(code, []).append(ongoing_change.to_dict())

    return MedicationJourney(
        per_medication=medication_history,
        visits=updated_visits,
    )


def _collect_latest_medications(medications: Iterable[dict]) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    for entry in medications:
        code = entry.get("code")
        if not code:
            continue
        current_time = _parse_time(entry.get("recorded_time"))
        existing = latest.get(code)
        if existing is None:
            latest[code] = entry
        else:
            existing_time = _parse_time(existing.get("recorded_time"))
            if current_time and (existing_time is None or current_time > existing_time):
                latest[code] = entry
    return latest


def _build_signature(entry: dict) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    return (
        entry.get("text_value"),
        entry.get("numeric_value"),
        entry.get("unit"),
    )


def _format_value(entry: dict) -> Optional[str]:
    text_value = entry.get("text_value")
    numeric_value = entry.get("numeric_value")
    unit = entry.get("unit")

    parts: List[str] = []
    if numeric_value is not None:
        parts.append(f"{numeric_value:g}")
    if unit:
        parts.append(unit)
    if text_value:
        if not parts or any(keyword in text_value.upper() for keyword in DOSAGE_KEYWORDS):
            parts.append(text_value)
    if not parts:
        return None
    return " ".join(parts)


def _classify_signature_change(
    previous: Optional[str],
    current: Optional[str],
) -> Tuple[str, Optional[Dict]]:
    if previous is None or current is None:
        return "adjusted", {"reason": "details_changed"}

    def _extract_numeric(value: str) -> Optional[float]:
        tokens = value.replace("/", " ").replace("-", " ").split()
        for token in tokens:
            try:
                return float(token)
            except ValueError:
                continue
        return None

    prev_num = _extract_numeric(previous)
    curr_num = _extract_numeric(current)
    if prev_num is not None and curr_num is not None and prev_num != curr_num:
        change_type = "increased" if curr_num > prev_num else "decreased"
        return change_type, {"previous": prev_num, "current": curr_num}

    if previous != current:
        return "adjusted", {"reason": "text_changed"}

    return "continued", None


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


__all__ = ["MedicationJourney", "MedicationChange", "track_medication_changes"]

