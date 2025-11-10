"""
Build a structured timeline from visit-level event groupings.

The resulting payload is a list of visit dictionaries that can be fed into the
reasoning engine and, ultimately, to the LLM summariser.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import polars as pl

from summarizer.visit_grouper import Visit

CodeDescriber = Callable[[str], Optional[str]]

# Common vital sign concept codes (LOINC + SNOMED)
VITAL_CODES = {
    "LOINC/29463-7",  # Body weight
    "LOINC/8302-2",  # Body height
    "LOINC/39156-5",  # BMI
    "LOINC/8480-6",  # Systolic BP
    "LOINC/8462-4",  # Diastolic BP
    "LOINC/8867-4",  # Heart rate
    "LOINC/9279-1",  # Respiratory rate
    "LOINC/8310-5",  # Body temperature
    "LOINC/59408-5",  # Oxygen saturation in Arterial blood
    "LOINC/2708-6",  # Ejection fraction (where available)
    "SNOMED/271649006",  # Systolic BP
    "SNOMED/271650006",  # Diastolic BP
    "SNOMED/301898006",  # Height
    "SNOMED/366199006",  # Weight
    "SNOMED/78564009",  # Heart rate
}


@dataclass(slots=True)
class Timeline:
    """Structured representation of a patient's visit timeline."""

    visits: List[Dict]
    start_time: Optional[str]
    end_time: Optional[str]
    visit_count: int

    def to_dict(self) -> dict:
        return {
            "visit_count": self.visit_count,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "visits": self.visits,
        }


def build_timeline(
    visits: Sequence[Visit],
    *,
    describe_code: Optional[CodeDescriber] = None,
) -> Timeline:
    """Convert visit groupings into a serialisable timeline object."""
    if not visits:
        return Timeline(visits=[], start_time=None, end_time=None, visit_count=0)

    structured_visits: List[Dict] = []
    for visit in visits:
        structured_visits.append(
            _serialize_visit(visit, describe_code=describe_code)
        )

    start_time = structured_visits[0]["start_time"]
    end_time = structured_visits[-1]["end_time"]

    return Timeline(
        visits=structured_visits,
        start_time=start_time,
        end_time=end_time,
        visit_count=len(structured_visits),
    )


def _serialize_visit(
    visit: Visit,
    *,
    describe_code: Optional[CodeDescriber],
) -> Dict:
    events = visit.events
    measurements = events.filter(pl.col("table") == "measurement")
    vitals, labs = _split_measurements(measurements, describe_code)

    visit_payload: Dict = {
        "key": visit.key,
        "start_time": _to_iso(visit.start_time),
        "end_time": _to_iso(visit.end_time),
        "visit_type": visit.visit_type,
        "source_visit_id": visit.source_visit_id,
        "is_synthetic": visit.is_synthetic,
        "vitals": vitals,
        "labs": labs,
        "diagnoses": _rows_to_payloads(
            events.filter(pl.col("table") == "condition"), describe_code
        ),
        "procedures": _rows_to_payloads(
            events.filter(pl.col("table") == "procedure"), describe_code
        ),
        "medications": _rows_to_payloads(
            events.filter(pl.col("table") == "drug_exposure"), describe_code
        ),
        "observations": _rows_to_payloads(
            events.filter(pl.col("table") == "observation"), describe_code
        ),
        "notes": _rows_to_payloads(events.filter(pl.col("table") == "note"), describe_code),
        "other_events": _rows_to_payloads(
            events.filter(~pl.col("table").is_in(
                ["measurement", "condition", "procedure", "drug_exposure", "observation", "note"]
            )),
            describe_code,
        ),
    }
    return visit_payload


def _split_measurements(
    measurements: pl.DataFrame,
    describe_code: Optional[CodeDescriber],
) -> tuple[List[Dict], List[Dict]]:
    if measurements.is_empty():
        return [], []

    vitals: List[Dict] = []
    labs: List[Dict] = []

    for row in measurements.iter_rows(named=True):
        payload = _build_common_payload(row, describe_code=describe_code)
        code = (row.get("code") or "").upper()
        if code in VITAL_CODES:
            vitals.append(payload)
        else:
            labs.append(payload)

    return vitals, labs


def _rows_to_payloads(
    df: pl.DataFrame,
    describe_code: Optional[CodeDescriber],
) -> List[Dict]:
    if df.is_empty():
        return []
    return [
        _build_common_payload(row, describe_code=describe_code)
        for row in df.iter_rows(named=True)
    ]


def _build_common_payload(
    row: Dict,
    *,
    describe_code: Optional[CodeDescriber],
) -> Dict:
    code = row.get("code")
    description = describe_code(code) if (code and describe_code is not None) else None
    numeric_value = row.get("numeric_value")
    try:
        numeric_value = float(numeric_value) if numeric_value is not None else None
    except (TypeError, ValueError):
        numeric_value = None

    payload = {
        "code": code,
        "description": description,
        "recorded_time": _to_iso(row.get("time")),
        "numeric_value": numeric_value,
        "text_value": row.get("text_value"),
        "unit": row.get("unit"),
        "table": row.get("table"),
        "note_id": row.get("note_id"),
        "provider_id": row.get("provider_id"),
    }
    return payload


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    # Polars DateTime may return numpy datetime64
    try:
        return value.to_pydatetime().isoformat()  # type: ignore[attr-defined]
    except AttributeError:
        return str(value)


__all__ = ["Timeline", "build_timeline"]

