"""
Group patient events into chronologically ordered visits.

Visits are primarily determined by the `visit_id` column in the MEDS OMOP
export. When that identifier is missing, events are bucketed into rolling
24-hour windows to form synthetic visits. The module also provides a light
heuristic classifier to label visit types (routine follow-up, ED visit, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Iterable, List, Optional

import polars as pl

VisitTypeInferencer = Callable[[str], Optional[str]]

ED_KEYWORDS = (
    "ED",
    "EMERGENCY",
    "ER ",
    "TRAUMA",
    "CRITICAL",
    "EMERG",
)
HOSPITAL_KEYWORDS = (
    "HOSP",
    "INPATIENT",
    "ADMIT",
    "INPAT",
    "ADT",
    "IP ",
)
URGENT_KEYWORDS = (
    "URGENT",
    "ACUTE",
    "STAT",
    "IMMEDIATE",
    "SAME DAY",
)
SPECIALIST_KEYWORDS = (
    "CARDIO",
    "HEART",
    "NEURO",
    "ONCO",
    "ENDO",
    "DERM",
    "ORTHO",
    "PULMON",
    "GI ",
    "RENAL",
    "TRANSPLANT",
    "RHEUM",
)
PRIMARY_KEYWORDS = (
    "PRIMARY",
    "INTERNAL MEDICINE",
    "FAMILY",
    "GENERAL",
    "OUTPATIENT",
    "CLINIC",
    "ROUTINE",
)


@dataclass(slots=True)
class Visit:
    """Container for a single grouped visit."""

    key: str
    start_time: datetime
    end_time: datetime
    visit_type: str
    events: pl.DataFrame
    source_visit_id: Optional[str] = None
    is_synthetic: bool = False

    def to_dict(self) -> dict:
        """Convert the visit to a serialisable dictionary."""
        return {
            "key": self.key,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "visit_type": self.visit_type,
            "source_visit_id": self.source_visit_id,
            "is_synthetic": self.is_synthetic,
            "event_count": int(self.events.height),
        }


def group_events_into_visits(
    events: pl.DataFrame,
    *,
    fallback_window_hours: int = 24,
    describe_code: Optional[Callable[[str], Optional[str]]] = None,
) -> List[Visit]:
    """
    Group a patient's events into visit buckets.

    Args:
        events: Polars DataFrame containing all events for a single patient.
        fallback_window_hours: Number of hours to use when bucketing events
            that lack a `visit_id`.
        describe_code: Optional callable that maps a code to a textual
            description. When provided, it improves visit type inference.
    """
    if events.is_empty():
        return []

    normalized = _ensure_datetime(events)
    with_visit_id = normalized.filter(pl.col("visit_id").is_not_null())
    without_visit_id = normalized.filter(pl.col("visit_id").is_null())

    visits: List[Visit] = []

    if with_visit_id.height > 0:
        for chunk in with_visit_id.partition_by(
            "visit_id", maintain_order=True, as_dict=False
        ):
            visit_id_value = chunk["visit_id"][0]
            visit_df = chunk.sort("time")
            visits.append(
                _build_visit(
                    visit_df,
                    key=str(visit_id_value),
                    source_visit_id=str(visit_id_value),
                    is_synthetic=False,
                    describe_code=describe_code,
                )
            )

    if without_visit_id.height > 0:
        synthetic_visits = _build_synthetic_visits(
            without_visit_id.sort("time"),
            window_hours=fallback_window_hours,
            describe_code=describe_code,
        )
        visits.extend(synthetic_visits)

    visits.sort(key=lambda visit: visit.start_time)
    return visits


def _ensure_datetime(df: pl.DataFrame) -> pl.DataFrame:
    if df.schema.get("time") == pl.Datetime:
        return df
    # Cast to microsecond datetime for consistent arithmetic.
    return df.with_columns(pl.col("time").cast(pl.Datetime("us")))


def _build_visit(
    visit_df: pl.DataFrame,
    *,
    key: str,
    source_visit_id: Optional[str],
    is_synthetic: bool,
    describe_code: Optional[Callable[[str], Optional[str]]],
) -> Visit:
    visit_df_sorted = visit_df.sort("time")
    start_time = visit_df_sorted["time"].min()
    end_time = visit_df_sorted["time"].max()
    visit_type = _infer_visit_type(visit_df_sorted, describe_code=describe_code)
    return Visit(
        key=key,
        start_time=start_time,
        end_time=end_time,
        visit_type=visit_type,
        events=visit_df_sorted.clone(),
        source_visit_id=source_visit_id,
        is_synthetic=is_synthetic,
    )


def _build_synthetic_visits(
    df: pl.DataFrame,
    *,
    window_hours: int,
    describe_code: Optional[Callable[[str], Optional[str]]],
) -> List[Visit]:
    times = df["time"].to_list()
    buckets: list[int] = []
    if not times:
        return []

    current_bucket = 0
    bucket_start = times[0]
    max_delta = timedelta(hours=window_hours)

    for ts in times:
        if isinstance(ts, datetime):
            timestamp = ts
        else:
            timestamp = ts.to_pydatetime()  # type: ignore[union-attr]

        if bucket_start is None:
            bucket_start = timestamp
        elif timestamp - bucket_start > max_delta:
            current_bucket += 1
            bucket_start = timestamp
        buckets.append(current_bucket)

    bucketed = df.with_columns(pl.Series("synthetic_visit_index", buckets))

    visits: List[Visit] = []
    for idx, chunk in enumerate(
        bucketed.partition_by("synthetic_visit_index", maintain_order=True)
    ):
        visit_df = chunk.drop("synthetic_visit_index")
        key = f"auto-{idx:04d}"
        visits.append(
            _build_visit(
                visit_df,
                key=key,
                source_visit_id=None,
                is_synthetic=True,
                describe_code=describe_code,
            )
        )

    return visits


def _infer_visit_type(
    visit_df: pl.DataFrame,
    *,
    describe_code: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    values: List[str] = []

    for column in ("table", "clarity_table", "text_value"):
        if column in visit_df.columns:
            col_values = [
                str(val).upper()
                for val in visit_df[column].to_list()
                if val not in (None, "")
            ]
            values.extend(col_values)

    if "code" in visit_df.columns:
        code_values = [
            str(val).upper() for val in visit_df["code"].to_list() if val not in (None, "")
        ]
        values.extend(code_values)

        if describe_code is not None:
            descriptions = [
                describe_code(code)
                for code in visit_df["code"].to_list()
                if code is not None
            ]
            values.extend(
                desc.upper()
                for desc in descriptions
                if desc is not None and desc != ""
            )

    if _matches(values, ED_KEYWORDS):
        return "Emergency Department Visit"
    if _matches(values, HOSPITAL_KEYWORDS):
        return "Hospital Admission"
    if _matches(values, URGENT_KEYWORDS):
        return "Acute/Urgent Care Visit"
    if _matches(values, SPECIALIST_KEYWORDS):
        return "Specialist Consultation"
    if _matches(values, PRIMARY_KEYWORDS):
        return "Routine Follow-up"

    return "Other Encounter"


def _matches(values: Iterable[str], keywords: Iterable[str]) -> bool:
    for value in values:
        upper_value = value.upper()
        for keyword in keywords:
            if keyword in upper_value:
                return True
    return False


__all__ = ["Visit", "group_events_into_visits"]

