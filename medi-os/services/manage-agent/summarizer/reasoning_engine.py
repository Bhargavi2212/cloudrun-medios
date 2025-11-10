"""
Rule-based reasoning engine for medication adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from summarizer.medication_tracker import MedicationJourney
from summarizer.trend_analyzer import TrendAnalysis


@dataclass(slots=True)
class ReasoningResult:
    visits: List[dict]
    per_medication: Dict[str, List[dict]]

    def to_dict(self) -> dict:
        return {
            "visits": self.visits,
            "per_medication": self.per_medication,
        }


def infer_medication_reasons(
    journey: MedicationJourney,
    trends: TrendAnalysis,
) -> ReasoningResult:
    """
    Attach human-readable reasons to each medication change event.
    """
    visit_metric_index = _build_visit_metric_index(trends.metrics)
    updated_visits: List[dict] = []
    updated_history: Dict[str, List[dict]] = {}
    seen_events: set[Tuple[str, str, Optional[str]]] = set()

    previous_visit_key: Optional[str] = None
    for visit in journey.visits:
        visit_key = visit["key"]
        current_metrics = visit_metric_index.get(visit_key, [])
        previous_metrics = (
            visit_metric_index.get(previous_visit_key, []) if previous_visit_key else []
        )
        previous_visit_key = visit_key

        enriched_changes: List[dict] = []
        for change in visit.get("medication_changes", []):
            reason = _derive_reason(change, current_metrics, previous_metrics)
            change_with_reason = dict(change)
            change_with_reason["reason"] = reason
            enriched_changes.append(change_with_reason)
            event_key = (change["code"], change["change_type"], change.get("recorded_time"))
            seen_events.add(event_key)
            updated_history.setdefault(change["code"], []).append(change_with_reason)

        visit_copy = dict(visit)
        visit_copy["medication_changes"] = enriched_changes
        updated_visits.append(visit_copy)

    # Preserve non-visit entries (e.g., ongoing status) and add default reasons
    for code, events in journey.per_medication.items():
        for event in events:
            event_key = (event["code"], event["change_type"], event.get("recorded_time"))
            if event_key in seen_events:
                continue
            enriched_event = dict(event)
            enriched_event["reason"] = enriched_event.get(
                "reason", _default_reason(event["change_type"], event.get("details"))
            )
            updated_history.setdefault(code, []).append(enriched_event)

    return ReasoningResult(visits=updated_visits, per_medication=updated_history)


def _build_visit_metric_index(metrics: Dict[str, dict]) -> Dict[str, List[dict]]:
    index: Dict[str, List[dict]] = {}
    for code, series in metrics.items():
        description = series.get("description") or code
        for point in series.get("points", []):
            entry = {
                "code": code,
                "description": description,
                "value": point.get("value"),
                "trend": point.get("trend"),
                "abnormal": point.get("abnormal"),
                "unit": point.get("unit"),
            }
            key = point.get("visit_key")
            if key is not None:
                index.setdefault(key, []).append(entry)
    return index


def _derive_reason(
    change: dict,
    current_metrics: Sequence[dict],
    previous_metrics: Sequence[dict],
) -> str:
    change_type = change.get("change_type")
    if change_type in ("started", "increased", "adjusted"):
        metric = _select_metric(current_metrics, prefer_abnormal="high", prefer_trend="↑")
        if metric is None:
            metric = _select_metric(previous_metrics, prefer_abnormal="high", prefer_trend="↑")
        if metric:
            return f"{_lead_phrase(change_type)} because { _describe_metric(metric) }."
        return f"{_lead_phrase(change_type)} to improve long-term disease control."

    if change_type == "decreased":
        metric = _select_metric(current_metrics, prefer_abnormal="low", prefer_trend="↓")
        if metric:
            return f"Dose reduced as { _describe_metric(metric) } shows improvement."
        return "Dose reduced after assessing clinical stability."

    if change_type == "stopped":
        metric = _select_metric(current_metrics, prefer_abnormal="low", prefer_trend="↓")
        if metric:
            return f"Medication stopped following improvement in { _describe_metric(metric) }."
        last_seen = change.get("details", {}).get("last_seen")
        if last_seen:
            return f"Medication stopped after no administrations since {last_seen}."
        return "Medication discontinued per provider assessment."

    if change_type == "ongoing":
        metric = _select_metric(current_metrics, prefer_abnormal="normal", prefer_trend="→")
        if metric:
            return f"Medication continued with { _describe_metric(metric) } remaining stable."
        return "Medication maintained with no concerning trends."

    return _default_reason(change_type, change.get("details"))


def _select_metric(
    metrics: Sequence[dict],
    *,
    prefer_abnormal: Optional[str] = None,
    prefer_trend: Optional[str] = None,
) -> Optional[dict]:
    if prefer_abnormal:
        for metric in metrics:
            if metric.get("abnormal") == prefer_abnormal:
                return metric
    if prefer_trend:
        for metric in metrics:
            if metric.get("trend") == prefer_trend:
                return metric
    return metrics[0] if metrics else None


def _describe_metric(metric: dict) -> str:
    description = metric.get("description") or metric.get("code")
    value = metric.get("value")
    unit = metric.get("unit")
    abnormal = metric.get("abnormal")
    trend = metric.get("trend")

    value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
    if value is None:
        value_str = "values"

    parts = [description]
    if value is not None:
        parts.append(f"{value_str}{f' {unit}' if unit else ''}")

    if abnormal == "high":
        parts.append("above target")
    elif abnormal == "low":
        parts.append("below target")

    if trend == "↑":
        parts.append("rising trend")
    elif trend == "↓":
        parts.append("improving trend")

    return ", ".join(parts)


def _lead_phrase(change_type: Optional[str]) -> str:
    mapping = {
        "started": "Therapy initiated",
        "increased": "Dose increased",
        "adjusted": "Regimen adjusted",
    }
    return mapping.get(change_type, "Medication updated")


def _default_reason(change_type: Optional[str], details: Optional[dict]) -> str:
    if change_type == "stopped" and details and details.get("last_seen"):
        return f"Medication stopped after no doses since {details['last_seen']}."
    fallback = {
        "started": "Therapy initiated based on clinical judgement.",
        "increased": "Dose increased to optimise control.",
        "adjusted": "Regimen adjusted per provider plan.",
        "decreased": "Dose reduced after clinical improvement.",
        "stopped": "Medication discontinued by provider.",
        "ongoing": "Medication remains in place as part of maintenance therapy.",
    }
    return fallback.get(change_type, "Medication plan reviewed with provider.")


__all__ = ["infer_medication_reasons", "ReasoningResult"]

