"""
Trend analysis for vitals and laboratory measurements.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from summarizer.timeline_builder import Timeline

TREND_THRESHOLD = 0.05  # 5% change

ABNORMAL_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "LOINC/4548-4": {"high": 6.5},  # HbA1c %
    "LOINC/17856-6": {"high": 140.0},  # Fasting glucose mg/dL
    "LOINC/2160-0": {"high": 1.3},  # Creatinine mg/dL
    "LOINC/9842-6": {"high": 100.0},  # BNP pg/mL
    "LOINC/49580-4": {"high": 50.0},  # NT-proBNP (example threshold)
    "LOINC/2085-9": {"high": 130.0},  # LDL cholesterol mg/dL
    "LOINC/13457-7": {"high": 200.0},  # Total cholesterol mg/dL
    "LOINC/8480-6": {"high": 140.0},  # Systolic BP mmHg
    "LOINC/8462-4": {"high": 90.0},  # Diastolic BP mmHg
    "LOINC/8302-2": {"high": 78.0, "low": 48.0},  # Height inches (flag unrealistic)
    "LOINC/29463-7": {"high": 400.0, "low": 70.0},  # Weight pounds (sanity range)
}


@dataclass(slots=True)
class TrendSeries:
    code: str
    description: Optional[str]
    points: List[dict]
    trend_direction: Optional[str]
    overall_change: Optional[float]

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "description": self.description,
            "points": self.points,
            "trend_direction": self.trend_direction,
            "overall_change": self.overall_change,
        }


@dataclass(slots=True)
class TrendAnalysis:
    """Trend annotations for the entire timeline."""

    visits: List[dict]
    metrics: Dict[str, dict]

    def to_dict(self) -> dict:
        return {
            "visits": self.visits,
            "metrics": self.metrics,
        }


def analyse_trends(
    timeline: Timeline,
    *,
    relative_threshold: float = TREND_THRESHOLD,
) -> TrendAnalysis:
    """
    Enrich timeline visits with trend information for vitals and labs.
    """
    visits = copy.deepcopy(timeline.visits)
    metric_series: Dict[str, TrendSeries] = {}

    for visit in visits:
        _annotate_entries(
            visit_key=visit["key"],
            entries=visit.get("vitals", []),
            metric_series=metric_series,
            relative_threshold=relative_threshold,
        )
        _annotate_entries(
            visit_key=visit["key"],
            entries=visit.get("labs", []),
            metric_series=metric_series,
            relative_threshold=relative_threshold,
        )

    metrics_payload = {code: series.to_dict() for code, series in metric_series.items()}
    return TrendAnalysis(visits=visits, metrics=metrics_payload)


def _annotate_entries(
    *,
    visit_key: str,
    entries: Sequence[dict],
    metric_series: Dict[str, TrendSeries],
    relative_threshold: float,
) -> None:
    for entry in entries:
        code = entry.get("code")
        if code is None:
            continue
        numeric_value = entry.get("numeric_value")
        if numeric_value is None:
            entry["trend"] = None
            continue

        series = metric_series.get(code)
        if series is None:
            series = metric_series[code] = TrendSeries(
                code=code,
                description=entry.get("description"),
                points=[],
                trend_direction=None,
                overall_change=None,
            )

        prev_value = series.points[-1]["value"] if series.points else None
        trend_symbol, delta = _compute_trend(prev_value, numeric_value, relative_threshold)

        abnormal_flag = _evaluate_abnormal(code, numeric_value)
        entry["trend"] = trend_symbol
        entry["change"] = delta
        entry["abnormal"] = abnormal_flag

        series.points.append(
            {
                "visit_key": visit_key,
                "time": entry.get("recorded_time"),
                "value": numeric_value,
                "trend": trend_symbol,
                "abnormal": abnormal_flag,
                "unit": entry.get("unit"),
            }
        )

        if len(series.points) >= 2:
            overall_delta = series.points[-1]["value"] - series.points[0]["value"]
            series.overall_change = overall_delta
            series.trend_direction = trend_symbol


def _compute_trend(
    previous: Optional[float],
    current: float,
    threshold: float,
) -> Tuple[Optional[str], Optional[float]]:
    if previous is None:
        return None, None

    delta = current - previous
    denominator = abs(previous) if previous != 0 else 1.0
    relative_change = abs(delta) / denominator

    if relative_change < threshold:
        return "→", delta
    return ("↑" if delta > 0 else "↓"), delta


def _evaluate_abnormal(code: str, value: float) -> Optional[str]:
    thresholds = ABNORMAL_THRESHOLDS.get(code)
    if thresholds is None:
        return None
    if "high" in thresholds and value > thresholds["high"]:
        return "high"
    if "low" in thresholds and value < thresholds["low"]:
        return "low"
    return "normal"


__all__ = ["analyse_trends", "TrendAnalysis"]

