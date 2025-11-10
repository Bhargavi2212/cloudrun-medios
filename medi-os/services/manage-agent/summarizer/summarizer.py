"""
High-level orchestration for the AI summariser pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from summarizer.config import Settings, load_settings
from summarizer.data_loader import DataLoader, DataLoaderConfig
from summarizer.errors import (
    ConfigurationError,
    LLMGenerationError,
    SummarizerError,
)
from summarizer.metrics import MetricsRecorder
from summarizer.medication_tracker import track_medication_changes
from summarizer.reasoning_engine import ReasoningResult, infer_medication_reasons
from summarizer.timeline_builder import Timeline, build_timeline
from summarizer.trend_analyzer import TrendAnalysis, analyse_trends
from summarizer.visit_grouper import group_events_into_visits

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]


@dataclass
class SummaryResult:
    subject_id: int
    markdown_summary: str
    structured_timeline: Dict[str, Any]
    metrics: Dict[str, Any]


class LLMClient:
    """Wrapper around Gemini with a graceful fallback when unavailable."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = bool(settings.gemini_api_key) and not settings.use_fake_llm
        self._model = None

        if self.enabled:
            if genai is None:
                logger.warning(
                    "google-generativeai not installed; falling back to rule-based summary."
                )
                self.enabled = False
            else:
                genai.configure(api_key=settings.gemini_api_key)
                self._model = genai.GenerativeModel(settings.gemini_model)

    def generate(self, structured_payload: Dict[str, Any]) -> str:
        if not self.enabled or self._model is None:
            return self._fallback_summary(structured_payload)

        prompt = self._compose_prompt(structured_payload)
        try:  # pragma: no cover - depends on external API
            response = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.settings.summarizer_temperature,
                    "max_output_tokens": self.settings.summarizer_max_tokens,
                },
            )
            return response.text or self._fallback_summary(structured_payload)
        except Exception as exc:  # pragma: no cover - external API errors
            logger.exception("Gemini generation failed: %s", exc)
            raise LLMGenerationError("Gemini generation failed") from exc

    def _compose_prompt(self, payload: Dict[str, Any]) -> str:
        return (
            "You are a clinical AI generating a longitudinal visit-by-visit summary.\n"
            "Format the output as markdown with clear sections for each visit, followed "
            "by medication journey, key trends, and clinical insights.\n"
            "Use arrows ↑ ↓ → for trends and highlight acute events with 🚨.\n"
            "Structured data follows:\n"
            f"{json.dumps(payload, default=str)}"
        )

    def _fallback_summary(self, payload: Dict[str, Any]) -> str:
        timeline = payload.get("timeline", {})
        visits = timeline.get("visits", [])
        lines = [
            "# Longitudinal Patient Summary",
            f"Patient ID: {payload.get('subject_id', 'Unknown')}",
            f"Care Period: {timeline.get('start_time')} – {timeline.get('end_time')} "
            f"({timeline.get('visit_count', 0)} encounters)",
            "",
            "## Visit Highlights",
        ]
        for visit in visits[:10]:
            lines.append(
                f"### {visit.get('start_time')} – {visit.get('visit_type')}"
            )
            if visit.get("medication_changes"):
                lines.append("Med changes:")
                for change in visit["medication_changes"]:
                    reason = change.get("reason", "Reason not available")
                    lines.append(
                        f"- {change['change_type'].title()} {change['code']}: {reason}"
                    )
            if visit.get("labs"):
                top_lab = visit["labs"][0]
                trend = top_lab.get("trend")
                trend_str = f" ({trend})" if trend else ""
                lines.append(
                    f"Latest lab: {top_lab.get('description') or top_lab.get('code')} "
                    f"{top_lab.get('numeric_value')}{top_lab.get('unit') or ''}{trend_str}"
                )
            lines.append("")

        lines.append("## Medication Journey")
        for code, history in payload.get("medication_history", {}).items():
            lines.append(f"- {code}: {len(history)} change(s)")
        return "\n".join(lines)


class SummarizerService:
    """Coordinates the end-to-end summarisation workflow."""

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        data_loader: Optional[DataLoader] = None,
        metrics: Optional[MetricsRecorder] = None,
    ) -> None:
        self.settings = settings or load_settings()
        loader_config = DataLoaderConfig(
            data_glob=self.settings.data_glob,
            codes_path=self.settings.codes_path,
            max_cache_entries=self.settings.max_cache_entries,
        )
        self.loader = data_loader or DataLoader(loader_config)
        self.metrics = metrics or MetricsRecorder(
            slow_threshold=self.settings.slow_request_threshold
        )
        self.llm_client = LLMClient(self.settings)

    def summarize(
        self,
        subject_id: int,
        *,
        visit_limit: Optional[int] = None,
    ) -> SummaryResult:
        try:
            with self.metrics.time("load_events"):
                patient_events = self.loader.get_patient_events(subject_id)
        except SummarizerError:
            raise
        except Exception as exc:
            raise SummarizerError(str(exc)) from exc

        with self.metrics.time("group_visits"):
            visits = group_events_into_visits(
                patient_events, describe_code=self.loader.get_code_description
            )

        with self.metrics.time("build_timeline"):
            timeline = build_timeline(
                visits, describe_code=self.loader.get_code_description
            )

        with self.metrics.time("analyse_trends"):
            trend_analysis = analyse_trends(timeline)

        trend_enriched_timeline = Timeline(
            visits=trend_analysis.visits,
            start_time=timeline.start_time,
            end_time=timeline.end_time,
            visit_count=timeline.visit_count,
        )

        with self.metrics.time("medication_tracking"):
            medication_journey = track_medication_changes(
                trend_enriched_timeline,
                stop_gap_days=self.settings.stop_gap_days,
            )

        with self.metrics.time("reasoning"):
            reasoning = infer_medication_reasons(medication_journey, trend_analysis)

        structured_payload = self._build_payload(
            subject_id=subject_id,
            timeline=timeline,
            reasoning=reasoning,
            trend_analysis=trend_analysis,
            visit_limit=visit_limit,
        )

        with self.metrics.time("llm_generation"):
            markdown_summary = self.llm_client.generate(structured_payload)

        metrics_snapshot = self.metrics.snapshot()
        return SummaryResult(
            subject_id=subject_id,
            markdown_summary=markdown_summary,
            structured_timeline=structured_payload,
            metrics=metrics_snapshot,
        )

    def _build_payload(
        self,
        *,
        subject_id: int,
        timeline: Timeline,
        reasoning: ReasoningResult,
        trend_analysis: TrendAnalysis,
        visit_limit: Optional[int],
    ) -> Dict[str, Any]:
        visits = reasoning.visits
        if visit_limit is not None:
            visits = visits[-visit_limit:]

        return {
            "subject_id": subject_id,
            "timeline": {
                "start_time": timeline.start_time,
                "end_time": timeline.end_time,
                "visit_count": len(visits),
                "visits": visits,
            },
            "medication_history": reasoning.per_medication,
            "trend_metrics": trend_analysis.metrics,
        }


__all__ = ["SummarizerService", "SummaryResult"]

