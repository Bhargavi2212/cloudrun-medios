from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.services import summarizer_service as summarizer_module
from backend.services.summarizer_service import MedicalSummarizer


@dataclass
class FakeSummaryResult:
    subject_id: int
    markdown_summary: str
    structured_timeline: dict
    metrics: dict


class FakeCore:
    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, subject_id: int, visit_limit=None):
        self.calls += 1
        return FakeSummaryResult(
            subject_id=subject_id,
            markdown_summary=f"Summary for {subject_id}",
            structured_timeline={"visits": [{"id": 1}]},
            metrics={"duration": 1.2},
        )


@pytest.fixture(autouse=True)
def disable_metrics(monkeypatch):
    monkeypatch.setattr(summarizer_module, "record_service_metric", lambda **_: None)


def test_medical_summarizer_uses_cache(monkeypatch):
    service = MedicalSummarizer(core_service=FakeCore())

    first = service.summarize_patient(123)
    second = service.summarize_patient(123)

    assert first.cached is False
    assert second.cached is True
    assert service.core.calls == 1  # type: ignore[attr-defined]


def test_medical_summarizer_stub_when_core_missing(monkeypatch):
    monkeypatch.setattr(summarizer_module, "CoreSummarizer", None)
    service = MedicalSummarizer(core_service=None)

    result = service.summarize_patient(456)

    assert result.is_stub is True
    assert "Summarizer Unavailable" in result.summary_markdown
