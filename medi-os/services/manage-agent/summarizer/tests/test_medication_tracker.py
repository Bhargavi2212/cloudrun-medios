from datetime import datetime

from summarizer.medication_tracker import track_medication_changes
from summarizer.timeline_builder import Timeline


def build_visit(key: str, date: datetime, medications: list[dict]) -> dict:
    iso = date.isoformat()
    return {
        "key": key,
        "start_time": iso,
        "end_time": iso,
        "visit_type": "Routine Follow-up",
        "source_visit_id": key,
        "is_synthetic": False,
        "vitals": [],
        "labs": [],
        "diagnoses": [],
        "procedures": [],
        "medications": medications,
        "observations": [],
        "notes": [],
        "other_events": [],
    }


def test_medication_tracker_detects_start_and_stop():
    visits = [
        build_visit(
            "visit-1",
            datetime(2022, 1, 1),
            [
                {
                    "code": "RxNorm/100",
                    "description": "Test Med",
                    "recorded_time": "2022-01-01T00:00:00",
                    "numeric_value": 10.0,
                    "unit": "mg",
                    "text_value": "10 mg tablet",
                    "table": "drug_exposure",
                    "note_id": None,
                    "provider_id": None,
                }
            ],
        ),
        build_visit(
            "visit-2",
            datetime(2022, 5, 1),
            [],
        ),
    ]

    timeline = Timeline(
        visits=visits,
        start_time=visits[0]["start_time"],
        end_time=visits[-1]["end_time"],
        visit_count=len(visits),
    )

    journey = track_medication_changes(timeline, stop_gap_days=60)
    history = journey.per_medication["RxNorm/100"]
    change_types = {entry["change_type"] for entry in history}
    assert "started" in change_types
    assert "stopped" in change_types

