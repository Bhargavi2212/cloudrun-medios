from datetime import datetime, timedelta

import polars as pl

from summarizer.visit_grouper import group_events_into_visits


def test_group_events_into_visits_with_fallback_window():
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [
                datetime(2020, 1, 1, 9, 0),
                datetime(2020, 1, 1, 12, 0),
                datetime(2020, 1, 3, 10, 0),
            ],
            "visit_id": [None, None, None],
            "table": ["measurement", "measurement", "measurement"],
            "code": ["LOINC/29463-7", "LOINC/8480-6", "LOINC/8462-4"],
            "numeric_value": [180.0, 135.0, 85.0],
            "text_value": [None, None, None],
            "unit": ["lb", "mmHg", "mmHg"],
            "clarity_table": [None, None, None],
            "note_id": [None, None, None],
            "provider_id": [None, None, None],
        }
    )

    visits = group_events_into_visits(df, fallback_window_hours=24)
    assert len(visits) == 2
    assert visits[0].is_synthetic
    assert visits[1].is_synthetic
    assert visits[0].events.height == 2
    assert visits[1].events.height == 1


def test_group_events_by_visit_id_respects_identifier():
    df = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [datetime(2020, 2, 1, 9, 0), datetime(2020, 2, 1, 10, 0)],
            "visit_id": ["V123", "V123"],
            "table": ["measurement", "measurement"],
            "code": ["LOINC/8480-6", "LOINC/8462-4"],
            "numeric_value": [140.0, 88.0],
            "text_value": [None, None],
            "unit": ["mmHg", "mmHg"],
            "clarity_table": [None, None],
            "note_id": [None, None],
            "provider_id": [None, None],
        }
    )

    visits = group_events_into_visits(df)
    assert len(visits) == 1
    visit = visits[0]
    assert visit.source_visit_id == "V123"
    assert not visit.is_synthetic
    assert visit.events.height == 2

