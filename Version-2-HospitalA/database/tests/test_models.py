"""
Integration tests for database models.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import (
    DialogueTranscript,
    Encounter,
    Patient,
    SoapNote,
    Summary,
    TriageObservation,
)

pytestmark = pytest.mark.asyncio


async def test_create_patient_with_related_entities(db_session: AsyncSession) -> None:
    """
    Ensure that patient-related records persist and maintain relationships.
    """

    patient_id = uuid4()
    encounter_id = uuid4()

    patient = Patient(
        id=patient_id,
        mrn="TEST-0001",
        first_name="Morgan",
        last_name="Lee",
        dob=date(1991, 6, 23),
        sex="female",
        contact_info={"email": "morgan.lee@example.com"},
    )
    encounter = Encounter(
        id=encounter_id,
        patient_id=patient_id,
        arrival_ts=datetime.now(tz=UTC),
        disposition="observation",
        location="ED-2",
        acuity_level=2,
    )
    triage = TriageObservation(
        encounter_id=encounter_id,
        vitals={"hr": 92, "rr": 19, "temp_c": 38.1},
        chief_complaint="Fever and cough",
        triage_score=2,
        triage_model_version="triage_v0",
    )
    transcript = DialogueTranscript(
        encounter_id=encounter_id,
        transcript="Doctor: How long has the fever lasted?\nPatient: About two days.",
        speaker_segments=[
            {"speaker": "doctor", "content": "How long has the fever lasted?"},
            {"speaker": "patient", "content": "About two days."},
        ],
        source="scribe",
    )
    soap = SoapNote(
        encounter_id=encounter_id,
        subjective="Reports fever and mild cough since Monday.",
        assessment="Likely viral infection.",
        plan="Supportive care, hydration, rest.",
        model_version="scribe_v0",
        confidence_score=0.75,
    )
    summary = Summary(
        patient_id=patient_id,
        encounter_ids=[str(encounter_id)],
        summary_text="Patient observed for fever, supportive care recommended.",
        model_version="summary_v0",
        confidence_score=0.8,
    )

    db_session.add_all([patient, encounter, triage, transcript, soap, summary])
    await db_session.commit()

    result = await db_session.execute(
        select(Patient)
        .options(
            selectinload(Patient.encounters).selectinload(
                Encounter.triage_observations
            ),
            selectinload(Patient.summaries),
        )
        .where(Patient.id == patient_id)
    )
    stored_patient = result.scalars().one()

    assert stored_patient.mrn == "TEST-0001"
    assert len(stored_patient.encounters) == 1
    assert stored_patient.encounters[0].triage_observations[0].triage_score == 2
    assert stored_patient.summaries[0].summary_text.startswith("Patient observed")
