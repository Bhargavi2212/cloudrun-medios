"""
Database seed utilities for demo and test environments.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    AuditLog,
    DialogueTranscript,
    Encounter,
    Patient,
    SoapNote,
    Summary,
    TriageObservation,
)

logger = logging.getLogger(__name__)


async def seed_demo_data(session: AsyncSession) -> None:
    """
    Seed the database with synthetic demo data if empty.

    The demo dataset represents a hospital instance that regularly exchanges
    portable profiles with two partner facilities. Each patient has encounters
    handled by a receptionist, triage nurse, and attending physicians to mirror
    realistic workflows for intake, documentation, and federated sharing.
    """

    if await _has_existing_patients(session):
        logger.info("✅ Demo seed skipped: patient records already exist.")
        return

    logger.info("📥 Seeding synthetic demo data.")
    now = datetime.now(tz=UTC)

    patients_seed = [
        {
            "patient": {
                "mrn": "MED-000001",
                "first_name": "Alex",
                "last_name": "Kim",
                "dob": datetime(1989, 4, 12, tzinfo=UTC).date(),
                "sex": "other",
                "contact_info": {
                    "phone": "+1-555-0199",
                    "email": "alex.kim@example.com",
                    "primary_language": "English",
                },
            },
            "encounters": [
                {
                    "hours_ago": 4,
                    "disposition": "discharged",
                    "location": "ED-4",
                    "acuity_level": 3,
                    "triage": {
                        "vitals": {
                            "hr": 92,
                            "rr": 19,
                            "sbp": 130,
                            "dbp": 80,
                            "temp_c": 37.2,
                            "pain": 4,
                            "spo2": 97,
                        },
                        "chief_complaint": (
                            "Shortness of breath after community Kiroween event."
                        ),
                        "notes": (
                            "Triage nurse Elena Shaw reviewed vitals and "
                            "flagged moderate acuity."
                        ),
                        "triage_score": 3,
                        "triage_model_version": "triage_local_v1",
                    },
                    "transcript": {
                        "transcript": (
                            "Nurse: Alex, we're going to run a quick "
                            "pulmonary check.\n"
                            "Patient: Breathing still feels tight.\n"
                            "Doctor: We'll start a nebulizer and reassess "
                            "in thirty minutes."
                        ),
                        "speaker_segments": [
                            {
                                "speaker": "nurse",
                                "content": (
                                    "Alex, we're going to run a quick "
                                    "pulmonary check."
                                ),
                            },
                            {
                                "speaker": "patient",
                                "content": "Breathing still feels tight.",
                            },
                            {
                                "speaker": "doctor",
                                "content": (
                                    "We'll start a nebulizer and reassess "
                                    "in thirty minutes."
                                ),
                            },
                        ],
                        "source": "scribe",
                    },
                    "soap_note": {
                        "subjective": (
                            "Reports dyspnea onset during outdoor event, "
                            "minimal chest pain."
                        ),
                        "objective": (
                            "RR 19, O2 sat 97% on room air, " "mild expiratory wheeze."
                        ),
                        "assessment": (
                            "Exercise-induced bronchospasm with viral "
                            "trigger unlikely."
                        ),
                        "plan": (
                            "Nebulized bronchodilator, discharge with inhaler, "
                            "outpatient follow-up in 72h."
                        ),
                        "model_version": "scribe_v1",
                        "confidence_score": 0.84,
                    },
                },
                {
                    "hours_ago": 30,
                    "disposition": "tele-consult",
                    "location": "Telehealth - Northwind Regional",
                    "acuity_level": 2,
                    "triage": {
                        "vitals": {
                            "hr": 76,
                            "rr": 17,
                            "sbp": 122,
                            "dbp": 76,
                            "temp_c": 36.9,
                            "pain": 2,
                            "spo2": 99,
                        },
                        "chief_complaint": "Follow-up on bronchodilator response.",
                        "notes": (
                            "Remote nurse Priya Das confirmed symptom "
                            "improvement during tele-check."
                        ),
                        "triage_score": 2,
                        "triage_model_version": "triage_partner_v1",
                    },
                    "transcript": {
                        "transcript": (
                            "Remote Doctor: Alex, any shortness of breath today?\n"
                            "Patient: Only with intense exercise.\n"
                            "Remote Doctor: Continue inhaler pre-activity "
                            "and schedule spirometry."
                        ),
                        "speaker_segments": [
                            {
                                "speaker": "doctor_remote",
                                "content": "Alex, any shortness of breath today?",
                            },
                            {
                                "speaker": "patient",
                                "content": "Only with intense exercise.",
                            },
                            {
                                "speaker": "doctor_remote",
                                "content": (
                                    "Continue inhaler pre-activity and "
                                    "schedule spirometry."
                                ),
                            },
                        ],
                        "source": "federated_partner",
                    },
                    "soap_note": {
                        "subjective": (
                            "Telehealth review confirms decreased symptoms "
                            "post discharge."
                        ),
                        "objective": (
                            "Peak flow self-reported at 470 L/min, "
                            "no audible wheeze."
                        ),
                        "assessment": (
                            "Asthma control improving; continue current " "regimen."
                        ),
                        "plan": (
                            "Maintain preventive inhaler use, schedule "
                            "pulmonary function test next week."
                        ),
                        "model_version": "scribe_partner_v1",
                        "confidence_score": 0.78,
                    },
                },
            ],
            "summary": {
                "summary_text": (
                    "Two encounters within 2 days for bronchospasm "
                    "management. Local ED provided acute therapy; remote "
                    "partner conducted telehealth follow-up. Patient stable "
                    "with action plan."
                ),
                "model_version": "summary_v1",
                "confidence_score": 0.81,
            },
            "audit_logs": [
                {
                    "action": "patient_check_in",
                    "performed_by": "maya.reid@kiroween.local",
                    "details": {
                        "hospital_id": "kiroween-general",
                        "staff_role": "receptionist",
                        "reason": "walk_in_check_in",
                    },
                },
                {
                    "action": "portable_profile_viewed",
                    "performed_by": "elena.shaw@kiroween.local",
                    "details": {
                        "hospital_id": "kiroween-general",
                        "staff_role": "triage_nurse",
                        "federated": False,
                    },
                },
                {
                    "action": "portable_profile_shared",
                    "performed_by": "federation-proxy@kiroween.local",
                    "details": {
                        "hospital_id": "kiroween-general",
                        "staff_role": "federation_gateway",
                        "target_hospital": "northwind-regional",
                        "federated": True,
                    },
                },
            ],
        },
        {
            "patient": {
                "mrn": "MED-000002",
                "first_name": "Morgan",
                "last_name": "Patel",
                "dob": datetime(1976, 8, 2, tzinfo=UTC).date(),
                "sex": "female",
                "contact_info": {
                    "phone": "+1-555-4432",
                    "email": "morgan.patel@example.com",
                    "preferred_language": "English",
                },
            },
            "encounters": [
                {
                    "hours_ago": 9,
                    "disposition": "admitted",
                    "location": "ED-Resus 2",
                    "acuity_level": 1,
                    "triage": {
                        "vitals": {
                            "hr": 110,
                            "rr": 24,
                            "sbp": 150,
                            "dbp": 92,
                            "temp_c": 38.5,
                            "pain": 7,
                            "spo2": 94,
                        },
                        "chief_complaint": "High fever and productive cough.",
                        "notes": (
                            "Rapid intake performed by triage nurse Omar Lewis; "
                            "patient placed on sepsis pathway."
                        ),
                        "triage_score": 1,
                        "triage_model_version": "triage_local_v1",
                    },
                    "transcript": {
                        "transcript": (
                            "Doctor: We suspect community-acquired pneumonia.\n"
                            "Patient: Breathing feels heavy and chest aches.\n"
                            "Scribe: Documenting orders for labs and "
                            "chest X-ray."
                        ),
                        "speaker_segments": [
                            {
                                "speaker": "doctor",
                                "content": "We suspect community-acquired pneumonia.",
                            },
                            {
                                "speaker": "patient",
                                "content": "Breathing feels heavy and chest aches.",
                            },
                            {
                                "speaker": "scribe",
                                "content": (
                                    "Documenting orders for labs and " "chest X-ray."
                                ),
                            },
                        ],
                        "source": "scribe",
                    },
                    "soap_note": {
                        "subjective": (
                            "Three-day history of fever, chills, productive "
                            "cough with rust-colored sputum."
                        ),
                        "objective": (
                            "Crackles in right lower lobe, WBC 14.2K, " "lactate 2.4."
                        ),
                        "assessment": (
                            "Severe community-acquired pneumonia; " "rule out sepsis."
                        ),
                        "plan": (
                            "Initiate broad-spectrum antibiotics, admit for "
                            "telemetry, repeat labs in 6h."
                        ),
                        "model_version": "scribe_v1",
                        "confidence_score": 0.9,
                    },
                },
                {
                    "hours_ago": 80,
                    "disposition": "discharged",
                    "location": "Urgent Care - Lakeside Partner",
                    "acuity_level": 2,
                    "triage": {
                        "vitals": {
                            "hr": 96,
                            "rr": 20,
                            "sbp": 138,
                            "dbp": 88,
                            "temp_c": 37.9,
                            "pain": 6,
                            "spo2": 95,
                        },
                        "chief_complaint": (
                            "Initial urgent care visit prior to ED transfer."
                        ),
                        "notes": (
                            "Lakeside RN Hannah Ortiz escalated case to "
                            "Kiroween ED for higher acuity care."
                        ),
                        "triage_score": 2,
                        "triage_model_version": "triage_partner_v1",
                    },
                    "transcript": {
                        "transcript": (
                            "Partner Physician: Symptoms concerning for "
                            "pneumonia, transferring to Kiroween ED.\n"
                            "Patient: I've never had breathing trouble "
                            "like this.\n"
                            "Partner Physician: Ambulance en route; records "
                            "synced to your digital passport."
                        ),
                        "speaker_segments": [
                            {
                                "speaker": "doctor_partner",
                                "content": (
                                    "Symptoms concerning for pneumonia, "
                                    "transferring to Kiroween ED."
                                ),
                            },
                            {
                                "speaker": "patient",
                                "content": (
                                    "I've never had breathing trouble like this."
                                ),
                            },
                            {
                                "speaker": "doctor_partner",
                                "content": (
                                    "Ambulance en route; records synced to "
                                    "your digital passport."
                                ),
                            },
                        ],
                        "source": "federated_partner",
                    },
                    "soap_note": {
                        "subjective": (
                            "Patient presented with fever onset 36h ago and "
                            "persistent cough."
                        ),
                        "objective": (
                            "Mild tachycardia, focal crackles; initial labs "
                            "pending at transfer."
                        ),
                        "assessment": (
                            "Moderate respiratory infection, concern for "
                            "lower lobe pneumonia."
                        ),
                        "plan": (
                            "Transfer to tertiary center, upload vitals to "
                            "portable profile for continuity."
                        ),
                        "model_version": "scribe_partner_v1",
                        "confidence_score": 0.75,
                    },
                },
            ],
            "summary": {
                "summary_text": (
                    "Urgent care escalation to Kiroween ED confirmed severe "
                    "pneumonia. Admission triggered sepsis protocol with "
                    "antibiotics and telemetry monitoring."
                ),
                "model_version": "summary_v1",
                "confidence_score": 0.88,
            },
            "audit_logs": [
                {
                    "action": "patient_check_in",
                    "performed_by": "ian.wells@kiroween.local",
                    "details": {
                        "hospital_id": "kiroween-general",
                        "staff_role": "receptionist",
                        "reason": "ems_arrival",
                    },
                },
                {
                    "action": "portable_profile_viewed",
                    "performed_by": "omar.lewis@kiroween.local",
                    "details": {
                        "hospital_id": "kiroween-general",
                        "staff_role": "triage_nurse",
                        "federated": False,
                    },
                },
                {
                    "action": "portable_profile_viewed",
                    "performed_by": "hannah.ortiz@lakeside.health",
                    "details": {
                        "hospital_id": "lakeside-partner",
                        "staff_role": "triage_nurse",
                        "federated": True,
                        "reason": "pre_transfer_review",
                    },
                },
                {
                    "action": "portable_profile_shared",
                    "performed_by": "federation-proxy@kiroween.local",
                    "details": {
                        "hospital_id": "kiroween-general",
                        "staff_role": "federation_gateway",
                        "target_hospital": "lakeside-partner",
                        "federated": True,
                    },
                },
            ],
        },
        {
            "patient": {
                "mrn": "MED-000003",
                "first_name": "Chris",
                "last_name": "Rivera",
                "dob": datetime(2003, 1, 23, tzinfo=UTC).date(),
                "sex": "male",
                "contact_info": {
                    "phone": "+1-555-9912",
                    "email": "chris.rivera@example.com",
                    "primary_language": "Spanish",
                },
            },
            "encounters": [
                {
                    "hours_ago": 14,
                    "disposition": "discharged",
                    "location": "Urgent Care - Kiroween South",
                    "acuity_level": 4,
                    "triage": {
                        "vitals": {
                            "hr": 68,
                            "rr": 14,
                            "sbp": 118,
                            "dbp": 74,
                            "temp_c": 36.7,
                            "pain": 3,
                            "spo2": 99,
                        },
                        "chief_complaint": (
                            "Laceration from seasonal Kiroween festival setup."
                        ),
                        "notes": (
                            "Nurse Lucia Mendez cleaned wound; no tendon "
                            "involvement."
                        ),
                        "triage_score": 4,
                        "triage_model_version": "triage_local_v1",
                    },
                    "transcript": {
                        "transcript": (
                            "Nurse: We'll irrigate and close the cut with "
                            "adhesive strips.\n"
                            "Patient: It happened while moving props.\n"
                            "Doctor: No stitches required; keep the area dry "
                            "for 48 hours."
                        ),
                        "speaker_segments": [
                            {
                                "speaker": "nurse",
                                "content": (
                                    "We'll irrigate and close the cut with "
                                    "adhesive strips."
                                ),
                            },
                            {
                                "speaker": "patient",
                                "content": "It happened while moving props.",
                            },
                            {
                                "speaker": "doctor",
                                "content": (
                                    "No stitches required; keep the area dry "
                                    "for 48 hours."
                                ),
                            },
                        ],
                        "source": "scribe",
                    },
                    "soap_note": {
                        "subjective": (
                            "Minor superficial laceration to left forearm, "
                            "bleeding controlled."
                        ),
                        "objective": (
                            "2.5 cm superficial cut, no foreign body, "
                            "intact neurovascular exam."
                        ),
                        "assessment": "Simple laceration without complication.",
                        "plan": (
                            "Cleanse, apply adhesive closure, provide tetanus "
                            "booster, discharge with wound care instructions."
                        ),
                        "model_version": "scribe_v1",
                        "confidence_score": 0.86,
                    },
                }
            ],
            "summary": {
                "summary_text": (
                    "Single urgent care visit for festival-related laceration. "
                    "Wound closed with adhesive, tetanus updated, no follow-up "
                    "required."
                ),
                "model_version": "summary_v1",
                "confidence_score": 0.73,
            },
            "audit_logs": [
                {
                    "action": "patient_check_in",
                    "performed_by": "julia.nguyen@kiroween.local",
                    "details": {
                        "hospital_id": "kiroween-south",
                        "staff_role": "receptionist",
                        "reason": "walk_in_check_in",
                    },
                },
                {
                    "action": "portable_profile_viewed",
                    "performed_by": "lucia.mendez@kiroween.local",
                    "details": {
                        "hospital_id": "kiroween-south",
                        "staff_role": "triage_nurse",
                        "federated": False,
                    },
                },
                {
                    "action": "portable_profile_viewed",
                    "performed_by": "dylan.grant@northwind.health",
                    "details": {
                        "hospital_id": "northwind-regional",
                        "staff_role": "emergency_physician",
                        "federated": True,
                        "reason": "quality_review",
                    },
                },
            ],
        },
    ]

    persisted_models = []
    encounter_counts: list[int] = []

    for patient_payload in patients_seed:
        patient = Patient(
            id=uuid4(),
            mrn=patient_payload["patient"]["mrn"],
            first_name=patient_payload["patient"]["first_name"],
            last_name=patient_payload["patient"]["last_name"],
            dob=patient_payload["patient"]["dob"],
            sex=patient_payload["patient"]["sex"],
            contact_info=patient_payload["patient"]["contact_info"],
        )
        persisted_models.append(patient)

        encounter_ids: list[str] = []
        encounters_seed = patient_payload["encounters"]
        encounter_counts.append(len(encounters_seed))

        for encounter_payload in encounters_seed:
            encounter = Encounter(
                id=uuid4(),
                patient_id=patient.id,
                arrival_ts=now - timedelta(hours=encounter_payload["hours_ago"]),
                disposition=encounter_payload["disposition"],
                location=encounter_payload["location"],
                acuity_level=encounter_payload["acuity_level"],
            )
            persisted_models.append(encounter)
            encounter_ids.append(str(encounter.id))

            triage_payload = encounter_payload.get("triage")
            if triage_payload:
                triage = TriageObservation(
                    id=uuid4(),
                    encounter_id=encounter.id,
                    vitals=triage_payload["vitals"],
                    chief_complaint=triage_payload.get("chief_complaint"),
                    notes=triage_payload.get("notes"),
                    triage_score=triage_payload.get("triage_score"),
                    triage_model_version=triage_payload.get("triage_model_version"),
                )
                persisted_models.append(triage)

            transcript_payload = encounter_payload.get("transcript")
            if transcript_payload:
                transcript = DialogueTranscript(
                    id=uuid4(),
                    encounter_id=encounter.id,
                    transcript=transcript_payload["transcript"],
                    speaker_segments=transcript_payload.get("speaker_segments"),
                    source=transcript_payload.get("source"),
                )
                persisted_models.append(transcript)

            soap_payload = encounter_payload.get("soap_note")
            if soap_payload:
                soap_note = SoapNote(
                    id=uuid4(),
                    encounter_id=encounter.id,
                    subjective=soap_payload.get("subjective"),
                    objective=soap_payload.get("objective"),
                    assessment=soap_payload.get("assessment"),
                    plan=soap_payload.get("plan"),
                    model_version=soap_payload.get("model_version"),
                    confidence_score=soap_payload.get("confidence_score"),
                )
                persisted_models.append(soap_note)

        summary_payload = patient_payload.get("summary")
        if summary_payload:
            summary = Summary(
                id=uuid4(),
                patient_id=patient.id,
                encounter_ids=summary_payload.get("encounter_ids", encounter_ids),
                summary_text=summary_payload["summary_text"],
                model_version=summary_payload.get("model_version"),
                confidence_score=summary_payload.get("confidence_score"),
            )
            persisted_models.append(summary)

        for audit_payload in patient_payload.get("audit_logs", []):
            audit_log = AuditLog(
                id=uuid4(),
                entity_type=audit_payload.get("entity_type", "patient_profile"),
                entity_id=audit_payload.get("entity_id", patient.id),
                action=audit_payload["action"],
                performed_by=audit_payload.get("performed_by"),
                details=audit_payload.get("details"),
            )
            persisted_models.append(audit_log)

    session.add_all(persisted_models)
    await session.commit()

    logger.info(
        "✅ Demo seed complete. Inserted %d patients with %d total encounters.",
        len(patients_seed),
        sum(encounter_counts),
    )


async def _has_existing_patients(session: AsyncSession) -> bool:
    """
    Check whether patient data already exists.
    """

    stmt: Select[tuple[int]] = select(func.count(Patient.id))
    result = await session.execute(stmt)
    return result.scalar_one() > 0
