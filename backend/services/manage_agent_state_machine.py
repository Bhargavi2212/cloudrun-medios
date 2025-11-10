"""
ManageAgent State Machine
LangGraph workflow for patient flow orchestration
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from backend.database.models import (Consultation, Patient, Role, User,
                                     UserRole, UserStatus)
from backend.database.session import get_session
from backend.dto.manage_agent_dto import TriageResult, VitalsSubmission
from backend.services.triage_engine import TriageEngine


class PatientStatus(Enum):
    """Patient flow states"""

    AWAITING_VITALS = "AWAITING_VITALS"
    AWAITING_DOCTOR_ASSIGNMENT = "AWAITING_DOCTOR_ASSIGNMENT"
    ASSIGNED_TO_DOCTOR = "ASSIGNED_TO_DOCTOR"
    IN_CONSULTATION = "IN_CONSULTATION"
    DISCHARGED = "DISCHARGED"


@dataclass
class PatientFlowState:
    """State for patient flow orchestration"""

    consultation_id: str
    patient_id: str
    status: PatientStatus
    triage_level: Optional[int] = None
    assigned_doctor_id: Optional[str] = None
    check_in_time: Optional[datetime] = None
    vitals: Optional[VitalsSubmission] = None
    triage_result: Optional[TriageResult] = None
    priority_score: Optional[float] = None
    wait_time_minutes: int = 0
    chief_complaint: str = ""
    error_message: str = ""


class ManageAgentStateMachine:
    """State machine for patient flow orchestration"""

    def __init__(self):
        """Initialize the state machine"""
        self.triage_engine = TriageEngine()
        # Configure memory saver with proper keys
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""

        # Create the state graph
        workflow = StateGraph(PatientFlowState)

        # Add nodes for each state transition
        workflow.add_node("check_in_patient", self._check_in_patient)
        workflow.add_node("submit_vitals", self._submit_vitals)
        workflow.add_node("calculate_triage", self._calculate_triage)
        workflow.add_node("assign_doctor", self._assign_doctor)
        workflow.add_node("start_consultation", self._start_consultation)
        workflow.add_node("discharge_patient", self._discharge_patient)

        # Define the flow
        workflow.set_entry_point("check_in_patient")

        # Patient check-in flow
        workflow.add_edge("check_in_patient", "submit_vitals")

        # Vitals submission flow
        workflow.add_edge("submit_vitals", "calculate_triage")
        workflow.add_edge("calculate_triage", "assign_doctor")

        # Doctor assignment flow
        workflow.add_edge("assign_doctor", END)

        # Consultation flow (can be triggered later)
        workflow.add_edge("start_consultation", "discharge_patient")
        workflow.add_edge("discharge_patient", END)

        # Compile with proper checkpointer configuration
        return workflow.compile()

    def _check_in_patient(self, state: PatientFlowState) -> PatientFlowState:
        """Handle patient check-in"""
        try:
            # Update state
            state.status = PatientStatus.AWAITING_VITALS
            state.check_in_time = datetime.now(timezone.utc)

            # Calculate preliminary triage based on chief complaint only
            if state.chief_complaint:
                # Create dummy vitals for preliminary triage (will be updated with real vitals later)
                dummy_vitals = VitalsSubmission(
                    heart_rate=80,  # Normal values for preliminary assessment
                    blood_pressure_systolic=120,
                    blood_pressure_diastolic=80,
                    respiratory_rate=16,
                    temperature_celsius=36.5,
                    oxygen_saturation=98.0,
                    weight_kg=70.0,
                )

                # Get preliminary triage based on chief complaint
                preliminary_triage = self.triage_engine.calculate_triage_level(
                    dummy_vitals, state.chief_complaint
                )
                state.triage_level = preliminary_triage.triage_level
                state.triage_result = preliminary_triage
                state.priority_score = preliminary_triage.priority_score

                print(
                    f"Preliminary triage calculated: Level {state.triage_level} for '{state.chief_complaint}'"
                )

            print(f"Patient {state.patient_id} checked in at {state.check_in_time}")
            return state

        except Exception as e:
            state.error_message = f"Check-in failed: {str(e)}"
            print(f"Check-in error for patient {state.patient_id}: {str(e)}")
            return state

    def _submit_vitals(self, state: PatientFlowState) -> PatientFlowState:
        """Handle vitals submission"""
        try:
            if not state.vitals:
                state.error_message = "No vitals data provided"
                return state

            # Update state
            state.status = PatientStatus.AWAITING_DOCTOR_ASSIGNMENT

            print(f"Vitals submitted for patient {state.patient_id}")
            return state

        except Exception as e:
            state.error_message = f"Vitals submission failed: {str(e)}"
            return state

    def _calculate_triage(self, state: PatientFlowState) -> PatientFlowState:
        """Calculate triage level"""
        try:
            if not state.vitals or not state.chief_complaint:
                state.error_message = "Missing vitals or chief complaint for triage"
                return state

            # Calculate triage using the engine
            triage_result = self.triage_engine.calculate_triage_level(
                state.vitals, state.chief_complaint
            )

            # Update state
            state.triage_result = triage_result
            state.triage_level = triage_result.triage_level
            state.priority_score = triage_result.priority_score

            print(
                f"Triage calculated for patient {state.patient_id}: Level {state.triage_level}"
            )
            return state

        except Exception as e:
            state.error_message = f"Triage calculation failed: {str(e)}"
            return state

    def _assign_doctor(self, state: PatientFlowState) -> PatientFlowState:
        """Assign doctor to patient"""
        try:
            # This would typically query the database for available doctors
            # For now, we'll simulate assignment
            available_doctors = self._get_available_doctors()

            if available_doctors:
                # Assign to doctor with lowest patient load
                assigned_doctor = min(
                    available_doctors, key=lambda d: d["current_patient_load"]
                )
                state.assigned_doctor_id = assigned_doctor["user_id"]
                state.status = PatientStatus.ASSIGNED_TO_DOCTOR

                print(
                    f"Patient {state.patient_id} assigned to doctor {state.assigned_doctor_id}"
                )
            else:
                state.error_message = "No available doctors"

            return state

        except Exception as e:
            state.error_message = f"Doctor assignment failed: {str(e)}"
            return state

    def _start_consultation(self, state: PatientFlowState) -> PatientFlowState:
        """Start consultation with doctor"""
        try:
            state.status = PatientStatus.IN_CONSULTATION
            print(f"Consultation started for patient {state.patient_id}")
            return state

        except Exception as e:
            state.error_message = f"Failed to start consultation: {str(e)}"
            return state

    def _discharge_patient(self, state: PatientFlowState) -> PatientFlowState:
        """Discharge patient"""
        try:
            state.status = PatientStatus.DISCHARGED
            print(f"Patient {state.patient_id} discharged")
            return state

        except Exception as e:
            state.error_message = f"Failed to discharge patient: {str(e)}"
            return state

    def _get_available_doctors(self) -> List[Dict[str, Any]]:
        """Get list of available doctors from database"""
        try:
            with get_session() as db:
                return self._query_available_doctors(db)
        except Exception as exc:
            print(f"DEBUG: Error querying doctors from database: {str(exc)}")
            return []

    def process_patient_check_in(
        self, consultation_id: str, patient_id: str, chief_complaint: str
    ) -> PatientFlowState:
        """Process a new patient check-in"""
        initial_state = PatientFlowState(
            consultation_id=consultation_id,
            patient_id=patient_id,
            status=PatientStatus.AWAITING_VITALS,
            chief_complaint=chief_complaint,
        )

        # Generate unique thread ID for this consultation
        thread_id = f"consultation_{consultation_id}_{uuid.uuid4().hex[:8]}"

        # Run the workflow with proper configuration
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": f"medios_consultation_{consultation_id}",
                "checkpoint_id": f"checkpoint_{consultation_id}_{uuid.uuid4().hex[:8]}",
            }
        }

        result = self.graph.invoke(initial_state, config=config)

        # LangGraph returns a dictionary, convert back to PatientFlowState if needed
        if isinstance(result, dict):
            return PatientFlowState(
                consultation_id=result.get("consultation_id", consultation_id),
                patient_id=result.get("patient_id", patient_id),
                status=result.get("status", PatientStatus.AWAITING_VITALS),
                triage_level=result.get("triage_level"),
                assigned_doctor_id=result.get("assigned_doctor_id"),
                check_in_time=result.get("check_in_time"),
                vitals=result.get("vitals"),
                triage_result=result.get("triage_result"),
                priority_score=result.get("priority_score"),
                wait_time_minutes=result.get("wait_time_minutes", 0),
                chief_complaint=result.get("chief_complaint", chief_complaint),
                error_message=result.get("error_message", ""),
            )

        return result

    def process_vitals_submission(
        self, consultation_id: str, vitals: VitalsSubmission
    ) -> PatientFlowState:
        """Process vitals submission for an existing patient"""
        try:
            with get_session() as db:
                consultation = (
                    db.query(Consultation)
                    .filter(
                        Consultation.id == consultation_id,
                        Consultation.is_deleted.is_(False),
                    )
                    .first()
                )
                if not consultation:
                    raise ValueError(f"No consultation found with ID {consultation_id}")

                patient = (
                    db.query(Patient)
                    .filter(
                        Patient.id == consultation.patient_id,
                        Patient.is_deleted.is_(False),
                    )
                    .first()
                )
                if not patient:
                    raise ValueError(
                        f"No patient found with ID {consultation.patient_id}"
                    )

                current_state = PatientFlowState(
                    consultation_id=consultation_id,
                    patient_id=consultation.patient_id,
                    status=PatientStatus.AWAITING_DOCTOR_ASSIGNMENT,
                    chief_complaint=consultation.chief_complaint or "",
                    triage_level=consultation.triage_level,
                    assigned_doctor_id=consultation.assigned_doctor_id,
                    check_in_time=consultation.created_at,
                    vitals=vitals,
                )

            print(
                f"DEBUG: Created state for vitals processing - Patient: "
                f"{current_state.patient_id}, Status: {current_state.status}"
            )

            triage_result = self.triage_engine.calculate_triage_level(
                vitals, current_state.chief_complaint
            )
            current_state.triage_result = triage_result
            current_state.triage_level = triage_result.triage_level
            current_state.priority_score = triage_result.priority_score

            print(
                f"DEBUG: Calculated triage - Level: {current_state.triage_level}, "
                f"Priority: {current_state.priority_score}"
            )

            current_state = self._assign_doctor(current_state)

            return current_state

        except Exception as exc:
            print(f"ERROR in process_vitals_submission: {str(exc)}")
            raise ValueError(
                f"Failed to process vitals submission: {str(exc)}"
            ) from exc

    def _query_available_doctors(self, db: Session) -> List[Dict[str, Any]]:
        role = (
            db.query(Role)
            .filter(Role.name == "DOCTOR", Role.is_deleted.is_(False))
            .first()
        )
        if not role:
            return []

        doctor_user_ids = [
            user_role.user_id
            for user_role in db.query(UserRole).filter(UserRole.role_id == str(role.id))
        ]
        if not doctor_user_ids:
            return []

        doctors = (
            db.query(User)
            .filter(
                User.id.in_(doctor_user_ids),
                User.status == UserStatus.ACTIVE,
                User.is_deleted.is_(False),
            )
            .all()
        )

        available_doctors: List[Dict[str, Any]] = []
        for doctor in doctors:
            full_name = " ".join(
                part for part in [doctor.first_name, doctor.last_name] if part
            ).strip()
            if not full_name:
                full_name = doctor.email

            available_doctors.append(
                {
                    "user_id": doctor.id,
                    "username": doctor.email,
                    "full_name": full_name,
                    "current_patient_load": 0,
                }
            )

        print(f"DEBUG: Found {len(available_doctors)} available doctors")
        for doc in available_doctors:
            print(
                f"  - Dr. {doc['full_name']} "
                f"(ID: {doc['user_id']}, Load: {doc['current_patient_load']})"
            )

        return available_doctors

    def _get_current_state(self, consultation_id: int) -> Optional[PatientFlowState]:
        """Get current state from memory (simplified)"""
        # This would typically query the checkpoint memory
        # For now, return None to indicate new workflow
        return None

    def get_patient_status(self, consultation_id: int) -> Optional[PatientFlowState]:
        """Get current patient status"""
        return self._get_current_state(consultation_id)

    def get_queue_summary(self) -> Dict[str, Any]:
        """Get summary of all patients in queue"""
        # This would query all active states
        return {
            "total_patients": 0,
            "awaiting_vitals": 0,
            "awaiting_assignment": 0,
            "assigned_to_doctor": 0,
            "in_consultation": 0,
        }
