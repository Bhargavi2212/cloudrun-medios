"""
Wait Time Estimator
Provides estimated wait times for patients in the queue
"""

from datetime import datetime
from typing import Dict, List

from backend.dto.manage_agent_dto import PatientQueueItem


class WaitTimeEstimator:
    """Simple wait time estimation based on triage levels and queue position"""

    def __init__(self):
        # Average processing times by triage level (in minutes)
        self.avg_processing_times = {
            1: 15,  # Critical - immediate attention
            2: 20,  # Urgent - quick processing
            3: 25,  # Less urgent - standard time
            4: 30,  # Standard - routine processing
            5: 35,  # Non-urgent - can wait longer
        }

        # Base wait time multipliers based on status
        self.status_multipliers = {
            "waiting": 1.0,  # Awaiting vitals
            "triage": 1.2,  # Waiting for nurse/doctor assignment
            "scribe": 1.5,  # Under active consultation workflow
            "discharge": 0.1,  # Wrapping up
        }

    def estimate_wait_time(self, patient: PatientQueueItem, queue_patients: List[PatientQueueItem]) -> Dict:
        """
        Estimate wait time for a specific patient

        Args:
            patient: The patient to estimate wait time for
            queue_patients: All patients in the queue

        Returns:
            Dict with estimated_wait_minutes, queue_position, confidence_level
        """
        try:
            # Get patient's triage level (default to 3 if not set)
            triage_level = patient.triage_level or 3

            # Calculate base processing time for this patient
            base_time = self.avg_processing_times.get(triage_level, 25)

            # Find patients ahead of this patient (higher priority)
            patients_ahead = self._get_patients_ahead(patient, queue_patients)

            # Calculate estimated wait time
            estimated_minutes = 0

            # Add time for patients ahead in queue
            for ahead_patient in patients_ahead:
                ahead_triage = ahead_patient.triage_level or 3
                ahead_processing_time = self.avg_processing_times.get(ahead_triage, 25)

                # Apply status multiplier
                status_multiplier = self.status_multipliers.get(ahead_patient.status, 1.0)
                estimated_minutes += ahead_processing_time * status_multiplier

            # Add this patient's processing time
            patient_multiplier = self.status_multipliers.get(patient.status, 1.0)
            estimated_minutes += base_time * patient_multiplier

            # Account for current wait time (if they've been waiting a while, reduce estimate)
            current_wait = patient.wait_time_minutes
            if current_wait > estimated_minutes:
                estimated_minutes = max(5, estimated_minutes - (current_wait * 0.5))

            # Round to nearest 5 minutes
            estimated_minutes = max(5, round(estimated_minutes / 5) * 5)

            # Calculate confidence level
            confidence = self._calculate_confidence(patient, queue_patients)

            return {
                "estimated_wait_minutes": int(estimated_minutes),
                "queue_position": len(patients_ahead) + 1,
                "total_patients_in_queue": len(queue_patients),
                "confidence_level": confidence,
            }

        except Exception as e:
            print(f"Wait time estimation error: {str(e)}")
            # Fallback estimation
            return {
                "estimated_wait_minutes": 30,
                "queue_position": 1,
                "total_patients_in_queue": len(queue_patients),
                "confidence_level": "low",
            }

    def _get_patients_ahead(self, patient: PatientQueueItem, queue_patients: List[PatientQueueItem]) -> List[PatientQueueItem]:
        """Get list of patients ahead of this patient in priority order"""
        patient_triage = patient.triage_level or 5
        patient_wait_time = patient.wait_time_minutes

        patients_ahead = []

        for other_patient in queue_patients:
            if other_patient.consultation_id == patient.consultation_id:
                continue

            other_triage = other_patient.triage_level or 5
            other_wait_time = other_patient.wait_time_minutes

            # Patient is ahead if:
            # 1. Higher priority (lower triage number)
            # 2. Same priority but waiting longer
            # 3. Already in consultation/assigned
            if (
                other_triage < patient_triage
                or (other_triage == patient_triage and other_wait_time > patient_wait_time)
                or other_patient.status in ["scribe", "discharge"]
            ):
                patients_ahead.append(other_patient)

        return patients_ahead

    def _calculate_confidence(self, patient: PatientQueueItem, queue_patients: List[PatientQueueItem]) -> str:
        """Calculate confidence level for the estimate"""
        # Higher confidence for:
        # - Patients with clear triage level
        # - Smaller queue size
        # - Earlier in the day (more predictable)

        confidence_score = 0

        # Triage level confidence
        if patient.triage_level and 1 <= patient.triage_level <= 5:
            confidence_score += 30

        # Queue size confidence (smaller queue = higher confidence)
        queue_size = len(queue_patients)
        if queue_size <= 3:
            confidence_score += 40
        elif queue_size <= 6:
            confidence_score += 20
        elif queue_size <= 10:
            confidence_score += 10

        # Status confidence
        if patient.status in ["waiting", "triage"]:
            confidence_score += 30

        # Determine confidence level
        if confidence_score >= 70:
            return "high"
        elif confidence_score >= 40:
            return "medium"
        else:
            return "low"
