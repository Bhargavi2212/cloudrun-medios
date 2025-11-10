"""
ManageAgent Wait Time Prediction Service
Integrates the main ManageAgent's ML-based wait time prediction into MediOS
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Check if CatBoost models are available locally
try:
    import catboost

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not available, falling back to rule-based estimator")

from backend.dto.manage_agent_dto import PatientQueueItem

logger = logging.getLogger(__name__)


class ManageAgentWaitTimePredictor:
    """
    Advanced wait time prediction using the main ManageAgent's ML models
    """

    def __init__(self):
        self.wait_time_model = None
        self.triage_model = None
        self.fallback_estimator = None
        self.ml_available = False

        # Set up models directory
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")

        if CATBOOST_AVAILABLE:
            try:
                # Load the enhanced wait time model
                wait_time_model_path = os.path.join(
                    self.models_dir, "enhanced_wait_time_model.cbm"
                )
                if os.path.exists(wait_time_model_path):
                    self.wait_time_model = catboost.CatBoostRegressor()
                    self.wait_time_model.load_model(wait_time_model_path)
                    self.ml_available = True
                    logger.info(
                        "✅ Enhanced CatBoost wait time model loaded successfully"
                    )
                else:
                    logger.warning(
                        f"⚠️ Wait time model not found at {wait_time_model_path}"
                    )
            except Exception as e:
                logger.error(f"❌ Failed to load CatBoost models: {e}")
                self.ml_available = False

        if not self.ml_available:
            from backend.services.wait_time_estimator import WaitTimeEstimator

            self.fallback_estimator = WaitTimeEstimator()
            logger.warning("⚠️ Using fallback rule-based estimator")

    def predict_wait_time(
        self, patient: PatientQueueItem, queue_patients: List[PatientQueueItem]
    ) -> Dict[str, Any]:
        """
        Predict wait time using ML-based approach from main ManageAgent

        Args:
            patient: The patient to predict wait time for
            queue_patients: All patients in the queue

        Returns:
            Dict with predicted wait time and confidence metrics
        """
        try:
            if self.wait_time_model and self.ml_available:
                return self._ml_based_prediction(patient, queue_patients)
            else:
                return self._fallback_prediction(patient, queue_patients)
        except Exception as e:
            logger.error(f"Wait time prediction error: {e}")
            return self._fallback_prediction(patient, queue_patients)

    def _ml_based_prediction(
        self, patient: PatientQueueItem, queue_patients: List[PatientQueueItem]
    ) -> Dict[str, Any]:
        """
        Use local CatBoost model for wait time prediction
        """
        try:
            # Prepare features for ML prediction
            features = self._prepare_ml_features(patient, queue_patients)

            # Convert features list to DataFrame with exact column order expected by the model
            feature_names = [
                "urgency_level",
                "medical_complexity",
                "queue_length",
                "staff_available",
                "predicted_triage_level",
                "hour_of_day",
                "day_of_week",
                "is_weekend",
                "is_rush_hour",
                "is_night",
                "hour_sin",
                "hour_cos",
                "day_sin",
                "day_cos",
                "triage_complexity_interaction",
                "triage_queue_interaction",
                "triage_staff_interaction",
                "triage_queue_staff_interaction",
                "high_priority",
                "low_priority",
                "emergency_case",
                "critical_case",
                "patient_staff_ratio",
                "staff_efficiency",
                "load_factor",
                "staff_utilization",
                "queue_length_squared",
                "queue_length_cubed",
                "medical_complexity_squared",
                "triage_level_squared",
                "staff_available_squared",
                "complexity_queue_staff_interaction",
                "department_encoded",
            ]
            feature_df = pd.DataFrame([features], columns=feature_names)

            # Make prediction using local CatBoost model
            predicted_minutes = self.wait_time_model.predict(feature_df)[0]

            # Apply reasonable bounds for critical cases while maintaining dynamic behavior
            triage_level = patient.triage_level or 3
            queue_size = len(queue_patients)

            # For critical cases, cap the wait time but still allow some variation
            if triage_level == 1:  # Level 1 - Immediate
                max_wait = min(15, 5 + (queue_size * 2))  # 5-15 minutes based on queue
                predicted_minutes = min(predicted_minutes, max_wait)
            elif triage_level == 2:  # Level 2 - Very Urgent
                max_wait = min(
                    30, 10 + (queue_size * 3)
                )  # 10-30 minutes based on queue
                predicted_minutes = min(predicted_minutes, max_wait)
            elif triage_level == 3:  # Level 3 - Urgent
                max_wait = min(
                    60, 20 + (queue_size * 4)
                )  # 20-60 minutes based on queue
                predicted_minutes = min(predicted_minutes, max_wait)

            # Ensure prediction is reasonable
            predicted_minutes = max(5, min(480, predicted_minutes))  # 5 min to 8 hours

            # Calculate confidence based on model performance
            confidence = self._calculate_ml_confidence(patient, queue_patients)

            return {
                "estimated_wait_minutes": int(predicted_minutes),
                "queue_position": self._calculate_queue_position(
                    patient, queue_patients
                ),
                "total_patients_in_queue": len(queue_patients),
                "confidence_level": confidence,
                "prediction_method": "ml_model",
                "model_accuracy": "18.58_min_rmse",
            }

        except Exception as e:
            logger.error(f"ML-based prediction failed: {e}")
            return self._fallback_prediction(patient, queue_patients)

    def _prepare_ml_features(
        self, patient: PatientQueueItem, queue_patients: List[PatientQueueItem]
    ) -> Dict[str, Any]:
        """
        Prepare features for ML-based wait time prediction
        """
        # Calculate queue metrics
        queue_size = len(queue_patients)
        patients_ahead = self._get_patients_ahead(patient, queue_patients)
        patients_ahead_count = len(patients_ahead)

        # Calculate triage distribution
        triage_distribution = {}
        for p in queue_patients:
            triage = p.triage_level or 3
            triage_distribution[triage] = triage_distribution.get(triage, 0) + 1

        # Calculate average wait time for patients ahead
        avg_wait_ahead = 0
        if patients_ahead:
            avg_wait_ahead = sum(p.wait_time_minutes for p in patients_ahead) / len(
                patients_ahead
            )

        # Current time features
        current_hour = datetime.now().hour
        is_peak_hours = 8 <= current_hour <= 18

        # Calculate dynamic features based on queue conditions
        high_priority_count = sum(1 for p in queue_patients if p.triage_level in [1, 2])
        low_priority_count = sum(1 for p in queue_patients if p.triage_level in [4, 5])

        # Dynamic staff availability based on queue load
        base_staff = 3
        staff_available = max(
            1, base_staff - (queue_size // 5)
        )  # Reduce staff as queue grows

        # Dynamic medical complexity based on triage level and queue conditions
        medical_complexity = 2  # Default medium
        if patient.triage_level in [1, 2]:
            medical_complexity = 3  # High complexity for critical cases
        elif patient.triage_level in [4, 5]:
            medical_complexity = 1  # Low complexity for non-urgent cases

        # Prepare all 33 features in exact order expected by the CatBoost model
        features = [
            patient.triage_level or 3,  # 0: urgency_level
            medical_complexity,  # 1: medical_complexity (dynamic)
            queue_size,  # 2: queue_length
            staff_available,  # 3: staff_available (dynamic)
            patient.triage_level or 3,  # 4: predicted_triage_level (use actual triage)
            current_hour,  # 5: hour_of_day
            datetime.now().weekday(),  # 6: day_of_week
            1 if datetime.now().weekday() >= 5 else 0,  # 7: is_weekend
            1 if 8 <= current_hour <= 18 else 0,  # 8: is_rush_hour
            1 if current_hour < 6 or current_hour > 22 else 0,  # 9: is_night
            np.sin(2 * np.pi * current_hour / 24),  # 10: hour_sin
            np.cos(2 * np.pi * current_hour / 24),  # 11: hour_cos
            np.sin(2 * np.pi * datetime.now().weekday() / 7),  # 12: day_sin
            np.cos(2 * np.pi * datetime.now().weekday() / 7),  # 13: day_cos
            (patient.triage_level or 3)
            * medical_complexity,  # 14: triage_complexity_interaction
            (patient.triage_level or 3) * queue_size,  # 15: triage_queue_interaction
            (patient.triage_level or 3)
            * staff_available,  # 16: triage_staff_interaction
            (patient.triage_level or 3)
            * queue_size
            * staff_available,  # 17: triage_queue_staff_interaction
            1 if (patient.triage_level or 3) in [1, 2] else 0,  # 18: high_priority
            1 if (patient.triage_level or 3) in [4, 5] else 0,  # 19: low_priority
            1 if (patient.triage_level or 3) == 1 else 0,  # 20: emergency_case
            1 if (patient.triage_level or 3) == 2 else 0,  # 21: critical_case
            queue_size / max(staff_available, 1),  # 22: patient_staff_ratio (dynamic)
            0.8 - (queue_size * 0.02),  # 23: staff_efficiency (decreases with queue)
            queue_size / 10,  # 24: load_factor
            min(
                0.9, queue_size / (staff_available * 2)
            ),  # 25: staff_utilization (dynamic)
            queue_size**2,  # 26: queue_length_squared
            queue_size**3,  # 27: queue_length_cubed
            medical_complexity**2,  # 28: medical_complexity_squared (dynamic)
            (patient.triage_level or 3) ** 2,  # 29: triage_level_squared
            staff_available**2,  # 30: staff_available_squared (dynamic)
            medical_complexity
            * queue_size
            * staff_available,  # 31: complexity_queue_staff_interaction (dynamic)
            0,  # 32: department_encoded (default)
        ]

        return features

    def _calculate_ml_confidence(
        self, patient: PatientQueueItem, queue_patients: List[PatientQueueItem]
    ) -> str:
        """
        Calculate confidence level for ML-based prediction
        """
        confidence_score = 0

        # Model confidence (CatBoost models are generally reliable)
        confidence_score += 40

        # Queue size confidence (smaller queue = higher confidence)
        queue_size = len(queue_patients)
        if queue_size <= 3:
            confidence_score += 30
        elif queue_size <= 6:
            confidence_score += 20
        elif queue_size <= 10:
            confidence_score += 10

        # Triage level confidence
        if patient.triage_level and 1 <= patient.triage_level <= 5:
            confidence_score += 20

        # Status confidence
        normalized_status = (patient.status or "").lower()
        if normalized_status in {"waiting", "triage"}:
            confidence_score += 10

        # Determine confidence level
        if confidence_score >= 80:
            return "high"
        elif confidence_score >= 60:
            return "medium"
        else:
            return "low"

    def _fallback_prediction(
        self, patient: PatientQueueItem, queue_patients: List[PatientQueueItem]
    ) -> Dict[str, Any]:
        """
        Fallback to rule-based prediction if ML is not available
        """
        if self.fallback_estimator:
            result = self.fallback_estimator.estimate_wait_time(patient, queue_patients)
            result["prediction_method"] = "rule_based"
            result["model_accuracy"] = "basic_rules"
            return result
        else:
            # Ultimate fallback
            return {
                "estimated_wait_minutes": 30,
                "queue_position": 1,
                "total_patients_in_queue": len(queue_patients),
                "confidence_level": "low",
                "prediction_method": "fallback",
                "model_accuracy": "basic_estimate",
            }

    def _calculate_queue_position(
        self, patient: PatientQueueItem, queue_patients: List[PatientQueueItem]
    ) -> int:
        """
        Calculate patient's position in the queue based on priority
        """
        patients_ahead = self._get_patients_ahead(patient, queue_patients)
        return len(patients_ahead) + 1

    def _get_patients_ahead(
        self, patient: PatientQueueItem, queue_patients: List[PatientQueueItem]
    ) -> List[PatientQueueItem]:
        """
        Get list of patients ahead of this patient in priority order
        """
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
            other_status = (other_patient.status or "").lower()
            if (
                other_triage < patient_triage
                or (
                    other_triage == patient_triage
                    and other_wait_time > patient_wait_time
                )
                or other_status in {"scribe", "discharge"}
            ):
                patients_ahead.append(other_patient)

        return patients_ahead

    def get_prediction_info(self) -> Dict[str, Any]:
        """
        Get information about the prediction method being used
        """
        return {
            "ml_available": self.ml_available and self.wait_time_model is not None,
            "prediction_method": (
                "ml_model"
                if (self.ml_available and self.wait_time_model)
                else "rule_based"
            ),
            "model_accuracy": (
                "18.58_min_rmse"
                if (self.ml_available and self.wait_time_model)
                else "basic_rules"
            ),
            "catboost_model_loaded": self.wait_time_model is not None,
            "fallback_available": self.fallback_estimator is not None,
        }
