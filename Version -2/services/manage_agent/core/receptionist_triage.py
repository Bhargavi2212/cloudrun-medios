"""
Receptionist triage model - uses trained ML model with age and chief complaint
(no vitals).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Model paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent
MODEL_DIR = ROOT_DIR / "data" / "receptionist_models_v3" / "models"
PREPROCESSING_DIR = ROOT_DIR / "data" / "processed" / "transformers"


@dataclass
class ReceptionistTriagePayload:
    """
    Payload for receptionist triage - only age and chief complaint.
    """

    age: int | None
    chief_complaint: str
    ambulance_arrival: bool = False
    seen_72h: bool = False
    injury: bool = False


@dataclass
class ReceptionistTriageResult:
    """
    Triage result from receptionist model.
    """

    acuity_level: int
    model_version: str
    explanation: str


class ReceptionistTriageEngine:
    """
    Triage engine for receptionist check-in using trained ML model.
    Uses only age and chief complaint (no vitals).
    """

    def __init__(
        self,
        *,
        model_version: str = "receptionist-xgboost-v3-D",
        model_name: str = "xgboost_D.pkl",
    ) -> None:
        self.model_version = model_version
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.rfv_cluster_mapping = None

        # Load model and preprocessing artifacts
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained XGBoost model and preprocessing artifacts."""
        try:
            model_path = MODEL_DIR / self.model_name
            if not model_path.exists():
                logger.warning(
                    f"Trained model not found at {model_path}. Using fallback heuristic."  # noqa: E501
                )
                return

            # Load model
            self.model = joblib.load(model_path)
            logger.info(f"Loaded receptionist triage model: {self.model_name}")

            # Try to load scaler and feature names from preprocessing artifacts
            scaler_path = PREPROCESSING_DIR / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")

            # Load feature names if available
            feature_names_path = PREPROCESSING_DIR / "feature_names.pkl"
            if feature_names_path.exists():
                self.feature_names = joblib.load(feature_names_path)
                logger.info(f"Loaded feature names: {len(self.feature_names)} features")

            # Load RFV cluster mapping if available
            rfv_mapping_path = PREPROCESSING_DIR / "rfv_cluster_mapping.pkl"
            if rfv_mapping_path.exists():
                self.rfv_cluster_mapping = joblib.load(rfv_mapping_path)
                logger.info("Loaded RFV cluster mapping")

        except Exception as e:
            logger.error(f"Error loading trained model: {e}. Using fallback heuristic.")
            self.model = None

    def _map_chief_complaint_to_rfv_clusters(
        self, chief_complaint: str
    ) -> dict[str, int]:
        """
        Map chief complaint text to RFV cluster features.
        Returns a dictionary of cluster names to binary values (0 or 1).
        """
        complaint_lower = chief_complaint.lower()

        # Default: all clusters set to 0
        clusters = {
            "rfv1_cluster_Neurological": 0,
            "rfv1_cluster_Respiratory": 0,
            "rfv1_cluster_Trauma_Injury": 0,
            "rfv1_cluster_Fever_Infection": 0,
            "rfv1_cluster_Gastrointestinal": 0,
            "rfv1_cluster_Cardiovascular": 0,
            "rfv1_cluster_Pain": 0,
            "rfv1_cluster_Other": 0,
        }

        # Keyword-based mapping (simple heuristic)
        if any(
            kw in complaint_lower
            for kw in ["chest pain", "heart", "cardiac", "palpitation"]
        ):
            clusters["rfv1_cluster_Cardiovascular"] = 1
        elif any(
            kw in complaint_lower
            for kw in ["breathing", "shortness of breath", "cough", "respiratory"]
        ):
            clusters["rfv1_cluster_Respiratory"] = 1
        elif any(
            kw in complaint_lower
            for kw in ["headache", "seizure", "dizziness", "neurological", "stroke"]
        ):
            clusters["rfv1_cluster_Neurological"] = 1
        elif any(
            kw in complaint_lower
            for kw in ["trauma", "injury", "accident", "fall", "cut", "wound"]
        ):
            clusters["rfv1_cluster_Trauma_Injury"] = 1
        elif any(kw in complaint_lower for kw in ["fever", "infection", "sick", "flu"]):
            clusters["rfv1_cluster_Fever_Infection"] = 1
        elif any(
            kw in complaint_lower
            for kw in ["stomach", "nausea", "vomiting", "diarrhea", "abdominal"]
        ):
            clusters["rfv1_cluster_Gastrointestinal"] = 1
        elif any(kw in complaint_lower for kw in ["pain", "ache", "sore"]):
            clusters["rfv1_cluster_Pain"] = 1
        else:
            clusters["rfv1_cluster_Other"] = 1

        return clusters

    def _prepare_features(self, payload: ReceptionistTriagePayload) -> np.ndarray:
        """
        Prepare features for the trained model.
        Converts age + chief complaint to model's expected feature format.
        """
        # Map chief complaint to RFV clusters
        rfv_clusters = self._map_chief_complaint_to_rfv_clusters(
            payload.chief_complaint
        )

        # Build feature vector
        features = {
            "age": payload.age if payload.age is not None else 40,  # Default age
            "ambulance_arrival": 1 if payload.ambulance_arrival else 0,
            "seen_72h": 1 if payload.seen_72h else 0,
            "injury": 1 if payload.injury else 0,
        }

        # Add RFV cluster features
        features.update(rfv_clusters)

        # Convert to DataFrame
        feature_df = pd.DataFrame([features])

        # Apply scaling if available
        if self.scaler is not None:
            try:
                feature_df_scaled = self.scaler.transform(feature_df)
            except Exception as e:
                logger.warning(f"Scaling failed: {e}. Using unscaled features.")
                feature_df_scaled = feature_df.values
        else:
            feature_df_scaled = feature_df.values

        return feature_df_scaled

    async def classify(
        self, payload: ReceptionistTriagePayload
    ) -> ReceptionistTriageResult:
        """
        Classify triage level using trained ML model or fallback heuristic.
        Returns ESI level 1-5 (1=most urgent, 5=least urgent).
        """

        # Try to use trained model first
        if self.model is not None:
            try:
                # Prepare features
                features = self._prepare_features(payload)

                # Make prediction
                prediction = self.model.predict(features)[0]

                # Model outputs 0-4, convert to 1-5
                acuity = int(prediction) + 1

                # Ensure valid range
                acuity = max(1, min(5, acuity))

                explanation = (
                    f"ML model prediction (XGBoost v3-D) based on age ({payload.age or 'unknown'}) "  # noqa: E501
                    f"and chief complaint analysis. "
                    f"Model: {self.model_version}"
                )

                logger.info(
                    "Receptionist ML triage: acuity=%s, age=%s, complaint='%s'",
                    acuity,
                    payload.age,
                    payload.chief_complaint[:50],
                )

                return ReceptionistTriageResult(
                    acuity_level=acuity,
                    model_version=self.model_version,
                    explanation=explanation,
                )

            except Exception as e:
                logger.warning(
                    f"ML model prediction failed: {e}. Using fallback heuristic."
                )

        # Fallback to heuristic if ML model fails
        return self._heuristic_classify(payload)

    def _heuristic_classify(
        self, payload: ReceptionistTriagePayload
    ) -> ReceptionistTriageResult:
        """Fallback heuristic classification."""
        severity_score = 0
        complaint_lower = payload.chief_complaint.lower()

        # Urgent keywords
        urgent_keywords = [
            "chest pain",
            "difficulty breathing",
            "shortness of breath",
            "severe pain",
            "unconscious",
            "bleeding",
            "trauma",
            "heart attack",
            "stroke",
            "seizure",
        ]

        # Moderate keywords
        moderate_keywords = [
            "fever",
            "pain",
            "nausea",
            "vomiting",
            "dizziness",
            "headache",
        ]

        if any(keyword in complaint_lower for keyword in urgent_keywords):
            severity_score += 3
        elif any(keyword in complaint_lower for keyword in moderate_keywords):
            severity_score += 1

        # Age-based adjustments
        if payload.age is not None:
            if payload.age < 2 or payload.age > 75:
                severity_score += 1

        # Convert to ESI level
        if severity_score >= 3:
            acuity = 1
        elif severity_score == 2:
            acuity = 2
        elif severity_score == 1:
            acuity = 3
        else:
            acuity = 4

        explanation = (
            f"Heuristic triage based on age ({payload.age or 'unknown'}) "
            f"and chief complaint keywords. "
            f"Model: {self.model_version} (fallback)"
        )

        return ReceptionistTriageResult(
            acuity_level=acuity,
            model_version=f"{self.model_version}-heuristic",
            explanation=explanation,
        )
