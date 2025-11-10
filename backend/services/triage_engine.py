"""
Enhanced Triage Engine
Uses trained XGBoost model for AI-powered triage with rule-based fallback
"""

import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from backend.dto.manage_agent_dto import TriageResult, VitalsSubmission

# Try to import xgboost, fallback gracefully if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available. Using rule-based triage only.")


class TriageEngine:
    """Enhanced triage engine using trained XGBoost model with rule-based fallback"""
    
    def __init__(self, model_path: Optional[str] = None, artifacts_path: Optional[str] = None):
        """Initialize the triage engine with trained model"""
        # Set default paths relative to backend directory
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = model_path or os.path.join(backend_dir, 'triage_model_synthetic.xgb')
        self.artifacts_path = artifacts_path or os.path.join(backend_dir, 'triage_model_artifacts.pkl')
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        
        # Load the trained model and artifacts
        self._load_model()
        
        # Load ESI guidelines for fallback
        self.esi_guidelines = self._load_esi_guidelines()
    
    def _load_model(self):
        """Load the trained XGBoost model and preprocessing artifacts"""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available. Using rule-based triage only.")
            self.model = None
            return
            
        try:
            # Load XGBoost model
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            
            # Load preprocessing artifacts
            artifacts = joblib.load(self.artifacts_path)
            self.scaler = artifacts['scaler']
            self.label_encoders = artifacts['label_encoders']
            self.feature_columns = artifacts['feature_columns']
            
            print("AI Triage Model loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load AI model ({e}). Falling back to rule-based triage.")
            self.model = None
    
    def _prepare_features(self, vitals: VitalsSubmission, chief_complaint: str) -> np.ndarray:
        """Prepare features for the AI model"""
        # Create feature dictionary
        features = {
            'age': 45,  # Default age - could be enhanced with patient data
            'gender': 0,  # Default - could be enhanced with patient data
            'heart_rate': vitals.heart_rate,
            'blood_pressure_systolic': vitals.blood_pressure_systolic,
            'blood_pressure_diastolic': vitals.blood_pressure_diastolic,
            'respiratory_rate': vitals.respiratory_rate,
            'temperature_celsius': vitals.temperature_celsius,
            'oxygen_saturation': vitals.oxygen_saturation,
            'weight_kg': vitals.weight_kg,
            'bmi': vitals.weight_kg / ((45 / 100) ** 2),  # Simplified BMI calculation
            'pulse_pressure': vitals.blood_pressure_systolic - vitals.blood_pressure_diastolic,
            'mean_arterial_pressure': vitals.blood_pressure_diastolic + ((vitals.blood_pressure_systolic - vitals.blood_pressure_diastolic) / 3),
            'wait_time_minutes': 0  # Default - could be enhanced with actual wait time
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Select only the features used by the model
        if self.feature_columns:
            df = df[self.feature_columns]
        
        # Scale features
        if self.scaler:
            df_scaled = self.scaler.transform(df)
        else:
            df_scaled = df.values
        
        return df_scaled
    
    def calculate_triage_level(self, vitals: VitalsSubmission, chief_complaint: str) -> TriageResult:
        """
        Calculate triage level using AI model with rule-based fallback
        
        Returns:
            TriageResult with triage_level (1-5), priority_score, and reasoning
        """
        
        # Try AI model first
        if self.model is not None:
            try:
                # Prepare features
                features = self._prepare_features(vitals, chief_complaint)
                
                # Make prediction
                prediction = self.model.predict(features)[0]
                
                # Convert from 0-4 back to 1-5
                ai_triage_level = int(prediction) + 1
                
                # CRITICAL FIX: Validate AI prediction against rule-based logic
                rule_based_result = self._rule_based_triage(vitals, chief_complaint)
                rule_based_level = rule_based_result.triage_level
                
                # If AI gives Level 5 for critical keywords, use rule-based instead
                critical_keywords = ['accident', 'crash', 'trauma', 'injury', 'chest pain', 'severe pain']
                complaint_lower = chief_complaint.lower()
                
                if (ai_triage_level == 5 and 
                    any(keyword in complaint_lower for keyword in critical_keywords) and
                    rule_based_level in [1, 2]):
                    
                    print(f"AI model gave Level 5 for critical case '{chief_complaint}', using rule-based Level {rule_based_level}")
                    
                    # Use rule-based result for critical cases
                    priority_score = self._calculate_priority_score(rule_based_level, 0)
                    return TriageResult(
                        triage_level=rule_based_level,
                        confidence=0.90,  # High confidence for validated rule-based
                        priority_score=priority_score,
                        reasoning=f"Rule-based triage: Level {rule_based_level} - {rule_based_result.reasoning}"
                    )
                
                # Also check if AI gives Level 5 for Level 2 cases
                if (ai_triage_level == 5 and rule_based_level == 2):
                    print(f"AI model gave Level 5 for Level 2 case '{chief_complaint}', using rule-based Level 2")
                    
                    # Use rule-based result for Level 2 cases
                    priority_score = self._calculate_priority_score(rule_based_level, 0)
                    return TriageResult(
                        triage_level=rule_based_level,
                        confidence=0.90,  # High confidence for validated rule-based
                        priority_score=priority_score,
                        reasoning=f"Rule-based triage: Level {rule_based_level} - {rule_based_result.reasoning}"
                    )
                
                # Also check if AI gives Level 5 for Level 4 cases (simple cases)
                if (ai_triage_level == 5 and rule_based_level == 4):
                    print(f"AI model gave Level 5 for Level 4 case '{chief_complaint}', using rule-based Level 4")
                    
                    # Use rule-based result for Level 4 cases
                    priority_score = self._calculate_priority_score(rule_based_level, 0)
                    return TriageResult(
                        triage_level=rule_based_level,
                        confidence=0.90,  # High confidence for validated rule-based
                        priority_score=priority_score,
                        reasoning=f"Rule-based triage: Level {rule_based_level} - {rule_based_result.reasoning}"
                    )
                
                # Calculate priority score
                priority_score = self._calculate_priority_score(ai_triage_level, 0)  # 0 wait time for now
                
                reasoning = f"AI Model Prediction: Level {ai_triage_level} based on vital signs analysis"
                
                return TriageResult(
                    triage_level=ai_triage_level,
                    confidence=0.95,  # High confidence for AI model
                    priority_score=priority_score,
                    reasoning=reasoning
                )
                
            except Exception as e:
                print(f"AI model prediction failed: {e}. Using rule-based fallback.")
        
        # Fallback to rule-based triage
        return self._rule_based_triage(vitals, chief_complaint)
    
    def _rule_based_triage(self, vitals: VitalsSubmission, chief_complaint: str) -> TriageResult:
        """Rule-based triage using ESI guidelines as fallback"""
        
        # Check for immediate life-threatening conditions (Level 1)
        if self._check_level_1_criteria(vitals, chief_complaint):
            triage_level = 1
            reasoning = "Level 1: Immediate life-saving intervention required"
        
        # Check for simple cases first (Level 4) - CRITICAL FIX: Check simple cases before high-risk
        elif self._check_chief_complaint(chief_complaint, "simple"):
            triage_level = 4
            reasoning = "Level 4: One resource needed"
        
        # Check for routine cases (Level 5) - CRITICAL FIX: Check routine cases
        elif self._check_chief_complaint(chief_complaint, "routine"):
            triage_level = 5
            reasoning = "Level 5: No resources needed, routine care"
        
        # Check for high-risk situations (Level 2)
        elif self._check_level_2_criteria(vitals, chief_complaint):
            triage_level = 2
            reasoning = "Level 2: High risk situation, severe distress"
        
        # Check vital signs for moderate risk (Level 3)
        elif self._check_vital_signs(vitals, "level_3"):
            triage_level = 3
            reasoning = "Level 3: Multiple resources likely needed"
        
        # Default to routine care (Level 5)
        else:
            triage_level = 5
            reasoning = "Level 5: No resources needed, routine care"
        
        # Calculate priority score
        priority_score = self._calculate_priority_score(triage_level, 0)  # 0 wait time for now
        
        return TriageResult(
            triage_level=triage_level,
            confidence=0.85,  # Good confidence for rule-based
            priority_score=priority_score,
            reasoning=reasoning
        )
    
    def _calculate_priority_score(self, triage_level: int, wait_time_minutes: int) -> float:
        """Calculate priority score combining urgency and fairness"""
        # Weight factors
        weight_triage = 0.6
        weight_wait = 0.01
        
        priority_score = (triage_level * weight_triage) + (wait_time_minutes * weight_wait)
        return round(priority_score, 2)
    
    def calculate_priority_score(self, triage_level: int, wait_time_minutes: int) -> float:
        """
        Calculate priority score based on triage level and wait time
        
        Args:
            triage_level: Triage level (1-5, where 1 is most urgent)
            wait_time_minutes: Wait time in minutes
            
        Returns:
            Priority score (higher = more urgent)
        """
        # CRITICAL FIX: Ensure Level 1-2 patients get much higher priority
        if triage_level == 1:
            # Level 1 (Critical) - highest priority
            base_priority = 100.0
        elif triage_level == 2:
            # Level 2 (Urgent) - very high priority
            base_priority = 80.0
        elif triage_level == 3:
            # Level 3 (Less Urgent) - moderate priority
            base_priority = 50.0
        elif triage_level == 4:
            # Level 4 (Standard) - lower priority
            base_priority = 30.0
        else:  # Level 5
            # Level 5 (Non-urgent) - lowest priority
            base_priority = 10.0
        
        # Wait time factor (increases priority over time, but not for critical cases)
        if triage_level in [1, 2]:
            # Critical patients don't need wait time boost - they're already highest priority
            wait_factor = 0.0
        else:
            # For non-critical patients, wait time increases priority
            wait_factor = min(wait_time_minutes / 30.0, 2.0)  # Cap at 2x multiplier
        
        # Calculate final priority score
        priority_score = base_priority * (1 + wait_factor)
        
        return round(priority_score, 2)
    
    def _check_level_1_criteria(self, vitals: VitalsSubmission, chief_complaint: str) -> bool:
        """Check for Level 1 (immediate life-saving intervention) criteria"""
        # Check vital signs
        if (vitals.heart_rate < 50 or vitals.heart_rate > 120 or
            vitals.blood_pressure_systolic < 90 or
            vitals.oxygen_saturation < 90 or
            vitals.temperature_celsius > 39.0):
            return True
        
        # Check chief complaint keywords for CRITICAL conditions
        critical_keywords = [
            'cardiac arrest', 'respiratory arrest', 'unconscious', 'not breathing', 
            'severe bleeding', 'stroke', 'severe allergic reaction', 'anaphylaxis',
            'accident', 'crash', 'car accident', 'motor vehicle accident', 'mva',
            'trauma', 'severe trauma', 'head injury', 'brain injury', 'spinal injury',
            'gunshot', 'stab wound', 'penetrating injury', 'amputation',
            'chest pain', 'heart attack', 'myocardial infarction', 'mi',
            'severe pain', 'excruciating pain', 'unbearable pain'
        ]
        
        complaint_lower = chief_complaint.lower()
        if any(keyword in complaint_lower for keyword in critical_keywords):
            print(f"CRITICAL: Level 1 triage triggered by keyword in '{chief_complaint}'")
            return True
        
        return False
    
    def _check_level_2_criteria(self, vitals: VitalsSubmission, chief_complaint: str) -> bool:
        """Check for Level 2 (high risk situation) criteria"""
        # Check vital signs
        if (vitals.heart_rate < 60 or vitals.heart_rate > 100 or
            vitals.blood_pressure_systolic < 100 or
            vitals.oxygen_saturation < 95 or
            vitals.temperature_celsius > 37.5):
            return True
        
        # Check chief complaint keywords for HIGH RISK conditions
        high_risk_keywords = [
            'chest pain', 'shortness of breath', 'difficulty breathing', 'dyspnea',
            'severe pain', 'headache', 'migraine', 'abdominal pain', 'stomach pain',
            'dizziness', 'fainting', 'syncope', 'confusion', 'altered mental status',
            'injury', 'broken bone', 'fracture', 'dislocation', 'sprain', 'strain',
            'fall', 'fell', 'slipped', 'twisted', 'cut', 'laceration', 'wound',
            'burn', 'scald', 'electric shock', 'poisoning', 'overdose',
            'seizure', 'convulsion', 'fever', 'high fever', 'infection',
            'bleeding', 'hemorrhage', 'blood loss', 'weakness', 'numbness',
            'tingling', 'paralysis', 'vision problems', 'blindness',
            'hearing problems', 'deafness', 'speech problems', 'slurred speech'
        ]
        
        # CRITICAL FIX: Exclude simple/minor cases from high-risk classification
        simple_keywords = ['small cut', 'minor cut', 'tiny cut', 'paper cut', 'small scratch']
        complaint_lower = chief_complaint.lower()
        
        # If it's a simple case, don't classify as high risk
        if any(simple in complaint_lower for simple in simple_keywords):
            return False
        
        if any(keyword in complaint_lower for keyword in high_risk_keywords):
            print(f"HIGH RISK: Level 2 triage triggered by keyword in '{chief_complaint}'")
            return True
        
        return False
    
    def _check_vital_signs(self, vitals: VitalsSubmission, level: str) -> bool:
        """Check if vital signs match a specific level"""
        if level == "level_3":
            # Normal vital signs for moderate cases
            return (60 <= vitals.heart_rate <= 100 and
                    90 <= vitals.blood_pressure_systolic <= 140 and
                    95 <= vitals.oxygen_saturation <= 100 and
                    36.0 <= vitals.temperature_celsius <= 37.5)
        return False
    
    def _check_chief_complaint(self, chief_complaint: str, complexity: str) -> bool:
        """Check chief complaint complexity"""
        if complexity == "simple":
            simple_keywords = [
                'small cut', 'tiny cut', 'paper cut', 'small scratch',
                'minor injury', 'small injury', 'simple wound',
                'sore throat', 'ear pain', 'simple rash'
            ]
            complaint_lower = chief_complaint.lower()
            return any(keyword in complaint_lower for keyword in simple_keywords)
        elif complexity == "routine":
            routine_keywords = [
                'prescription refill', 'medication refill', 'refill',
                'check up', 'routine', 'follow up', 'minor complaint',
                'cold symptoms', 'minor cold', 'simple cold'
            ]
            complaint_lower = chief_complaint.lower()
            return any(keyword in complaint_lower for keyword in routine_keywords)
        return False
    
    def _load_esi_guidelines(self) -> Dict[str, Any]:
        """Load ESI triage guidelines for fallback"""
        return {
            "level_1": {
                "description": "Immediate, life-saving intervention",
                "criteria": [
                    "Cardiac/respiratory arrest",
                    "Severe respiratory distress",
                    "Critical trauma",
                    "Severe shock",
                    "Overdose with altered mental status"
                ]
            },
            "level_2": {
                "description": "High risk situation, confused/lethargic, severe pain/distress",
                "criteria": [
                    "Severe pain/distress",
                    "Altered mental status",
                    "Dangerous vital signs",
                    "High risk situation"
                ]
            },
            "level_3": {
                "description": "Many different resources needed",
                "criteria": [
                    "Multiple resources likely needed",
                    "Complex workup required"
                ]
            },
            "level_4": {
                "description": "One resource needed",
                "criteria": [
                    "Simple workup",
                    "Single resource needed"
                ]
            },
            "level_5": {
                "description": "No resources needed",
                "criteria": [
                    "Simple prescription",
                    "No resources needed"
                ]
            }
        }
