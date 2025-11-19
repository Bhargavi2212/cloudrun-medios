#!/usr/bin/env python3
"""
Integration tests for multi-hospital profile assembly.

This module tests patient profile import/export between hospitals,
append-only timeline functionality, and distributed hospital network simulation.
"""

import asyncio
import pytest
import json
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path

# Test configuration
TEST_HOSPITALS = [
    {
        "hospital_id": "hospital_a",
        "hospital_name": "General Hospital A",
        "api_endpoint": "http://localhost:8001",
        "capabilities": ["profile_import", "profile_export", "timeline_sync", "federated_learning"]
    },
    {
        "hospital_id": "hospital_b", 
        "hospital_name": "Medical Center B",
        "api_endpoint": "http://localhost:8002",
        "capabilities": ["profile_import", "profile_export", "timeline_sync"]
    },
    {
        "hospital_id": "hospital_c",
        "hospital_name": "Regional Hospital C", 
        "api_endpoint": "http://localhost:8003",
        "capabilities": ["profile_import", "profile_export", "emergency_access"]
    }
]

# Sample patient data for testing
SAMPLE_PATIENT_DATA = {
    "patient_id": "MED-integration-test-001",
    "demographics": {
        "first_name": "John",
        "last_name": "Doe", 
        "date_of_birth": "1980-01-15",
        "biological_sex": "male"
    },
    "clinical_timeline": [
        {
            "entry_id": "entry_001",
            "timestamp": datetime.utcnow() - timedelta(days=30),
            "event_type": "clinical_visit",
            "clinical_summary": "Initial consultation for chest pain evaluation",
            "structured_data": {
                "symptoms": ["chest pain", "shortness of breath"],
                "vital_signs": {"hr": 88, "bp": "140/90", "temp": 98.6},
                "medications_prescribed": ["aspirin 81mg daily"]
            }
        }
    ],
    "active_medications": [
        {
            "name": "Aspirin",
            "dosage": "81mg",
            "frequency": "daily",
            "start_date": "2024-10-15"
        }
    ],
    "known_allergies": [
        {
            "allergen": "Penicillin",
            "reaction": "Rash",
            "severity": "moderate"
        }
    ],
    "chronic_conditions": [
        {
            "condition_name": "Hypertension",
            "icd_code": "I10",
            "diagnosis_date": "2023-05-20",
            "status": "active"
        }
    ]
}
class 
MockHospitalService:
    """Mock hospital service for integration testing."""
    
    def __init__(self, hospital_config: Dict[str, Any]):
        self.hospital_id = hospital_config["hospital_id"]
        self.hospital_name = hospital_config["hospital_name"]
        self.api_endpoint = hospital_config["api_endpoint"]
        self.capabilities = hospital_config["capabilities"]
        
        # In-memory storage for testing
        self.patients: Dict[str, Dict[str, Any]] = {}
        self.peer_registry: Dict[str, Dict[str, Any]] = {}
        self.audit_logs: List[Dict[str, Any]] = []
        
        # Initialize with test data directory
        self.data_dir = Path(tempfile.mkdtemp(prefix=f"hospital_{self.hospital_id}_"))
        
    async def import_patient_profile(
        self,
        encrypted_profile_data: str,
        cryptographic_signature: str,
        source_hospital_id: str = None
    ) -> Dict[str, Any]:
        """Simulate patient profile import."""
        try:
            # Simulate decryption and verification
            profile_data = json.loads(encrypted_profile_data)  # Simplified for testing
            patient_id = profile_data["patient_id"]
            
            # Check if patient already exists
            if patient_id in self.patients:
                # Merge timeline entries (append-only)
                existing_timeline = self.patients[patient_id].get("clinical_timeline", [])
                new_timeline = profile_data.get("clinical_timeline", [])
                
                # Combine timelines and sort by timestamp
                combined_timeline = existing_timeline + new_timeline
                combined_timeline.sort(key=lambda x: x["timestamp"])
                
                # Update patient data
                self.patients[patient_id].update(profile_data)
                self.patients[patient_id]["clinical_timeline"] = combined_timeline
            else:
                # New patient
                self.patients[patient_id] = profile_data
            
            # Log audit event
            await self._log_audit_event(
                category="profile_import",
                message=f"Imported profile for patient {patient_id}",
                patient_id=patient_id,
                source_hospital=source_hospital_id
            )
            
            return {
                "success": True,
                "patient_id": patient_id,
                "timeline_entries_imported": len(profile_data.get("clinical_timeline", [])),
                "import_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def export_patient_profile(
        self,
        patient_id: str,
        privacy_level: str = "standard"
    ) -> Dict[str, Any]:
        """Simulate patient profile export."""
        try:
            if patient_id not in self.patients:
                return {
                    "success": False,
                    "error": f"Patient {patient_id} not found"
                }
            
            # Get patient data
            patient_data = self.patients[patient_id].copy()
            
            # Apply privacy filtering (remove hospital metadata)
            filtered_profile = await self._apply_privacy_filtering(patient_data)
            
            # Simulate encryption and signing
            encrypted_data = json.dumps(filtered_profile)  # Simplified for testing
            signature = f"signature_{self.hospital_id}_{patient_id}"
            
            # Log audit event
            await self._log_audit_event(
                category="profile_export",
                message=f"Exported profile for patient {patient_id}",
                patient_id=patient_id
            )
            
            return {
                "success": True,
                "patient_id": patient_id,
                "encrypted_profile_data": encrypted_data,
                "cryptographic_signature": signature,
                "export_timestamp": datetime.utcnow().isoformat(),
                "timeline_entries_count": len(filtered_profile.get("clinical_timeline", []))
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def append_timeline_entry(
        self,
        patient_id: str,
        clinical_event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate appending timeline entry."""
        try:
            if patient_id not in self.patients:
                return {
                    "success": False,
                    "error": f"Patient {patient_id} not found"
                }
            
            # Create timeline entry
            entry = {
                "entry_id": f"entry_{self.hospital_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.utcnow().isoformat(),
                "hospital_id": self.hospital_id,  # This will be filtered out in export
                **clinical_event
            }
            
            # Append to timeline (append-only)
            if "clinical_timeline" not in self.patients[patient_id]:
                self.patients[patient_id]["clinical_timeline"] = []
            
            self.patients[patient_id]["clinical_timeline"].append(entry)
            
            # Log audit event
            await self._log_audit_event(
                category="timeline_append",
                message=f"Appended timeline entry for patient {patient_id}",
                patient_id=patient_id
            )
            
            return {
                "success": True,
                "entry_id": entry["entry_id"],
                "patient_id": patient_id,
                "timestamp": entry["timestamp"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_patient_timeline(
        self,
        patient_id: str,
        include_hospital_metadata: bool = False
    ) -> Dict[str, Any]:
        """Get patient timeline."""
        try:
            if patient_id not in self.patients:
                return {
                    "success": False,
                    "error": f"Patient {patient_id} not found"
                }
            
            timeline = self.patients[patient_id].get("clinical_timeline", [])
            
            # Filter hospital metadata if requested
            if not include_hospital_metadata:
                filtered_timeline = []
                for entry in timeline:
                    filtered_entry = {k: v for k, v in entry.items() if k != "hospital_id"}
                    filtered_timeline.append(filtered_entry)
                timeline = filtered_timeline
            
            return {
                "success": True,
                "patient_id": patient_id,
                "timeline_entries": timeline,
                "total_entries": len(timeline)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def register_peer_hospital(self, peer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Register a peer hospital."""
        try:
            peer_id = peer_config["hospital_id"]
            self.peer_registry[peer_id] = {
                **peer_config,
                "registered_at": datetime.utcnow().isoformat(),
                "status": "active",
                "trust_score": 1.0
            }
            
            return {
                "success": True,
                "hospital_id": peer_id,
                "status": "active"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _apply_privacy_filtering(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply privacy filtering to remove hospital metadata."""
        filtered_data = patient_data.copy()
        
        # Filter timeline entries
        if "clinical_timeline" in filtered_data:
            filtered_timeline = []
            for entry in filtered_data["clinical_timeline"]:
                filtered_entry = {k: v for k, v in entry.items() if k != "hospital_id"}
                # Remove any hospital-identifying information from clinical summary
                if "clinical_summary" in filtered_entry:
                    summary = filtered_entry["clinical_summary"]
                    # Basic filtering (in production this would be more sophisticated)
                    summary = summary.replace(self.hospital_name, "[HOSPITAL]")
                    summary = summary.replace(self.hospital_id, "[HOSPITAL_ID]")
                    filtered_entry["clinical_summary"] = summary
                
                filtered_timeline.append(filtered_entry)
            
            filtered_data["clinical_timeline"] = filtered_timeline
        
        return filtered_data
    
    async def _log_audit_event(
        self,
        category: str,
        message: str,
        patient_id: str = None,
        **additional_data
    ) -> None:
        """Log audit event."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "hospital_id": self.hospital_id,
            "category": category,
            "message": message,
            "patient_id_hash": f"hash_{patient_id}" if patient_id else None,
            "additional_data": additional_data
        }
        
        self.audit_logs.append(audit_entry)
    
    def cleanup(self):
        """Cleanup test data."""
        import shutil
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)class
 TestMultiHospitalIntegration:
    """Integration tests for multi-hospital profile assembly."""
    
    @pytest.fixture
    async def hospital_network(self):
        """Create a network of mock hospitals for testing."""
        hospitals = {}
        
        for config in TEST_HOSPITALS:
            hospital = MockHospitalService(config)
            hospitals[config["hospital_id"]] = hospital
        
        # Register each hospital as a peer of the others
        for hospital_id, hospital in hospitals.items():
            for peer_config in TEST_HOSPITALS:
                if peer_config["hospital_id"] != hospital_id:
                    await hospital.register_peer_hospital(peer_config)
        
        yield hospitals
        
        # Cleanup
        for hospital in hospitals.values():
            hospital.cleanup()
    
    @pytest.mark.asyncio
    async def test_patient_profile_import_export_between_hospitals(self, hospital_network):
        """Test patient profile import/export between hospitals."""
        hospital_a = hospital_network["hospital_a"]
        hospital_b = hospital_network["hospital_b"]
        
        # Hospital A creates initial patient profile
        patient_data = SAMPLE_PATIENT_DATA.copy()
        patient_id = patient_data["patient_id"]
        
        # Simulate patient visit at Hospital A
        hospital_a.patients[patient_id] = patient_data
        
        # Add a clinical event at Hospital A
        await hospital_a.append_timeline_entry(patient_id, {
            "event_type": "clinical_visit",
            "clinical_summary": "Follow-up visit for chest pain, symptoms improved",
            "structured_data": {
                "symptoms": ["mild chest discomfort"],
                "vital_signs": {"hr": 72, "bp": "130/85"},
                "medications_adjusted": ["aspirin 81mg daily continued"]
            }
        })
        
        # Hospital A exports patient profile
        export_result = await hospital_a.export_patient_profile(patient_id)
        assert export_result["success"] == True
        assert export_result["patient_id"] == patient_id
        assert "encrypted_profile_data" in export_result
        assert "cryptographic_signature" in export_result
        
        # Hospital B imports the profile
        import_result = await hospital_b.import_patient_profile(
            encrypted_profile_data=export_result["encrypted_profile_data"],
            cryptographic_signature=export_result["cryptographic_signature"],
            source_hospital_id="hospital_a"
        )
        
        assert import_result["success"] == True
        assert import_result["patient_id"] == patient_id
        assert import_result["timeline_entries_imported"] >= 1
        
        # Verify patient exists in Hospital B
        assert patient_id in hospital_b.patients
        
        # Verify timeline was imported correctly
        timeline_result = await hospital_b.get_patient_timeline(patient_id)
        assert timeline_result["success"] == True
        assert timeline_result["total_entries"] >= 2  # Original + Hospital A's addition
        
        print(f"✅ Successfully imported patient {patient_id} from Hospital A to Hospital B")
        print(f"   Timeline entries: {timeline_result['total_entries']}")
    
    @pytest.mark.asyncio
    async def test_append_only_timeline_across_hospitals(self, hospital_network):
        """Test append-only timeline functionality across multiple hospitals."""
        hospital_a = hospital_network["hospital_a"]
        hospital_b = hospital_network["hospital_b"] 
        hospital_c = hospital_network["hospital_c"]
        
        patient_data = SAMPLE_PATIENT_DATA.copy()
        patient_id = patient_data["patient_id"]
        
        # Patient starts at Hospital A
        hospital_a.patients[patient_id] = patient_data
        
        # Add event at Hospital A
        await hospital_a.append_timeline_entry(patient_id, {
            "event_type": "clinical_visit",
            "clinical_summary": "Initial cardiac evaluation completed",
            "structured_data": {"procedures": ["ECG", "chest X-ray"], "results": "normal"}
        })
        
        # Export from A and import to B
        export_a = await hospital_a.export_patient_profile(patient_id)
        await hospital_b.import_patient_profile(
            export_a["encrypted_profile_data"],
            export_a["cryptographic_signature"],
            "hospital_a"
        )
        
        # Add event at Hospital B
        await hospital_b.append_timeline_entry(patient_id, {
            "event_type": "follow_up",
            "clinical_summary": "Cardiology follow-up, stress test ordered",
            "structured_data": {"procedures": ["stress test"], "specialist": "cardiology"}
        })
        
        # Export from B and import to C
        export_b = await hospital_b.export_patient_profile(patient_id)
        await hospital_c.import_patient_profile(
            export_b["encrypted_profile_data"],
            export_b["cryptographic_signature"],
            "hospital_b"
        )
        
        # Add event at Hospital C
        await hospital_c.append_timeline_entry(patient_id, {
            "event_type": "emergency",
            "clinical_summary": "Emergency visit for acute chest pain, ruled out MI",
            "structured_data": {"diagnosis": "non-cardiac chest pain", "disposition": "discharged"}
        })
        
        # Verify timeline integrity across all hospitals
        timeline_a = await hospital_a.get_patient_timeline(patient_id)
        timeline_b = await hospital_b.get_patient_timeline(patient_id)
        timeline_c = await hospital_c.get_patient_timeline(patient_id)
        
        # Hospital A should have original + its addition (2 entries)
        assert timeline_a["total_entries"] == 2
        
        # Hospital B should have original + A's addition + its addition (3 entries)
        assert timeline_b["total_entries"] == 3
        
        # Hospital C should have all entries (4 entries)
        assert timeline_c["total_entries"] == 4
        
        # Verify chronological order in Hospital C (complete timeline)
        entries = timeline_c["timeline_entries"]
        timestamps = [entry["timestamp"] for entry in entries]
        assert timestamps == sorted(timestamps), "Timeline entries should be in chronological order"
        
        print(f"✅ Append-only timeline verified across 3 hospitals")
        print(f"   Hospital A: {timeline_a['total_entries']} entries")
        print(f"   Hospital B: {timeline_b['total_entries']} entries") 
        print(f"   Hospital C: {timeline_c['total_entries']} entries")
    
    @pytest.mark.asyncio
    async def test_privacy_filtering_in_profile_export(self, hospital_network):
        """Test that hospital metadata is filtered during profile export."""
        hospital_a = hospital_network["hospital_a"]
        
        patient_data = SAMPLE_PATIENT_DATA.copy()
        patient_id = patient_data["patient_id"]
        
        # Create patient with hospital-specific data
        hospital_a.patients[patient_id] = patient_data
        
        # Add timeline entry with hospital metadata
        await hospital_a.append_timeline_entry(patient_id, {
            "event_type": "clinical_visit",
            "clinical_summary": f"Patient seen at {hospital_a.hospital_name} by Dr. Smith",
            "structured_data": {
                "attending_physician": "Dr. Smith",
                "department": "Cardiology",
                "hospital_mrn": "MRN123456"
            }
        })
        
        # Export profile
        export_result = await hospital_a.export_patient_profile(patient_id)
        assert export_result["success"] == True
        
        # Parse exported data
        exported_data = json.loads(export_result["encrypted_profile_data"])
        
        # Verify hospital metadata is filtered
        timeline_entries = exported_data.get("clinical_timeline", [])
        for entry in timeline_entries:
            # Hospital ID should be removed
            assert "hospital_id" not in entry
            
            # Hospital name should be replaced in clinical summary
            if "clinical_summary" in entry:
                assert hospital_a.hospital_name not in entry["clinical_summary"]
                assert "[HOSPITAL]" in entry["clinical_summary"] or hospital_a.hospital_name not in entry["clinical_summary"]
        
        print(f"✅ Privacy filtering verified - hospital metadata removed from export")
    
    @pytest.mark.asyncio
    async def test_multi_hospital_patient_journey_simulation(self, hospital_network):
        """Simulate a complete patient journey across multiple hospitals."""
        hospital_a = hospital_network["hospital_a"]
        hospital_b = hospital_network["hospital_b"]
        hospital_c = hospital_network["hospital_c"]
        
        patient_data = SAMPLE_PATIENT_DATA.copy()
        patient_id = patient_data["patient_id"]
        
        print(f"\n🏥 Simulating patient journey for {patient_id}")
        
        # Day 1: Initial visit at Hospital A
        print("   Day 1: Initial visit at Hospital A")
        hospital_a.patients[patient_id] = patient_data
        await hospital_a.append_timeline_entry(patient_id, {
            "event_type": "clinical_visit",
            "clinical_summary": "Initial presentation with chest pain, workup initiated",
            "structured_data": {
                "chief_complaint": "chest pain",
                "procedures": ["ECG", "chest X-ray", "blood work"],
                "disposition": "discharged home"
            }
        })
        
        # Day 5: Follow-up at Hospital B (patient moved)
        print("   Day 5: Follow-up at Hospital B (patient moved)")
        export_a = await hospital_a.export_patient_profile(patient_id)
        await hospital_b.import_patient_profile(
            export_a["encrypted_profile_data"],
            export_a["cryptographic_signature"],
            "hospital_a"
        )
        
        await hospital_b.append_timeline_entry(patient_id, {
            "event_type": "follow_up",
            "clinical_summary": "Cardiology follow-up, stress test results normal",
            "structured_data": {
                "procedures": ["stress test"],
                "results": "normal",
                "plan": "continue current medications"
            }
        })
        
        # Day 10: Emergency visit at Hospital C
        print("   Day 10: Emergency visit at Hospital C")
        export_b = await hospital_b.export_patient_profile(patient_id)
        await hospital_c.import_patient_profile(
            export_b["encrypted_profile_data"],
            export_b["cryptographic_signature"],
            "hospital_b"
        )
        
        await hospital_c.append_timeline_entry(patient_id, {
            "event_type": "emergency",
            "clinical_summary": "Emergency presentation with severe chest pain, ruled out acute MI",
            "structured_data": {
                "triage_level": "ESI-2",
                "procedures": ["ECG", "troponins", "CT angiogram"],
                "diagnosis": "atypical chest pain",
                "disposition": "discharged"
            }
        })
        
        # Verify complete patient journey
        final_timeline = await hospital_c.get_patient_timeline(patient_id)
        assert final_timeline["success"] == True
        assert final_timeline["total_entries"] >= 4  # Original + 3 hospital visits
        
        # Verify audit logs across all hospitals
        total_audit_events = (
            len(hospital_a.audit_logs) + 
            len(hospital_b.audit_logs) + 
            len(hospital_c.audit_logs)
        )
        assert total_audit_events >= 6  # Imports, exports, and timeline appends
        
        print(f"✅ Complete patient journey simulation successful")
        print(f"   Final timeline entries: {final_timeline['total_entries']}")
        print(f"   Total audit events: {total_audit_events}")
    
    @pytest.mark.asyncio
    async def test_concurrent_timeline_updates(self, hospital_network):
        """Test concurrent timeline updates from multiple hospitals."""
        hospital_a = hospital_network["hospital_a"]
        hospital_b = hospital_network["hospital_b"]
        
        patient_data = SAMPLE_PATIENT_DATA.copy()
        patient_id = patient_data["patient_id"]
        
        # Set up patient in both hospitals
        hospital_a.patients[patient_id] = patient_data
        hospital_b.patients[patient_id] = patient_data.copy()
        
        # Simulate concurrent timeline updates
        tasks = []
        
        # Hospital A adds multiple entries
        for i in range(3):
            task = hospital_a.append_timeline_entry(patient_id, {
                "event_type": "clinical_visit",
                "clinical_summary": f"Hospital A visit {i+1}",
                "structured_data": {"visit_number": i+1, "hospital": "A"}
            })
            tasks.append(task)
        
        # Hospital B adds multiple entries
        for i in range(3):
            task = hospital_b.append_timeline_entry(patient_id, {
                "event_type": "clinical_visit", 
                "clinical_summary": f"Hospital B visit {i+1}",
                "structured_data": {"visit_number": i+1, "hospital": "B"}
            })
            tasks.append(task)
        
        # Execute all updates concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify all updates succeeded
        for result in results:
            assert result["success"] == True
        
        # Verify timeline integrity
        timeline_a = await hospital_a.get_patient_timeline(patient_id)
        timeline_b = await hospital_b.get_patient_timeline(patient_id)
        
        assert timeline_a["total_entries"] == 4  # Original + 3 additions
        assert timeline_b["total_entries"] == 4  # Original + 3 additions
        
        print(f"✅ Concurrent timeline updates successful")
        print(f"   Hospital A timeline: {timeline_a['total_entries']} entries")
        print(f"   Hospital B timeline: {timeline_b['total_entries']} entries")