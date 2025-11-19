"""
Simple integration test for database repositories.
"""

import asyncio
import os
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from shared.database import (
    Base,
    DatabaseManager,
    ProfileRepository,
    ClinicalEventRepository,
    ClinicalEventType
)


async def test_database_integration():
    """Test basic database operations with repositories."""
    
    print("Testing database integration...")
    
    # Use PostgreSQL for testing (requires running database)
    database_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://medi_user:medi_password@localhost:5432/medi_os_test")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(database_url, "test-hospital")
        await db_manager.initialize()
        await db_manager.create_tables()
        
        print("✓ Database initialized")
        
        # Test profile repository
        async with db_manager.get_session() as session:
            profile_repo = ProfileRepository(session)
            
            # Create a test profile
            patient_id = "MED-550e8400-e29b-41d4-a716-446655440000"
            profile = await profile_repo.create_profile(
                patient_id=patient_id,
                first_name="John",
                last_name="Doe",
                date_of_birth=datetime(1980, 1, 1),
                biological_sex="M",
                active_medications={"medications": [{"name": "aspirin", "dose": "81mg"}]},
                known_allergies={"allergies": [{"allergen": "penicillin"}]}
            )
            
            print(f"✓ Created profile: {profile.patient_id}")
            
            # Retrieve the profile
            retrieved_profile = await profile_repo.get_by_id(patient_id)
            assert retrieved_profile is not None
            assert retrieved_profile.first_name == "John"
            
            print("✓ Retrieved profile successfully")
            
            # Test clinical event repository
            event_repo = ClinicalEventRepository(session)
            
            # Create a clinical event
            event = await event_repo.create_clinical_event(
                patient_id=patient_id,
                event_type=ClinicalEventType.VISIT,
                clinical_summary="Patient presented with chest pain",
                structured_data={"vitals": {"bp": "120/80", "hr": 72}},
                ai_generated_insights="Low risk chest pain",
                confidence_score=0.85
            )
            
            print(f"✓ Created clinical event: {event.event_id}")
            
            # Get patient timeline
            timeline = await event_repo.get_patient_timeline(patient_id)
            assert len(timeline) == 1
            assert timeline[0].clinical_summary == "Patient presented with chest pain"
            
            print("✓ Retrieved patient timeline")
            
            # Get timeline summary
            summary = await event_repo.get_timeline_summary(patient_id)
            assert summary["total_events"] == 1
            assert summary["event_counts"]["visit"] == 1
            
            print("✓ Generated timeline summary")
        
        await db_manager.close()
        
        print("\n🎉 All database integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Database integration test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("Medi OS Database Integration Test")
    print("=================================")
    
    success = await test_database_integration()
    
    if success:
        print("\n✅ Database repositories are working correctly!")
        print("Ready to proceed with service integration.")
        return 0
    else:
        print("\n❌ Database integration issues detected")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)