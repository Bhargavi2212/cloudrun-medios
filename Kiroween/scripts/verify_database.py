#!/usr/bin/env python3
"""
Database verification script for Medi OS Kiroween Edition v2.0

This script validates database schema, connectivity, and medical data integrity
across all hospital instances.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, List, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def verify_database_connection(database_url: str, hospital_id: str) -> Tuple[bool, str]:
    """
    Verify database connection and basic functionality.
    
    Args:
        database_url: PostgreSQL connection URL
        hospital_id: Hospital identifier
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Create async engine
        engine = create_async_engine(database_url, echo=False)
        
        async with engine.begin() as conn:
            # Test basic connectivity
            result = await conn.execute(text("SELECT 1"))
            if result.scalar() != 1:
                return False, "Basic connectivity test failed"
                
            # Test database health check function
            result = await conn.execute(text("SELECT database_health_check()"))
            health_data = result.fetchone()
            
            if not health_data:
                return False, "Health check function not available"
                
            # Verify required extensions
            result = await conn.execute(text("""
                SELECT extname FROM pg_extension 
                WHERE extname IN ('uuid-ossp', 'pgcrypto')
                ORDER BY extname
            """))
            extensions = [row[0] for row in result.fetchall()]
            
            required_extensions = ['pgcrypto', 'uuid-ossp']
            missing_extensions = set(required_extensions) - set(extensions)
            
            if missing_extensions:
                return False, f"Missing extensions: {', '.join(missing_extensions)}"
                
            # Verify custom functions exist
            result = await conn.execute(text("""
                SELECT proname FROM pg_proc 
                WHERE proname IN (
                    'validate_patient_id',
                    'generate_patient_id', 
                    'generate_event_id',
                    'update_profile_integrity_hash',
                    'log_profile_access'
                )
                ORDER BY proname
            """))
            functions = [row[0] for row in result.fetchall()]
            
            required_functions = [
                'generate_event_id',
                'generate_patient_id',
                'log_profile_access',
                'update_profile_integrity_hash',
                'validate_patient_id'
            ]
            missing_functions = set(required_functions) - set(functions)
            
            if missing_functions:
                return False, f"Missing functions: {', '.join(missing_functions)}"
                
            # Test patient ID generation and validation
            result = await conn.execute(text("SELECT generate_patient_id()"))
            patient_id = result.scalar()
            
            result = await conn.execute(text("SELECT validate_patient_id(:pid)"), {"pid": patient_id})
            is_valid = result.scalar()
            
            if not is_valid:
                return False, f"Patient ID validation failed for: {patient_id}"
                
        await engine.dispose()
        return True, f"Database verification successful for {hospital_id}"
        
    except Exception as e:
        return False, f"Database verification failed for {hospital_id}: {str(e)}"


async def verify_table_structure(database_url: str, hospital_id: str) -> Tuple[bool, str]:
    """
    Verify that all required tables exist with correct structure.
    
    Args:
        database_url: PostgreSQL connection URL
        hospital_id: Hospital identifier
        
    Returns:
        Tuple of (success, message)
    """
    try:
        engine = create_async_engine(database_url, echo=False)
        
        async with engine.begin() as conn:
            # Check if tables exist
            result = await conn.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """))
            existing_tables = [row[0] for row in result.fetchall()]
            
            required_tables = [
                'portable_profiles',
                'clinical_events', 
                'local_patient_records',
                'profile_signatures',
                'profile_access_log'
            ]
            
            missing_tables = set(required_tables) - set(existing_tables)
            
            if missing_tables:
                return False, f"Missing tables: {', '.join(missing_tables)}"
                
            # Verify key constraints exist
            result = await conn.execute(text("""
                SELECT constraint_name, table_name 
                FROM information_schema.table_constraints 
                WHERE constraint_type = 'CHECK' 
                AND table_schema = 'public'
                AND constraint_name LIKE 'ck_%'
                ORDER BY table_name, constraint_name
            """))
            constraints = [(row[0], row[1]) for row in result.fetchall()]
            
            # Check for critical medical data constraints
            constraint_names = [c[0] for c in constraints]
            required_constraints = [
                'ck_patient_id_format',
                'ck_biological_sex_values',
                'ck_confidence_score_range',
                'ck_signature_algorithm'
            ]
            
            missing_constraints = set(required_constraints) - set(constraint_names)
            
            if missing_constraints:
                return False, f"Missing constraints: {', '.join(missing_constraints)}"
                
        await engine.dispose()
        return True, f"Table structure verification successful for {hospital_id}"
        
    except Exception as e:
        return False, f"Table structure verification failed for {hospital_id}: {str(e)}"


async def verify_medical_data_integrity(database_url: str, hospital_id: str) -> Tuple[bool, str]:
    """
    Verify medical data integrity constraints and validation.
    
    Args:
        database_url: PostgreSQL connection URL
        hospital_id: Hospital identifier
        
    Returns:
        Tuple of (success, message)
    """
    try:
        engine = create_async_engine(database_url, echo=False)
        
        async with engine.begin() as conn:
            # Test patient ID validation
            test_cases = [
                ("MED-550e8400-e29b-41d4-a716-446655440000", True),
                ("INVALID-ID", False),
                ("MED-invalid-uuid", False),
                ("", False)
            ]
            
            for test_id, expected in test_cases:
                result = await conn.execute(
                    text("SELECT validate_patient_id(:pid)"), 
                    {"pid": test_id}
                )
                is_valid = result.scalar()
                
                if is_valid != expected:
                    return False, f"Patient ID validation failed for: {test_id}"
                    
            # Test that invalid biological sex values are rejected
            try:
                await conn.execute(text("""
                    INSERT INTO portable_profiles (
                        patient_id, 
                        biological_sex, 
                        integrity_hash
                    ) VALUES (
                        'MED-' || gen_random_uuid()::TEXT,
                        'INVALID_SEX',
                        'test_hash'
                    )
                """))
                # If we get here, the constraint didn't work
                return False, "Biological sex constraint not working"
                
            except Exception:
                # This is expected - constraint should reject invalid values
                pass
                
            # Test confidence score range validation
            try:
                await conn.execute(text("""
                    INSERT INTO clinical_events (
                        event_id,
                        patient_id,
                        timestamp,
                        event_type,
                        clinical_summary,
                        confidence_score,
                        cryptographic_signature,
                        signing_key_fingerprint
                    ) VALUES (
                        'EVT-' || gen_random_uuid()::TEXT,
                        'MED-' || gen_random_uuid()::TEXT,
                        NOW(),
                        'visit',
                        'Test summary',
                        1.5,
                        'test_signature',
                        'test_fingerprint'
                    )
                """))
                # If we get here, the constraint didn't work
                return False, "Confidence score constraint not working"
                
            except Exception:
                # This is expected - constraint should reject values > 1.0
                pass
                
        await engine.dispose()
        return True, f"Medical data integrity verification successful for {hospital_id}"
        
    except Exception as e:
        return False, f"Medical data integrity verification failed for {hospital_id}: {str(e)}"


def get_hospital_databases() -> Dict[str, str]:
    """
    Get database URLs for all configured hospitals.
    
    Returns:
        Dictionary mapping hospital_id to database_url
    """
    hospitals = {}
    
    # Check for hospital-specific URLs
    for hospital_id in ['hospital-a', 'hospital-b', 'hospital-c']:
        url_key = f"DATABASE_URL_{hospital_id.upper().replace('-', '_')}"
        database_url = os.getenv(url_key)
        
        if database_url:
            hospitals[hospital_id] = database_url
            
    # Fall back to generic DATABASE_URL if no specific URLs found
    if not hospitals:
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            hospitals["default"] = database_url
            
    return hospitals


async def main() -> int:
    """Main verification function."""
    logger.info("🏥 Starting Medi OS database verification...")
    
    hospitals = get_hospital_databases()
    
    if not hospitals:
        logger.error("❌ No database URLs configured")
        logger.error("   Set DATABASE_URL or hospital-specific URLs:")
        logger.error("   - DATABASE_URL_HOSPITAL_A")
        logger.error("   - DATABASE_URL_HOSPITAL_B") 
        logger.error("   - DATABASE_URL_HOSPITAL_C")
        return 1
        
    total_tests = 0
    passed_tests = 0
    
    for hospital_id, database_url in hospitals.items():
        logger.info(f"\n🔍 Verifying database for {hospital_id}...")
        
        # Convert to async URL if needed
        if not database_url.startswith("postgresql+asyncpg://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            
        # Test 1: Basic connectivity
        success, message = await verify_database_connection(database_url, hospital_id)
        total_tests += 1
        if success:
            logger.info(f"  ✅ Connectivity: {message}")
            passed_tests += 1
        else:
            logger.error(f"  ❌ Connectivity: {message}")
            
        # Test 2: Table structure
        success, message = await verify_table_structure(database_url, hospital_id)
        total_tests += 1
        if success:
            logger.info(f"  ✅ Tables: {message}")
            passed_tests += 1
        else:
            logger.error(f"  ❌ Tables: {message}")
            
        # Test 3: Medical data integrity
        success, message = await verify_medical_data_integrity(database_url, hospital_id)
        total_tests += 1
        if success:
            logger.info(f"  ✅ Integrity: {message}")
            passed_tests += 1
        else:
            logger.error(f"  ❌ Integrity: {message}")
            
    logger.info(f"\n📊 Database Verification Results:")
    logger.info(f"   Hospitals tested: {len(hospitals)}")
    logger.info(f"   Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        logger.info("✅ All database verifications passed!")
        return 0
    else:
        logger.error("❌ Some database verifications failed!")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))