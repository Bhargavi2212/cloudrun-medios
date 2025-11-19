-- Initial database setup for Medi OS Kiroween Edition v2.0
-- This script creates the basic database structure and extensions

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create custom types for medical data validation
DO $$ BEGIN
    CREATE TYPE clinical_event_type AS ENUM (
        'visit',
        'procedure', 
        'diagnosis',
        'medication',
        'allergy',
        'lab_result',
        'imaging',
        'summary',
        'emergency'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE profile_sync_status AS ENUM (
        'synced',
        'pending',
        'conflict',
        'error'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create function to validate MED-{uuid4} format
CREATE OR REPLACE FUNCTION validate_patient_id(patient_id TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN patient_id ~ '^MED-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create function to generate MED-{uuid4} patient IDs
CREATE OR REPLACE FUNCTION generate_patient_id()
RETURNS TEXT AS $$
BEGIN
    RETURN 'MED-' || gen_random_uuid()::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Create function to generate event IDs
CREATE OR REPLACE FUNCTION generate_event_id()
RETURNS TEXT AS $$
BEGIN
    RETURN 'EVT-' || gen_random_uuid()::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Create function to update profile integrity hash
CREATE OR REPLACE FUNCTION update_profile_integrity_hash()
RETURNS TRIGGER AS $$
BEGIN
    -- Update integrity hash when profile data changes
    NEW.integrity_hash = encode(
        digest(
            COALESCE(NEW.first_name, '') ||
            COALESCE(NEW.last_name, '') ||
            COALESCE(NEW.date_of_birth::TEXT, '') ||
            COALESCE(NEW.biological_sex, '') ||
            COALESCE(NEW.active_medications::TEXT, '{}') ||
            COALESCE(NEW.known_allergies::TEXT, '{}') ||
            COALESCE(NEW.chronic_conditions::TEXT, '{}') ||
            NEW.last_updated::TEXT,
            'sha256'
        ),
        'hex'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create audit logging function (no PHI)
CREATE OR REPLACE FUNCTION log_profile_access(
    patient_id TEXT,
    action_type TEXT,
    user_id TEXT DEFAULT 'system',
    hospital_id TEXT DEFAULT 'unknown'
)
RETURNS VOID AS $$
BEGIN
    -- Log profile access for audit purposes (no PHI)
    INSERT INTO profile_access_log (
        patient_id_hash,
        action_type,
        user_id_hash,
        hospital_id_hash,
        access_timestamp
    ) VALUES (
        encode(digest(patient_id, 'sha256'), 'hex'),
        action_type,
        encode(digest(user_id, 'sha256'), 'hex'),
        encode(digest(hospital_id, 'sha256'), 'hex'),
        NOW()
    );
END;
$$ LANGUAGE plpgsql;

-- Create audit log table (no PHI stored)
CREATE TABLE IF NOT EXISTS profile_access_log (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id_hash VARCHAR(64) NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    user_id_hash VARCHAR(64) NOT NULL,
    hospital_id_hash VARCHAR(64) NOT NULL,
    access_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes for audit queries
    INDEX idx_access_log_timestamp (access_timestamp),
    INDEX idx_access_log_patient_hash (patient_id_hash),
    INDEX idx_access_log_action (action_type)
);

-- Create database health check function
CREATE OR REPLACE FUNCTION database_health_check()
RETURNS TABLE(
    status TEXT,
    database_name TEXT,
    connection_count INTEGER,
    last_stats_reset TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'healthy'::TEXT as status,
        current_database()::TEXT as database_name,
        (SELECT count(*)::INTEGER FROM pg_stat_activity WHERE datname = current_database()) as connection_count,
        (SELECT stats_reset FROM pg_stat_database WHERE datname = current_database()) as last_stats_reset;
END;
$$ LANGUAGE plpgsql;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO PUBLIC;
GRANT CREATE ON SCHEMA public TO PUBLIC;

-- Create indexes for performance
-- These will be created by Alembic migrations, but included here for reference

COMMENT ON DATABASE CURRENT_DATABASE() IS 'Medi OS Kiroween Edition v2.0 - Patient-controlled medical records with federated learning';

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Medi OS database initialized successfully for hospital: %', COALESCE(current_setting('app.hospital_id', true), 'unknown');
END $$;