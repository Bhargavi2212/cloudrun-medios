#!/usr/bin/env bash
set -euo pipefail

CONN=${DATABASE_URL:-postgresql://medi_os:medi_os@localhost:5432/medi_os}

psql "${CONN}" <<'SQL'
TRUNCATE TABLE audit_logs CASCADE;
TRUNCATE TABLE triage_observations CASCADE;
TRUNCATE TABLE dialogue_transcripts CASCADE;
TRUNCATE TABLE soap_notes CASCADE;
TRUNCATE TABLE summaries CASCADE;
TRUNCATE TABLE encounters CASCADE;
TRUNCATE TABLE patients CASCADE;
SQL

echo "✅ Demo data reset."

