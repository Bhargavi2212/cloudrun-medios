#!/usr/bin/env bash
set -euo pipefail

API_URL=${FEDERATION_API_URL:-http://localhost:8010}
SHARED_SECRET=${FEDERATION_SHARED_SECRET:-super-secret}
HOSPITAL_ID=${HOSPITAL_ID:-hospital-a}

if [[ -z "${1:-}" ]]; then
  echo "Usage: ./scripts/run_federated_round.sh model_name"
  exit 1
fi

MODEL_NAME="$1"
ROUND_ID=${ROUND_ID:-1}
WEIGHTS=${WEIGHTS:-"0.1,0.2,0.3"}

payload=$(cat <<EOF
{
  "model_name": "${MODEL_NAME}",
  "round_id": ${ROUND_ID},
  "hospital_id": "${HOSPITAL_ID}",
  "weights": {
    "dense": [${WEIGHTS}]
  }
}
EOF
)

curl -sS -X POST "${API_URL}/federation/submit" \
  -H "Authorization: Bearer ${SHARED_SECRET}" \
  -H "X-Hospital-ID: ${HOSPITAL_ID}" \
  -H "Content-Type: application/json" \
  -d "${payload}"

