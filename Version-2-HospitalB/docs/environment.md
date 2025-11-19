# Environment Variables Reference

This document lists the key environment variables required across Medi OS Version -2 services.

## Common

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `DATABASE_URL` | Async SQLAlchemy connection string for the local hospital database. | `postgresql+asyncpg://medi_os:medi_os@localhost:5432/medi_os` |

## Manage-Agent

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `MANAGE_AGENT_MODEL_VERSION` | Identifier for the triage model in use. | `triage_v0` |
| `MANAGE_AGENT_CORS_ORIGINS` | Comma-separated origins for frontend access. | `http://localhost:5173` |
| `MANAGE_AGENT_DOL_URL` | Base URL for the DOL service. | `http://localhost:8004` |
| `MANAGE_AGENT_DOL_SECRET` | Shared secret when calling the DOL. | – (required) |
| `MANAGE_AGENT_HOSPITAL_ID` | Hospital identifier sent to the DOL. | `hospital-a` |

## Scribe-Agent

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `SCRIBE_AGENT_MODEL_VERSION` | Identifier for SOAP generation model. | `scribe_v0` |
| `SCRIBE_AGENT_CORS_ORIGINS` | Allowed origins. | `http://localhost:5173` |

## Summarizer-Agent

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `SUMMARIZER_AGENT_MODEL_VERSION` | Identifier for summarizer model. | `summary_v0` |
| `SUMMARIZER_AGENT_CORS_ORIGINS` | Allowed origins. | `http://localhost:5173` |

## Data Orchestration Layer (DOL)

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `DOL_HOSPITAL_ID` | Unique identifier for the hospital. | `hospital-a` |
| `DOL_SHARED_SECRET` | Shared secret for federated requests. | – (required) |
| `DOL_CORS_ORIGINS` | Allowed origins. | `http://localhost:5173` |
| `DOL_PEERS` | JSON array of peer configurations. | `[]` |

## Federation Aggregator

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `FEDERATION_SHARED_SECRET` | Shared secret for aggregator endpoints. | – (required) |
| `FEDERATION_CORS_ORIGINS` | Allowed origins. | `http://localhost:5173` |

## Frontend

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `VITE_MANAGE_API_URL` | Manage-agent base URL. | `http://localhost:9001` |
| `VITE_SCRIBE_API_URL` | Scribe-agent base URL. | `http://localhost:9002` |
| `VITE_SUMMARIZER_API_URL` | Summarizer-agent base URL. | `http://localhost:9003` |
| `VITE_DOL_API_URL` | DOL base URL. | `http://localhost:8004` |
| `VITE_FEDERATION_API_URL` | Federation aggregator base URL. | `http://localhost:8010` |

