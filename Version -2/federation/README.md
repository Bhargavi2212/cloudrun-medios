# Federation Infrastructure

This package provides the services and client utilities that power Medi OS federated learning. It exposes an aggregator that performs FedAvg-style model combination and a reusable client used by hospital agents.

## Aggregator API

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/health` | Service health probe. |
| POST | `/federation/submit` | Submit model weights for aggregation. |
| GET | `/federation/global-model/{model_name}` | Retrieve the latest aggregated model. |

### Example: Submit Update

```http
POST /federation/submit
Authorization: Bearer <shared_secret>
X-Hospital-ID: hospital-a
Content-Type: application/json

{
  "model_name": "triage",
  "round_id": 5,
  "hospital_id": "hospital-a",
  "weights": {
    "layer1": [0.12, 0.24],
    "layer2": [0.04]
  }
}
```

## Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r federation/requirements.txt

cp federation/env.example federation/.env

uvicorn federation.aggregator.main:app --reload --port 8010
```

## Environment Variables

| Variable | Description |
| -------- | ----------- |
| `FEDERATION_SHARED_SECRET` | Shared secret used for Authorization bearer token. |
| `FEDERATION_CORS_ORIGINS` | Optional comma-separated list of allowed origins. |

## Client Usage

```python
from federation.client import FederationClient
from federation.schemas import ModelUpdate

client = FederationClient(
    base_url="http://localhost:8010",
    shared_secret="replace-me",
    hospital_id="hospital-a",
)

update = ModelUpdate(
    model_name="triage",
    round_id=1,
    hospital_id="hospital-a",
    weights={"layer1": [0.1, 0.2]}
)
ack = await client.submit_update(update)
global_model = await client.fetch_global_model("triage")
```

## Testing

```bash
poetry run pytest federation/tests
```

## Docker

```bash
docker build -f federation/Dockerfile -t federation-aggregator:latest .
docker run --env-file federation/env.example -p 8010:8010 federation-aggregator:latest
```

