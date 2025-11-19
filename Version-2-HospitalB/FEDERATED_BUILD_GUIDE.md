# MEDI OS KIROWEEN EDITION v2.0 – END-TO-END BUILD & DEMO GUIDE

Architecture: Federated Data + Federated Learning  
Date: November 15, 2025

---

## 1. Project Vision

Enable portable, privacy-preserving patient profiles and continuous, collaborative learning across multiple hospitals—without ever sharing raw patient data.

- Each hospital owns and stores their own clinical data locally.
- Portable patient profiles are dynamically assembled via federated data orchestration.
- AI models are trained and improved via federated learning: each hospital trains on its own data, shares only model updates for global model aggregation and improvement.

---

## 2. High-Level Architecture

### Federated Data Flow (Portable Patient Profile)

```
Doctor at Hospital B → Requests Patient Profile for ID: MED-xxxxxx
        |
        ↓
DOL (Hospital B) orchestrates queries to DOL at Hospital A, C, ...N
        |
        ↓
Each hospital’s DOL:
    -  Retrieves local profile, timeline, summaries (if exists)
    -  Applies privacy filter: removes hospital/doctor/location info
        |
        ↓
DOL (Hospital B) aggregates responses, merges/filters into single unified timeline/profile
        |
        ↓
Portable profile returned to Doctor’s frontend (Hospital B)
```

### Federated Learning Flow

```
Each Hospital:
    1. Trains AI models (triage, summarizer, etc.) on local patient data
    2. Shares only model weights/gradients (never raw data) at regular intervals
Aggregator (central or peer-coordinated):
    3. Combines (averages/aggregates) models into global update
    4. Sends updated global model parameters back to all hospitals
Each Hospital:
    5. Updates local models to reap benefits of global learning
Process repeats for continuous, privacy-preserving improvement
```

---

## 3. System Components

### Per Hospital

- **Local PostgreSQL Database:** All local patient/timeline/event data.
- **AI Agents (Local):** Triage, Scribe, Summarizer—train and run on local data.
- **DOL - Data Orchestration Layer:**  
  - API for federated data queries (Patient ID-based, privacy-filtered)  
  - Receives/sends requests from/to peer hospitals  
  - No persistent storage except for audit logs (non-PHI)
- **Federated Parameter Client:** Node that handles model update, shares with aggregator, updates AI models.
- **Access Control & Audit:** Only authenticated, authorized requests allowed. Audit logs maintained for access.

### Aggregator (Central or Rotating Peer)

- **Model Aggregator Service:** Receives, aggregates, and distributes model updates.  
  - Can be hosted by a neutral party or via secure multi-party computation (future).

---

## 4. Tech Stack

- **Database:** PostgreSQL (per hospital)
- **Backend:** FastAPI Python 3.11+, async ORM
- **AI Models:** XGBoost, LightGBM, PyTorch, or TensorFlow; wrapped for federated learning
- **Federated Learning Framework:** FedML, Flower, PySyft, or custom (for demo, can simulate aggregation)
- **Storage:** Hospital’s local disk or cloud bucket
- **Security:** JWT + mutual TLS for federated comms; CORS, RBAC, encryption everywhere

---

## 5. Hospital Data Layer Interface

### Key DOL API Endpoints (per hospital)

- `POST /api/federated/patient` — Retrieve portable profile for Patient ID (returns clinical data, privacy-filtered)
- `POST /api/federated/timeline` — Retrieve portable timeline for Patient ID
- `POST /api/federated/model_update` — Receive/share model weights (federated learning)
- `GET /api/dol/health` — Service health check

### Peer Discovery

- Hospitals maintain a list of trusted peer DOLs (manual or via central registry, NOT public).

---

## 6. Federated Learning Protocol

### Training Cycle

1. **Local Training:** Each hospital trains (or fine-tunes) its own AI model on local data for a set number of epochs/rounds.
2. **Model Update:** On schedule, model weights or gradients (NOT raw data) are sent to the aggregator service.
3. **Aggregation:** Aggregator averages or aggregates the parameters/gradients (Federated Averaging).
4. **Global Model Broadcast:** Aggregator returns global model to each hospital.
5. **Local Update:** Each hospital updates local model to new global.
6. **Repeat:** Cycle begins again with new local training on new batches.

### Security & Privacy

- All model updates encrypted in transit (TLS).
- Only model params/weights/gradients shared—never patient, visit, or raw encounter data.
- Hospitals never see each other’s patient data.

---

## 7. Demo Scenarios

### Scenario 1: Federated Profile Retrieval

- Patient registered at Hospital A (gets MED-uuid4).
- Patient visits Hospital B, provides Patient ID.
- DOL Hospital B queries DOLs at A, C, ...N.
- Each DOL returns timeline/summaries—privacy-filtered.
- Hospital B’s DOL merges and presents a portable profile without any hospital metadata.

### Scenario 2: Federated Learning in Action

- All hospitals train triage model on local data for 5 epochs.
- Model weights sent to aggregator.
- Aggregator computes the federated average/global model.
- New global model distributed to all hospitals; each sees improved validation accuracy on their own test set (documented in logs/demo).
- (Optional) Show that no raw patient data ever leaves any hospital during the process.

---

## 8. Testing & Validation

### For Federated Data

- Simulate 3 or more hospital instances with distinct patient sets.
- Register synthetic patients at each hospital.
- Perform federated profile lookup from a new hospital—verify successful cross-hospital timeline aggregation and privacy filtering.
- Ensure audit logs are written for each federated request/response.

### For Federated Learning

- Each hospital logs model accuracy before and after global update.
- Confirm improved or at least stable model accuracy after each round.
- Document time taken for local training, update sharing, and global aggregation.
- Verify no hospital receives another’s raw data.

---

## 9. Security & Governance

- JWT authentication for all DOL-to-DOL and federated model communications.
- Only hospitals in the allow-list/trusted registry may participate.
- All network communications encrypted via mutual TLS.
- Patient privacy filter ensures no hospital, provider, or location details appear in the portable profile.
- Audit logging of patient ID lookups (timestamp, requesting hospital, NOT clinical data).
- Hospitals can opt-out at any time.

---

## 10. Deployment Plan

1. **Setup N distinct hospital instances** (can be Docker containers, VMs, or folders).
2. **Each hospital:** Set up local DB, DOL backend, AI agent services.
3. **Aggregator service:** Set up simple model combiner (FedAvg), REST endpoint for model aggregation.
4. **Initial model training:** Each hospital trains initial model to convergence on local data.
5. **Federated model training loop:** Scripted or REST-driven model update, aggregation, and distribution (repeat for several cycles).
6. **Demo Day:** Show both federated profile assembly and federated model improvement.

---

## 11. Success Criteria

- Demonstrable federated profile lookup across hospitals—privacy preserved.
- Demonstrable federated learning rounds resulting in improved (or at least non-degrading) model performance for all hospitals.
- No patient data leaves hospital DBs at any point.
- All logs show only model parameters and event-level metadata.
- Judges see evidence of distributed, privacy-first, collaborative learning system.

---

## 12. Live Demonstration Checklist

- Register a patient at Hospital A, then retrieve portable profile at Hospital B (and C if time permits).
- Trigger a federated learning cycle—show logs: local accuracy, send weights, global update, new accuracy.
- Show privacy filter (no hospital metadata in portable view).
- Highlight that no patient data ever left any hospital.

---

## 13. Operational Commands & Tooling

- Bootstrap a fresh hospital database and seed demo data:
  ```bash
  export DATABASE_URL=postgresql+asyncpg://medi_os:medi_os@localhost:5432/medi_os
  poetry run python scripts/bootstrap_hospital_instance.py
  ```
- Launch the full stack for a local demo:
  ```bash
  docker compose -f docker-compose.demo.yml up --build
  ```
- Submit a synthetic federated update from the command line:
  ```bash
  FEDERATION_SHARED_SECRET=super-secret \
  ./scripts/run_federated_round.sh triage
  ```
- Reset demo data between rehearsals:
  ```bash
  DATABASE_URL=postgresql://medi_os:medi_os@localhost:5432/medi_os \
  ./scripts/reset_demo_data.sh
  ```

---

End of Federated Architecture & Demo Guide.

