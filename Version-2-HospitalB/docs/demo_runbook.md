# Demo Runbook – Medi OS Version -2

This runbook outlines the sequence of steps, roles, and validation checks for the hackathon demonstration.

## Roles

- **Operator A (Hospital A):** Runs manage-agent and demonstrates triage + scribe workflows.
- **Operator B (Hospital B):** Retrieves portable profiles via the DOL.
- **Operator C (Federation Lead):** Shows aggregator dashboard and federated update submission.
- **Narrator:** Guides judges through the flow and highlights privacy guarantees.

## Pre-Demo Checklist

- [ ] `docker compose -f docker-compose.demo.yml up --build` completed without errors.
- [ ] `scripts/bootstrap_hospital_instance.py` executed to seed baseline data.
- [ ] Frontend accessible at `http://localhost:5173`.
- [ ] Health endpoints return 200 for manage, scribe, summarizer, DOL, federation aggregator.
- [ ] Authorization headers verified for DOL and federation endpoints (`super-secret` shared secret).

## Seeded Demo Dataset

- **Patient – Alex Kim:** Bronchospasm episode treated locally, followed by telehealth review from Northwind Regional. Audit trail includes receptionist, triage nurse, and federation gateway actors.
- **Patient – Morgan Patel:** Escalated pneumonia case transferred from Lakeside urgent care to Kiroween ED, illustrating cross-hospital portability and sepsis pathway activation.
- **Patient – Chris Rivera:** Festival-related laceration managed at Kiroween South with remote quality review by Northwind Regional.
- **Staff Roles:** Receptionists handle automated check-in, triage nurses populate vitals, physicians and scribes author notes, federation proxy shares profiles. All actions land in `audit_logs` for traceability.
- **Usage:** Run `poetry run python scripts/seed_database.py` after setting `DATABASE_URL` to populate the data set above for demos and smoke tests.

## Demo Flow

1. **Introduction (Narrator)**
   - Briefly recap the Medi OS blueprint and federated goals.
   - Show high-level architecture slide or reference `FEDERATED_BUILD_GUIDE.md`.

2. **Hospital A: Patient Intake and Triage (Operator A)**
   - In the frontend, create a new patient via the “Add Patient” button.
   - Run triage classification with vitals to demonstrate immediate acuity scoring.
   - Confirm triage event visible in local patient timeline.
   - Select an existing patient by universal ID to show the automatic check-in fetching the federated profile (no manual request needed).

3. **Hospital A: Scribe Workflow**
   - Capture a short transcript and generate a SOAP note.
   - Trigger summary generation; point out success banner and resulting timeline updates.

4. **Hospital B: Portable Profile Retrieval (Operator B)**
   - Switch to another browser profile or incognito session to simulate Hospital B.
   - Retrieve the same patient’s portable profile via DOL.
   - Highlight privacy filters (no hospital identifiers) and aggregated timeline entries.

5. **Federated Learning Update (Operator C)**
   - Open the federated dashboard tab.
   - Submit a synthetic model update using the UI.
   - Display contributor count increment and averaged weights.
   - Optionally run `scripts/run_federated_round.sh triage` in a terminal to show CLI path.

6. **Recap & Q&A (Narrator)**
   - Reiterate success criteria: data locality, federated learning benefits, audit logging.
   - Be ready to show audit log entries in the database if judges ask.

## Post-Demo Cleanup

- Run `scripts/reset_demo_data.sh` if preparing for another rehearsal.
- Bring down the stack: `docker compose -f docker-compose.demo.yml down`.

## Troubleshooting Tips

- If DOL requests fail, confirm `Authorization` header matches `DOL_SHARED_SECRET`.
- For frontend CORS issues, verify service environments expose `http://localhost:5173`.
- Aggregator 404 on global model indicates no updates yet—submit at least one round.

