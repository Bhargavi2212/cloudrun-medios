# Medi OS Frontend (Version -2)

This Vite + React + TypeScript application powers the Medi OS demo portal. It surfaces patient timelines, triage simulations, scribe workflows, and the federated learning dashboard.

## Getting Started

```bash
cd apps/frontend
pnpm install # or npm install / yarn install
cp env.example .env
pnpm run dev
```

## Scripts

| Command | Description |
| ------- | ----------- |
| `pnpm run dev` | Start the Vite development server. |
| `pnpm run build` | Produce a production build. |
| `pnpm run preview` | Preview the production bundle locally. |
| `pnpm run lint` | Run ESLint against all source files. |
| `pnpm run test` | Execute unit tests with Vitest. |

## Features

- Patient profile explorer combining manage-agent, DOL, and summarizer outputs.
- Triage simulator hitting the manage-agent classification endpoint.
- Scribe workflow to capture transcripts, generate SOAP notes, and trigger summary generation.
- Federated learning dashboard to submit synthetic model updates and view the latest global model.

Ensure backend services are running locally (or update the URLs in `.env`) before interacting with the frontend.

