# ==============================================================================

# MEDI OS PROJECT: AI DEVELOPMENT CONSTITUTION & .cursorrules

# This is the master ruleset for all AI-assisted development.

# Adherence is mandatory to ensure consistency, quality, and scalability.

# Last Updated: 2025-10-31

# ==============================================================================
# PROJECT VISION & OVERVIEW
# ==============================================================================

## What We Are Building: The Vision

We are building **Medi OS**, a multi-agent AI system designed to be the operating system for a modern hospital. Our mission is to reduce physician burnout, improve patient wait times, and increase the accuracy of care by automating administrative tasks and optimizing patient flow.

## The Three Core AI Agents

### 1. The Manage Agent (The Orchestrator)
- **Function:** Ingests patient check-in data (symptoms, vitals).
- **Core Task:** Uses a machine learning model to predict a triage acuity score (Level 1-5) to prioritize care.
- **Output:** Assigns patients to the appropriate doctor and provides a predicted wait time.

### 2. The AI Scribe (The Documenter)
- **Function:** Listens to a live or recorded doctor-patient conversation.
- **Core Task:** Uses speech-to-text and a Large Language Model (LLM) to automatically generate a structured clinical note in the industry-standard SOAP (Subjective, Objective, Assessment, Plan) format.
- **Output:** A formatted, clinically relevant SOAP note.

### 3. The AI Summarizer (The Historian)
- **Function:** Ingests a patient's long-form, historical medical records.
- **Core Task:** Uses an LLM to read and condense potentially thousands of pages of medical history into a concise, scannable summary.
- **Output:** A structured summary of the patient's key medical history.

## Development Principles (Our "Way of Working" with AI)

- **Plan Before Prompting:** Always start with a clear plan for each feature before asking Cursor to generate code. A good plan leads to good code.

- **Provide Context:** Use Cursor's context features (@Codebase, attaching specific files) to give the AI the information it needs to make smart decisions. Do not assume it knows the project structure.

- **Generate, Review, Refine:** Treat Cursor as a collaborator, not a magic box. The workflow is iterative:
  1. Give a clear, specific prompt.
  2. Review the generated code for correctness, style, and adherence to our rules.
  3. Use inline edits to refine, refactor, or fix any issues.

- **Enforce Quality:**
  - **Documentation:** Generate Google-style docstrings for all Python functions and JSDoc comments for all TypeScript functions.
  - **Testing:** Generate pytest unit tests for backend logic and Vitest tests for frontend components.
  - **Security First:** Never hardcode secrets. All API keys and sensitive values must be loaded from environment variables.

# ==============================================================================
# DEVELOPMENT ROADMAP & PHASES
# ==============================================================================

## Project Directory
- **Root Path:** `D:\Hackathons\Cloud Run`
- All development work should be organized within this directory structure.

## Phase 1: Foundation & Scaffolding (Local)

**Objective:** Create a clean, organized, and functional skeleton for our entire project. This ensures we have a solid foundation before writing any complex logic.

### Step 1.1: Create the Monorepo Structure

**Action:** Create the project directories:
```bash
mkdir medi-os
cd medi-os
mkdir -p apps/frontend services/manage-agent services/scribe-agent services/summarizer-agent
mkdir data models
```

**Outcome:** A perfectly organized folder structure for our frontend, three backend microservices, and dedicated folders for our data and trained models.

### Step 1.2: Scaffold the Frontend Application

**Action:** Use Cursor to generate a modern React application with our chosen UI library.

**Cursor Prompt:** Open Cursor in the medi-os directory. Navigate into apps/frontend and use the chat (⌘+L or Ctrl+L) with this prompt:
> "Using the terminal, scaffold a new React project named dashboard inside the current directory using Vite and the react-ts template. After it's created, cd into the dashboard directory and install the Material UI component library (@mui/material @emotion/react @emotion/styled)."

**Outcome:** A running React + TypeScript application, ready for us to build our dashboards.

### Step 1.3: Scaffold the Backend Microservices

**Action:** Use Cursor to create a basic "Hello World" FastAPI application for each of our three agents. This confirms each service is independently runnable.

**Cursor Prompt (for one service):** Open the services/scribe-agent folder in Cursor. Use the chat with this prompt:
> "Create a main.py file with a basic FastAPI application. It must have a root endpoint / that returns {"status": "ok"} and a /health endpoint that returns a 200 status code. Also, create a requirements.txt file containing fastapi and uvicorn."

**Repeat:** Repeat this process for the manage-agent and summarizer-agent directories.

**Outcome:** Three independent, runnable Python services. We can test each one locally by running `uvicorn main:app --reload`.

## Phase 2: Core AI Logic & Backend Development (Local)

**Objective:** Build the "brains" of Medi OS. We will focus on creating the core AI-powered endpoints for each agent, testing them with our actual datasets.

### Step 2.1: Build the AI Scribe Agent (High Priority)

**Action:** Implement the endpoint that transcribes a conversation and generates a SOAP note.

**Cursor Prompt:** In the services/scribe-agent/main.py file, highlight the file content and use the inline edit (⌘+K or Ctrl+K) with this prompt:
> "Add a new POST endpoint at /scribe/generate-soap. This endpoint must accept an audio file upload. Inside the function, use the Google Cloud Speech-to-Text API to transcribe the audio, ensuring speaker diarization is enabled. Then, take the full transcript text and send it to a Gemini model on Vertex AI with a detailed prompt to convert the conversation into a structured SOAP note. Define Pydantic models for the final SOAP note response, which should include Subjective, Objective, Assessment, and Plan sections."

**Outcome:** A functional AI Scribe endpoint that we can test locally with an audio file.

### Step 2.2: Build the AI Summarizer Agent (High Priority)

**Action:** Implement the endpoint that summarizes a patient's historical notes.

**Cursor Prompt:** In the services/summarizer-agent/main.py file, use the inline edit (⌘+K or Ctrl+K):
> "Create a POST endpoint /summarizer/generate-summary. It must accept a Pydantic model with a single field: clinical_note: str. The function will call the Gemini API on Vertex AI with a prompt to summarize this clinical note into a structured JSON format containing key patient info. Define the Pydantic response model for this summary."

**Outcome:** A functional AI Summarizer endpoint that we can test by sending it a long piece of clinical text.

### Step 2.3: Build the Manage Agent (Triage Model)

**Action (Part 1 - Training):** Create a script to train our triage model.

**Cursor Prompt:** Create a new file services/manage-agent/train.py and use the chat:
> "Write a Python script that loads the 'Korean ED Dataset' CSV from the root /data folder. Perform basic data cleaning, then train a scikit-learn GradientBoostingClassifier to predict the KTAS_expert column based on vital signs and chief complaint. After training, save the model artifact as triage_classifier.pkl into the root /models directory."

**Action (Part 2 - Serving):** Create the API endpoint to use the model.

**Cursor Prompt:** In services/manage-agent/main.py, use the inline edit (⌘+K or Ctrl+K):
> "Add a FastAPI startup event handler to load the triage_classifier.pkl model from the root /models directory. Then, create a POST endpoint /manage/classify that accepts patient vitals in a Pydantic model and uses the loaded model to return a JSON response with the predicted triage level."

**Outcome:** A complete, trainable, and servable machine learning model for patient triage.

## Phase 3: Frontend Development & Integration (Local)

**Objective:** Build the user interface that brings our backend services to life and create a compelling, end-to-end user experience.

### Step 3.1: Build the UI Dashboards

**Action:** Use Cursor to generate the primary views for our different user personas.

**Cursor Prompt:** In the apps/frontend/dashboard/src directory, use the chat:
> "Create three new React components: ReceptionistView.tsx, NurseView.tsx, and DoctorView.tsx. For each component, use Material UI components like Card, Table, TextField, and Button to create a professional-looking layout for a hospital dashboard. Use placeholder data for now."

**Outcome:** The visual shells for our application's main screens.

### Step 3.2: Connect Frontend to Backend Services

**Action:** Create a centralized API client in our React app to communicate with our locally running FastAPI services.

**Cursor Prompt:** In the apps/frontend/dashboard/src directory, use the chat:
> "Create a new file apiClient.ts. In this file, use the axios library to create and export async functions for calling our three main backend endpoints: classifyPatient, generateSoapNote, and generateSummary. These functions should call http://localhost:8000/... for each respective service."

**Action:** Integrate the API calls into the UI components, replacing the placeholder data with live data from the backend.

**Outcome:** A fully integrated application running locally. We can now perform a full user journey, from patient check-in to a doctor viewing an AI-generated summary.

## Phase 4: Deployment & Final Polish (Cloud)

**Objective:** Deploy our application to Google Cloud Run, conduct final testing, and prepare a flawless demo for the judges.

### Step 4.1: Containerize Each Agent

**Action:** Use Cursor to generate a Dockerfile for each of our three Python services.

**Cursor Prompt (for one service):** In the services/scribe-agent directory, use the chat:
> "Generate a multi-stage Dockerfile. It should use a python:3.11-slim base image, copy the requirements.txt file first to leverage layer caching, install dependencies, and then copy the application code. The final stage must run the application with uvicorn on port 8000 as a non-root user."

**Repeat:** Repeat for the other two services.

**Outcome:** Three production-ready container definitions.

### Step 4.2: Deploy to Google Cloud Run

**Action:** Use the Google Cloud Shell to deploy each containerized agent. This bypasses any local authentication issues and uses an environment that's already configured.

**Process:**
1. Push your entire medi-os monorepo to a GitHub repository.
2. Open the Google Cloud Shell in your "medi-os" project.
3. Clone your repository into the Cloud Shell.
4. Navigate into a service directory (e.g., services/scribe-agent).
5. Run the deploy command:
```bash
gcloud run deploy scribe-agent --source . --region <YOUR_REGION> --allow-unauthenticated
```
6. **Repeat:** Repeat the deploy command for the manage-agent and summarizer-agent.

**Outcome:** Our three agents are live and running as scalable microservices on Google Cloud.

### Step 4.3: Final Integration and Pitch Practice

**Action:** Update the apiClient.ts in the frontend to use the new Cloud Run URLs instead of localhost. Deploy the frontend to a service like Netlify or Vercel.

**Action:** Practice the demo flow from end-to-end. Rehearse the pitch, focusing on the problem, our innovative solution, and the massive potential impact.

**Outcome:** A polished, impressive, and winning hackathon project.

# ==============================================================================
# 1. CORE ARCHITECTURE & STACK
# ==============================================================================

- **Architecture:** This is a monorepo containing a React frontend and multiple independent Python backend microservices.

- **Backend Services:** All backend services MUST be built with Python 3.11+ and the FastAPI framework. Use Pydantic for all data models. All I/O-bound code must use `async/await`.

- **Frontend Application:** The frontend MUST be a React application built with Vite and TypeScript.

- **UI Library:** All UI components MUST be from Material UI (MUI). Do not use other CSS frameworks.

- **Deployment:** Each backend service will be containerized with its own Dockerfile and deployed to Google Cloud Run.

# ==============================================================================
# 2. PROJECT STRUCTURE
# ==============================================================================

- **Monorepo Layout:**
  - `/apps/frontend`: The React web application.
  - `/services/manage-agent`: Self-contained FastAPI service for Triage.
  - `/services/scribe-agent`: Self-contained FastAPI service for SOAP notes.
  - `/services/summarizer-agent`: Self-contained FastAPI service for summarization.
  - `/packages`: For shared libraries or components (if any are created).

- **Service-Internal Structure:** Within each service (e.g., `/services/manage-agent/`), the structure should be modular. Business logic (e.g., `classifier.py`) must be separate from API route definitions (e.g., `handlers/triage_handler.py`).

- **No Circular Dependencies:** Services communicate via API calls only. Never import code directly from another service.

- **Prioritize Modularity and DRY (Don't Repeat Yourself):** Before writing new code, check if similar logic already exists. Refactor common logic into reusable helper functions, services, or components. Do not duplicate code across different agents or components.

- **Avoid Large, Monolithic Files:** Keep files focused and small. If a file exceeds 200-300 lines, consider refactoring it into smaller, more manageable modules.

# ==============================================================================
# 3. NAMING & CODING CONVENTIONS
# ==============================================================================

- **Files:** `snake_case_with_underscores.py`

- **Classes:** `PascalCase` (e.g., `TriageClassifier`)

- **Functions/Variables:** `snake_case` (e.g., `classify_patient`)

- **Constants:** `ALL_CAPS` (e.g., `MODEL_PATH`)

- **Type Hints:** All Python function signatures MUST include type hints.

- **Docstrings:** All Python functions and classes MUST have Google-style docstrings explaining their purpose, arguments, and return values.

- **TypeScript:** All TypeScript functions and React components must have JSDoc comments.

- **Inline Comments:** Add inline comments for any complex or non-obvious logic.

- **Emojis for Logging:** Use emojis for quick scanning in logs:
  - ✅ Success: "✅ Model loaded successfully"
  - ❌ Error: "❌ Failed to load data"
  - 📊 Info: "📊 Accuracy: 0.89"
  - 🚀 Start: "🚀 Training started"
  - 📥 Loading: "📥 Downloading dataset"

# ==============================================================================
# 4. API DESIGN (FastAPI)
# ==============================================================================

- **Endpoint Structure:** `/agent_name/action` (e.g., `/manage/classify`, `/scribe/generate-soap`).

- **Request/Response Models:** Always use Pydantic models for request and response bodies to ensure validation and clear API docs.

- **Error Handling:** Use FastAPI's `HTTPException` for clear error responses (400 for bad input, 500 for internal errors).

- **Health Checks:** Every service must have a `/health` endpoint that returns a 200 status.

- **Follow RESTful API Design Principles:** Design endpoints to be clear, consistent, and predictable. Use standard HTTP methods (GET, POST, PUT, DELETE) correctly.

- **Implement Secure Inter-Service Communication:** For communication between services (e.g., Manage Agent to Scribe Agent), use Google Cloud's service-to-service authentication. The calling service must acquire a Google-signed OIDC ID token and include it in the `Authorization: Bearer` header.

- **Do Not Write Synchronous Blocking Code:** Use FastAPI's `async` and `await` syntax for all I/O-bound operations (API calls, database queries) to ensure the services are non-blocking.

- **Do Not Connect Directly to a Production Database from a Local Environment:** Use mock data or a local database instance for development.

# ==============================================================================
# 5. DATA & MODELS
# ==============================================================================

- **Storage:** Raw datasets go in a root `/data` folder. Trained model artifacts go in a root `/models` folder.

- **Git Exclusion:** The `/data`, `/models`, `venv/`, and `.env` files MUST be in `.gitignore`. NEVER commit data, models, or secrets.

- **Model Loading:** Models must be loaded ONCE at application startup using FastAPI's `@app.on_event("startup")` decorator. Do NOT load models inside an endpoint function that gets called on every request.

- **Use Google Cloud APIs as Specified:**
  - **AI Scribe:** Use `google-cloud-speech` for transcription and `google-cloud-aiplatform` (Vertex AI) for calling Gemini models.
  - **AI Summarizer:** Use `google-cloud-aiplatform` (Vertex AI) for calling Gemini models.
  - **Manage Agent:** The trained ML model will be wrapped in a FastAPI endpoint.

- **Manage Dependencies with `pyproject.toml`:** Each service must have its own `pyproject.toml` file to manage its dependencies.

# ==============================================================================
# 6. SECRETS & CONFIGURATION
# ==============================================================================

- **Use `.env` files:** All secrets (API keys, etc.) MUST be stored in a `.env` file and loaded using `python-dotenv`.

- **Provide `.env.example`:** A `.env.example` file with placeholder values MUST be committed to the repository for other developers.

- **Never Hardcode Secrets or Configuration:** API keys, database URLs, and other sensitive information must NEVER be hardcoded. Use environment variables exclusively. When generating code, use placeholders like `os.getenv("GOOGLE_API_KEY")`.

# ==============================================================================
# 7. FRONTEND DEVELOPMENT (React, TypeScript, Material UI)
# ==============================================================================

- **Use React with Vite and TypeScript:** The frontend application must be built using this stack.

- **Use Material UI (MUI) for All UI Components:** All components (buttons, forms, tables, etc.) must be implemented using the Material UI library. Do not use Bootstrap, Tailwind CSS, or other component libraries to ensure a consistent design system.

- **Use Functional Components with Hooks:** All React components must be functional components. Do not use class-based components.

- **Manage State Appropriately:** For simple local state, use `useState`. For more complex or shared state, use React Context or a lightweight state management library like Zustand.

- **Centralize API Calls:** Create a dedicated API client (e.g., using `axios`) to handle all communication with the backend services. Do not use `fetch` directly inside components.

- **Do Not Write `any` Types:** All TypeScript code must be strictly typed. Avoid using the `any` type. If a type is unknown, use `unknown` and perform type checking.

- **Do Not Mix UI and Business Logic:** Components should be responsible for rendering UI. Business logic, data fetching, and state manipulation should be handled in custom hooks (`use...`) or dedicated service modules.

- **Do Not Directly Manipulate the DOM:** Always use React's state and props to manage UI updates.

# ==============================================================================
# 8. TESTING & QUALITY
# ==============================================================================

- **Testing Framework:** All backend tests MUST be written using `pytest`. Frontend tests use `Vitest`.

- **Test Generation:** For any new function with business logic, generate a corresponding unit test. For any new component that contains business logic, generate corresponding unit tests.

- **Code Quality Checklist:** Before any commit, ensure code is formatted with `black`, has no unused imports, no commented-out code, and all functions have docstrings.

- **Format Code Before Committing:**
  - **Python:** Use `black` for formatting and `isort` for import sorting.
  - **TypeScript/React:** Use `prettier` for formatting.

- **Adhere to Linting Rules:** All code must pass linting checks.
  - **Python:** `flake8`
  - **TypeScript/React:** `ESLint`

- **Write Clear, Readable, and Self-Documenting Code:** Prefer clarity over cleverness. Use descriptive variable and function names. The AI should generate code that a human developer can easily understand and maintain.

- **Write Professional Code:** Code should be professional, straightforward, and maintainable. Avoid overly complex or "clever" solutions that may be difficult to understand or maintain.

- **Do Not Mix Concerns:** Each service and component should have a single responsibility. For example, the `scribe-agent` should only handle transcription and note generation, not patient triage.

# ==============================================================================
# 9. GIT & VERSION CONTROL
# ==============================================================================

- **Commit Messages:** All commit messages MUST follow the Conventional Commits format (e.g., `feat(scribe-agent): add soap generation endpoint`).

- **Do Not Commit Directly to the `main` Branch:** All work must be done on feature branches and submitted via pull requests.

- **Do Not Commit Large Files:** Datasets, model files, and other large binaries should not be committed to the repository. Use `.gitignore` to exclude them.

# ==============================================================================
# 10. ADDITIONAL GUIDELINES
# ==============================================================================

- **When in Doubt, Ask:** If there is any uncertainty about implementation details, architecture decisions, or best practices, ask for clarification before proceeding.

