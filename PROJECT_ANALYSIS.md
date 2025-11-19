# Medi OS Version 2 - Complete Project Analysis

## Overview

This project is **Medi OS Version 2**, a comprehensive healthcare management platform with **federated learning** and **multi-hospital support**. The system consists of three main folders representing different deployment configurations:

1. **Version -2** - Main development version with multi-hospital frontend
2. **Version-2-HospitalA** - Hospital A instance (ports 8001-8003)
3. **Version-2-HospitalB** - Hospital B instance (ports 8011-8013 or 9001-9003)

---

## 🏗️ What is Being Built

### Core System: **Medi OS - Healthcare Operating System**

Medi OS is a multi-agent AI system designed to be the **operating system for modern hospitals**. It combines:

1. **AI-Powered Triage System** - Predicts patient acuity and prioritizes care
2. **AI Clinical Scribe** - Automatically generates SOAP notes from doctor-patient conversations
3. **AI Medical Summarizer** - Condenses patient medical history into concise summaries
4. **Queue Management System** - Real-time patient queue with updates
5. **Role-Based Dashboards** - Tailored interfaces for each healthcare role
6. **Patient Management** - Complete patient lifecycle from check-in to discharge
7. **Federated Learning** - Collaborative AI model training across hospitals without sharing raw data
8. **Federated Data Orchestration** - Cross-hospital patient profile retrieval with privacy preservation

---

## 📁 Folder Structure Analysis

### 1. **Version -2** (Main Development Version)

**Purpose**: Main development codebase with multi-hospital frontend support

**Key Features**:
- **Multi-hospital frontend** with hospital selector
- **Cross-hospital profile retrieval** via DOL (Data Orchestration Layer)
- **Federated learning** infrastructure
- **Complete microservices architecture**

**Services**:
- **Manage-Agent** (Port 8001): Patient triage, check-in, queue management
- **Scribe-Agent** (Port 8002): SOAP note generation from transcripts
- **Summarizer-Agent** (Port 8003): Patient history summarization
- **DOL Service** (Port 8004): Data Orchestration Layer for federated data queries
- **Federation Aggregator** (Port 8010): Federated learning model aggregation

**Frontend**:
- React 18 + TypeScript + Vite + Material UI
- Hospital selector component
- Dynamic API URL switching based on selected hospital
- Role-based dashboards (Receptionist, Nurse, Doctor, Admin)

**Database**:
- PostgreSQL with SQLAlchemy ORM
- Core tables: `patients`, `encounters`, `triage_observations`, `dialogue_transcripts`, `soap_notes`, `summaries`, `file_assets`, `timeline_events`
- Alembic migrations for schema management

**Key Files**:
- `apps/frontend/src/shared/config/hospitalConfig.ts` - Hospital configuration
- `services/manage_agent/services/profile_merge_service.py` - Cross-hospital profile merging
- `services/manage_agent/services/check_in_service.py` - DOL integration
- `dol_service/` - Data Orchestration Layer
- `federation/` - Federated learning client

---

### 2. **Version-2-HospitalA** (Hospital A Instance)

**Purpose**: Standalone instance configured for Hospital A

**Configuration**:
- **Ports**: 8001 (manage), 8002 (scribe), 8003 (summarizer), 8004 (DOL)
- **Database**: `medi_os_v2_a`
- **Storage**: `./storage/hospital_a`
- **Hospital ID**: `hospital-a`

**Key Differences from Version -2**:
- **No multi-hospital frontend** - Single hospital configuration
- **Simplified frontend** - No hospital selector
- **Standalone deployment** - Can run independently
- **Same core services** - All three agents + DOL + Federation

**Use Case**: 
- Production deployment for Hospital A
- Testing Hospital A in isolation
- Federated learning participant

**Key Files**:
- `services/manage_agent/config.py` - Hospital A specific configuration
- `START_ALL.ps1` - PowerShell script to start all services
- `FINAL_STATUS.md` - Deployment status

---

### 3. **Version-2-HospitalB** (Hospital B Instance)

**Purpose**: Standalone instance configured for Hospital B

**Configuration**:
- **Ports**: 9001 (manage), 9002 (scribe), 9003 (summarizer), 8004 (DOL - shared)
- **Database**: `medi_os_v2_b`
- **Storage**: `./storage/hospital_b`
- **Hospital ID**: `hospital-b`

**Key Differences from Version -2**:
- **No multi-hospital frontend** - Single hospital configuration
- **Simplified frontend** - No hospital selector
- **Standalone deployment** - Can run independently
- **Different ports** - Avoids conflicts with Hospital A
- **Same core services** - All three agents + DOL + Federation

**Use Case**:
- Production deployment for Hospital B
- Testing Hospital B in isolation
- Federated learning participant
- Demonstrating cross-hospital federated profile retrieval

**Key Files**:
- `services/manage_agent/config.py` - Hospital B specific configuration
- `START_ALL.ps1` - PowerShell script to start all services
- `FINAL_STATUS.md` - Deployment status

---

## 🔄 System Architecture

### Microservices Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + MUI)                   │
│  - Hospital Selector (Version -2 only)                      │
│  - Role-based Dashboards                                     │
│  - Patient Profile, Queue, Documents                        │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Manage-Agent │   │ Scribe-Agent │   │Summarizer-   │
│   (Port      │   │   (Port      │   │  Agent       │
│   8001/9001) │   │   8002/9002) │   │  (Port       │
│              │   │              │   │  8003/9003)   │
│ - Triage     │   │ - SOAP Notes │   │ - Summaries  │
│ - Check-in   │   │ - Transcripts│   │ - Timeline   │
│ - Queue      │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  DOL Service │   │  Federation  │   │  PostgreSQL   │
│   (Port      │   │  Aggregator  │   │   Database    │
│   8004)      │   │   (Port      │   │               │
│              │   │   8010)      │   │               │
│ - Federated  │   │ - Model      │   │ - Patients     │
│   Data Query │   │   Aggregation│   │ - Encounters  │
│ - Privacy    │   │ - Federated  │   │ - Documents   │
│   Filtering  │   │   Learning   │   │               │
└──────────────┘   └──────────────┘   └──────────────┘
```

### Federated Learning Flow

1. **Local Training**: Each hospital trains models on local data
2. **Model Update**: Model weights (not raw data) sent to aggregator
3. **Aggregation**: Aggregator combines model updates (Federated Averaging)
4. **Global Model**: Aggregated model distributed back to hospitals
5. **Repeat**: Continuous improvement cycle

### Federated Data Flow (Cross-Hospital Profile Retrieval)

1. **Patient Check-in** at Hospital B with patient ID
2. **DOL Query**: Hospital B's DOL queries Hospital A's DOL
3. **Privacy Filter**: Hospital A's DOL removes hospital/location metadata
4. **Profile Merge**: Hospital B merges local + federated profile
5. **Unified Timeline**: Patient sees complete history from all hospitals

---

## 🤖 Core AI Agents

### 1. **Manage-Agent** (Triage & Patient Management)

**Responsibilities**:
- Patient check-in and registration
- Triage classification (acuity level prediction)
- Queue management
- Cross-hospital profile retrieval via DOL
- Profile merging (local + federated)
- Document upload and management

**AI Models**:
- **Receptionist Triage Model**: XGBoost/LightGBM for initial triage
- **Nurse Triage Model**: Enhanced model with vitals
- **Federated Learning**: Collaborative model improvement

**Key Endpoints**:
- `POST /manage/check-in` - Patient check-in with DOL query
- `POST /manage/triage/classify` - Triage classification
- `GET /manage/queue` - Patient queue
- `POST /manage/documents/upload` - Document upload
- `GET /manage/patients/{patient_id}/profile` - Patient profile (merged)

### 2. **Scribe-Agent** (Clinical Documentation)

**Responsibilities**:
- Dialogue transcript ingestion
- SOAP note generation using Google Gemini
- Clinical note management

**AI Models**:
- **Google Gemini 1.5 Pro**: SOAP note generation from transcripts

**Key Endpoints**:
- `POST /scribe/transcript` - Upload dialogue transcript
- `POST /scribe/generate-soap` - Generate SOAP note
- `GET /scribe/notes/{encounter_id}` - Retrieve SOAP notes

### 3. **Summarizer-Agent** (Medical Summarization)

**Responsibilities**:
- Patient history summarization
- Document processing (PDF/image extraction)
- Timeline event creation
- Multi-step Gemini pipeline

**AI Models**:
- **Google Gemini 1.5 Pro**: Multi-step document processing and summarization

**Key Endpoints**:
- `POST /summarizer/generate-summary` - Generate patient summary
- `POST /summarizer/documents/{file_id}/process` - Process uploaded document
- `GET /summarizer/history/{patient_id}` - Summary history

---

## 🗄️ Database Schema

### Core Tables

1. **patients**: Patient demographics and information
2. **encounters**: Hospital visits/encounters
3. **triage_observations**: Triage data with vitals and scores
4. **dialogue_transcripts**: Doctor-patient conversation transcripts
5. **soap_notes**: AI-generated SOAP notes
6. **summaries**: Patient history summaries
7. **file_assets**: Uploaded documents (PDFs, images)
8. **timeline_events**: Chronological patient timeline
9. **audit_logs**: System audit trail

### Federated Learning Tables

- **model_rounds**: Federated learning training rounds
- **model_updates**: Model weight updates from hospitals
- **global_models**: Aggregated global models

---

## 🎨 Frontend Features

### Role-Based Dashboards

1. **Receptionist Dashboard**:
   - Patient check-in
   - Queue management
   - Cross-hospital profile retrieval feedback

2. **Nurse Dashboard**:
   - Vitals capture
   - Triage classification
   - Document upload

3. **Doctor Dashboard**:
   - Patient consultation workspace
   - SOAP note review and approval
   - Patient profile with timeline

4. **Admin Dashboard**:
   - User management
   - Analytics
   - System configuration

### Multi-Hospital Support (Version -2 only)

- **Hospital Selector**: Dropdown to switch between hospitals
- **Dynamic API URLs**: API calls automatically switch based on selection
- **Hospital Context**: React context for hospital state management
- **Source Indicators**: Timeline shows "Local" vs "Federated" events

---

## 🔐 Security & Privacy

### Privacy Preservation

- **Privacy Filter**: Removes hospital/location/provider metadata from federated profiles
- **No Raw Data Sharing**: Only model weights shared in federated learning
- **Encrypted Communications**: TLS for all inter-hospital communications
- **Audit Logging**: All federated queries logged (non-PHI)

### Authentication

- JWT-based authentication
- Service-level API keys for inter-service communication
- Shared secrets for DOL and Federation services

---

## 🚀 Deployment Configuration

### Version -2 (Development)

- **Frontend**: Single frontend with hospital selector
- **Backend**: All services on standard ports (8001-8010)
- **Database**: Single database for development
- **Use Case**: Development, testing, demo

### Version-2-HospitalA (Production A)

- **Frontend**: Hospital A specific (no selector)
- **Backend**: Ports 8001-8003
- **Database**: `medi_os_v2_a`
- **Storage**: `./storage/hospital_a`
- **Use Case**: Production deployment for Hospital A

### Version-2-HospitalB (Production B)

- **Frontend**: Hospital B specific (no selector)
- **Backend**: Ports 9001-9003
- **Database**: `medi_os_v2_b`
- **Storage**: `./storage/hospital_b`
- **Use Case**: Production deployment for Hospital B

---

## 📊 Key Technologies

### Backend
- **Python 3.11+** with FastAPI
- **PostgreSQL 15** with SQLAlchemy ORM
- **Alembic** for database migrations
- **Google Gemini 1.5 Pro** for AI features
- **XGBoost/LightGBM** for triage models
- **Async/Await** for performance

### Frontend
- **React 18** with TypeScript
- **Vite** for build tooling
- **Material UI (MUI)** for components
- **React Query** for data fetching
- **Zustand** for state management
- **Axios** for API calls

### Infrastructure
- **Docker** for containerization
- **Docker Compose** for local orchestration
- **Google Cloud Run** target for deployment
- **Poetry** for Python dependency management
- **npm** for frontend dependencies

---

## 🎯 Use Cases

### 1. Single Hospital Operation
- Use **Version-2-HospitalA** or **Version-2-HospitalB**
- Standalone deployment
- Local database and storage
- No cross-hospital features needed

### 2. Multi-Hospital Development/Demo
- Use **Version -2**
- Multi-hospital frontend
- Test cross-hospital profile retrieval
- Demonstrate federated learning

### 3. Federated Learning Demonstration
- Run both Hospital A and Hospital B
- Show model improvement across hospitals
- Demonstrate privacy-preserving collaboration

### 4. Cross-Hospital Patient Profile
- Patient visits Hospital B
- System retrieves history from Hospital A
- Unified timeline with source indicators
- Privacy-preserved data sharing

---

## 📝 Summary

**Version -2** is the main development version with:
- Multi-hospital frontend support
- Cross-hospital profile retrieval
- Complete federated learning infrastructure
- All microservices

**Version-2-HospitalA** and **Version-2-HospitalB** are:
- Production-ready standalone instances
- Configured for specific hospitals
- Can participate in federated learning
- Simplified frontends (no hospital selector)

All three versions share the same core architecture and services, but differ in:
- Frontend configuration (multi-hospital vs single)
- Port assignments
- Database names
- Storage paths
- Hospital IDs

The system enables **privacy-preserving collaboration** between hospitals through:
1. **Federated Learning**: Collaborative AI model improvement
2. **Federated Data**: Cross-hospital patient profile retrieval with privacy filtering

---

## 🔗 Key Documentation Files

- `README.md` - Project overview and setup
- `FEDERATED_BUILD_GUIDE.md` - Federated architecture guide
- `WHAT_WAS_BUILT.md` - Multi-hospital features documentation
- `FINAL_STATUS.md` - System completion status
- `QUICK_START.md` - Quick start guide
- `DEPLOYMENT_GUIDE.md` - Deployment instructions

---

**Status**: Production-ready infrastructure with complete feature set for multi-hospital federated healthcare system.

