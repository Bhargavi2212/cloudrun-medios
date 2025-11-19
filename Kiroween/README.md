# Medi OS Kiroween Edition v2.0

## Patient-Controlled, Privacy-First Medical Records with Federated Learning

**Revolutionary healthcare system that empowers patients to carry their complete clinical history as portable "digital medical passports" while enabling AI improvement across hospitals without sharing patient data.**

---

## 🌟 Vision Statement

Medi OS Kiroween Edition v2.0 transforms healthcare by putting patients in complete control of their medical data through:

- **🎒 Portable Medical Passports**: Patients carry complete clinical history with zero hospital metadata
- **🔒 Privacy-First Architecture**: Absolute guarantee of no hospital/provider identification in profiles  
- **🌐 Offline-First Design**: Full functionality without network dependencies or external systems
- **🤖 Federated AI Learning**: Global AI improvement while maintaining absolute patient privacy
- **🏥 Universal Compatibility**: Works at any hospital worldwide with identical interface

---

## 🏗️ Architecture Overview

### Patient-Controlled Data Flow

```
Patient Journey:
Hospital A → Digital Passport → Hospital B → Updated Passport → Hospital C
    ↓              ↓                ↓              ↓              ↓
Local DB      QR/Mobile/USB    Local DB      QR/Mobile/USB    Local DB
    ↓              ↓                ↓              ↓              ↓
AI Models ←→ Federated Learning ←→ AI Models ←→ Federated Learning ←→ AI Models
```

### Core Principles

1. **Patient Sovereignty**: Patients own and control their complete medical data
2. **Privacy by Design**: Zero hospital/provider metadata in portable profiles
3. **Offline Resilience**: Full functionality without internet connectivity
4. **Federated Learning**: AI improves globally without exposing patient data
5. **Universal Access**: Same interface and workflow at every hospital worldwide

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Node.js 18+ (for frontend)
- PostgreSQL 15+ (or use Docker)

### Local Development Setup

```bash
# Clone and setup
git clone <repository-url>
cd Kiroween

# Install Python dependencies
pip install -e ".[dev,ml,ai]"

# Setup pre-commit hooks for medical safety
pre-commit install

# Start all services with Docker Compose
docker-compose up -d

# Verify all services are running
curl http://localhost:8001/health  # Hospital A - Manage Agent
curl http://localhost:8002/health  # Hospital A - Scribe Agent  
curl http://localhost:8003/health  # Hospital A - Summarizer Agent
curl http://localhost:8004/health  # Hospital A - DOL Service

# Access frontend
open http://localhost:5173
```

### Multi-Hospital Demo Setup

```bash
# Start 3-hospital simulation
docker-compose up postgres-hospital-a postgres-hospital-b postgres-hospital-c
docker-compose up manage-agent-a manage-agent-b manage-agent-c
docker-compose up dol-service-a dol-service-b dol-service-c

# Load synthetic demo data
python scripts/load_demo_data.py

# Run demo scenarios
python scripts/demo_global_medical_tourism.py
python scripts/demo_emergency_care.py
python scripts/demo_federated_learning.py
```

---

## 🏥 Service Architecture

### Hospital Services (Per Hospital)

| Service | Port | Purpose |
|---------|------|---------|
| **Manage Agent** | 8001/8011/8021 | Patient triage and management |
| **Scribe Agent** | 8002/8012/8022 | SOAP note generation |
| **Summarizer Agent** | 8003/8013/8023 | Clinical summarization |
| **DOL Service** | 8004/8014/8024 | Data orchestration & profile management |

### Shared Infrastructure

| Service | Port | Purpose |
|---------|------|---------|
| **Frontend** | 5173 | Universal hospital interface |
| **Federated Aggregator** | 9000 | AI model parameter coordination |
| **PostgreSQL** | 5432-5434 | Per-hospital databases |
| **Redis** | 6379 | Caching and session management |

---

## 📊 Demo Scenarios

### 1. Global Medical Tourism Journey

**Scenario**: Patient travels from US → Europe → Asia with cardiac history

```bash
# Patient starts at Hospital A (US)
curl -X POST http://localhost:8004/api/federated/patient \
  -H "Content-Type: application/json" \
  -d '{"demographics": {"name": "Demo Patient", "dob": "1980-01-01"}}'

# Export portable profile
curl http://localhost:8004/api/federated/patient/MED-{uuid}/export

# Import at Hospital B (Europe) 
curl -X POST http://localhost:8014/api/federated/patient/import \
  -H "Content-Type: application/json" \
  -d @portable_profile.json

# Verify complete timeline preservation
curl http://localhost:8014/api/federated/patient/MED-{uuid}/timeline
```

### 2. Emergency Care with QR Code

**Scenario**: Unconscious patient with QR code medical passport

```bash
# Emergency team scans QR code
python scripts/scan_qr_emergency.py --qr-data "encrypted_profile_data"

# Instant access to critical information
# - Allergies: Penicillin, Latex
# - Medications: Warfarin 5mg daily
# - Conditions: Atrial fibrillation, Diabetes Type 2
# - Emergency contacts: Available

# Provide care and update profile
curl -X POST http://localhost:8004/api/federated/patient/MED-{uuid}/timeline \
  -d '{"event": "emergency_treatment", "details": "..."}'
```

### 3. Federated AI Learning

**Scenario**: 3 hospitals improve AI without sharing patient data

```bash
# Each hospital trains locally
python services/manage-agent/train.py --hospital-id hospital-a
python services/manage-agent/train.py --hospital-id hospital-b  
python services/manage-agent/train.py --hospital-id hospital-c

# Submit model parameters (no patient data)
curl -X POST http://localhost:9000/federated/submit-parameters \
  -d @hospital_a_parameters.json

# Receive improved global model
curl http://localhost:9000/federated/global-model

# Verify privacy: no patient data in federated exchange
python scripts/verify_federated_privacy.py
```

---

## 🔒 Privacy & Security

### Privacy Guarantees

- ✅ **Zero Hospital Metadata**: No hospital names, provider names, or locations in profiles
- ✅ **Patient-Controlled**: Patients own and control all medical data
- ✅ **Cryptographic Integrity**: All profile entries cryptographically signed
- ✅ **Tamper Detection**: Hash chains detect any profile modifications
- ✅ **Selective Sharing**: Patients choose what information to include

### Security Features

- 🔐 **End-to-End Encryption**: All portable profiles encrypted with patient keys
- 🔏 **Digital Signatures**: Each clinical event signed for authenticity
- 🛡️ **Access Auditing**: Complete audit trails without PHI exposure
- 🚫 **No Central Storage**: No centralized patient database or single point of failure
- 🔍 **Privacy Validation**: Automated checks ensure no PHI in code or logs

### Compliance

- **HIPAA**: Full compliance with patient data protection requirements
- **GDPR**: Patient right to data portability and erasure
- **International**: Compliance with global healthcare data protection regulations

---

## 🤖 Federated Learning

### How It Works

1. **Local Training**: Each hospital trains AI models on local patient data only
2. **Parameter Extraction**: Only model weights/gradients extracted (no patient data)
3. **Secure Aggregation**: FedAvg algorithm combines parameters from all hospitals
4. **Global Distribution**: Improved model distributed back to all hospitals
5. **Privacy Verification**: Cryptographic proofs ensure no patient data shared

### AI Models

- **Triage Classification**: Emergency severity index (ESI) prediction
- **Clinical Summarization**: Automated SOAP note generation
- **Diagnostic Support**: Pattern recognition for common conditions
- **Risk Assessment**: Early warning systems for patient deterioration

### Benefits

- 🎯 **Improved Accuracy**: Global learning improves model performance
- 🔒 **Privacy Preserved**: No patient data ever leaves hospital premises  
- 🌍 **Global Knowledge**: Benefits from worldwide medical experience
- ⚡ **Real-Time Updates**: Continuous model improvement without data sharing

---

## 🧪 Testing & Quality Assurance

### Medical Safety Validation

```bash
# Run medical safety checks
python scripts/validate_medical_safety.py

# Check for privacy compliance
python scripts/validate_privacy.py

# Verify no PHI in codebase
python scripts/check_no_phi.py
```

### Test Coverage

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=shared --cov=services --cov-report=html

# Run specific test categories
pytest -m medical_safety
pytest -m privacy
pytest -m federated
```

### Integration Testing

```bash
# Test multi-hospital workflows
pytest tests/integration/test_multi_hospital.py

# Test federated learning
pytest tests/integration/test_federated_learning.py

# Test offline functionality
pytest tests/integration/test_offline_mode.py
```

---

## 📚 Development Guidelines

### Code Quality Standards

- ✅ **Type Hints**: Required on all function signatures
- ✅ **Docstrings**: Google-style docstrings on all functions
- ✅ **Error Handling**: Comprehensive error handling for medical functions
- ✅ **Async/Await**: All I/O operations must be async
- ✅ **Medical Safety**: Automated validation for medical safety requirements

### Pre-Commit Hooks

- 🔍 **Medical Safety**: Validates medical function safety
- 🔒 **Privacy Check**: Ensures no PHI in code
- 🎨 **Code Formatting**: Black, flake8, mypy validation
- 🧪 **Test Validation**: Ensures tests pass before commit

### Development Workflow

1. **Spec-Driven Development**: Follow requirements → design → tasks workflow
2. **Medical Safety First**: All medical functions must have proper error handling
3. **Privacy by Design**: No hospital metadata in any exported data
4. **Production Ready**: Code must work in real deployment environments
5. **Test Integrity**: Tests must validate actual functionality, not fake outputs

---

## 🚀 Deployment

### Local Development

```bash
# Start all services
docker-compose up

# Start specific hospital
docker-compose up postgres-hospital-a manage-agent-a dol-service-a

# View logs
docker-compose logs -f manage-agent-a
```

### Production Deployment

```bash
# Build production images
docker build -t medi-os/manage-agent services/manage-agent/
docker build -t medi-os/dol-service services/dol-service/

# Deploy to Google Cloud Run
gcloud run deploy manage-agent --image medi-os/manage-agent
gcloud run deploy dol-service --image medi-os/dol-service
```

### Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Configure required variables
GOOGLE_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://user:pass@host:port/db
HOSPITAL_ID=hospital-a
```

---

## 📈 Success Metrics

### Technical Innovation
- **Patient Privacy**: Zero hospital metadata in portable profiles ✅
- **Offline Functionality**: Complete operation without network ✅
- **Federated Learning**: AI improvement with privacy guarantees ✅
- **Global Compatibility**: Universal hospital interface ✅

### Demo Impact
- **Patient Empowerment**: Complete patient control over medical data
- **Care Continuity**: Seamless history access across hospitals globally
- **Privacy Leadership**: New standard for medical data privacy
- **AI Advancement**: Federated learning without data sharing

---

## 🏆 Kiroween 2025 Goals

### Revolutionary Vision
Present the future of patient-controlled, privacy-first medical records that will transform healthcare worldwide.

### Technical Excellence
Demonstrate sophisticated federated learning, cryptographic security, and privacy-preserving architecture.

### Global Impact
Show potential to revolutionize healthcare with portable medical passports that work anywhere.

### Hackathon Victory
Win Kiroween 2025 with groundbreaking healthcare innovation that empowers patients and advances AI.

---

## 📞 Support & Contributing

### Getting Help

- 📖 **Documentation**: See service-specific READMEs in each directory
- 🐛 **Issues**: Report bugs and feature requests via GitHub issues
- 💬 **Discussions**: Join community discussions for questions and ideas

### Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Follow development guidelines and run pre-commit hooks
4. Ensure all tests pass and medical safety validation succeeds
5. Submit pull request with comprehensive description

### Code of Conduct

This project follows healthcare industry standards for:
- **Patient Privacy**: Absolute protection of patient information
- **Medical Safety**: No compromises on patient safety or care quality
- **Professional Standards**: Maintain highest standards of medical software development

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

**Built for Kiroween 2025 Hackathon**  
*Transforming healthcare through patient-controlled, privacy-first medical records*

---

*Medi OS Kiroween Edition v2.0 - The future of medical records is portable, private, and patient-controlled.*