# Docker Setup - Complete ✅

## What Was Created

### 1. Frontend Dockerfile (`frontend/Dockerfile`)

Multi-stage build for production-optimized frontend:
- **Stage 1 (Builder)**: Installs dependencies and builds the React app
- **Stage 2 (Production)**: Serves static files with nginx
- **Features**:
  - Environment variable support for API URLs
  - Optimized nginx configuration
  - Health check endpoint
  - Gzip compression
  - Security headers

### 2. Frontend Nginx Config (`frontend/nginx.conf`)

Production-ready nginx configuration:
- ✅ SPA routing support (React Router)
- ✅ Static asset caching
- ✅ Gzip compression
- ✅ Security headers
- ✅ Health check endpoint

### 3. Docker Compose (`docker-compose.yml`)

Complete local development environment:
- ✅ **PostgreSQL**: Database service
- ✅ **Backend**: FastAPI service
- ✅ **Frontend**: React app served by nginx
- ✅ **Nginx** (optional): Reverse proxy for production-like setup

### 4. Nginx Reverse Proxy (`nginx/nginx.conf`)

Production-like reverse proxy:
- ✅ Routes `/api/` to backend
- ✅ Routes `/` to frontend
- ✅ Rate limiting
- ✅ WebSocket support
- ✅ Security headers

### 5. Docker Ignore (`.dockerignore`)

Excludes unnecessary files from Docker builds:
- Git files
- Documentation
- IDE files
- Test files
- Build artifacts

## Usage

### Local Development

```bash
# Start all services
docker-compose up

# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Build Individual Services

```bash
# Build backend
docker build -t medios-backend ./backend

# Build frontend
docker build -t medios-frontend \
  --build-arg VITE_API_BASE_URL=http://localhost:8000/api/v1 \
  ./frontend
```

### Environment Variables

Create a `.env` file in the root directory:

```env
# GCP/Gemini (optional for local dev)
GEMINI_API_KEY=your-key-here
HF_TOKEN=your-token-here

# Database (defaults in docker-compose.yml)
POSTGRES_USER=medios_user
POSTGRES_PASSWORD=medios_password
POSTGRES_DB=medios_db
```

### Access Services

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **PostgreSQL**: localhost:5432
- **With Nginx**: http://localhost (frontend), http://api.localhost (backend)

### Health Checks

All services have health check endpoints:
- Backend: `http://localhost:8000/health`
- Frontend: `http://localhost:3000/health`

## Production Build

### Frontend with Environment Variables

```bash
docker build -t medios-frontend \
  --build-arg VITE_API_BASE_URL=https://api.medios.example.com/api/v1 \
  --build-arg VITE_WS_URL=wss://api.medios.example.com/ws \
  --build-arg VITE_ENVIRONMENT=production \
  ./frontend
```

### Backend

```bash
docker build -t medios-backend ./backend
```

## Docker Compose Services

| Service | Port | Description |
|---------|------|-------------|
| postgres | 5432 | PostgreSQL database |
| backend | 8000 | FastAPI backend API |
| frontend | 3000 | React frontend (nginx) |
| nginx | 80 | Reverse proxy (optional) |

## Volumes

- `postgres_data`: Database persistence
- `backend_uploads`: Uploaded files
- `backend_models`: ML model cache

## Networks

All services are on the `medios-network` bridge network for internal communication.

## Notes

- Backend depends on PostgreSQL being healthy before starting
- Frontend depends on backend being available
- Nginx reverse proxy is optional (use `--profile production`)
- Health checks ensure services are ready before marking as healthy
- All services restart automatically unless stopped

