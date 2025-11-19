# Environment Setup Guide

## Setting Up Gemini API Key

The document processing system requires a Gemini API key for the multi-step extraction pipeline.

### Option 1: Environment Variable (Recommended)

Set the environment variable before starting the services:

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
```

**Windows CMD:**
```cmd
set GEMINI_API_KEY=AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
```

### Option 2: .env File

Create a `.env` file in the `Version -2/services/summarizer_agent/` directory:

```env
GEMINI_API_KEY=AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/summarizer_agent
SUMMARIZER_AGENT_STORAGE_ROOT=./storage
```

### Option 3: System Environment (Permanent)

**Windows:**
1. Open System Properties > Environment Variables
2. Add new variable: `GEMINI_API_KEY` = `AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo`

**Linux/Mac:**
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
```

## Verify Setup

After setting the API key, verify it's loaded:

```bash
cd "Version -2"
python -c "import os; print('GEMINI_API_KEY set:', bool(os.getenv('GEMINI_API_KEY')))"
```

## Important Notes

- Never commit the API key to version control
- The `.env` file should be in `.gitignore`
- For production, use secure secret management (e.g., Google Secret Manager)
- The API key is only needed for the `summarizer-agent` service

## Storage Configuration

Both services need to use the same storage root:

```env
# In manage-agent .env
MANAGE_AGENT_STORAGE_ROOT=./storage

# In summarizer-agent .env
SUMMARIZER_AGENT_STORAGE_ROOT=./storage
```

This ensures the summarizer can access files uploaded by manage-agent.

