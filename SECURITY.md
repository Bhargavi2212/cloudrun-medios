# Security Guide

## ⚠️ CRITICAL: Secrets Management

**NEVER commit passwords, API keys, or any secrets to git!**

This repository has been audited and cleaned of hardcoded secrets. All secrets must be managed through environment variables or secret management systems.

## Exposed Secrets (Action Required)

If you find any of these secrets were previously committed to git, you **MUST**:

1. **Rotate them immediately** - The exposed secrets are compromised
2. **Review git history** - Secrets in commit history remain accessible
3. **Update all systems** - Change passwords/keys everywhere they're used

### Previously Exposed Secrets:

1. **Google Gemini API Key**: `AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo`
   - **Action**: Revoke and regenerate in Google Cloud Console
   - **Impact**: Unauthorized API usage, potential billing charges

2. **Database Password**: `Anuradha`
   - **Action**: Change PostgreSQL password immediately
   - **Impact**: Unauthorized database access

## Secret Management Best Practices

### Local Development

1. **Use `.env` files** (already in `.gitignore`)
   ```bash
   # .env (never commit this file!)
   DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/dbname
   GEMINI_API_KEY=your_api_key_here
   ```

2. **Use environment variables**
   ```powershell
   # PowerShell
   $env:DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/dbname"
   $env:GEMINI_API_KEY="your_api_key_here"
   ```
   ```bash
   # Bash
   export DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/dbname"
   export GEMINI_API_KEY="your_api_key_here"
   ```

### Production / Cloud Run

1. **Use Google Cloud Secret Manager** (recommended)
   - Store secrets in Secret Manager
   - Access via environment variables in Cloud Run
   - See CI/CD workflow for implementation

2. **Use Cloud Run environment variables** (encrypted at rest)
   - Set via `gcloud run services update`
   - Or via Terraform/Cloud Deployment Manager

### CI/CD (GitHub Actions)

Secrets are stored in GitHub Secrets and accessed via:
```yaml
env:
  GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
  DATABASE_URL: ${{ secrets.DATABASE_URL }}
```

## What to Never Commit

- ✅ API keys (GEMINI_API_KEY, GOOGLE_API_KEY, etc.)
- ✅ Database passwords
- ✅ Private keys (`.pem`, `.key` files)
- ✅ Service account JSON files
- ✅ JWT secrets
- ✅ OAuth client secrets
- ✅ Encryption keys

## Current `.gitignore` Protection

The following patterns are ignored:
- `.env`, `.env.*` files (except `.env.example`)
- `*secret*`, `*password*`, `*credential*` files
- `*.key`, `*.pem` files
- `*_key.json`, `service-account*.json` files

## Security Checklist

- [ ] All hardcoded secrets removed from code
- [ ] All secrets removed from documentation
- [ ] `.gitignore` properly configured
- [ ] Exposed secrets rotated (if any were found)
- [ ] Environment variables used for all secrets
- [ ] Production secrets stored in Secret Manager
- [ ] CI/CD uses GitHub Secrets
- [ ] No secrets in commit history (consider `git filter-branch` if needed)

## Verifying Secrets Are Not Committed

Check git history for exposed secrets:
```bash
# Search for API keys
git log --all -p -S "AIza" | grep -i "api"

# Search for passwords
git log --all -p -S "password" | grep -E "(password|passwd)"

# List files that might contain secrets
git log --all --full-history -- "*secret*" "*password*" "*credential*"
```

## Reporting Security Issues

If you discover a security vulnerability:

1. **DO NOT** create a public issue
2. Email the repository maintainers privately
3. Do not commit fixes that expose the vulnerability further

## Additional Resources

- [OWASP Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [Google Cloud Secret Manager](https://cloud.google.com/secret-manager)
- [GitHub Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)

