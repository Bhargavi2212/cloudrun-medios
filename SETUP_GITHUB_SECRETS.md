# Setup GitHub Secrets for Database Connection

## Required Secrets

You **MUST** add these secrets to your GitHub repository before deployment:

1. Go to: https://github.com/Bhargavi2212/cloudrun-medios/settings/secrets/actions
2. Click "New repository secret" for each:

### DB_PASSWORD
- **Name**: `DB_PASSWORD`
- **Value**: `SHJCrvBWL8if2Z4e3GUlDxcN`
- **Description**: PostgreSQL database password (rotated for security)

### DB_HOST
- **Name**: `DB_HOST`
- **Value**: `104.198.58.247`
- **Description**: PostgreSQL database host (VM external IP)

## Why These Are Needed

- The old database password was exposed in git history and has been rotated
- Database host was hardcoded and has been moved to secrets for security
- The CI/CD workflow will use these secrets if Secret Manager access fails
- This provides a fallback mechanism for deployments

## Verification

After adding the secrets, the next deployment should work correctly. The workflow will:
1. Try to get credentials from Secret Manager first
2. Fall back to GitHub Secrets if Secret Manager access fails
3. Fail with clear error if neither source is available

