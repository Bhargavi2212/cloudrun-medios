# Security Fix: Database Credentials

## Issue
The PostgreSQL password was hardcoded in the CI/CD workflow file and exposed in git history.

## Actions Taken
1. ✅ Removed hardcoded password from `.github/workflows/version-2-ci-cd.yml`
2. ✅ Updated workflow to use GitHub Secrets: `${{ secrets.DB_PASSWORD }}` and `${{ secrets.DB_HOST }}`
3. ✅ Rotated PostgreSQL password on the database instance
4. ✅ Updated password in GCP Secret Manager

## Required: Add GitHub Secrets

You **MUST** add these secrets to your GitHub repository before the next deployment:

1. Go to: https://github.com/Bhargavi2212/cloudrun-medios/settings/secrets/actions
2. Click "New repository secret"
3. Add the following secrets:

### DB_PASSWORD
- **Name**: `DB_PASSWORD`
- **Value**: `q5obs6NQpa2zAU7VFEtkcP0ln` (or check `new-db-password.txt`)

### DB_HOST
- **Name**: `DB_HOST`
- **Value**: `104.198.58.247`

## Important Notes
- ⚠️ The old password is still in git history. Consider using `git filter-branch` or BFG Repo-Cleaner if this is a security concern.
- ⚠️ The password file `new-db-password.txt` should be deleted after adding to GitHub Secrets.
- ✅ The new password is stored in GCP Secret Manager as `db-password-free`

## Verification
After adding the secrets, the next deployment should work correctly. The workflow will use the secrets instead of hardcoded values.

