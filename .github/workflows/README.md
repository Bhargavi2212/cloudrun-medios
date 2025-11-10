# GitHub Actions Workflows

This directory contains CI/CD workflows for the Medi OS project.

## Workflows

### `ci-cd.yml` - Main CI/CD Pipeline

Runs on every push and pull request to `main` or `develop` branches.

**Jobs:**
1. **backend-lint-test**: Lints and tests backend code
   - Runs Black (code formatter)
   - Runs isort (import sorter)
   - Runs Flake8 (linter)
   - Runs MyPy (type checker)
   - Runs pytest with coverage

2. **frontend-lint-test**: Lints and tests frontend code
   - Runs ESLint
   - Runs TypeScript type check
   - Runs Vitest unit tests with coverage

3. **build-images**: Builds Docker images (only on push)
   - Builds backend Docker image
   - Builds frontend Docker image
   - Pushes to GCR if on main branch

4. **deploy**: Deploys to Cloud Run (only on main branch)
   - Deploys backend service
   - Deploys frontend service
   - Runs smoke tests

### `test.yml` - Tests Only

Lightweight workflow that only runs tests. Useful for quick feedback on PRs.

**Jobs:**
1. **backend-tests**: Runs backend tests with coverage
2. **frontend-tests**: Runs frontend tests with coverage

## Required Secrets

To use the full CI/CD pipeline, you need to set up these GitHub secrets:

- `GCP_SA_KEY`: Google Cloud Service Account JSON key
- `GCP_PROJECT_ID`: Google Cloud Project ID

## Setup Instructions

1. **Create a Service Account in GCP:**
   ```bash
   gcloud iam service-accounts create github-actions \
     --display-name="GitHub Actions Service Account"
   
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/run.admin"
   
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/storage.admin"
   
   gcloud iam service-accounts keys create key.json \
     --iam-account=github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

2. **Add secrets to GitHub:**
   - Go to your repository → Settings → Secrets and variables → Actions
   - Add `GCP_SA_KEY` with the contents of `key.json`
   - Add `GCP_PROJECT_ID` with your GCP project ID

3. **Enable Container Registry API:**
   ```bash
   gcloud services enable containerregistry.googleapis.com
   ```

## Workflow Triggers

- **Push to main/develop**: Runs full CI/CD pipeline
- **Pull Request**: Runs linting and tests only (no deployment)
- **Manual trigger**: Use `workflow_dispatch` in test.yml

## Notes

- The deploy job only runs on pushes to `main` branch
- Docker images are cached using GitHub Actions cache
- Coverage reports are uploaded to Codecov (optional)
- All jobs run in parallel where possible for faster feedback

