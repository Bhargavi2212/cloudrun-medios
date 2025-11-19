# GitHub Actions Workflow Strategy

## Overview

Your repository now has **two sets of CI/CD workflows** that can coexist:

1. **Legacy Workflows** - For old `backend/` and `frontend/` structure
2. **Version 2 Workflows** - For new `Version-2-HospitalA/` and `Version-2-HospitalB/` structure

## How They Work Together

### Path-Based Triggers

Both sets of workflows use **path filters** so they only trigger when relevant files change:

**Legacy Workflows** (`ci-cd.yml`, `test.yml`):
- ✅ Trigger only when `backend/**` or `frontend/**` changes
- ✅ Won't trigger for Version-2 changes

**Version 2 Workflows** (`version-2-ci-cd.yml`, `version-2-test.yml`):
- ✅ Trigger only when `Version-2-HospitalA/**` or `Version-2-HospitalB/**` changes
- ✅ Won't trigger for legacy backend/frontend changes

### Workflow Execution

```
Push to main/develop
    ↓
Which paths changed?
    ↓
┌─────────────────────┬─────────────────────┐
│ Legacy paths        │ Version-2 paths     │
│ (backend/, frontend/│ (Version-2-Hospital*│
│                     │                     │
│ → Old workflows run │ → New workflows run │
│   OR                │   OR                │
│                     │                     │
│ → Nothing runs      │ → Nothing runs      │
└─────────────────────┴─────────────────────┘
```

## Recommendation

### Option 1: Keep Both (Recommended if maintaining both structures)

**Keep both sets of workflows** if you:
- Still have active code in `backend/` or `frontend/`
- Want to maintain backward compatibility
- Are gradually migrating to Version 2

**Advantages:**
- Both structures work independently
- No conflicts between workflows
- Safe migration path

### Option 2: Remove Legacy Workflows (Recommended if fully migrated)

**Remove legacy workflows** if you:
- Have fully migrated to Version 2
- No longer use `backend/` or `frontend/` at root
- Want to simplify your CI/CD

**Steps to remove:**
```bash
# Delete legacy workflows
rm .github/workflows/ci-cd.yml
rm .github/workflows/test.yml

# Commit the changes
git add .github/workflows/
git commit -m "Remove legacy CI/CD workflows - migrated to Version 2"
git push
```

## Current Status

✅ **Both workflows are configured** with path filters
✅ **GCP secrets added**: `GCP_SA_KEY` and `GCP_PROJECT_ID`
✅ **Version 2 workflows ready** to deploy to Cloud Run

## What Happens When You Push

### Scenario 1: Push to Version-2-HospitalA
```yaml
Changes: Version-2-HospitalA/services/manage_agent/main.py
    ↓
✅ version-2-ci-cd.yml triggers
✅ version-2-test.yml triggers
❌ ci-cd.yml does NOT trigger
❌ test.yml does NOT trigger
```

### Scenario 2: Push to backend/
```yaml
Changes: backend/main.py
    ↓
✅ ci-cd.yml triggers
✅ test.yml triggers
❌ version-2-ci-cd.yml does NOT trigger
❌ version-2-test.yml does NOT trigger
```

### Scenario 3: Push to both
```yaml
Changes: 
  - Version-2-HospitalA/services/manage_agent/main.py
  - backend/main.py
    ↓
✅ ALL workflows trigger (both legacy and Version 2)
```

## Deployment Strategy

### Version 2 Deployment (New)

When you push to `main` branch with Version-2 changes:
1. ✅ Tests run for HospitalA and HospitalB
2. ✅ Docker images built for all services
3. ✅ Deploys to Cloud Run:
   - Hospital A services → `us-central1`
   - Hospital B services → `us-east1`

### Legacy Deployment (Old)

When you push to `main` branch with backend/frontend changes:
1. ✅ Tests run
2. ✅ Docker images built
3. ✅ Deploys to Cloud Run (old structure)

## Recommended Next Steps

1. **Test the Version 2 workflow:**
   ```bash
   # Make a small change to Version-2-HospitalA
   echo "# Test" >> Version-2-HospitalA/README.md
   git add Version-2-HospitalA/README.md
   git commit -m "Test Version 2 CI/CD"
   git push
   ```

2. **Check GitHub Actions:**
   - Go to: https://github.com/Bhargavi2212/cloudrun-medios/actions
   - Verify `version-2-ci-cd.yml` workflow runs

3. **Monitor deployment:**
   - After pushing to `main`, check Cloud Run services
   - Verify all services deployed correctly

4. **Remove legacy workflows** (when ready):
   - If you're fully migrated to Version 2
   - Delete `ci-cd.yml` and `test.yml`

## Troubleshooting

### Both workflows trigger when they shouldn't

**Cause:** Path filters might not be working correctly

**Fix:** Check that path filters are set correctly:
```yaml
# Legacy workflow should have:
paths:
  - 'backend/**'
  - 'frontend/**'

# Version 2 workflow should have:
paths:
  - 'Version-2-HospitalA/**'
  - 'Version-2-HospitalB/**'
```

### Workflow doesn't trigger

**Cause:** Path filters might be too restrictive

**Fix:** Ensure the changed files match the path patterns in the workflow

### Deployment fails

**Cause:** GCP secrets might not be configured correctly

**Fix:** Verify secrets are set:
- Repository Settings → Secrets and variables → Actions
- Check `GCP_SA_KEY` and `GCP_PROJECT_ID` exist

## Summary

✅ **Both workflows can coexist** - they use path filters to avoid conflicts
✅ **Version 2 workflows are ready** - will deploy to Cloud Run on main branch
✅ **Legacy workflows still work** - for old backend/frontend structure
⚠️ **Remove legacy workflows** when you fully migrate to Version 2

