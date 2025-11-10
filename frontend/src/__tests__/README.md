# Frontend Tests

This directory contains unit and integration tests for the Medi OS frontend.

## Test Structure

- `setup.ts` - Test configuration and global setup
- `utils.tsx` - Testing utilities and custom render functions
- `mocks/` - Mock data and API handlers
- `components/` - Component unit tests

## Running Tests

### Unit Tests (Vitest)

```bash
# Run tests in watch mode
npm run test

# Run tests once
npm run test:coverage

# Run tests with UI
npm run test:ui
```

### E2E Tests (Playwright)

```bash
# Run E2E tests
npm run test:e2e

# Run E2E tests with UI
npm run test:e2e:ui

# Run all tests
npm run test:all
```

## Writing Tests

### Component Tests

Use the custom `render` function from `utils.tsx` which includes all necessary providers:

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '../utils'
import { MyComponent } from '@/components/MyComponent'

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />)
    expect(screen.getByText('Hello')).toBeInTheDocument()
  })
})
```

### E2E Tests

E2E tests should be placed in `tests/e2e/` and follow Playwright patterns:

```ts
import { test, expect } from '@playwright/test'

test('user can login', async ({ page }) => {
  await page.goto('/login')
  // ... test implementation
})
```

## Test Coverage Goals

- **Target**: ≥70% coverage for critical components
- **Critical Components**: Auth, Dashboards, AI workflows
- **E2E**: All role-based workflows should have E2E tests

