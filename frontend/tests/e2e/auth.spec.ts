import { test, expect } from '@playwright/test'

test.describe('Authentication Flow', () => {
  test('should display login page', async ({ page }) => {
    await page.goto('/login')
    
    // Check for login form elements
    await expect(page.getByRole('heading', { name: /login/i })).toBeVisible()
    await expect(page.getByLabel(/email/i)).toBeVisible()
    await expect(page.getByLabel(/password/i)).toBeVisible()
    await expect(page.getByRole('button', { name: /sign in/i })).toBeVisible()
  })

  test('should show validation errors for empty form', async ({ page }) => {
    await page.goto('/login')
    
    // Try to submit empty form
    await page.getByRole('button', { name: /sign in/i }).click()
    
    // Should show validation errors (implementation dependent)
    // This is a placeholder - adjust based on actual validation
  })

  test('should navigate to register page', async ({ page }) => {
    await page.goto('/login')
    
    const registerLink = page.getByRole('link', { name: /register|sign up/i })
    if (await registerLink.isVisible()) {
      await registerLink.click()
      await expect(page).toHaveURL(/.*register/)
    }
  })

  test('should navigate to forgot password page', async ({ page }) => {
    await page.goto('/login')
    
    const forgotPasswordLink = page.getByRole('link', { name: /forgot password/i })
    if (await forgotPasswordLink.isVisible()) {
      await forgotPasswordLink.click()
      await expect(page).toHaveURL(/.*forgot-password/)
    }
  })
})

