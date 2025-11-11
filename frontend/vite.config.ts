/// <reference types="vitest" />
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'
import { fileURLToPath } from 'url'
import { existsSync } from 'fs'

// Get the directory of the current file
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Resolve the src directory - normalize the path for cross-platform compatibility
const srcDir = path.resolve(__dirname, 'src')

// Verify the src directory exists (helpful for debugging)
if (!existsSync(srcDir)) {
  throw new Error(`Source directory not found: ${srcDir}. Current __dirname: ${__dirname}`)
}

// https://vitejs.dev/config/
export default defineConfig({
  // @ts-ignore - Type compatibility issue between @vitejs/plugin-react and vitest's vite types, but works at runtime
  plugins: [react()],
  resolve: {
    alias: {
      '@': srcDir,
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/__tests__/setup.ts',
    css: true,
    include: ['src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
    exclude: [
      'node_modules',
      'dist',
      '.idea',
      '.git',
      '.cache',
      'tests/e2e', // Exclude Playwright E2E tests
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/__tests__/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/mockData',
        'tests/e2e',
      ],
    },
  },
})
