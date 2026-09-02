import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// GitHub Pages는 https://<사용자>.github.io/<저장소>/ 아래에 배포되므로
// 배포 워크플로가 BASE_PATH=/transformer_explain/ 을 넘긴다. 로컬 개발은 '/'.
export default defineConfig({
  base: process.env.BASE_PATH ?? '/',
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false,
  },
})
