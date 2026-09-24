import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// GitHub Pages는 https://<사용자>.github.io/<저장소>/ 아래에 배포되므로
// 배포 워크플로가 BASE_PATH=/transformer_explainer/ 을 넘긴다. 로컬 개발은 '/'.
export default defineConfig({
  base: process.env.BASE_PATH ?? '/',
  plugins: [react()],
  build: {
    // 설명 페이지(index.html), 트랜스포머 시뮬레이터(simulator.html), DeepSeek-V4.1 시뮬레이터 세 페이지를 만든다.
    rollupOptions: {
      input: {
        main: 'index.html',
        simulator: 'simulator.html',
        dsv41: 'DCv4.1-simulator.html',
      },
    },
    // 내장 모델(313 KB, 1.1 MB)이 번들에 포함되므로 청크 크기 경고 기준을 올린다.
    chunkSizeWarningLimit: 2400,
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false,
  },
})
