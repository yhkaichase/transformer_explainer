import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// 테스트마다 렌더링된 DOM을 정리한다.
afterEach(() => {
  cleanup()
})
