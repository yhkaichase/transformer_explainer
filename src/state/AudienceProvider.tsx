import { useCallback, useMemo, useState, type ReactNode } from 'react'
import type { Audience } from '../content/types'
import { AudienceContext } from './audience-context'

export const AUDIENCE_STORAGE_KEY = 'transformer-explain:audience'

function readStoredAudience(): Audience {
  try {
    return window.localStorage.getItem(AUDIENCE_STORAGE_KEY) === 'engineer'
      ? 'engineer'
      : 'executive'
  } catch {
    // 저장소를 쓸 수 없는 환경(사생활 보호 모드 등)에서는 기본값을 쓴다.
    return 'executive'
  }
}

export function AudienceProvider({ children }: { children: ReactNode }) {
  const [audience, setAudienceState] = useState<Audience>(readStoredAudience)

  const setAudience = useCallback((next: Audience) => {
    setAudienceState(next)
    try {
      window.localStorage.setItem(AUDIENCE_STORAGE_KEY, next)
    } catch {
      // 저장에 실패해도 화면 전환은 계속 동작해야 한다.
    }
  }, [])

  const value = useMemo(() => ({ audience, setAudience }), [audience, setAudience])

  return <AudienceContext.Provider value={value}>{children}</AudienceContext.Provider>
}
