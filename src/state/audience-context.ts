import { createContext, useContext } from 'react'
import type { Audience } from '../content/types'

export interface AudienceState {
  audience: Audience
  setAudience: (audience: Audience) => void
}

export const AudienceContext = createContext<AudienceState | null>(null)

export function useAudience(): AudienceState {
  const state = useContext(AudienceContext)
  if (!state) {
    throw new Error('useAudience 는 AudienceProvider 안에서만 사용할 수 있습니다.')
  }
  return state
}
