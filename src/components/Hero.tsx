import { useAudience } from '../state/audience-context'

export function Hero() {
  const { audience } = useAudience()

  return (
    <header className="hero">
      <p className="hero-kicker">ChatGPT 같은 AI 의 핵심 구조</p>
      <h1>트랜스포머, 쉽게 이해하기</h1>
      <p className="hero-lead">
        수식 없이 비유로 시작해서, 원하면 수식과 구조까지 내려갑니다. 오른쪽 위에서 설명 수준을 바꿀
        수 있습니다.
      </p>
      <p className="hero-mode" aria-live="polite">
        지금은 <strong>{audience === 'executive' ? '임원용' : '엔지니어용'}</strong> 설명을 보고
        있습니다.
        {audience === 'executive'
          ? ' 각 섹션의 비유와 핵심만 보입니다.'
          : ' 각 섹션 끝에 수식과 구조 설명이 추가됩니다.'}
      </p>
    </header>
  )
}
