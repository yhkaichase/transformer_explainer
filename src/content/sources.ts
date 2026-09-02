import type { Source } from './types'

/**
 * 페이지 하단에 표시하는 참고 자료.
 * 여기에 넣는 URL 은 실제로 열리는지 확인한 것만 넣는다 (docs/plan.md 의 출처 규칙 참고).
 */
export const sources: Source[] = [
  {
    label: 'Vaswani et al., "Attention Is All You Need" (2017)',
    detail: '트랜스포머를 처음 제안한 논문. 이 페이지의 수식과 구조 설명의 근거.',
    url: 'https://arxiv.org/abs/1706.03762',
  },
  {
    label: 'Transformer Explainer (Georgia Tech Polo Club)',
    detail:
      '브라우저에서 실제 GPT-2 를 실행하며 내부를 보여 주는 인터랙티브 시각화. 이 프로젝트의 출발점.',
    url: 'https://github.com/poloclub/transformer-explainer',
  },
]
