import type { Locale } from '../../content/types'
import type { AttentionDataset } from '../../lib/attentionData'

/**
 * 테스트 전용 가짜 데이터. 실제 모델 출력이 아니며 화면에는 절대 쓰지 않는다.
 * 토큰 3개, 2층 × 2헤드. 각 행의 합은 1 이고 인과 마스크대로 뒤 토큰은 0 이다.
 */
export function attentionFixture(locale: Locale = 'ko'): AttentionDataset {
  return {
    schemaVersion: 1,
    model: 'test-fixture',
    locale,
    generatedAt: '2026-01-01T00:00:00+00:00',
    examples: [
      {
        id: 'fixture',
        text: 'a b c',
        tokens: ['a', ' b', ' c'],
        layers: 2,
        heads: 2,
        attention: [
          [
            [
              [1, 0, 0],
              [0.5, 0.5, 0],
              [0.2, 0.3, 0.5],
            ],
            [
              [1, 0, 0],
              [0.1, 0.9, 0],
              [0.6, 0.2, 0.2],
            ],
          ],
          [
            [
              [1, 0, 0],
              [0.3, 0.7, 0],
              [0.1, 0.1, 0.8],
            ],
            [
              [1, 0, 0],
              [0.8, 0.2, 0],
              [0.4, 0.4, 0.2],
            ],
          ],
        ],
      },
    ],
  }
}
