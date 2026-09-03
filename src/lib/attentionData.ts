import type { Locale } from '../content/types'

/**
 * scripts/precompute_attention.py 가 만든 실제 어텐션 값 파일을 읽고 검사한다.
 * 파일이 없으면 데모는 안내 상태를 보여 준다. 값은 절대 손으로 만들지 않는다.
 */

export interface AttentionExample {
  id: string
  text: string
  /** 사람이 읽을 수 있게 바꾼 BPE 조각. 앞 공백이 있는 조각은 공백으로 시작한다. */
  tokens: string[]
  layers: number
  heads: number
  /** [층][헤드][보는 토큰][보이는 토큰]. 각 행의 합은 1. */
  attention: number[][][][]
}

export interface AttentionDataset {
  schemaVersion: 1
  model: string
  locale: Locale
  generatedAt: string
  transformersVersion?: string
  torchVersion?: string
  examples: AttentionExample[]
}

export class AttentionDataError extends Error {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function fail(message: string): never {
  throw new AttentionDataError(message)
}

function validateExample(raw: unknown, index: number): AttentionExample {
  const where = `examples[${index}]`
  if (!isRecord(raw)) fail(`${where} 가 객체가 아닙니다.`)
  const { id, text, tokens, layers, heads, attention } = raw
  if (typeof id !== 'string' || id === '') fail(`${where}.id 가 없습니다.`)
  if (typeof text !== 'string') fail(`${where}.text 가 없습니다.`)
  if (
    !Array.isArray(tokens) ||
    tokens.length === 0 ||
    !tokens.every((t) => typeof t === 'string')
  ) {
    fail(`${where}.tokens 가 문자열 배열이 아닙니다.`)
  }
  if (!Number.isInteger(layers) || (layers as number) < 1)
    fail(`${where}.layers 가 양의 정수가 아닙니다.`)
  if (!Number.isInteger(heads) || (heads as number) < 1)
    fail(`${where}.heads 가 양의 정수가 아닙니다.`)
  const n = tokens.length
  if (!Array.isArray(attention) || attention.length !== layers) {
    fail(`${where}.attention 의 층 수가 layers(${String(layers)})와 다릅니다.`)
  }
  attention.forEach((layer, l) => {
    if (!Array.isArray(layer) || layer.length !== heads) {
      fail(`${where}.attention[${l}] 의 헤드 수가 heads(${String(heads)})와 다릅니다.`)
    }
    layer.forEach((head, h) => {
      if (!Array.isArray(head) || head.length !== n) {
        fail(`${where}.attention[${l}][${h}] 의 행 수가 토큰 수(${n})와 다릅니다.`)
      }
      head.forEach((row, q) => {
        if (!Array.isArray(row) || row.length !== n) {
          fail(`${where}.attention[${l}][${h}][${q}] 의 열 수가 토큰 수(${n})와 다릅니다.`)
        }
        row.forEach((weight, k) => {
          if (typeof weight !== 'number' || !(weight >= 0 && weight <= 1.001)) {
            fail(`${where}.attention[${l}][${h}][${q}][${k}] 가 0~1 사이의 수가 아닙니다.`)
          }
        })
      })
    })
  })
  return {
    id,
    text,
    tokens: tokens as string[],
    layers: layers as number,
    heads: heads as number,
    attention: attention as number[][][][],
  }
}

/** JSON 을 검사해 타입이 보장된 데이터로 돌려준다. 문제가 있으면 AttentionDataError 를 던진다. */
export function validateDataset(raw: unknown): AttentionDataset {
  if (!isRecord(raw)) fail('JSON 최상위가 객체가 아닙니다.')
  if (raw.schemaVersion !== 1) fail(`지원하지 않는 schemaVersion: ${String(raw.schemaVersion)}`)
  if (typeof raw.model !== 'string' || raw.model === '') fail('model 이 없습니다.')
  if (raw.locale !== 'ko' && raw.locale !== 'en')
    fail(`locale 이 ko 또는 en 이 아닙니다: ${String(raw.locale)}`)
  if (typeof raw.generatedAt !== 'string') fail('generatedAt 이 없습니다.')
  if (!Array.isArray(raw.examples)) fail('examples 가 배열이 아닙니다.')
  return {
    schemaVersion: 1,
    model: raw.model,
    locale: raw.locale,
    generatedAt: raw.generatedAt,
    transformersVersion:
      typeof raw.transformersVersion === 'string' ? raw.transformersVersion : undefined,
    torchVersion: typeof raw.torchVersion === 'string' ? raw.torchVersion : undefined,
    examples: raw.examples.map((example, index) => validateExample(example, index)),
  }
}

export interface LoadedDatasets {
  datasets: Partial<Record<Locale, AttentionDataset>>
  /** 읽지 못한 파일과 이유. 화면 안내 상태에 표시한다. */
  errors: string[]
}

/** 파일 경로 → JSON 내용 맵을 검사해 언어별 데이터로 정리한다. */
export function loadDatasets(files: Record<string, unknown>): LoadedDatasets {
  const datasets: Partial<Record<Locale, AttentionDataset>> = {}
  const errors: string[] = []
  for (const [path, raw] of Object.entries(files)) {
    try {
      const dataset = validateDataset(raw)
      datasets[dataset.locale] = dataset
    } catch (error) {
      errors.push(`${path}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  return { datasets, errors }
}

// src/data/attention/*.json 이 있으면 빌드에 포함되고, 없으면 빈 객체가 된다 (오류 없음).
const files = import.meta.glob('../data/attention/*.json', {
  eager: true,
  import: 'default',
}) as Record<string, unknown>

export const attentionData: LoadedDatasets = loadDatasets(files)

export function getAttentionDataset(locale: Locale): AttentionDataset | null {
  return attentionData.datasets[locale] ?? null
}

/** 0~1 가중치를 다섯 단계로 나눈다 (히트맵 색 단계). */
export function weightBin(weight: number): 0 | 1 | 2 | 3 | 4 {
  if (weight < 0.05) return 0
  if (weight < 0.15) return 1
  if (weight < 0.3) return 2
  if (weight < 0.5) return 3
  return 4
}

/** BPE 조각을 화면에 보일 형태로 다듬는다. 공백만 있는 조각은 ␣ 로 표시한다. */
export function displayToken(token: string): string {
  const trimmed = token.trim()
  return trimmed === '' ? '␣' : trimmed
}
