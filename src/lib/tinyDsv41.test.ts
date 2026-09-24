import { describe, expect, it } from 'vitest'
import {
  e4m3,
  entryValue,
  fp4Store,
  kvBytesPerToken,
  loadTinyDsv41,
  mxfp4Store,
  nearestE2M1,
} from './tinyDsv41'

const model = loadTinyDsv41()

describe('FP4 / MXFP4 저장 규칙', () => {
  it('E4M3 은 가수 3비트로 반올림하고 448 을 넘지 않는다', () => {
    expect(e4m3(1)).toBe(1)
    expect(e4m3(1.0625)).toBe(1.125) // 반올림 (half up)
    expect(e4m3(1.05)).toBe(1)
    expect(e4m3(3)).toBe(3)
    expect(e4m3(1000)).toBe(448)
    expect(e4m3(0)).toBe(0)
  })

  it('E2M1 격자에서 가장 가까운 값을 고른다', () => {
    expect(nearestE2M1(0.24).value).toBe(0)
    expect(nearestE2M1(0.25).value).toBe(0.5)
    expect(nearestE2M1(-2.6).value).toBe(-3)
    expect(nearestE2M1(5).value).toBe(6)
    expect(nearestE2M1(100).code).toBe(7)
  })

  it('FP4 저장은 그룹마다 scale 하나를 두고 코드 × scale 로 복원한다', () => {
    const vec = Float32Array.from([0.1, -0.5, 2, 4, 0, 1, -1, 0.3])
    const stored = fp4Store(vec, 4)
    expect(stored.scales).toHaveLength(2)
    expect(stored.codes.every((c) => c >= -7 && c <= 7)).toBe(true)
    for (let i = 0; i < vec.length; i++) {
      const scale = stored.scales[Math.floor(i / 4)]
      expect(Math.abs(stored.dequantized[i] - vec[i])).toBeLessThanOrEqual(scale * 1.01)
    }
    expect(stored.dequantized[3]).toBeCloseTo(6 * stored.scales[0], 6) // 4 / 0.6875 ≈ 5.8 → 6
  })

  it('MXFP4 는 벡터당 2의 거듭제곱 scale 을 쓴다', () => {
    const stored = mxfp4Store([0.2, -7, 3, 1])
    expect(Math.log2(stored.scale) % 1).toBe(0)
    expect(stored.scale).toBe(2) // 2^ceil(log2(7 / 6))
    expect(stored.dequantized[1]).toBe(-8) // -7 / 2 = -3.5 → -4
  })
})

describe('캐시 용량 계산', () => {
  it('매뉴얼의 유도식을 작은 구조에 적용하면 토큰당 30 B 다', () => {
    const bytes = kvBytesPerToken(model.config)
    expect(bytes.mainEntry).toBe(10) // 16채널 × 4bit + scale 2개
    expect(bytes.indexerEntry).toBe(5) // 8채널 × 4bit + E8M0 scale
    expect(bytes.sources.map((s) => s.layer)).toEqual([1, 3, 5])
    expect(bytes.perToken).toBe(30) // (10 + 5) × (2 × ½ + 1)
  })
})

describe('TinyDsv41 추론', () => {
  it('학습 스크립트가 남긴 기준값과 같은 결과를 낸다', () => {
    expect(model.reference.length).toBeGreaterThan(0)
    for (const ref of model.reference) {
      const ids = model.encode(ref.prompt)
      expect(ids).toEqual(ref.ids)
      expect(model.decoderStartFor(ids.length)).toBe(ref.decoderStart)
      const trace = model.forwardTrace(ids, ref.decoderStart)
      const T = ids.length
      const t = T - 1
      const top = model.nextTokenDistribution(trace.logits)
      // 기준값 top5 는 ␀ 를 제외하지 않고 뽑은 것이므로 토큰별 확률로 비교한다
      const probs = softmaxOf(trace.logits)
      for (const item of ref.top5) {
        const id = model.encode(item.token)[0]
        expect(probs[id]).toBeCloseTo(item.probability, 3)
      }
      expect(top[0].token).toBe(ref.top5[0].token)

      const E = model.config.experts
      for (const [key, layerRef] of Object.entries(ref.layers)) {
        const layer = trace.layers[Number(key)]
        const n = layer.n
        const experts = Array.from({ length: E }, (_, e) => e).filter(
          (e) => layer.expertMask[t * E + e] === 1,
        )
        expect(experts, `layer ${key} experts`).toEqual(layerRef.experts)
        experts.forEach((e, i) => expect(layer.gates[t * E + e]).toBeCloseTo(layerRef.gates[i], 3))
        for (let s = 0; s < T; s++)
          expect(layer.attnLocal[t * T + s], `layer ${key} local ${s}`).toBeCloseTo(
            layerRef.local_h0[s],
            3,
          )
        if (layerRef.selected) {
          const selected = Array.from({ length: n }, (_, j) => j).filter(
            (j) => layer.selected![t * n + j] === 1,
          )
          expect(selected, `layer ${key} selected`).toEqual(layerRef.selected)
          for (let j = 0; j < n; j++)
            expect(layer.attnGlobal[t * n + j]).toBeCloseTo(layerRef.global_h0![j], 3)
        }
        if (layerRef.scores)
          for (let j = 0; j < n; j++)
            expect(layer.scores![t * n + j], `layer ${key} score ${j}`).toBeCloseTo(
              layerRef.scores[j],
              3,
            )
        if (layerRef.pool) {
          const pool = Array.from({ length: n }, (_, j) => j).filter(
            (j) => layer.pool![t * n + j] === 1,
          )
          expect(pool, `layer ${key} pool`).toEqual(layerRef.pool)
        }
        if (layerRef.entry_last) {
          const last = layerRef.selected![layerRef.selected!.length - 1]
          const value = entryValue(layer.entries!, t, last)
          layerRef.entry_last.forEach((v, d) => expect(value[d]).toBeCloseTo(v, 3))
        }
      }

      // 탐욕적 이어 쓰기도 같아야 한다
      const greedyIds = ids.slice()
      let text = ref.prompt
      for (let step = 0; step < 12 && greedyIds.length < model.config.context; step++) {
        const logits = model.forwardTrace(greedyIds, ref.decoderStart).logits
        let best = 1
        for (let v = 2; v < logits.length; v++) if (logits[v] > logits[best]) best = v
        greedyIds.push(best)
        text += model.vocab[best]
      }
      expect(text).toBe(ref.greedy)
    }
  })

  it('뒤에 토큰을 붙여도 앞 위치의 결과는 변하지 않는다 (decode = 이어 붙이기)', () => {
    const ids = model.encode('은행에 가서 돈을 찾았다')
    const P = 8
    const ds = model.decoderStartFor(P)
    const short = model.forwardTrace(ids.slice(0, P), ds)
    const long = model.forwardTrace(ids, ds)
    const D = model.config.dModel
    for (let i = 0; i < P * D; i++) expect(long.final[i]).toBeCloseTo(short.final[i], 6)
    // encoder 의 2토큰 압축: 짝이 안 온 엔트리는 그 토큰만으로 만든 부분 상태를 쓴다
    const entries = long.layers[1].entries!
    expect(entries.m).toBe(2)
    expect(entries.n).toBe(Math.ceil(ids.length / 2))
    expect(entryValue(entries, 2, 1)).toEqual(entries.cPart.subarray(16, 32))
    expect(entryValue(entries, 3, 1)).toEqual(entries.c.subarray(16, 32))
  })

  it('decoder 층은 decoderStart 이전 위치를 처리하지 않는다', () => {
    const ids = model.encode('DeepSeek-V4.1-Flash의 KV cache는')
    const ds = model.decoderStartFor(ids.length)
    expect(ds).toBe(ids.length - model.config.window)
    const trace = model.forwardTrace(ids, ds)
    const decoderLayer = trace.layers[model.config.encoderLayers]
    const D = model.config.dModel
    for (let d = 0; d < D; d++) expect(decoderLayer.x2[d]).toBe(trace.layers[4].x2[d])
    expect(decoderLayer.start).toBe(ds)
    expect(trace.layers[0].start).toBe(0)
    // 모드 배치
    expect(trace.layers.map((l) => l.mode)).toEqual(model.config.layers.map((l) => l.mode))
    expect(trace.layers[2].entries).toBe(trace.layers[1].entries)
    expect(trace.layers[2].selectedFrom).toBe(1)
    expect(trace.layers[7].pool).toBeNull()
    expect(trace.layers[5].pool).not.toBeNull()
  })

  it('입력 검사와 생성', () => {
    expect(() => model.forwardTrace([])).toThrow(RangeError)
    expect(() => model.forwardTrace(new Array(65).fill(1))).toThrow(RangeError)
    expect(model.decode(model.encode('은행'))).toBe('은행')
    const out = model.generate('은행에 ', 3, 1, () => 0)
    expect(Array.from(out)).toHaveLength(7)
  })
})

function softmaxOf(logits: Float32Array): number[] {
  const max = Math.max(...logits)
  const exps = Array.from(logits, (v) => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((v) => v / sum)
}
