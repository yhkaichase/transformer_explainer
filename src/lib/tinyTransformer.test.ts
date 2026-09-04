import { describe, expect, it } from 'vitest'
import modelFile from '../data/model/tiny-transformer.json'
import { softmax } from './math'
import { decodeTensor, halfToFloat, loadTinyModel, type TinyModelFile } from './tinyTransformer'

const file = modelFile as unknown as TinyModelFile
const model = loadTinyModel()

describe('fp16 디코딩', () => {
  it('반정밀도 값을 정확히 복원한다', () => {
    expect(halfToFloat(0x3c00)).toBe(1)
    expect(halfToFloat(0xc000)).toBe(-2)
    expect(halfToFloat(0x0000)).toBe(0)
    expect(halfToFloat(0x3555)).toBeCloseTo(0.333, 3)
    expect(halfToFloat(0x0001)).toBeCloseTo(2 ** -24, 12) // 가장 작은 서브노멀
  })

  it('텐서 모양과 원소 수가 맞는다', () => {
    const emb = decodeTensor(file.weights.tok_emb)
    expect(emb.length).toBe(file.vocab.length * file.config.dModel)
    expect(() => decodeTensor({ ...file.weights.lnf_g, shape: [3] })).toThrow()
  })
})

describe('TinyTransformer', () => {
  it('학습 코드(JAX)가 내보낸 기준값과 같은 확률·어텐션을 낸다', () => {
    for (const reference of file.reference) {
      expect(model.encode(reference.prompt)).toEqual(reference.ids)
      const result = model.forward(reference.ids)
      const probabilities = softmax(Array.from(result.logits))
      for (const { token, probability } of reference.top5) {
        const id = file.vocab.indexOf(token)
        expect(probabilities[id]).toBeCloseTo(probability, 3)
      }
      result.attentions[0][0].forEach((row, i) =>
        row.forEach((value, j) => expect(value).toBeCloseTo(reference.attention_l0_h0[i][j], 3)),
      )
      const lastLayer = result.attentions[result.attentions.length - 1]
      lastLayer[1][reference.ids.length - 1].forEach((value, j) =>
        expect(value).toBeCloseTo(reference.attention_last_layer_h1_last_row[j], 3),
      )
    }
  })

  it('어텐션은 인과 마스크를 지키고 각 행의 합이 1 이다', () => {
    const result = model.forward(model.encode('은행에 가서'))
    for (const layer of result.attentions) {
      for (const head of layer) {
        head.forEach((row, i) => {
          expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
          row.forEach((value, j) => {
            if (j > i) expect(value).toBe(0)
          })
        })
      }
    }
    expect(result.embeddings).toHaveLength(6)
    expect(result.embeddings[0]).toHaveLength(file.config.dModel)
  })

  it('어휘에 없는 글자는 id 0 이 되고, 문맥 길이를 넘으면 뒤쪽만 남긴다', () => {
    expect(model.encode('🙂')).toEqual([0])
    const long = 'a'.repeat(file.config.context + 10)
    expect(model.encode(long)).toHaveLength(file.config.context)
    expect(() => model.forward([])).toThrow(RangeError)
  })

  it('다음 글자 분포는 ␀ 를 제외하고 합이 1 이며, 온도가 낮을수록 1등 확률이 커진다', () => {
    const { logits } = model.forward(model.encode('은행에 가서 돈을'))
    const cold = model.nextTokenDistribution(logits, 0.3)
    const hot = model.nextTokenDistribution(logits, 2)
    expect(cold.reduce((a, c) => a + c.probability, 0)).toBeCloseTo(1, 6)
    expect(cold.every((c) => c.id !== 0)).toBe(true)
    expect(cold[0].probability).toBeGreaterThan(hot[0].probability)
    // 페이지 본문으로 학습했으므로 "은행에 가서 돈을" 다음은 공백을 예측한다
    expect(cold[0].token).toBe(' ')
  })

  it('generate 는 주어진 난수로 결정적으로 이어 쓴다', () => {
    const text = model.generate('은행에 가서 돈을', 4, 0.5, () => 0)
    expect(text.startsWith('은행에 가서 돈을')).toBe(true)
    expect(Array.from(text)).toHaveLength(Array.from('은행에 가서 돈을').length + 4)
  })

  it('모델 정보가 파일에 기록되어 있다', () => {
    expect(model.training.paramCount).toBeGreaterThan(100_000)
    expect(model.config.nLayers).toBe(3)
    expect(model.vocab[0]).toBe(model.unkToken)
  })
})

describe('forwardTrace', () => {
  it('중간 활성값의 크기가 맞고, forward 와 같은 로짓과 어텐션을 낸다', () => {
    const ids = model.encode('은행에 가서')
    const t = ids.length
    const { dModel, dFf, nLayers, nHeads } = model.config
    const trace = model.forwardTrace(ids)
    const plain = model.forward(ids)
    expect(trace.layers).toHaveLength(nLayers)
    for (const layer of trace.layers) {
      expect(layer.ln1).toHaveLength(t * dModel)
      expect(layer.q).toHaveLength(t * dModel)
      expect(layer.attentions).toHaveLength(nHeads)
      expect(layer.headsConcat).toHaveLength(t * dModel)
      expect(layer.ffnPre).toHaveLength(t * dFf)
      expect(layer.ffnHidden.every((value, i) => value === Math.max(0, layer.ffnPre[i]))).toBe(true)
      expect(layer.residual2).toHaveLength(t * dModel)
    }
    expect(trace.final).toHaveLength(t * dModel)
    expect(Array.from(trace.logits)).toEqual(Array.from(plain.logits))
    expect(trace.attentions).toEqual(plain.attentions)
    // 잔차: residual1 = 입력 x + projected, residual2 = residual1 + ffnOut (첫 층)
    const first = trace.layers[0]
    for (let i = 0; i < 5; i++) {
      expect(first.residual1[i]).toBeCloseTo(trace.embeddings[0][i] + first.projected[i], 5)
      expect(first.residual2[i]).toBeCloseTo(first.residual1[i] + first.ffnOut[i], 5)
    }
    // 임베딩 = 토큰 임베딩 + 위치 임베딩
    expect(trace.embeddings[1][3]).toBeCloseTo(
      trace.tokenEmbeddings[1][3] + trace.positionEmbeddings[1][3],
      6,
    )
  })

  it('가중치 접근자는 올바른 크기의 행렬을 돌려준다', () => {
    const { dModel, dFf } = model.config
    const w = model.layerWeights(0)
    expect(w.wq).toHaveLength(dModel * dModel)
    expect(w.w1).toHaveLength(dModel * dFf)
    expect(w.w2).toHaveLength(dFf * dModel)
    expect(model.tokenEmbedding).toHaveLength(model.vocab.length * dModel)
    expect(model.finalNorm.g).toHaveLength(dModel)
  })
})
