# 실제 어텐션 값 사전 계산

섹션 4 의 "실제 모델의 어텐션 보기" 와 섹션 5 의 헤드 비교는 `src/data/attention/<locale>.json` 을 읽는다.
이 파일은 손으로 쓰지 않고, 아래 스크립트로 실제 모델의 출력을 저장해 만든다. 파일이 없으면 화면에 안내문이 나온다.

## 실행 (사용자 PC 에서 한 번)

```bash
pip install -r scripts/requirements.txt
python scripts/precompute_attention.py --locale en --model gpt2
python scripts/precompute_attention.py --locale ko --model <한국어 GPT-2 계열 모델 id>
```

생성된 `src/data/attention/en.json`, `ko.json` 을 커밋하면 페이지에 히트맵이 나타난다.
예문은 `scripts/examples.json` 에 있다. 바꾸면 스크립트를 다시 실행한다.

## 주의

- GPT-2(`gpt2`)는 영어로 학습된 모델이라 한국어 문장은 바이트 조각으로 잘게 쪼개져 의미 있는 패턴이 나오지 않는다.
  한국어 예문에는 한국어로 학습된 GPT-2 계열 모델을 써야 한다. 모델 id 는 Hugging Face 에서 확인한 뒤 `--model` 로 넘긴다.
- `--max-tokens`(기본 24)보다 긴 예문은 건너뛴다. 어텐션 행렬은 토큰 수의 제곱으로 커지기 때문이다.
- 출력 JSON 에는 모델 id, 계산 시각, 라이브러리 버전이 함께 기록되며 화면에 표시된다.
- 이 저장소를 만든 원격 세션에서는 Hugging Face 접속이 막혀 있어 여기서는 실행할 수 없었다.

## 출력 형식 (schemaVersion 1)

```json
{
  "schemaVersion": 1,
  "model": "gpt2",
  "locale": "en",
  "generatedAt": "2026-09-03T00:00:00+00:00",
  "transformersVersion": "...",
  "torchVersion": "...",
  "examples": [
    {
      "id": "bank-money",
      "text": "I went to the bank to withdraw money",
      "tokens": ["I", " went", " to", " the", " bank", " to", " withdraw", " money"],
      "layers": 12,
      "heads": 12,
      "attention": "[층][헤드][보는 토큰][보이는 토큰] 크기의 실수 배열, 소수 셋째 자리 반올림"
    }
  ]
}
```

`src/lib/attentionData.ts` 가 이 형식을 검사한다. 형식이 맞지 않으면 빌드 시 오류 대신 화면 안내문이 나온다.

# 내장 소형 모델 학습 (train_tiny_model.py)

"직접 넣어 보기" 데모는 `src/data/model/tiny-transformer.json` 의 가중치를 브라우저에서 직접 실행한다.
이 파일은 `scripts/train_tiny_model.py` 가 페이지 본문(`src/content/ko.ts`, `en.ts` 의 문자열과 `scripts/examples.json`)으로
문자 단위 트랜스포머(3층 × 3헤드, d_model 48, 문맥 64)를 학습해 만든다. 손으로 만들거나 고치지 않는다.

```bash
pip install -r scripts/requirements-train.txt   # jax, numpy (CPU)
python scripts/train_tiny_model.py                # 몇 분. 손실과 이어 쓰기 예가 출력된다
npm test                                          # tinyTransformer.test.ts 가 파일의 reference 값과 대조
```

- 파일에는 fp16 가중치(base64), 어휘, 학습 기록, 그리고 TypeScript 구현 검증용 기준값(`reference`)이 들어 있다.
- 모델 구조를 바꾸면 `src/lib/tinyTransformer.ts` 의 forward 도 같은 순서로 바꿔야 한다.
- 이 세션에서는 이 스크립트를 실제로 실행해 파일을 만들었다 (Hugging Face 접속이 필요 없으므로 원격 환경에서도 가능).

# DeepSeek-V4.1 축소 모델 학습 (train_tiny_dsv41.py)

`DCv4.1-simulator.html` 은 `src/data/model/tiny-dsv41.json` 의 가중치를 브라우저에서 직접 실행한다.
이 파일은 `scripts/train_tiny_dsv41.py` 가 페이지 본문에 `DeepseekV4.1_manual.docx` 의 문단 텍스트를 더한 말뭉치로
V4.1-Flash 구조의 축소판(10층 = causal encoder 5 + decoder 5, 헤드 4 × latent 16, 윈도우 8, Top-4, 인덱서 2 × 8,
후보 풀 2블록 × 4, MoE shared 1 + routed 2/8, 문맥 64)을 학습해 만든다. 손으로 만들거나 고치지 않는다.

```bash
pip install -r scripts/requirements-train.txt   # jax, numpy (CPU)
python scripts/train_tiny_dsv41.py                # 40분 안팎. 손실·인덱서 KL·전문가 부하가 출력된다
npm test                                          # tinyDsv41.test.ts 가 파일의 reference 값과 대조
```

- 층 모드는 스크립트의 `LAYERS` 에 고정돼 있다: L0 SWA 전용, L1·L3 Full(encoder, 압축비 2), L2·L4 Reuse, L5 Full(decoder, CED),
  L6·L8 Reuse, L7·L9 Reindex. 실제 모델의 배치(40층, Full 4층, Reindex 4층)는 매뉴얼 그림 6 을 따른 것이다.
- 학습은 세 단계다: dense 워밍업(전체 엔트리에 어텐션, 인덱서는 KL 로 정렬) → 30% 지점부터 Top-K 희소 선택 → 60% 지점부터 FP4/MXFP4 QAT(straight-through).
  MoE 부하 균형은 손실 없이 선택용 편향을 매 스텝 조정하는 방식이다.
- 기준값(`reference`)은 fp16 으로 반올림한 가중치로 float64 에서 계산하며, 브라우저 구현이 같은 Top-K·후보 풀·전문가를 고르는지까지 검사한다.
- 모델 구조를 바꾸면 `src/lib/tinyDsv41.ts` 의 forwardTrace 도 같은 순서로 바꿔야 한다. 양자화 규칙(E2M1 격자, E4M3 scale, 2의 거듭제곱 scale)은 두 구현이 글자 그대로 같아야 한다.
