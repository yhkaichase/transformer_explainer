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
