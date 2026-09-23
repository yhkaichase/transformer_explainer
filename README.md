# 트랜스포머 쉽게 이해하기

_English: [README.en.md](README.en.md)_

ChatGPT 같은 AI 의 핵심 구조인 **트랜스포머(Transformer)** 를 주니어 엔지니어와 임원 눈높이에서 설명하는 인터랙티브 웹 페이지입니다.

- **임원용** 모드: 수식 없이 비유와 결과 중심으로 읽습니다.
- **엔지니어용** 모드: 같은 페이지에서 수식과 구조 설명이 추가됩니다.
- **한국어 / English**: 상단 토글로 바꾸거나, `?lang=en` 을 붙인 주소로 영어 버전을 바로 열 수 있습니다.
- **직접 넣어 보기**: 맨 위에서 글을 바꾸면 페이지 안에 내장된 작은 트랜스포머(문자 단위, 3층, 파라미터 약 11만 개)가 브라우저에서 토큰 → 임베딩 → 어텐션 → 다음 글자 확률을 실제로 계산합니다. 서버나 인터넷이 필요 없습니다.

참고 프로젝트 [Transformer Explainer](https://github.com/poloclub/transformer-explainer)(Georgia Tech Polo Club)가 브라우저에서 실제 GPT-2 를 실행하며 내부를 보여 준다면, 이 프로젝트는 그보다 한 단계 쉬운 설명을 목표로 합니다. 자세한 계획은 [docs/plan.md](docs/plan.md) 에 있습니다.

## 시작하기

Node.js 22.12 이상이 필요합니다 (`.nvmrc` 참고).

```bash
npm install
npm run dev        # http://localhost:5173
```

## 시뮬레이터 (transformer-simulator.html)

설명 글 없이 구조만 보고 싶을 때 쓰는 두 번째 페이지입니다. 개발 서버에서는 `http://localhost:5173/simulator.html`, 빌드 뒤에는 `dist/simulator.html`, 단일 파일은 `dist/transformer-simulator.html` 입니다.
입력 → 임베딩 → (층) LayerNorm → W_Q·W_K·W_V → 헤드별 어텐션 → W_O → 잔차 → LayerNorm → W₁ → ReLU → W₂ → 잔차 → 최종 LayerNorm → 로짓 → softmax → 다음 토큰을 한 화면에 가로로 펼치고, 실제 가중치 행렬과 활성값을 히트맵으로 그립니다. 상단 오른쪽 결과 카드에 생성된 글과 다음 토큰 후보 확률이 바로 갱신되고, 단계 스테퍼(← → 키, 자동 재생 0.5×~4× 배속, 마지막 단계 뒤 토큰 이어 쓰기 옵션)로 흐름을 따라가고, Prefill(프롬프트 전체 병렬 계산)과 Decode(새 토큰 1개, K·V 캐시 재사용)를 구분해 표시합니다.

## Node.js 없이 보기

페이지는 정적 파일이라 보는 데는 브라우저만 있으면 됩니다. Node.js 는 빌드할 때만 필요합니다.

- **파일 하나로 열기**: `npm run build && npm run build:single` 을 실행하면 `dist/transformer-explain.html`(설명 페이지)과 `dist/transformer-simulator.html`(시뮬레이터)이 생깁니다. CSS 와 JS 가 모두 들어 있어 더블클릭으로 열리고, 인트라넷이나 메일로 전달할 수 있습니다. 영어는 파일 주소 뒤에 `?lang=en` 을 붙입니다.
- **빌드 결과 내려받기**: 푸시할 때마다 GitHub Actions 의 CI 가 `site` 아티팩트(빌드 폴더 전체와 위 단일 파일)를 남깁니다. 저장소 Actions 탭 → 해당 실행 → Artifacts 에서 받을 수 있습니다.
- **링크로 공유하기**: 아래 "배포" 대로 GitHub Pages 를 켜면 주소 하나로 공유할 수 있고, 같은 주소 뒤에 `/transformer-explain.html` 을 붙이면 오프라인용 단일 파일을 받을 수 있습니다.

## 스크립트

| 명령              | 설명                                                    |
| ----------------- | ------------------------------------------------------- |
| `npm run dev`     | 개발 서버                                               |
| `npm run build`   | 타입 검사 후 `dist/` 생성                               |
| `npm run preview` | 빌드 결과 미리보기                                      |
| `npm run check`   | lint, 포맷 검사, 타입 검사, 단위 테스트, 빌드를 한 번에 |
| `npm run lint`    | oxlint                                                  |
| `npm run format`  | Prettier 로 전체 포맷                                   |
| `npm test`        | Vitest 단위 테스트                                      |
| `npm run e2e`     | Playwright 브라우저 테스트 (빌드 결과에 대해 실행)      |

처음 `npm run e2e` 를 실행하기 전에 브라우저를 한 번 설치합니다.

```bash
npx playwright install chromium
```

## 구조

```
src/content/     본문 데이터. ko.ts / en.ts 에 언어별 본문·참고 자료·UI 문자열
src/components/  화면 조각과 인터랙티브 데모
src/lib/         softmax, 어텐션 등 순수 계산 함수 (테스트 포함)
src/state/       설명 수준(임원용/엔지니어용) 상태
e2e/             Playwright 스모크 테스트
docs/plan.md     콘텐츠 계획과 열린 질문
```

## 내장 소형 모델 다시 학습하기

"직접 넣어 보기" 데모의 모델은 `scripts/train_tiny_model.py` 가 이 페이지의 한국어·영어 본문으로 학습한 것입니다. 본문을 크게 바꿨거나 모델 크기를 바꾸고 싶을 때 다시 학습합니다 (CPU 로 몇 분).

```bash
pip install -r scripts/requirements-train.txt
python scripts/train_tiny_model.py
npm test   # TypeScript 구현이 새 가중치의 기준값과 일치하는지 확인
```

## 실제 어텐션 값 만들기

섹션 4 의 "실제 모델의 어텐션 보기" 와 섹션 5 의 헤드 비교는 실제 모델 출력 파일이 있어야 나타납니다. 없으면 안내문이 보입니다.

```bash
pip install -r scripts/requirements.txt
python scripts/precompute_attention.py --locale en --model gpt2
python scripts/precompute_attention.py --locale ko --model <한국어 GPT-2 계열 모델 id>
```

생성된 `src/data/attention/*.json` 을 커밋하면 됩니다. 자세한 내용은 [scripts/README.md](scripts/README.md) 에 있습니다.

## 배포

`main` 브랜치에 푸시하면 `.github/workflows/deploy.yml` 이 GitHub Pages 로 배포합니다.
저장소 **Settings → Pages → Build and deployment → Source** 를 **GitHub Actions** 로 한 번 설정해야 합니다.
배포 주소는 `https://<사용자>.github.io/transformer_explainer/` 형태이고, 영어 버전은 뒤에 `?lang=en` 을 붙입니다.

## 참고 자료

- Vaswani et al., "Attention Is All You Need" (2017) — https://arxiv.org/abs/1706.03762
- Transformer Explainer — https://github.com/poloclub/transformer-explainer
