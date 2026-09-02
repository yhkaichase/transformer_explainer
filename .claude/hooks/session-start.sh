#!/bin/bash
# Claude Code on the web 세션이 시작될 때 의존성을 설치한다.
# 로컬 세션에서는 아무것도 하지 않는다 (CLAUDE_CODE_REMOTE 가 "true"일 때만 실행).
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

# npm ci 대신 npm install: 컨테이너 캐시를 재사용하므로 두 번째 실행부터 빠르다.
npm install --no-audit --no-fund
