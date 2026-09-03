#!/usr/bin/env python3
"""예문 몇 개에 대한 실제 언어 모델의 어텐션 가중치를 미리 계산해 JSON 으로 저장한다.

웹 페이지는 이 JSON 을 읽어 어텐션 히트맵(섹션 4)과 헤드 비교(섹션 5)를 그린다.
값은 모델의 실제 출력이며, 이 스크립트 외의 방법으로 만든 파일은 저장소에 넣지 않는다.

사용법 (저장소 루트에서):
    pip install -r scripts/requirements.txt
    python scripts/precompute_attention.py --locale en --model gpt2
    python scripts/precompute_attention.py --locale ko --model <한국어 GPT-2 계열 모델 id>

출력: src/data/attention/<locale>.json  (기본값. --out 으로 바꿀 수 있다)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXAMPLES = ROOT / "scripts" / "examples.json"
DEFAULT_OUT_DIR = ROOT / "src" / "data" / "attention"
ROUND_DIGITS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locale", choices=["ko", "en"], required=True, help="examples.json 의 언어 키")
    parser.add_argument("--model", default="gpt2", help="Hugging Face 모델 id (기본: gpt2)")
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES, help="예문 파일")
    parser.add_argument("--out", type=Path, default=None, help="출력 JSON 경로")
    parser.add_argument("--max-tokens", type=int, default=24, help="이보다 긴 예문은 건너뛴다")
    return parser.parse_args()


def readable_tokens(tokenizer, ids: list[int]) -> list[str]:
    """BPE 조각을 사람이 읽을 수 있는 문자열로 바꾼다 (GPT-2 의 'Ġ' 는 앞 공백)."""
    pieces = tokenizer.convert_ids_to_tokens(ids)
    return [tokenizer.convert_tokens_to_string([piece]) for piece in pieces]


def main() -> int:
    args = parse_args()
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - 안내 메시지
        print(f"필요한 패키지가 없습니다: {error}. pip install -r scripts/requirements.txt", file=sys.stderr)
        return 1

    examples = json.loads(args.examples.read_text(encoding="utf-8"))[args.locale]
    out_path = args.out or (DEFAULT_OUT_DIR / f"{args.locale}.json")

    print(f"모델 불러오는 중: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # sdpa 구현은 어텐션 행렬을 돌려주지 않으므로 eager 로 강제한다.
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    results = []
    for example in examples:
        encoded = tokenizer(example["text"], return_tensors="pt")
        ids = encoded["input_ids"][0].tolist()
        if len(ids) > args.max_tokens:
            print(f"건너뜀 ({len(ids)} 토큰 > {args.max_tokens}): {example['text']}")
            continue
        with torch.no_grad():
            output = model(**{k: v.to(device) for k, v in encoded.items()}, output_attentions=True)
        # output.attentions: 층 수 × [batch, heads, query, key]
        stacked = torch.stack(output.attentions)[:, 0].to("cpu")  # [layers, heads, q, k]
        layers, heads, _, _ = stacked.shape
        attention = [
            [[[round(float(w), ROUND_DIGITS) for w in row] for row in head] for head in layer]
            for layer in stacked.tolist()
        ]
        results.append(
            {
                "id": example["id"],
                "text": example["text"],
                "tokens": readable_tokens(tokenizer, ids),
                "layers": layers,
                "heads": heads,
                "attention": attention,
            }
        )
        print(f"완료: {example['id']} ({len(ids)} 토큰, {layers} 층 × {heads} 헤드)")

    dataset = {
        "schemaVersion": 1,
        "model": args.model,
        "locale": args.locale,
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "transformersVersion": transformers.__version__,
        "torchVersion": torch.__version__,
        "examples": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"저장: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
