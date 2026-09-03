#!/usr/bin/env python3
"""페이지 본문(한국어·영어)으로 아주 작은 문자 단위 트랜스포머를 학습해 가중치를 내보낸다.

웹 페이지의 "직접 넣어 보기" 데모가 이 가중치를 브라우저 안에서 그대로 실행한다.
모델 구조는 GPT-2 와 같은 디코더 전용(pre-LN) 트랜스포머이며, 크기만 수천 배 작다.

    pip install -r scripts/requirements-train.txt
    python scripts/train_tiny_model.py

출력: src/data/model/tiny-transformer.json
  - config, vocab, fp16(base64) 가중치, 학습 기록
  - reference: TypeScript 구현이 같은 결과를 내는지 검사하는 기준값
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "data" / "model" / "tiny-transformer.json"

CONFIG = {
    "dModel": 48,
    "nHeads": 3,
    "nLayers": 3,
    "dFf": 192,
    "context": 64,
    "layerNormEps": 1e-5,
}
STEPS = 2500
BATCH = 32
LR = 2e-3
SEED = 7
UNK = "␀"  # ␀ : 어휘에 없는 글자를 나타내는 특수 토큰 (id 0)
REFERENCE_PROMPTS = ["은행에 가서 돈을", "I went to the bank to", "트랜스포머는 "]


# ----------------------------------------------------------------------------- 말뭉치
def extract_strings(ts_source: str) -> list[str]:
    """TS 파일의 작은따옴표/큰따옴표 문자열 리터럴을 꺼낸다 (한 줄짜리)."""
    found = []
    for match in re.finditer(r"'((?:[^'\\\n]|\\.)*)'|\"((?:[^\"\\\n]|\\.)*)\"", ts_source):
        text = match.group(1) if match.group(1) is not None else match.group(2)
        text = text.replace("\\'", "'").replace('\\"', '"')
        if len(text) < 12 or text.startswith("http") or "${" in text:
            continue
        if not re.search(r"[가-힣A-Za-z]", text):
            continue
        found.append(text)
    return found


def build_corpus() -> str:
    parts: list[str] = []
    for name in ("ko.ts", "en.ts"):
        parts.extend(extract_strings((ROOT / "src" / "content" / name).read_text(encoding="utf-8")))
    examples = json.loads((ROOT / "scripts" / "examples.json").read_text(encoding="utf-8"))
    for locale in examples.values():
        parts.extend(item["text"] for item in locale)
    # 중복 제거 (순서 유지)
    seen: set[str] = set()
    unique = [p for p in parts if not (p in seen or seen.add(p))]
    return "\n".join(unique) + "\n"


# ----------------------------------------------------------------------------- 모델
def init_params(key, vocab_size: int):
    d, f, n_layers, ctx = CONFIG["dModel"], CONFIG["dFf"], CONFIG["nLayers"], CONFIG["context"]
    keys = iter(jax.random.split(key, 4 + n_layers * 6))
    normal = lambda k, shape, std=0.02: std * jax.random.normal(k, shape, dtype=jnp.float32)
    params = {
        "tok_emb": normal(next(keys), (vocab_size, d)),
        "pos_emb": normal(next(keys), (ctx, d)),
        "lnf_g": jnp.ones(d),
        "lnf_b": jnp.zeros(d),
        "layers": [],
    }
    for _ in range(n_layers):
        params["layers"].append(
            {
                "ln1_g": jnp.ones(d),
                "ln1_b": jnp.zeros(d),
                "wq": normal(next(keys), (d, d)),
                "bq": jnp.zeros(d),
                "wk": normal(next(keys), (d, d)),
                "bk": jnp.zeros(d),
                "wv": normal(next(keys), (d, d)),
                "bv": jnp.zeros(d),
                "wo": normal(next(keys), (d, d)),
                "bo": jnp.zeros(d),
                "ln2_g": jnp.ones(d),
                "ln2_b": jnp.zeros(d),
                "w1": normal(next(keys), (d, f)),
                "b1": jnp.zeros(f),
                "w2": normal(next(keys), (f, d)),
                "b2": jnp.zeros(d),
            }
        )
    return params


def layer_norm(x, g, b):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + CONFIG["layerNormEps"]) * g + b


def forward(params, ids):
    """ids: [T] 정수. logits [T, V] 와 층별 어텐션 [L, H, T, T] 를 돌려준다."""
    d, h = CONFIG["dModel"], CONFIG["nHeads"]
    dh = d // h
    t = ids.shape[0]
    x = params["tok_emb"][ids] + params["pos_emb"][:t]
    mask = jnp.tril(jnp.ones((t, t), dtype=bool))
    attentions = []
    for layer in params["layers"]:
        hn = layer_norm(x, layer["ln1_g"], layer["ln1_b"])
        q = (hn @ layer["wq"] + layer["bq"]).reshape(t, h, dh).transpose(1, 0, 2)
        k = (hn @ layer["wk"] + layer["bk"]).reshape(t, h, dh).transpose(1, 0, 2)
        v = (hn @ layer["wv"] + layer["bv"]).reshape(t, h, dh).transpose(1, 0, 2)
        scores = q @ k.transpose(0, 2, 1) / jnp.sqrt(dh)
        scores = jnp.where(mask[None], scores, -1e9)
        att = jax.nn.softmax(scores, axis=-1)
        attentions.append(att)
        out = (att @ v).transpose(1, 0, 2).reshape(t, d)
        x = x + out @ layer["wo"] + layer["bo"]
        hn = layer_norm(x, layer["ln2_g"], layer["ln2_b"])
        x = x + jax.nn.relu(hn @ layer["w1"] + layer["b1"]) @ layer["w2"] + layer["b2"]
    x = layer_norm(x, params["lnf_g"], params["lnf_b"])
    logits = x @ params["tok_emb"].T
    return logits, jnp.stack(attentions)


def loss_fn(params, batch_x, batch_y):
    logits = jax.vmap(lambda ids: forward(params, ids)[0])(batch_x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(log_probs, batch_y[..., None], axis=-1)[..., 0]
    return -picked.mean()


# ----------------------------------------------------------------------------- 학습
def train(corpus_ids: np.ndarray, vocab_size: int):
    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, vocab_size)
    ctx = CONFIG["context"]

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    beta1, beta2, eps = 0.9, 0.99, 1e-8

    @jax.jit
    def step(params, m, v, batch_x, batch_y, t):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_x, batch_y)
        m = jax.tree_util.tree_map(lambda a, g: beta1 * a + (1 - beta1) * g, m, grads)
        v = jax.tree_util.tree_map(lambda a, g: beta2 * a + (1 - beta2) * g * g, v, grads)
        lr_t = LR * jnp.sqrt(1 - beta2**t) / (1 - beta1**t)
        params = jax.tree_util.tree_map(lambda p, a, b: p - lr_t * a / (jnp.sqrt(b) + eps), params, m, v)
        return params, m, v, loss

    rng = np.random.default_rng(SEED)
    history = []
    started = time.time()
    for t in range(1, STEPS + 1):
        starts = rng.integers(0, len(corpus_ids) - ctx - 1, size=BATCH)
        batch_x = np.stack([corpus_ids[s : s + ctx] for s in starts])
        batch_y = np.stack([corpus_ids[s + 1 : s + ctx + 1] for s in starts])
        params, m, v, loss = step(params, m, v, jnp.asarray(batch_x), jnp.asarray(batch_y), t)
        if t % 250 == 0 or t == 1:
            history.append({"step": t, "loss": round(float(loss), 4)})
            print(f"step {t:5d}  loss {float(loss):.4f}  ({time.time() - started:.0f}s)")
    return params, history


# ----------------------------------------------------------------------------- 내보내기
def to_fp16_roundtrip(params):
    """fp16 으로 저장할 것이므로, 기준값 계산도 fp16 으로 반올림한 가중치로 한다."""
    return jax.tree_util.tree_map(lambda p: jnp.asarray(np.asarray(p, dtype=np.float16).astype(np.float32)), params)


def encode_array(array) -> dict:
    arr = np.asarray(array, dtype=np.float16)
    return {"shape": list(arr.shape), "dtype": "f16", "data": base64.b64encode(arr.tobytes()).decode("ascii")}


def count_params(params) -> int:
    return int(sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params)))


def generate(params, vocab: list[str], stoi: dict[str, int], prompt: str, steps: int) -> str:
    text = prompt
    for _ in range(steps):
        ids = jnp.asarray([stoi.get(ch, 0) for ch in text[-CONFIG["context"] :]])
        logits, _ = forward(params, ids)
        text += vocab[int(jnp.argmax(logits[-1]))]
    return text


def main() -> int:
    corpus = build_corpus()
    chars = sorted(set(corpus))
    vocab = [UNK] + chars
    stoi = {ch: i for i, ch in enumerate(vocab)}
    corpus_ids = np.asarray([stoi[ch] for ch in corpus], dtype=np.int32)
    print(f"말뭉치 {len(corpus):,} 글자, 어휘 {len(vocab)} 개, jax {jax.__version__}")

    params, history = train(corpus_ids, len(vocab))
    params = to_fp16_roundtrip(params)
    print("파라미터 수:", count_params(params))

    for prompt in REFERENCE_PROMPTS:
        print(f"이어 쓰기 예: {generate(params, vocab, stoi, prompt, 24)!r}")

    references = []
    for prompt in REFERENCE_PROMPTS:
        ids = jnp.asarray([stoi.get(ch, 0) for ch in prompt])
        logits, attentions = forward(params, ids)
        probs = np.asarray(jax.nn.softmax(logits[-1]))
        top = np.argsort(-probs)[:5]
        references.append(
            {
                "prompt": prompt,
                "ids": [int(i) for i in ids],
                "top5": [{"token": vocab[int(i)], "probability": round(float(probs[i]), 5)} for i in top],
                "attention_l0_h0": np.round(np.asarray(attentions[0, 0]), 5).tolist(),
                "attention_last_layer_h1_last_row": np.round(np.asarray(attentions[-1, 1, -1]), 5).tolist(),
            }
        )

    weights = {"tok_emb": encode_array(params["tok_emb"]), "pos_emb": encode_array(params["pos_emb"]),
               "lnf_g": encode_array(params["lnf_g"]), "lnf_b": encode_array(params["lnf_b"]), "layers": []}
    for layer in params["layers"]:
        weights["layers"].append({name: encode_array(value) for name, value in layer.items()})

    payload = {
        "schemaVersion": 1,
        "description": "Character-level decoder-only transformer trained on this page's Korean and English text by scripts/train_tiny_model.py.",
        "config": CONFIG,
        "vocab": vocab,
        "unkToken": UNK,
        "training": {
            "steps": STEPS,
            "batch": BATCH,
            "learningRate": LR,
            "seed": SEED,
            "corpusChars": len(corpus),
            "paramCount": count_params(params),
            "lossHistory": history,
            "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "jaxVersion": jax.__version__,
        },
        "weights": weights,
        "reference": references,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"저장: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
