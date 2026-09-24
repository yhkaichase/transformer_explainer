#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash 의 어텐션·KV cache 구조를 아주 작게 흉내 낸 문자 단위 모델을 학습해 내보낸다.

DCv4.1-simulator.html 페이지가 이 가중치를 브라우저 안에서 그대로 실행한다.
구조는 저장소의 DeepseekV4.1_manual.docx 가 설명하는 V4.1-Flash 를 따르되 크기만 수천 배 작다.

  - CED: 앞쪽 층은 causal encoder, 뒤쪽 층은 decoder. decoder 의 global KV 는 encoder 마지막 출력에서 투영.
  - 층마다 CSA2 모드: swa(SWA 전용) / full(KV·indexer K 생성, Top-K 선택) / reindex(KV 재사용, Top-K 재선택) /
    reuse(KV 와 Top-K 모두 재사용, query 와 SWA 만 새로).
  - global KV 엔트리 c_j 는 latent 하나가 key 와 value 를 겸한다 (MLA 계열). encoder 는 인접 2토큰을 1엔트리로 합친다.
  - 인덱서: S_{t,j} = Σ_h w_{t,h} · ReLU(q^I_{t,h} · K^I_j) 로 점수를 매겨 Top-K 를 고른다.
    decoder 첫 Full 층은 블록 최대 점수로 후보 풀을 만들고 Reindex 층은 그 안에서만 다시 고른다.
  - global 로짓과 SWA 로짓을 이어 붙여 softmax 하나로 처리한다.
  - main KV 는 FP4(E2M1) + 그룹별 E4M3 scale, indexer K 는 MXFP4 로 저장 (학습 후반에 QAT).
  - FFN 은 MoE: shared 전문가 1개 + routed 전문가 중 Top-2 (sigmoid 게이트, 편향으로 부하 균형).

    pip install -r scripts/requirements-train.txt
    python scripts/train_tiny_dsv41.py

출력: src/data/model/tiny-dsv41.json (config, vocab, fp16 가중치, 학습 기록, reference 기준값)
"""

from __future__ import annotations

import base64
import datetime as dt
import functools
import html
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("DSV41_OUT", ROOT / "src" / "data" / "model" / "tiny-dsv41.json"))
MANUAL = ROOT / "DeepseekV4.1_manual.docx"

# ----------------------------------------------------------------------------- 구조
D = 48  # d_model
CTX = 64  # 문맥 길이
H = 4  # main attention 헤드 수 (실제 64)
DC = 16  # 공유 latent 차원 c_j (실제 512)
W = 8  # SWA 윈도우 (실제 128)
K = 4  # Top-K (실제 512)
POOL_BLOCK = 4  # 후보 풀 블록 크기 (실제 8)
POOL_BLOCKS = 2  # 후보 풀 블록 수 (실제 2,048)
HI = 2  # 인덱서 헤드 수
DI = 8  # 인덱서 차원 (실제 128)
E = 8  # routed 전문가 수 (실제 384)
ETOP = 2  # 활성 routed 전문가 수 (실제 6)
FE = 24  # routed 전문가 hidden
FS = 48  # shared 전문가 hidden
FP4_GROUP = 8  # main KV 의 E4M3 scale 하나가 맡는 채널 수 (실제 16)
EPS = 1e-5
ENCODER_LAYERS = 5
LAYERS = [
    {"mode": "swa"},
    {"mode": "full", "m": 2},
    {"mode": "reuse", "src": 1, "idx": 1},
    {"mode": "full", "m": 2},
    {"mode": "reuse", "src": 3, "idx": 3},
    {"mode": "full", "m": 1},
    {"mode": "reuse", "src": 5, "idx": 5},
    {"mode": "reindex", "src": 5},
    {"mode": "reuse", "src": 5, "idx": 7},
    {"mode": "reindex", "src": 5},
]
CONFIG = {
    "dModel": D,
    "context": CTX,
    "nHeads": H,
    "dLatent": DC,
    "window": W,
    "topK": K,
    "poolBlock": POOL_BLOCK,
    "poolBlocks": POOL_BLOCKS,
    "indexerHeads": HI,
    "dIndexer": DI,
    "experts": E,
    "expertsTopK": ETOP,
    "dExpert": FE,
    "dShared": FS,
    "fp4Group": FP4_GROUP,
    "layerNormEps": EPS,
    "encoderLayers": ENCODER_LAYERS,
    "layers": LAYERS,
}

STEPS = int(os.environ.get("DSV41_STEPS", 3000))
BATCH = 32
LR = 2e-3
SEED = 11
SPARSE_FROM = 0.3  # 이 비율부터 Top-K 희소 선택 사용 (앞은 dense 워밍업)
QUANT_FROM = 0.6  # 이 비율부터 FP4/MXFP4 QAT
KL_WEIGHT = 0.05  # 인덱서 정렬 손실 가중치
BIAS_RATE = 0.01  # MoE 부하 균형 편향 갱신 폭
CLIP = 1.0
UNK = "␀"
REFERENCE_PROMPTS = ["은행에 가서 돈을", "DeepSeek-V4.1-Flash의 KV cache는", "I went to the bank to", "트랜스포머는 "]


# ----------------------------------------------------------------------------- 말뭉치
def extract_strings(ts_source: str) -> list[str]:
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


def docx_text(path: Path) -> str:
    """docx 의 본문 문단 텍스트만 꺼낸다 (수식 이미지는 제외)."""
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    lines = []
    for paragraph in xml.split("</w:p>"):
        texts = re.findall(r"<w:t(?:\s[^>]*)?>([^<]*)</w:t>", paragraph)
        line = html.unescape("".join(texts)).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def build_corpus() -> str:
    parts: list[str] = []
    if MANUAL.exists():
        parts.extend(docx_text(MANUAL).split("\n"))
    for name in ("ko.ts", "en.ts"):
        parts.extend(extract_strings((ROOT / "src" / "content" / name).read_text(encoding="utf-8")))
    examples = json.loads((ROOT / "scripts" / "examples.json").read_text(encoding="utf-8"))
    for locale in examples.values():
        parts.extend(item["text"] for item in locale)
    seen: set[str] = set()
    unique = [p for p in parts if not (p in seen or seen.add(p))]
    return "\n".join(unique) + "\n"


# ----------------------------------------------------------------------------- 양자화 (TS 구현과 같은 규칙)
E2M1 = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e4m3(v):
    """0 이상 실수를 E4M3 (지수 4비트, 가수 3비트) 격자로 반올림한다. scale 저장용."""
    mant, exp = jnp.frexp(jnp.maximum(v, 1e-30))  # v = mant · 2^exp, mant ∈ [0.5, 1)
    exp = exp - 1
    frac = jnp.floor((mant * 2 - 1) * 8 + 0.5) / 8
    exp = jnp.where(frac >= 1, exp + 1, exp)
    frac = jnp.where(frac >= 1, 0.0, frac)
    exp = jnp.maximum(exp, -6)
    out = jnp.minimum((1 + frac) * 2.0**exp, 448.0)  # 448 = E4M3 최댓값에서 포화
    return jnp.where(v > 0, out, 0.0)


def nearest_e2m1(y):
    a = jnp.abs(y)
    idx = (
        (a >= 0.25).astype(jnp.int32)
        + (a >= 0.75)
        + (a >= 1.25)
        + (a >= 1.75)
        + (a >= 2.5)
        + (a >= 3.5)
        + (a >= 5.0)
    )
    return jnp.sign(y) * E2M1[idx]


def fp4_store(c):
    """main KV: FP4(E2M1) 값 + 채널 그룹별 E4M3 scale. 저장 후 다시 읽은(dequantize) 값을 돌려준다."""
    shape = c.shape
    groups = c.reshape(shape[:-1] + (shape[-1] // FP4_GROUP, FP4_GROUP))
    amax = jnp.max(jnp.abs(groups), axis=-1, keepdims=True)
    scale = e4m3(amax / 6.0)
    scale = jnp.where(scale > 0, scale, 1.0)
    q = nearest_e2m1(groups / scale)
    return (q * scale).reshape(shape)


def mxfp4_store(k):
    """indexer K: E2M1 값 + 벡터당 2의 거듭제곱 scale (E8M0)."""
    amax = jnp.max(jnp.abs(k), axis=-1, keepdims=True)
    scale = jnp.where(amax > 0, 2.0 ** jnp.ceil(jnp.log2(jnp.maximum(amax, 1e-30) / 6.0)), 1.0)
    q = nearest_e2m1(k / scale)
    return q * scale


def ste(value, quantizer):
    """straight-through estimator: 값은 양자화, 기울기는 그대로."""
    return value + jax.lax.stop_gradient(quantizer(value) - value)


# ----------------------------------------------------------------------------- 모델
def init_params(key, vocab_size: int):
    keys = iter(jax.random.split(key, 8 + len(LAYERS) * 16))
    normal = lambda shape, std=0.02: std * jax.random.normal(next(keys), shape, dtype=jnp.float32)
    params = {
        "tok_emb": normal((vocab_size, D)),
        "pos_emb": normal((CTX, D)),
        "lnf_g": jnp.ones(D),
        "lnf_b": jnp.zeros(D),
        "layers": [],
    }
    for spec in LAYERS:
        layer = {
            "ln1_g": jnp.ones(D),
            "ln1_b": jnp.zeros(D),
            "wq": normal((D, H * DC)),
            "bq": jnp.zeros(H * DC),
            "wk": normal((D, H * DC)),
            "bk": jnp.zeros(H * DC),
            "wv": normal((D, H * DC)),
            "bv": jnp.zeros(H * DC),
            "wo": normal((H * DC, D)),
            "bo": jnp.zeros(D),
            "ln2_g": jnp.ones(D),
            "ln2_b": jnp.zeros(D),
            "wr": normal((D, E)),
            "br": jnp.zeros(E),
            "rbias": jnp.zeros(E),
            "ws1": normal((D, FS)),
            "bs1": jnp.zeros(FS),
            "ws2": normal((FS, D)),
            "bs2": jnp.zeros(D),
            "we1": normal((E, D, FE)),
            "be1": jnp.zeros((E, FE)),
            "we2": normal((E, FE, D)),
            "be2": jnp.zeros((E, D)),
        }
        if spec["mode"] == "full":
            layer["wc"] = normal((spec["m"] * D, DC))
            layer["bc"] = jnp.zeros(DC)
            layer["wi"] = normal((DC, DI), 0.05)
        if spec["mode"] in ("full", "reindex"):
            layer["wqi"] = normal((D, HI * DI), 0.05)
            layer["bqi"] = jnp.zeros(HI * DI)
            layer["ww"] = normal((D, HI))
            layer["bw"] = jnp.ones(HI)
        params["layers"].append(layer)
    return params


def layer_norm(x, g, b):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS) * g + b


NEG = -1e9


def forward(params, ids, ds, sparse, quant, want_trace=False):
    """ids: [T] 정수, ds: decoder 가 처리를 시작하는 위치 (prefill 은 max(0, P - W), 학습은 0).

    logits [T, V], 인덱서 KL 손실, 전문가 부하 [L, E], (want_trace 면) 층별 중간값을 돌려준다.
    """
    T = ids.shape[0]
    pos = jnp.arange(T)
    qpos, spos = pos[:, None], pos[None, :]
    local_mask = (spos <= qpos) & (spos > qpos - W)
    x = params["tok_emb"][ids] + params["pos_emb"][:T]
    scale = 1.0 / np.sqrt(DC)
    kv = {}
    sel_store = {}
    pool = None
    kl_total = 0.0
    kl_count = 0.0
    loads = []
    trace = []
    for l, spec in enumerate(LAYERS):
        p = params["layers"][l]
        mode = spec["mode"]
        is_dec = l >= ENCODER_LAYERS
        start = ds if is_dec else 0
        hn = layer_norm(x, p["ln1_g"], p["ln1_b"])
        q = (hn @ p["wq"] + p["bq"]).reshape(T, H, DC)
        kw = (hn @ p["wk"] + p["bk"]).reshape(T, H, DC)
        vw = (hn @ p["wv"] + p["bv"]).reshape(T, H, DC)
        info = {}
        c = ki = vis = entry_pos = None
        n = 0
        if mode == "full":
            if spec["m"] == 2:
                n = (T + 1) // 2
                hpad = jnp.concatenate([hn, jnp.zeros((2 * n - T, D), hn.dtype)], 0)
                pairs = hpad.reshape(n, 2 * D)
                partial = jnp.concatenate([hpad[0::2], jnp.zeros((n, D), hn.dtype)], 1)
                c_full = pairs @ p["wc"] + p["bc"]
                c_part = partial @ p["wc"] + p["bc"]
                entry_pos = 2 * jnp.arange(n)
                use_part = entry_pos[None, :] == qpos  # 짝이 아직 안 온 엔트리는 [h; 0] 으로
                c = jnp.where(use_part[..., None], c_part[None], c_full[None])
            else:
                n = T
                c = jnp.broadcast_to((hn @ p["wc"] + p["bc"])[None], (T, n, DC))
                entry_pos = jnp.arange(n)
            vis = entry_pos[None, :] <= qpos
            if quant:
                c = ste(c, fp4_store)
            ki = c @ p["wi"]
            if quant:
                ki = ste(ki, mxfp4_store)
            kv[l] = (c, ki, vis, n, entry_pos)
        elif mode in ("reuse", "reindex"):
            c, ki, vis, n, entry_pos = kv[spec["src"]]
        sel = None
        rng = None
        s_masked = None
        if mode in ("full", "reindex"):
            qi = (hn @ p["wqi"] + p["bqi"]).reshape(T, HI, DI)
            wgt = hn @ p["ww"] + p["bw"]
            s = jax.nn.relu(jnp.einsum("thd,tnd->thn", qi, ki))
            s = jnp.einsum("th,thn->tn", wgt, s) - 1e-6 * jnp.arange(n)[None, :]  # 동점은 앞 엔트리 우선
            rng = vis if mode == "full" else (pool & vis)
            s_masked = jnp.where(rng, s, NEG)
            if mode == "full" and is_dec:
                nb = -(-n // POOL_BLOCK)
                s_pad = jnp.concatenate([s_masked, jnp.full((T, nb * POOL_BLOCK - n), NEG, s.dtype)], 1)
                block_max = s_pad.reshape(T, nb, POOL_BLOCK).max(-1)
                _, top_b = jax.lax.top_k(block_max, min(POOL_BLOCKS, nb))
                bmask = (jax.nn.one_hot(top_b, nb).sum(1) > 0) & (block_max > NEG / 2)
                pool = jnp.repeat(bmask, POOL_BLOCK, axis=1)[:, :n] & vis
                info["pool"] = pool
            if sparse:
                _, top_i = jax.lax.top_k(s_masked, min(K, n))
                sel = (jax.nn.one_hot(top_i, n).sum(1) > 0) & rng
            else:
                sel = rng
            sel_store[l] = sel
            info.update(qi=qi, wgt=wgt, scores=s_masked, rng=rng)
        elif mode == "reuse":
            sel = sel_store[spec["idx"]]
        lm = local_mask & (spos >= start)
        el = jnp.where(lm[:, None, :], jnp.einsum("thd,shd->ths", q, kw) * scale, NEG)
        if sel is not None:
            eg = jnp.where(sel[:, None, :], jnp.einsum("thd,tnd->thn", q, c) * scale, NEG)
            a = jax.nn.softmax(jnp.concatenate([eg, el], -1), -1)
            ag, al = a[..., :n], a[..., n:]
            o = jnp.einsum("thn,tnd->thd", ag, c) + jnp.einsum("ths,shd->thd", al, vw)
        else:
            a = jax.nn.softmax(el, -1)
            ag, al = None, a
            o = jnp.einsum("ths,shd->thd", al, vw)
        y = o.reshape(T, H * DC) @ p["wo"] + p["bo"]
        x1 = x + y
        if mode in ("full", "reindex"):
            target = jnp.where(rng, ag.mean(1), 0.0)
            mass = target.sum(-1, keepdims=True)
            target = jax.lax.stop_gradient(target / jnp.maximum(mass, 1e-9))
            logp = jax.nn.log_softmax(s_masked, -1)
            kl = jnp.where(rng, target * (jnp.log(jnp.maximum(target, 1e-9)) - logp), 0.0).sum(-1)
            valid = (mass[:, 0] > 1e-6) & (pos >= start)
            kl_total = kl_total + jnp.where(valid, kl, 0.0).sum()
            kl_count = kl_count + valid.sum()
        h2 = layer_norm(x1, p["ln2_g"], p["ln2_b"])
        r = jax.nn.sigmoid(h2 @ p["wr"] + p["br"])
        _, top_e = jax.lax.top_k(r + jax.lax.stop_gradient(p["rbias"]), ETOP)
        emask = jax.nn.one_hot(top_e, E).sum(1) > 0
        gates = jnp.where(emask, r, 0.0)
        gates = gates / gates.sum(-1, keepdims=True)
        shared = jax.nn.relu(h2 @ p["ws1"] + p["bs1"]) @ p["ws2"] + p["bs2"]
        hidden = jax.nn.relu(jnp.einsum("td,edf->tef", h2, p["we1"]) + p["be1"][None])
        outs = jnp.einsum("tef,efd->ted", hidden, p["we2"]) + p["be2"][None]
        routed = jnp.einsum("te,ted->td", gates, outs)
        x2 = x1 + shared + routed
        active = (pos >= start)[:, None]
        x = jnp.where(active, x2, x) if is_dec else x2
        loads.append(jnp.where(active, emask, 0).sum(0) / jnp.maximum(active.sum(), 1))
        if want_trace:
            info.update(
                sel=sel, c=c, ki=ki, ag=ag, al=al, gates=gates, emask=emask, r=r, x=x
            )
            trace.append(info)
    final = layer_norm(x, params["lnf_g"], params["lnf_b"])
    logits = final @ params["tok_emb"].T
    return logits, kl_total / jnp.maximum(kl_count, 1.0), jnp.stack(loads), trace


# 기준값 계산과 이어 쓰기는 문맥 길이로 채워(pad) 모양을 고정한 뒤 jit 한다. 뒤에 붙는 채움 토큰은
# 인과 마스크 때문에 앞 위치의 결과에 영향을 주지 않는다.
forward_jit = jax.jit(forward, static_argnames=("sparse", "quant", "want_trace"))


def padded(ids: list[int]) -> jnp.ndarray:
    return jnp.asarray(ids + [0] * (CTX - len(ids)), dtype=jnp.int32)


def loss_fn(params, batch_x, batch_y, sparse, quant):
    def one(ids, targets):
        logits, kl, loads, _ = forward(params, ids, 0, sparse, quant)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        picked = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
        return -picked.mean(), kl, loads

    ce, kl, loads = jax.vmap(one)(batch_x, batch_y)
    return ce.mean() + KL_WEIGHT * kl.mean(), (ce.mean(), kl.mean(), loads.mean(0))


# ----------------------------------------------------------------------------- 학습
def train(corpus_ids: np.ndarray, vocab_size: int):
    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, vocab_size)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    beta1, beta2, eps = 0.9, 0.99, 1e-8

    @functools.partial(jax.jit, static_argnames=("sparse", "quant"))
    def step(params, m, v, batch_x, batch_y, t, sparse, quant):
        (loss, (ce, kl, loads)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch_x, batch_y, sparse, quant
        )
        norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
        grads = jax.tree_util.tree_map(lambda g: g * jnp.minimum(1.0, CLIP / (norm + 1e-6)), grads)
        m = jax.tree_util.tree_map(lambda a, g: beta1 * a + (1 - beta1) * g, m, grads)
        v = jax.tree_util.tree_map(lambda a, g: beta2 * a + (1 - beta2) * g * g, v, grads)
        lr_t = LR * jnp.sqrt(1 - beta2**t) / (1 - beta1**t)
        params = jax.tree_util.tree_map(lambda p, a, b: p - lr_t * a / (jnp.sqrt(b) + eps), params, m, v)
        # MoE 부하 균형: 많이 뽑힌 전문가의 편향은 내리고 적게 뽑힌 전문가는 올린다 (aux-loss-free).
        for l, layer in enumerate(params["layers"]):
            layer["rbias"] = layer["rbias"] + BIAS_RATE * jnp.sign(ETOP / E - loads[l])
        return params, m, v, loss, ce, kl, loads

    rng = np.random.default_rng(SEED)
    history = []
    started = time.time()
    for t in range(1, STEPS + 1):
        sparse = t > STEPS * SPARSE_FROM
        quant = t > STEPS * QUANT_FROM
        starts = rng.integers(0, len(corpus_ids) - CTX - 1, size=BATCH)
        batch_x = np.stack([corpus_ids[s : s + CTX] for s in starts])
        batch_y = np.stack([corpus_ids[s + 1 : s + CTX + 1] for s in starts])
        params, m, v, loss, ce, kl, loads = step(
            params, m, v, jnp.asarray(batch_x), jnp.asarray(batch_y), t, sparse=sparse, quant=quant
        )
        if t % 250 == 0 or t == 1 or t == STEPS:
            entry = {
                "step": t,
                "loss": round(float(ce), 4),
                "indexerKl": round(float(kl), 4),
                "sparse": bool(sparse),
                "quant": bool(quant),
            }
            history.append(entry)
            spread = float(jnp.max(loads)) / max(float(jnp.min(loads)), 1e-6)
            print(
                f"step {t:5d}  ce {float(ce):.4f}  kl {float(kl):.4f}  sparse={int(sparse)} quant={int(quant)}"
                f"  expert load max/min {spread:.1f}  ({time.time() - started:.0f}s)"
            )
    return params, history


# ----------------------------------------------------------------------------- 내보내기
def to_fp16_roundtrip(params):
    return jax.tree_util.tree_map(
        lambda p: jnp.asarray(np.asarray(p, dtype=np.float16).astype(np.float64)), params
    )


def encode_array(array) -> dict:
    arr = np.asarray(array, dtype=np.float16)
    return {"shape": list(arr.shape), "dtype": "f16", "data": base64.b64encode(arr.tobytes()).decode("ascii")}


def count_params(params) -> int:
    return int(sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params)))


def decoder_start(prompt_len: int) -> int:
    return max(0, prompt_len - W)


def generate(params, vocab, stoi, prompt: str, steps: int) -> str:
    ids = [stoi.get(ch, 0) for ch in prompt]
    ds = decoder_start(len(ids))
    text = prompt
    for _ in range(steps):
        if len(ids) >= CTX:
            break
        logits, _, _, _ = forward_jit(params, padded(ids), ds, True, True, False)
        nxt = int(jnp.argmax(logits[len(ids) - 1].at[0].set(-jnp.inf)))
        ids.append(nxt)
        text += vocab[nxt]
    return text


def rounded(array, digits=5):
    return np.round(np.asarray(array, dtype=np.float64), digits).tolist()


def reference_for(params, vocab, stoi, prompt: str) -> dict:
    ids = [stoi.get(ch, 0) for ch in prompt]
    T = len(ids)
    ds = decoder_start(T)
    logits, _, _, trace = forward_jit(params, padded(ids), ds, True, True, True)
    t = T - 1
    probs = np.asarray(jax.nn.softmax(logits[t]))
    top = np.argsort(-probs)[:5]
    layers = {}
    for l in (0, 1, 3, 5, 7, 9):
        info = trace[l]
        spec = LAYERS[l]
        m = LAYERS[spec.get("src", l)].get("m")
        n_vis = T if m == 1 else (T + 1) // 2  # TS 구현이 만드는 엔트리 수 (채움 토큰 제외)
        entry = {
            "experts": [int(i) for i in np.flatnonzero(np.asarray(info["emask"][t]))],
            "gates": rounded(np.asarray(info["gates"][t])[np.asarray(info["emask"][t])]),
            "local_h0": rounded(info["al"][t, 0][:T]),
        }
        if info["sel"] is not None:
            entry["selected"] = [int(i) for i in np.flatnonzero(np.asarray(info["sel"][t]))]
            entry["global_h0"] = rounded(info["ag"][t, 0][:n_vis])
        if "scores" in info:
            entry["scores"] = rounded(
                np.where(np.asarray(info["rng"][t]), np.asarray(info["scores"][t]), 0)[:n_vis]
            )
        if "pool" in info:
            entry["pool"] = [int(i) for i in np.flatnonzero(np.asarray(info["pool"][t]))]
        if l in (1, 5):
            entry["entry_last"] = rounded(info["c"][t, int(np.flatnonzero(np.asarray(info["sel"][t]))[-1])])
        layers[str(l)] = entry
    return {
        "prompt": prompt,
        "ids": [int(i) for i in ids],
        "decoderStart": ds,
        "top5": [{"token": vocab[int(i)], "probability": round(float(probs[i]), 5)} for i in top],
        "layers": layers,
        "greedy": generate(params, vocab, stoi, prompt, 12),
    }


def main() -> int:
    corpus = build_corpus()
    chars = sorted(set(corpus))
    vocab = [UNK] + chars
    stoi = {ch: i for i, ch in enumerate(vocab)}
    corpus_ids = np.asarray([stoi[ch] for ch in corpus], dtype=np.int32)
    print(f"말뭉치 {len(corpus):,} 글자, 어휘 {len(vocab)} 개, jax {jax.__version__}, steps {STEPS}")

    params, history = train(corpus_ids, len(vocab))
    print("파라미터 수:", count_params(params))

    # 기준값은 fp16 으로 반올림한 가중치로 float64 에서 계산한다 (브라우저의 계산 정밀도와 맞춤).
    jax.config.update("jax_enable_x64", True)
    params = to_fp16_roundtrip(params)
    references = [reference_for(params, vocab, stoi, prompt) for prompt in REFERENCE_PROMPTS]
    for ref in references:
        print(f"이어 쓰기 예: {ref['greedy']!r}")

    weights = {
        "tok_emb": encode_array(params["tok_emb"]),
        "pos_emb": encode_array(params["pos_emb"]),
        "lnf_g": encode_array(params["lnf_g"]),
        "lnf_b": encode_array(params["lnf_b"]),
        "layers": [{name: encode_array(value) for name, value in layer.items()} for layer in params["layers"]],
    }
    payload = {
        "schemaVersion": 1,
        "description": "Character-level model with the DeepSeek-V4.1-Flash attention/KV structure (CED, CSA2 modes, indexer Top-K, SWA merge, FP4 KV, MoE) at toy scale, trained by scripts/train_tiny_dsv41.py.",
        "config": CONFIG,
        "vocab": vocab,
        "unkToken": UNK,
        "training": {
            "steps": STEPS,
            "batch": BATCH,
            "learningRate": LR,
            "seed": SEED,
            "sparseFromStep": int(STEPS * SPARSE_FROM),
            "quantFromStep": int(STEPS * QUANT_FROM),
            "indexerKlWeight": KL_WEIGHT,
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
