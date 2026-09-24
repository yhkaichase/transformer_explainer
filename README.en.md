# Transformers, explained simply

_한국어: [README.md](README.md)_

An interactive web page that explains the **Transformer**, the architecture behind AI such as ChatGPT, at the level of junior engineers and executives.

- **Executive** mode: analogies and outcomes, no equations.
- **Engineer** mode: the same page gains equations and structural detail.
- **한국어 / English**: switch with the toggle at the top, or open the English version directly with `?lang=en`.
- **Try it live**: at the top of the page, edit the text and a small transformer embedded in the page (character-level, 3 layers, about 115k parameters) computes tokens → embeddings → attention → next-character probabilities in the browser. No server or internet needed.

Where the reference project [Transformer Explainer](https://github.com/poloclub/transformer-explainer) (Georgia Tech Polo Club) runs a real GPT-2 in the browser and shows its internals, this project aims one step simpler. The content plan is in [docs/plan.md](docs/plan.md) (Korean).

## Getting started

Node.js 22.12 or later is required (see `.nvmrc`).

```bash
npm install
npm run dev        # http://localhost:5173
```

## Simulator (transformer-simulator.html)

A second page for looking at the structure without explanatory text: `http://localhost:5173/simulator.html` on the dev server, `dist/simulator.html` after a build, or the single file `dist/transformer-simulator.html`.
It lays out input → embedding → (layer) LayerNorm → W_Q·W_K·W_V → per-head attention → W_O → residual → LayerNorm → W₁ → ReLU → W₂ → residual → final LayerNorm → logits → softmax → next token horizontally on one screen, drawing the real weight matrices and activations as heatmaps. An output card at the top right updates the generated text and next-token probabilities immediately, a stage stepper (← → keys, autoplay at 0.5×–4× speed with an option to decode a token after the last stage) walks through the flow, and Prefill (all prompt tokens in parallel) is distinguished from Decode (one new token, cached K·V reused).

## DeepSeek-V4.1-Flash simulator (DCv4.1-simulator.html)

A third page in which a toy-scale model with the attention/KV-cache structure of DeepSeek-V4.1-Flash, as described by `DeepseekV4.1_manual.docx` in this repository (2026-09-21), computes for real: `http://localhost:5173/DCv4.1-simulator.html` on the dev server, `dist/DCv4.1-simulator.html` after a build, and the single file of the same name.
It shows CED (a causal encoder of 5 layers plus a 5-layer decoder whose global KV is projected from the encoder output; in prefill the decoder processes only the last window of tokens), the fixed CSA2 mode of every layer (SWA-only / Full / Reuse / Reindex), global KV entries in which one latent serves as both key and value (the encoder merges two adjacent tokens into one entry), FP4 (E2M1) storage with E4M3 scales and MXFP4 indexer keys, indexer scores → candidate pool (block max) → Top-K, the merged softmax over global and SWA logits, MoE (shared 1 + routed 2 of 8), and a KV-cache card that derives bytes per token the same way the manual derives its 890 B/token and counts the entries read for the last token. Picking a layer in the layer-map panel separates what that layer recomputes from what it inherits. The differences from the real model (size, character tokens, points the manual itself assumes) are listed at the bottom of the page.

## Viewing without Node.js

The page is a static site: a browser is all you need to view it. Node.js is required only to build it.

- **Open a single file**: `npm run build && npm run build:single` produces `dist/transformer-explain.html` (explanation page), `dist/transformer-simulator.html` (transformer simulator), and `dist/DCv4.1-simulator.html` (DeepSeek-V4.1-Flash simulator) with all CSS and JS inlined. It opens by double-click and can be shared over an intranet or by email. Append `?lang=en` to the file address for English.
- **Download the build**: every push makes the CI workflow upload a `site` artifact (the build folder plus the single file). Find it under the repository's Actions tab → the run → Artifacts.
- **Share a link**: enable GitHub Pages as described under "Deployment"; the same address with `/transformer-explain.html` appended serves the offline single file.

## Scripts

| Command           | What it does                                               |
| ----------------- | ---------------------------------------------------------- |
| `npm run dev`     | Development server                                         |
| `npm run build`   | Typecheck, then build `dist/`                              |
| `npm run preview` | Preview the build                                          |
| `npm run check`   | Lint, format check, typecheck, unit tests, build, in order |
| `npm run lint`    | oxlint                                                     |
| `npm run format`  | Format everything with Prettier                            |
| `npm test`        | Vitest unit tests                                          |
| `npm run e2e`     | Playwright browser tests against the build                 |

Install the browser once before the first `npm run e2e`:

```bash
npx playwright install chromium
```

## Layout

```
src/content/     Text data. ko.ts / en.ts hold each language's sections, references, and UI strings
src/components/  UI pieces and interactive demos
src/lib/         Pure functions such as softmax and attention, plus the embedded-model inference (tinyTransformer.ts, tinyDsv41.ts), with tests
src/simulator/   Transformer simulator page and the canvas heatmap / display parts shared by both simulators
src/dsv41/       DeepSeek-V4.1-Flash simulator page
src/state/       Level-of-detail (executive/engineer) and language state
e2e/             Playwright smoke tests
docs/plan.md     Content plan and open questions (Korean)
```

Both languages must always be edited together: a change to `ko.ts` needs the matching change in `en.ts`, and a unit test checks that paragraph and key-term counts stay equal.

## Retraining the embedded small model

The model behind "Try it live" is trained by `scripts/train_tiny_model.py` on this page's Korean and English text. Retrain it after large text changes or to change the model size (a few minutes on CPU).

```bash
pip install -r scripts/requirements-train.txt
python scripts/train_tiny_model.py
npm test   # checks that the TypeScript implementation matches the new reference values
```

The DeepSeek-V4.1-Flash simulator's model is trained by `scripts/train_tiny_dsv41.py` on the same text plus the text of `DeepseekV4.1_manual.docx` (about 40 minutes on CPU; output `src/data/model/tiny-dsv41.json`). Sparse Top-K selection and FP4 QAT are switched on in the later part of training.

```bash
python scripts/train_tiny_dsv41.py
npm test   # tinyDsv41.test.ts compares against the reference values
```

## Generating the real attention values

"Attention in a real model" in section 4 and the head comparison in section 5 appear only when real model output files exist. Otherwise a notice is shown.

```bash
pip install -r scripts/requirements.txt
python scripts/precompute_attention.py --locale en --model gpt2
python scripts/precompute_attention.py --locale ko --model <a Korean GPT-2-style model id>
```

Commit the generated `src/data/attention/*.json`. Details are in [scripts/README.md](scripts/README.md) (Korean).

## Deployment

Pushing to `main` deploys to GitHub Pages through `.github/workflows/deploy.yml`.
Set **Settings → Pages → Build and deployment → Source** to **GitHub Actions** once in the repository.
The site is served at `https://<user>.github.io/transformer_explainer/`; append `?lang=en` for the English version.

## References

- Vaswani et al., "Attention Is All You Need" (2017) — https://arxiv.org/abs/1706.03762
- Transformer Explainer — https://github.com/poloclub/transformer-explainer
