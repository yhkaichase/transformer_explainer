# Transformers, explained simply

_한국어: [README.md](README.md)_

An interactive web page that explains the **Transformer**, the architecture behind AI such as ChatGPT, at the level of junior engineers and executives.

- **Executive** mode: analogies and outcomes, no equations.
- **Engineer** mode: the same page gains equations and structural detail.
- **한국어 / English**: switch with the toggle at the top, or open the English version directly with `?lang=en`.

Where the reference project [Transformer Explainer](https://github.com/poloclub/transformer-explainer) (Georgia Tech Polo Club) runs a real GPT-2 in the browser and shows its internals, this project aims one step simpler. The content plan is in [docs/plan.md](docs/plan.md) (Korean).

## Getting started

Node.js 22.12 or later is required (see `.nvmrc`).

```bash
npm install
npm run dev        # http://localhost:5173
```

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
src/lib/         Pure functions such as softmax and attention, with tests
src/state/       Level-of-detail (executive/engineer) and language state
e2e/             Playwright smoke tests
docs/plan.md     Content plan and open questions (Korean)
```

Both languages must always be edited together: a change to `ko.ts` needs the matching change in `en.ts`, and a unit test checks that paragraph and key-term counts stay equal.

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
The site is served at `https://<user>.github.io/transformer_explain/`; append `?lang=en` for the English version.

## References

- Vaswani et al., "Attention Is All You Need" (2017) — https://arxiv.org/abs/1706.03762
- Transformer Explainer — https://github.com/poloclub/transformer-explainer
