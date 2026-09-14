# DeepSafe

Benchmark, model zoo, and adaptation toolkit for deepfake detection. 24 models
(19 detection, 5 provenance) in one process.

## Read First

- `CONTRIBUTING.md` for contribution rules and licensing terms
- `docs/ARCHITECTURE.md` for the system design
- `docs/styleguides/` for code style (Google Python / TypeScript style guides)

## Key Rules

- **TDD:** Write the failing test first. Do not skip the red phase.
- **Package managers:** `uv` for Python, `pnpm` for Node. Never `pip` or `npm`.
- **Non-interactive:** Prefer non-interactive commands. Pass `CI=true` for
  watch-mode tools.
- **No secrets:** Never commit credentials. CI runs `gitleaks`.
- **No large files:** Nothing over 5 MB, no Git LFS. Assets live on HuggingFace.
- **Commit format:** `<type>(<scope>): <description>`, with `-s` to sign off.

## Honesty Rules

These are project invariants, not preferences.

- `eval` must always report in-distribution and held-out-generator performance
  separately. Never collapse them into one number.
- Never state an accuracy claim without a reproducible eval run behind it.
- Never describe this project as "open source". The license is non-commercial,
  which makes it source-available. Say "free for non-commercial use".
- Do not claim a language count for the eval tier. The manifest has no
  `language` field yet.

## Common Commands

```bash
# Inference server
cd apps/inference && PYTHONPATH=.:../../packages/shared python server.py
python smoke_test.py

# Gateway
cd apps/gateway && PYTHONPATH=.:../../packages/shared uv run uvicorn main:app --reload
PYTHONPATH=.:../../packages/shared uv run pytest tests/

# Frontend / docs
cd apps/web && pnpm dev
cd apps/docs && pnpm dev

# Benchmark
deepsafe eval --model <detector.py> --tier small
```

## Architecture

- `src/deepsafe/` - the `detect` / `eval` / `fit` package
- `apps/inference/` - inference server, 24 predictor wrappers
- `apps/gateway/` - FastAPI gateway (auth and API keys optional, off by default)
- `apps/web/` - landing page and static benchmark explorer
- `apps/docs/` - Fumadocs documentation
- `eval/` - benchmark harness
- `packages/shared/` - shared Python (ensemble, device utils, constants)
- `infrastructure/` - systemd units, nginx config, self-hosting scripts
