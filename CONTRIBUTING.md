# Contributing

## Licensing of Contributions

By submitting a contribution you agree that it is licensed under
[PolyForm Noncommercial 1.0.0](LICENSE), and you grant the maintainer a
perpetual, irrevocable license to use, modify, and relicense your contribution.
The relicensing grant exists so the project can move to a more permissive
license later without tracking down every contributor. Without it, that door
closes permanently.

Sign off your commits to confirm you have the right to submit the work:

```bash
git commit -s -m "feat(models): Add adapter for X"
```

## Good First Contributions

**Add a `TrainableAdapter` for a model.** Only 7 of the 19 detection models
support Tier 3 fine-tuning today. The remaining ones raise `NotImplementedError`
with a pointer to upstream training instructions. Implementing one is
self-contained and needs no changes elsewhere. See `src/deepsafe/fit/`.

**Submit a leaderboard entry.** Any detector implementing
`predict(path: Path) -> float` can be scored:

```bash
deepsafe eval --model your_detector.py --tier small
```

Open a PR with the generated report card.

**Add a generator to the eval set.** New generative models ship constantly and
the benchmark is only useful if it keeps up. Add samples plus a manifest entry.

## Before your first push

Enable the pre-push secret scan:

```bash
git config core.hooksPath .githooks
brew install gitleaks    # or: https://github.com/gitleaks/gitleaks
```

CI scans too, but CI runs *after* the push. This repository is public, so a
credential that reaches the remote is compromised the moment it lands, even if
you delete it seconds later. The hook is the gate that actually protects you.

## Rules

- Follow the style guides in `docs/styleguides/`.
- Write the failing test first. This project uses TDD; do not skip the red phase.
- Use `uv` for Python and `pnpm` for Node. Never `pip` or `npm` directly.
- No secrets, ever. CI runs `gitleaks` and fails the build on any hit.
- No file over 5 MB and no Git LFS. Large assets belong on HuggingFace.

## What We Will Not Merge

- Changes that make the benchmark report only in-distribution performance.
  Separating in-distribution from held-out-generator results is the point of
  this project, not a display option.
- Accuracy claims without a reproducible eval run behind them.
