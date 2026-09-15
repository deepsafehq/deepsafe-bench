# Security Policy

## Reporting a Vulnerability

Report security issues privately via GitHub's [private vulnerability
reporting](https://github.com/deepsafehq/deepsafe-bench/security/advisories/new).
Please do not open a public issue for a security problem.

Expect an acknowledgement within 7 days and an assessment within 30 days.

## If you find a credential in this repository

Report it privately using the link above, do not open a public issue, and we
will rotate it. Every push is scanned by `.githooks/pre-push` and by CI, but
neither is a guarantee.

## Scope

In scope:

- The inference server (`apps/inference/`) and API gateway (`apps/gateway/`)
- The `deepsafe` package (`src/deepsafe/`)
- Authentication and API key handling
- Any path where untrusted media reaches a model

Out of scope:

- Vulnerabilities in third-party model code mirrored from upstream research
  repositories. Report those to the original authors; see
  `THIRD_PARTY_NOTICES.md` for upstream URLs. We will mirror a fix once
  upstream publishes one.
- Detection accuracy. A model failing to catch a deepfake is a research
  limitation, not a vulnerability. See `BENCHMARK.md` for measured limits.

## A Note on Threat Model

This project processes untrusted media files with a large stack of third-party
research code that was never written with adversarial input in mind. Treat the
inference server as untrusted-input-facing and sandbox it accordingly. Do not
expose it directly to the public internet without isolation.
