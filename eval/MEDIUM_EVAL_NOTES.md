---
name: Medium eval dataset and inference run
description: Medium eval dataset built from full source data, inference running on 15.5K samples with 23 models on A100 80GB
type: project
originSessionId: 84c551b1-cf98-4291-8c9d-9cca99d68266
---
Medium eval dataset built 2026-04-10 from full source at `/deepsafe/data/master_eval_full/master_eval_full/`.

**Why:** Need 15.5K samples (vs 50 in small) to properly train meta-learners. Small dataset RF meta-learners are overfitting (50 samples, 8 features for video = EPV 2.5).

**How to apply:** Use prediction results for meta-learner retraining. Never retrain on the small dataset again.

## Dataset location
- Built by: `PYTHONPATH=eval:$PYTHONPATH python -m eval.build_master_eval --root /deepsafe/data/master_eval_full/master_eval_full --tier medium --seed 42`
- Output: `/deepsafe/data/master_eval_full/master_eval_full/master_eval/` (8.6GB)
- Symlink: `/deepsafe/dataset/master_eval` → above path
- Manifest: `metadata.json` with 15,500 entries

## Dataset composition
| Modality | Real | Fake | Total | Generators |
|----------|------|------|-------|------------|
| Images | 5,000 | 5,000 | 10,000 | 7 real sources, 47 fake generators |
| Audio | 1,000 | 2,500 | 3,500 | 10 real sources, 325 fake generators |
| Video | 1,000 | 1,000 | 2,000 | 4 real sources, 19 fake generators |

## Eval run
- Script: `python eval/run_monolith_eval.py --workers 4`
- Server: monolith with 23 models, A100 80GB, 18.5GB VRAM
- Output files (in `eval/results/`):
  - `monolith_eval_YYYYMMDD_HHMMSS.json` — aggregate metrics
  - `monolith_eval_YYYYMMDD_HHMMSS_predictions.json` — per-file raw predictions (sample_id, label, per-model probabilities)
- The predictions JSON is the key artifact for meta-learner training
- Previous small eval: `eval/results/monolith_eval_20260409_154247.json` (198 samples only)

## Full dataset source
- `/deepsafe/data/master_eval_full/master_eval_full/` (25GB) — contains `images/`, `audio/`, `video/` with all generators
- `/deepsafe/data/master_eval_full.zip` (26.7GB) — backup archive
