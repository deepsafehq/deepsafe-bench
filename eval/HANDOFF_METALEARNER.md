---
name: Handoff — meta-learner retraining on MacOS
description: Continuation prompt for retraining ensemble meta-learners on M2 Max MacOS after A100 eval completed
type: project
originSessionId: 84c551b1-cf98-4291-8c9d-9cca99d68266
---
## Continuation prompt for new session on MacOS M2 Max

Copy-paste this to start a new session:

---

I'm continuing work on the DeepSafe ensemble meta-learner retraining. The A100 GPU work is done — we ran inference on 15,500 samples (medium eval dataset) with all 23 models and saved per-file predictions.

**What's done (on A100):**
1. Built medium eval dataset: 10K images, 3.5K audio, 2K video from `/data/master_eval_full/`
2. Ran all 23 models on 15,499 samples (~7 hours), results at:
   - `eval/results/monolith_eval_20260410_102244.json` — aggregate metrics
   - `eval/results/monolith_eval_20260410_102244_predictions.json` — per-file raw predictions (13MB, per-sample model probabilities)
3. Created `eval/ensemble_experiment.py` — exhaustive meta-learner search script

**Medium dataset results (current ensemble is BAD):**
- Image ensemble: AUC=0.947 (down from 0.997 on 50-sample small set — RF was overfitting)
- Audio ensemble: AUC=0.829 (down from 0.898 — RF overfitting on 50 samples)
- Video ensemble: AUC=0.669, F1=0.281 (terrible — RF on 50 samples was useless)

**Per-model AUCs on medium dataset:**
- Image: FSD=0.896, CO-SPY=0.868, Effort=0.835, AIDE=0.748, Universal=0.671, NPR=0.637, Yermandy=0.591
- Audio: Nes2Net=0.841, ShiftySpeech=0.735, SafeEar=0.604
- Video: NPR-Video=0.798(!), SBI=0.676, LipFD=0.626, MINTIME=0.581, DFD-FCG=0.554, UnivFD-Video=0.469, PwTF-DVD=0.417(anti!), RECCE=0.332(anti!), FakeSTormer=insufficient data

**What to do now (all CPU, no GPU needed):**

1. **Run `eval/ensemble_experiment.py`** on the predictions file — tries 400+ algorithm/feature/hyperparameter combos per modality:
   ```bash
   python eval/ensemble_experiment.py --predictions eval/results/monolith_eval_20260410_102244_predictions.json --modality video --output eval/results/ensemble_exp_video.json
   python eval/ensemble_experiment.py --predictions eval/results/monolith_eval_20260410_102244_predictions.json --modality audio --output eval/results/ensemble_exp_audio.json
   python eval/ensemble_experiment.py --predictions eval/results/monolith_eval_20260410_102244_predictions.json --modality images --output eval/results/ensemble_exp_images.json
   ```
   Run video first (2K samples, fastest). Then audio (3.5K). Then images (10K, slowest).

2. **Based on results, retrain the production meta-learners** at `models/ensemble/artifacts/`:
   - Replace RF with best algorithm per modality (likely LR for audio/video, RF or XGBoost for images)
   - Drop anti-correlated models: PwTF-DVD (AUC=0.417), RECCE (0.332) from video
   - Use optimal threshold per modality (not 0.5 for all)
   - Save new `{modality}_meta_learner.pkl`, `{modality}_scaler.pkl`, `{modality}_config.json`

3. **Update `apps/gateway/ensemble.py`** with new model orders, AUC weights, and thresholds

4. **Implement provenance tiered overrides** (replace additive boost):
   - C2PA AI manifest >= 0.80 → override to 0.95
   - Watermark >= 0.70 → override to 0.85

5. **Add cross-modal fusion** for video files with audio tracks

**Key research findings (see memory: project_metalearner_research.md):**
- P0: Use LogisticRegression(C=0.01, L2) for audio/video instead of RF
- P0: Drop video models with AUC < 0.55 (FakeSTormer, RECCE, PwTF-DVD, UnivFD-Video)
- P1: Provenance tiered overrides instead of additive boost
- P1: Cross-modal fusion max(video_score, audio_score) for video+audio files
- LR outputs are inherently calibrated — may not need separate Platt scaling

**Dependencies:** `pip install scikit-learn pandas numpy xgboost lightgbm` (no PyTorch/CUDA needed)

---
