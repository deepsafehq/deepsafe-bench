---
name: Meta-learner research findings and strategy
description: Literature-backed ensemble strategy decisions — LR over RF for small data, provenance tiered overrides, cross-modal fusion
type: project
originSessionId: 84c551b1-cf98-4291-8c9d-9cca99d68266
---
Comprehensive literature study completed 2026-04-10 on ensemble meta-learner strategies.

**Why:** Current RF meta-learners on 50 samples are overfitting. Need evidence-based strategy for retraining on medium dataset.

**How to apply:** When retraining meta-learners after medium eval completes, follow priority order below.

## Key findings

### P0: Switch audio/video from RF to Logistic Regression
- RF(300 trees, max_depth=8) on 50 samples has hundreds of effective parameters fitting noise
- EPV (events per variable): video=2.5 (min 10), audio=6.7 (marginal)
- Literature consensus (Wolpert 1992, Super Learner, Niculescu-Mizil 2005): LR with L2 is standard for meta-learners with <100 samples
- Use `LogisticRegression(C=0.01, penalty='l2')` — only K+1 parameters
- Image RF (1465 samples, 7 features) is fine — keep it

### P0: Drop weak video features
- FakeSTormer (AUC=0.50), RECCE (0.527), LipFD (0.545) add noise not signal
- Reduce from 8→5 features: MINTIME, SBI, NPR-Video, PwTF-DVD, DFD-FCG

### P0: Use RepeatedStratifiedKFold or LOOCV for small datasets
- 5-fold CV on 50 samples = 10 samples/fold = high-variance AUC
- `RepeatedStratifiedKFold(n_splits=5, n_repeats=10)` for stability

### P1: Provenance — tiered override instead of additive boost
- Current boost formula: `(max_prov - 0.5) × 0.10 × (1 - score)` = max ~2% shift — too weak for cryptographic evidence
- C2PA AI manifest ≥ 0.80 → override to 0.95
- Watermark (AudioSeal/VideoSeal) ≥ 0.70 → override to 0.85
- Valid C2PA chain + no AI markers → modest -0.05 downward
- Phase 2: add provenance as features IN meta-learner

### P1: Cross-modal fusion for video+audio
- Video files have audio tracks — currently only video models run
- Audio AUC=0.898 >> Video AUC=0.773
- Simple fusion: `max(video_score, audio_score)` for high recall
- Or AUC-weighted average for balanced precision/recall

### P2: Calibration
- Image: Platt scaling works (ECE 0.018→0.014)
- Audio/video: LR outputs are already calibrated (optimizes log-loss) — may not need separate calibrator
- Never use isotonic regression with <1000 samples

### P2: Pattern-specific video sub-ensembles
- Pre-train 2 meta-learners: full (all models) vs no-face (NPR-Video, PwTF-DVD, UnivFD-Video only)
- Select at inference based on which models returned valid results

## Key papers referenced
- Wolpert 1992 (Stacked Generalization), Super Learner (van der Laan)
- Niculescu-Mizil & Caruana ICML 2005 (RF calibration)
- AVFF CVPR 2024 (audio-visual fusion, 98.6% on FakeAVCeleb)
- DF40 NeurIPS 2024 (40 generator benchmark)
- NSA/CISA Content Credentials Jan 2025
- Nemecek et al. 2026 (Authenticated Contradictions — C2PA+watermark conflict matrix)
