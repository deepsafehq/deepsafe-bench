# Session Learnings — A100 Eval Run (2026-04-10)

Comprehensive findings from running 23 models on 15,499 samples (medium eval dataset)
on A100 80GB. This document captures everything learned that isn't obvious from the
code or metrics JSON alone.

## 1. The Small Dataset Was Lying To Us

The previous meta-learners were trained on 50 samples. The medium dataset (15.5K) revealed
massive overfitting:

| Modality | Small Eval AUC | Medium Eval AUC | Delta | Cause |
|----------|---------------|-----------------|-------|-------|
| Image    | 0.997         | 0.947           | -0.050 | RF memorized 50 samples |
| Audio    | 0.898         | 0.829           | -0.069 | RF memorized 50 samples |
| Video    | 0.773         | 0.669           | -0.104 | RF completely useless on 50 samples |

**Lesson:** Never trust metrics from <200 samples. The RF meta-learners were fitting noise.
The video ensemble (F1=0.281 on medium) is barely better than random.

## 2. Model Rankings Changed Dramatically

### Image Models (on 15.5K vs 50 samples)
| Model | Small AUC | Medium AUC | Change |
|-------|-----------|------------|--------|
| FSD | 0.858 | **0.896** | +0.038 (best overall) |
| CO-SPY | 0.843 | **0.868** | +0.025 |
| Effort | 0.789 | **0.835** | +0.046 |
| AIDE | 0.765 | 0.748 | -0.017 |
| Universal | 0.583 | 0.671 | +0.088 |
| NPR | 0.576 | 0.637 | +0.061 |
| Yermandy | 0.549 | 0.591 | +0.042 |

**Lesson:** FSD is the single best image model, not CO-SPY as the small eval suggested.

### Audio Models
| Model | Small AUC | Medium AUC | Change |
|-------|-----------|------------|--------|
| Nes2Net | 0.948 | **0.841** | -0.107 |
| ShiftySpeech | 0.908 | 0.735 | -0.173 |
| SafeEar | 0.627 | 0.604 | -0.023 |

**Lesson:** All audio models degraded significantly on diverse data. ShiftySpeech dropped
most — the small eval was biased toward English/asvspoof generators where it excels. The
medium dataset has 325 generators across dozens of languages.

### Video Models  
| Model | Small AUC | Medium AUC | Change | Verdict |
|-------|-----------|------------|--------|---------|
| NPR-Video | 0.612 | **0.798** | +0.186 | BEST video model! |
| SBI | 0.638 | 0.676 | +0.038 | Solid |
| LipFD | 0.545 | 0.626 | +0.081 | Improved |
| MINTIME | 0.722 | 0.581 | -0.141 | Overstated |
| DFD-FCG | 0.582 | 0.554 | -0.028 | Marginal |
| UnivFD-Video | 0.562 | 0.469 | -0.093 | Anti-correlated |
| PwTF-DVD | 0.589 | **0.417** | -0.172 | **ANTI-CORRELATED** |
| RECCE | 0.527 | **0.332** | -0.195 | **STRONGLY ANTI-CORRELATED** |
| FakeSTormer | 0.500 | insufficient | — | Crashes on most videos |

**Critical lesson:** NPR-Video (full-frame, no face detection) is BY FAR the best video 
model at AUC=0.798. The face-dependent models (MINTIME, DFD-FCG) are much weaker than 
the small eval suggested. PwTF-DVD and RECCE are **actively harmful** — they predict the
WRONG direction. They must be excluded from the ensemble.

## 3. Anti-Correlated Models Are Dangerous

Three models have AUC < 0.5 on medium data, meaning they are anti-correlated:
- **RECCE**: AUC=0.332 (predicts real as fake, fake as real)
- **PwTF-DVD**: AUC=0.417 (same problem)
- **UnivFD-Video**: AUC=0.469 (borderline)

Including these in a weighted average or RF drags ensemble performance DOWN. The current
production ensemble includes all of them with positive weights, which explains the
terrible video F1=0.281.

**Action:** Remove PwTF-DVD and RECCE from the video ensemble entirely. Consider removing
UnivFD-Video or using it inverted (1 - prob).

## 4. Generator-Specific Blind Spots

### Image: Generators We Completely Miss (acc < 20%)
| Generator | Accuracy | Avg Score | Samples |
|-----------|----------|-----------|---------|
| Ideogram 2.0 | 15.5% | 0.255 | 97 |
| Grok 2 Image | 17.7% | 0.196 | 113 |
| Recraft V3 | 18.6% | 0.266 | 113 |
| Fairness Eval | 24.6% | 0.312 | 114 |
| Aurora | 39.1% | 0.446 | 92 |
| SD 1.5 DreamShaper | 40.7% | 0.415 | 113 |
| AI Upscaled | 42.1% | 0.454 | 114 |

**Lesson:** Newest generators (Grok, Ideogram, Recraft V3) completely evade all 7 image
models. These are the frontier we need to address. Adversarial compression (Instagram,
WhatsApp) also degrades detection significantly.

### Image: Generators We Nail (acc = 100%)
DALLE-2, Firefly, Flux Schnell, GLIDE, MidJourney V5, SD 1.3, SD 1.4 — all older
generators are trivially detected.

### Audio: Generators We Completely Miss (acc = 0%)
- **LLaSA family** (1B, 3B, 8B, multilingual): 0% accuracy across all sizes
- **Kokoro** (en, fr, it, ja): 0% accuracy
- **Spark TTS 0.5B**: 0% accuracy  
- **Higgs Audio V2**: 0% accuracy
- **Index TTS 1.5**: 0% accuracy
- **Kyutai TTS**: 0% accuracy

**Lesson:** Latest TTS models (LLaSA, Kokoro, Spark) completely evade all 3 audio models.
These are the next-gen neural codecs that sound indistinguishable from real speech.

### Audio: False Positive Hotspots (real classified as fake)
| Real Source | Accuracy | Avg Score | Samples |
|-------------|----------|-----------|---------|
| Environmental | 49.0% | 0.550 | 100 |
| Music | 55.0% | 0.521 | 100 |
| Common Voice | 59.0% | 0.486 | 100 |

**Lesson:** Non-speech audio (environmental sounds, music) triggers false positives. Our
audio models are speech-focused and misclassify non-speech as fake. Common Voice (diverse
accents/quality) also causes issues.

### Video: Almost Nothing Works Well
No video generator has >90% accuracy. Best results:
- Veo: moderate detection
- Sora: moderate detection
- Face swap videos: moderate detection

The T2V generators (gen2, crafter, hotshot, hunyuan, kling, moonvalley) all have 
poor detection rates. NPR-Video is the only model that provides meaningful discrimination.

## 5. Infrastructure Learnings

### A100 80GB Performance
- 23 models loaded: 18.5 GB VRAM (23% utilization)
- Throughput: 1.0–1.5 samples/s for images, 0.05–0.1 for video
- DEEPSAFE_MAX_WORKERS=3 is optimal (tested)
- Total eval time: ~7 hours for 15.5K samples with 4 HTTP workers

### Latency by Model (p50, ms)
| Model | P50 Latency | Notes |
|-------|------------|-------|
| NPR | 21ms | Fastest |
| NPR-Video | 48ms | Very fast for video |
| UnivFD-Video | 65ms | Fast |
| Universal | 62ms | Fast |
| Yermandy | 67ms | Fast |
| SBI | 77ms | Fast |
| FakeSTormer | 89ms | Moderate |
| CO-SPY | 109ms | Moderate |
| ShiftySpeech | 116ms | Moderate |
| SafeEar | 116ms | Moderate |
| Nes2Net | 124ms | Moderate |
| Effort | 224ms | Moderate |
| FSD | 236ms | Moderate |
| AIDE | 317ms | Slow |
| RECCE | 158ms | Moderate (but anti-correlated!) |
| LipFD | 1355ms | Slow |
| DFD-FCG | 1856ms | Very slow |
| MINTIME | 1667ms | Very slow |
| PwTF-DVD | 2983ms | Extremely slow (and anti-correlated!) |

**Lesson:** PwTF-DVD is both the slowest model AND anti-correlated. It's the #1 candidate
for removal — eliminating it would save ~3 seconds per video AND improve accuracy.

### Video Inference is the Bottleneck
- Video files can take 30–285 seconds for a single sample
- The 4 slowest video models (LipFD, DFD-FCG, MINTIME, PwTF-DVD) account for >90% of 
  video inference time
- NPR-Video (the best model) takes only 48ms — 62x faster than PwTF-DVD

## 6. Ensemble Architecture Findings

### Current Ensemble is Wrong for Video
The current RF meta-learner was trained on 50 samples with 8 features (models). On medium
data it produces F1=0.281 — worse than just using NPR-Video alone (AUC=0.798).

**Root cause:** The RF learned weights that amplify anti-correlated models (RECCE, PwTF-DVD)
because they happened to work on the 50-sample small eval.

### Provenance Signals Are Near-Useless on This Dataset
All provenance models (C2PA, SDXL Watermark, AudioSeal, VideoSeal) showed AUC≈0.500 on 
the medium dataset — zero discrimination. This is expected: the medium dataset doesn't
contain content with provenance signals (no C2PA manifests, no SDXL watermarks in the
test images).

**Lesson:** Provenance is valuable in production (real users uploading Veo/DALL-E content
with C2PA) but doesn't help on academic eval datasets. Don't train meta-learners to rely 
on provenance signals from eval data.

### Optimal Thresholds Are NOT 0.5
From baseline analysis:
- Image: optimal F1 at threshold=0.310 (AUC-weighted avg), not 0.500
- Audio: optimal F1 at threshold ~0.350
- Video: optimal F1 at threshold ~0.300

**Lesson:** Most models output probabilities biased toward low values (real content scores 
0.02-0.10). A 0.5 threshold misses many fakes that score 0.3-0.5. Per-modality threshold
optimization is critical.

## 7. Key Strategic Recommendations

### For the Meta-Learner Retraining (Next Session)
1. **Video:** Use only NPR-Video (0.798), SBI (0.676), LipFD (0.626), MINTIME (0.581), 
   DFD-FCG (0.554). Drop RECCE, PwTF-DVD, UnivFD-Video, FakeSTormer.
2. **Audio:** All 3 models are useful but SafeEar is marginal (0.604). Consider LR with
   L2 to learn the right weighting.
3. **Image:** All 7 models contribute. FSD+CO-SPY+Effort are the powerhouse trio.
4. **Threshold:** Optimize per-modality. Image ~0.31, audio ~0.35, video TBD.
5. **Algorithm:** LR(C=0.01, L2) for audio/video (fewer features, need regularization).
   RF or XGBoost for images (more data, more features).

### For Production API Improvements
1. **Remove PwTF-DVD from production** — anti-correlated AND slowest model. Saves 3s/video.
2. **Remove RECCE from production** — anti-correlated. Saves 0.2s/video.
3. **Add cross-modal fusion** — extract audio from video files, run audio models too. 
   Audio AUC=0.829 >> Video AUC=0.669.
4. **Add confidence-based fast path** — if NPR-Video scores >0.95, skip slow face models.
5. **Add "uncertain" verdict** — when models strongly disagree, return "uncertain" instead
   of forcing a binary fake/real. Better for LEA/media use cases.

### Models to Investigate Adding
The generator blind spots (Grok, Ideogram, Recraft V3, LLaSA, Kokoro) suggest we need:
- **Image:** A model trained on 2025-era generators (Grok, Ideogram 3, Recraft V3)
- **Audio:** A model that detects neural codec TTS (LLaSA, Kokoro, Spark TTS)
- **Video:** More full-frame detectors like NPR-Video (since it's by far the best)

## 8. Data Quality Issues Discovered

1. **FakeSTormer crashes on almost all videos** — only 1 out of 1978 videos produced a 
   valid prediction. The model likely has very narrow input requirements.
2. **Image models run on video files** — AIDE, CO-SPY, Effort, FSD, NPR, Universal, 
   Yermandy all ran on ~21 video samples (likely keyframes). This is noise, not signal.
3. **Audio provenance models run on all modalities** — AudioSeal ran on 728 video samples
   and 3497 audio samples, which is correct. But VideoSeal and SDXL Watermark ran on
   all modalities including audio, which adds latency for no value.

## 9. Reproducibility Notes

### To Reproduce the Medium Eval
```bash
# 1. Build medium dataset (needs full source at /data/master_eval_full/master_eval_full/)
PYTHONPATH=eval:$PYTHONPATH python -m eval.build_master_eval \
    --root /data/master_eval_full/master_eval_full --tier medium --seed 42

# 2. Update symlink
ln -sf /path/to/medium/dataset dataset/master_eval

# 3. Start monolith server (needs GPU + all model weights)
python -m monolith.server

# 4. Run eval
python eval/run_monolith_eval.py --workers 4
```

### To Reproduce Meta-Learner Experiments (no GPU needed)
```bash
pip install scikit-learn pandas numpy xgboost lightgbm
python eval/ensemble_experiment.py \
    --predictions eval/results/monolith_eval_20260410_102244_predictions.json \
    --modality video \
    --output eval/results/ensemble_exp_video.json
```
