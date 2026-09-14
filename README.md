# DeepSafe

**A benchmark, model zoo, and adaptation toolkit for deepfake detection.**

![Models](https://img.shields.io/badge/models-24-06b6d4)
![Generators](https://img.shields.io/badge/generators-411-8b5cf6)
![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial-orange)

DeepSafe runs 24 detection and provenance models behind one interface, scores
any detector across 411 generative models, and lets you adapt the stack to your
own data. It is free for research, education, and personal use.

## Why this exists

Published deepfake detectors report near-perfect accuracy. Most of that accuracy
does not survive contact with a generator the model has not seen.

Here is our own ensemble of 19 published detectors, on 15,499 held-out samples:

- **Recall on fakes: 66.2%**
- **False positive rate on real media: 6.5%**

A third of fake media walks straight through. And the failures are not evenly
distributed:

| Generator | Caught |
|---|---|
| Hunyuan (video) | **1.7%** |
| Sora (video) | **7.0%** |
| Veo (video) | **8.0%** |
| Kling (video) | **12.0%** |
| Stable Diffusion 1.4 (image) | 100% |
| Midjourney v5 (image) | 100% |
| GLIDE (image) | 100% |

The stack is near-perfect on the 2022-2023 diffusion models the literature was
built around, and close to blind on the video generators people actually worry
about today.

Ensemble AUC, held out: image 0.9466, audio 0.8290, **video 0.6694**. Those are
lower than the cross-validated numbers from meta-learner training (0.9705 /
0.8748 / 0.8898), and that gap is itself the finding. Cross-validated scores on
the training distribution overstate what a detector does in the wild.

Almost nobody publishes this. Papers compare against three or four baselines on
one or two datasets because getting baselines to run is genuinely brutal, not
because researchers are careless.

DeepSafe exists so that stops being the bottleneck. Full numbers, reproducible
from the shipped prediction matrix, are in [BENCHMARK.md](BENCHMARK.md).

## What you get

```bash
deepsafe detect  suspicious.mp4           # 24 models + calibrated ensemble verdict
deepsafe eval    --model mine.py          # score any detector, 411 generators, 6 attacks
deepsafe fit     --data ./labeled/        # adapt to your domain, then prove it generalized
```

`eval` always reports in-distribution and held-out-generator performance
separately. That separation is not configurable.

`fit` runs a generalization check after every adaptation and tells you when your
in-distribution gains did not transfer. It is designed to catch your own
overfitting rather than flatter it.

## Models

| Modality | Count | Models |
|---|---|---|
| Image | 7 | NPR, Yermandy, Universal, AIDE, FSD, Effort, CO-SPY |
| Video | 9 | FakeSTormer, SBI, DFD-FCG, PwTF-DVD, LipFD, RECCE, MINTIME, NPR-Video, UnivFD-Video |
| Audio | 3 | ShiftySpeech, SafeEar, Nes2Net |
| Provenance | 5 | C2PA, SDXL Watermark, TrustMark, AudioSeal, VideoSeal |

All 24 run in a single process with one CUDA context. Model code and weights are
mirrored on HuggingFace so setup keeps working after upstream repositories and
paper download links rot, which in this field they reliably do.

Every model is the work of its original authors under its own license. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), which includes an unconditional
takedown policy.

## Setup

```bash
git clone https://github.com/deepsafehq/deepsafe-bench.git
cd deepsafe-bench
bash setup.sh
```

Requires a CUDA GPU with 24 GB or more of VRAM for the full lineup, NVIDIA driver
535+, CUDA 12.1+, and Python 3.11 or 3.12. Blackwell (sm_120) is not supported.

`eval` against a stub detector and the small dataset tier runs on CPU.

## Limitations

Read these before trusting any output.

- **Detection does not generalize.** Expect a large drop on any generator absent
  from the training distribution. This is a property of the field, not a bug in
  this implementation.
- **Not suitable for high-stakes decisions.** Do not use this to accuse a person
  of anything. A confident score is not evidence.
- **Audio is weakest outside English.** See the multilingual numbers above.
- **Provenance signals are stronger than detection**, when present. A C2PA
  manifest or a watermark is real evidence; a model score is an estimate.

## License

[PolyForm Noncommercial 1.0.0](LICENSE). Free for research, education, personal
projects, and use by nonprofits, educational institutions, and government
bodies. Commercial use is not permitted.

This is a source-available license, not an OSI-approved open source license.

## Citation

See [CITATION.cff](CITATION.cff), or use GitHub's "Cite this repository" button.
`deepsafe eval` also writes a `CITATIONS.bib` next to every report containing
entries for DeepSafe and for every model it ran.
