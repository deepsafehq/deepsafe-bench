# Third-Party Notices

DeepSafe orchestrates 24 models. **The first-party code in this repository is
licensed under PolyForm Noncommercial 1.0.0. Nothing in this file is.** Each
model below is the work of its original authors and retains its own license.

Model code and weights are mirrored on HuggingFace at
`huggingface.co/deepsafe/model-code` and `huggingface.co/deepsafe/model-weights`
so that setup keeps working when upstream repositories and paper download links
disappear. Each mirrored model carries a `SOURCE.md` recording its upstream URL,
commit SHA, pull date, authors, and citation.

## Takedown Policy

If you are an author of any mirrored work and want it removed, open an issue or
email the maintainer listed in `CITATION.cff`. **We will remove it within 48
hours, no questions asked and no justification required.** We mirror to preserve
reproducibility, not to claim ownership.

## Detection Models

| Model | License | Upstream |
|---|---|---|
| AIDE | MIT | https://github.com/shilinyan99/AIDE (arXiv:2406.19435) |
| CO-SPY | MIT | https://github.com/Megum1/CO-SPY (arXiv:2503.18286) |
| Universal (UniversalFakeDetect) | MIT | https://github.com/Yuheng-Li/UniversalFakeDetect (arXiv:2302.10174) |
| Yermandy (GenD) | MIT | https://github.com/yermandy/GenD (arXiv:2508.06248) |
| ShiftySpeech | MIT | *upstream URL to be verified during mirroring* |
| PwTF-DVD | MIT | arXiv:2507.02398 (*repo URL to be verified*) |
| RECCE | MIT | arXiv (*repo URL to be verified*) |
| SafeEar | CC-BY-4.0 | arXiv:2409.09272 (*repo URL to be verified*) |
| FSD (Forensic Self-Descriptions) | **CC-BY-NC-SA-4.0** | https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25 (arXiv:2503.21003) |
| Effort | *no license file upstream* | https://github.com/YZY-stack/Effort-AIGI-Detection (arXiv:2411.15633) |
| NPR | *no license file upstream* | https://github.com/chuangchuangtan/NPR-DeepfakeDetection (arXiv:2312.10461) |
| MINTIME | *no license file upstream* | https://github.com/davide-coccomini/MINTIME-Multi-Identity-size-iNvariant-TIMEsformer-for-Video-Deepfake-Detection (arXiv:2206.13829) |
| DFD-FCG | *no license file upstream* | arXiv:2404.05583 (*repo URL to be verified*) |
| LipFD | *no license file upstream* | arXiv:2401.15668 (*repo URL to be verified*) |
| Nes2Net | *no license file upstream* | *upstream URL to be verified during mirroring* |
| FakeSTormer | *no license file upstream* | *upstream URL to be verified during mirroring* |
| NPR-Video | *inherits NPR* | same as NPR |
| UnivFD-Video | *inherits Universal* | same as Universal |

**SBI** is not third-party code. `apps/inference/models/sbi.py` is a first-party
reimplementation of the inference-time detector on `efficientnet_pytorch`. Only
its weights originate upstream (Self-Blended Images, CVPR 2022).

### On models with no upstream license

Several upstream repositories ship no license file. Under copyright law that
means all rights are reserved, and mirroring them is redistribution without
explicit permission. We mirror them anyway, with full attribution, because paper
and repository links in this field rot quickly and reproducibility depends on the
code remaining reachable. The takedown policy above is unconditional and exists
precisely for this reason.

If you are an author and would prefer to add a license rather than have the
mirror removed, we would rather that outcome, and we will update this file the
same day.

## Provenance Services

These install from PyPI and are not mirrored beyond their weights.

| Service | Package | License |
|---|---|---|
| C2PA | `c2pa-python==0.32.0` | Apache-2.0 / MIT (per upstream) |
| SDXL Watermark | `invisible-watermark==0.2.0` | MIT |
| TrustMark | `trustmark>=0.9.0` | MIT (Adobe/CAI) |
| AudioSeal | `audioseal==0.1.4` | MIT (Meta) |
| VideoSeal | `videoseal==1.0.1` | MIT (Meta) |

VideoSeal installs with `--no-deps` because of a `timm` version conflict. This is
required; a naive reinstall will break the environment.

## Datasets

The evaluation set is assembled from roughly 20 upstream sources with
incompatible terms. Freely redistributable: LibriSpeech, LJSpeech, VCTK, Common
Voice, COCO. Research-only, EULA-bound, or non-redistributable: ImageNet,
ASVspoof, DF40, LAV-DF, MSR-VTT, MLAAD.

The medium and full tiers are **gated** on HuggingFace for this reason. Accepting
the gate does not relieve you of complying with each upstream source's own terms.
Full per-source attribution is in the dataset card at
`huggingface.co/datasets/deepsafe/evaluation-dataset`.

Audio and video splits contain recordings of real people. Treat them as
biometric data.
