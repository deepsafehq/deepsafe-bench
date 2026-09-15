# Reproducing the benchmark

Every number in [BENCHMARK.md](BENCHMARK.md) is reproducible. This page gives
three levels, cheapest first, and is explicit about what each one actually
proves.

| Level | Needs | Time | Proves |
|---|---|---|---|
| 1. Verify the numbers | nothing | ~30 s | the reported figures follow from the published predictions |
| 2. Regenerate the report | nothing | ~1 min | the whole document derives from one data file |
| 3. Re-run the models | GPU + dataset | hours | the predictions themselves are reproducible |

Level 1 and 2 check our arithmetic. Only level 3 checks our models.

---

## Level 1: Verify the published numbers

No GPU, no download, no account.

```bash
pip install deepsafe-bench
deepsafe eval --baseline ensemble --modality video
```

You should see:

```
  AUC                   0.6694
  recall on fakes        17.4%  @ threshold 0.5
  ...
  Weakest generators (this is the number that matters):
    hunyuan                        1.7%   n=58
    sora                           7.0%   n=57
    veo                            8.0%   n=50
```

The package ships the full prediction matrix (15,499 rows, 23 model score
columns) as bundled data, which is why this works offline. Check any model or
modality:

```bash
deepsafe eval --list-baselines
deepsafe eval --baseline npr --modality images
deepsafe eval --baseline ensemble --modality audio
```

The metric code has **no third-party dependencies**: no numpy, pandas or
scikit-learn. AUC is computed directly in `src/deepsafe/metrics.py` via the
Mann-Whitney statistic with tie correction, so you can read the arithmetic
rather than trust a library. It is about 40 lines.

## Level 2: Regenerate BENCHMARK.md from scratch

```bash
git clone https://github.com/deepsafehq/deepsafe-bench.git
cd deepsafe-bench
python eval/scripts/generate_benchmark.py
git diff BENCHMARK.md      # should be empty
```

This rebuilds the entire document from
`eval/results/predictions_medium.csv.gz`. An empty diff means every table,
percentage and caveat in it is derived, not typed by hand.

The same file drives the [benchmark explorer](https://deepsafehq.github.io/deepsafe-bench/benchmark),
regenerated in CI, so the site cannot drift from the document either.

## Level 3: Re-run the models

This regenerates the predictions themselves. It needs the models and the
evaluation media.

### Get the dataset

```bash
bash scripts/download_dataset.sh small     # 198 samples, ~1.7 GB
bash scripts/download_dataset.sh medium    # 15,454 samples, ~10 GB
```

The medium tier is what `BENCHMARK.md` reports on. The dataset is gated
because it aggregates sources with incompatible terms (ASVspoof, DF40,
LAV-DF and others). Accepting is one click at
[huggingface.co/datasets/deepsafe/evaluation-dataset](https://huggingface.co/datasets/deepsafe/evaluation-dataset),
then `hf auth login`.

### Get the models

```bash
bash setup.sh
```

Pulls ~44 GB of weights and model code from HuggingFace. No token needed.
Requires a CUDA GPU with 24 GB+ VRAM for the full lineup, driver 535+,
CUDA 12.1+, Python 3.11 or 3.12. On a smaller card, load a subset with
`DEEPSAFE_MODELS=npr,aide`.

### Run it

```bash
# Start the inference server
cd apps/inference && PYTHONPATH=.:../../packages/shared python server.py

# Score the dataset (writes eval/results/*.json)
python eval/run_monolith_eval.py

# Rebuild the report from your own predictions
python eval/scripts/generate_benchmark.py
```

### What to expect

Exact figures will not match to four decimal places, and that is normal. Face
detection, frame sampling and FP16 autocast all introduce nondeterminism, and
GPU architecture affects results. **The finding is the pattern, not the
decimals**: near-total detection on 2022-2023 diffusion images, near-total
failure on recent video generators. If Sora comes out at 6% or 9% rather than
7.0%, the conclusion is unchanged. If it comes out at 70%, please open an
issue, because one of us has a bug.

---

## Scoring your own detector

Any Python file exposing `predict(path) -> float` is scorable:

```python
# my_detector.py
import pathlib

def predict(path: pathlib.Path) -> float:
    """Return P(synthetic) in [0, 1]."""
    ...
```

```bash
deepsafe eval --model my_detector.py --tier small --out ./results
```

You get a report card and a `CITATIONS.bib` containing entries for DeepSafe
and every model your run touched. See `examples/` for two worked detectors,
including one that deliberately exploits a dataset artifact to show why
per-generator breakdowns matter.

## If a number looks wrong

Open an issue with the command you ran and its output. Every figure here comes
from one committed file, so disagreements are resolvable rather than a matter
of opinion.
