#!/usr/bin/env python3
"""
Cross-Language Audio Deepfake Detection Evaluation

Measures per-language AUC for ShiftySpeech, SafeEar, and SONICS
to reveal English bias in audio detection models.

Dataset: 439 fake TTS (5 languages) + 140 real speech (7 languages)
"""

import base64
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from deepsafe_eval.config import DATASET_ROOT as _DATASET_ROOT
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

DATASET_ROOT = _DATASET_ROOT / "audio"
TIMEOUT = 300
MAX_WORKERS = 8

AUDIO_MODELS = {
    "shiftyspeech": "http://localhost:8001/predict",
    "safeear": "http://localhost:8002/predict",
    "sonics": "http://localhost:8003/predict",
    "nes2net": "http://localhost:8004/predict",
}

# Language → fake directory name mapping
FAKE_LANGS = {
    "arabic": "multilingual_arabic",
    "chinese": "multilingual_chinese",
    "hindi": "multilingual_hindi",
    "japanese": "multilingual_japanese",
    "spanish": "multilingual_spanish",
}

# Languages present in real/multilingual/ (filename pattern: real_{lang}_NNN.wav)
REAL_LANGS = [
    "arabic",
    "chinese",
    "english",
    "german",
    "hindi",
    "japanese",
    "spanish",
]


def b64_file(path: Path) -> str:
    """Base64-encode a file."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def call_model(
    url: str,
    filepath: Path,
) -> tuple[float | None, str | None, float]:
    """Returns (probability, error_message, inference_time_seconds)."""
    t0 = time.time()
    try:
        payload = {"audio_data": b64_file(filepath)}
        resp = requests.post(url, json=payload, timeout=TIMEOUT)
        elapsed = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        for key in ("probability", "fake_probability", "score"):
            if key in data and isinstance(data[key], (int, float)):
                return float(data[key]), None, elapsed
        return None, f"unexpected keys: {list(data.keys())}", elapsed
    except requests.exceptions.Timeout:
        return None, "timeout", time.time() - t0
    except Exception as e:
        return None, str(e)[:120], time.time() - t0


def warmup(name: str, url: str, test_file: Path) -> bool:
    """Health check + single warm-up inference."""
    health_url = url.replace("/predict", "/health")
    try:
        requests.get(health_url, timeout=5).raise_for_status()
    except Exception:
        print(f"  [{name}] OFFLINE", flush=True)
        return False
    print(f"  [{name}] warming up...", end=" ", flush=True)
    prob, err, _ = call_model(url, test_file)
    if prob is not None:
        print(f"ok (prob={prob:.4f})", flush=True)
        return True
    print(f"FAILED: {err}", flush=True)
    return False


def compute_metrics(
    labels: list[int],
    probs: list[float],
    threshold: float = 0.5,
) -> dict:
    """Compute AUC, FPR, FNR, accuracy, and other metrics."""
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    n = len(labels)

    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0

    auc_roc = 0.0
    auc_pr = 0.0
    eer = 1.0
    optimal_threshold = threshold
    if len(set(labels)) == 2 and n >= 2:
        try:
            auc_roc = roc_auc_score(labels, probs)
            auc_pr = average_precision_score(labels, probs)
            fpr_arr, tpr_arr, thresholds = roc_curve(labels, probs)
            fnr_arr = 1 - tpr_arr
            eer_idx = np.abs(fpr_arr - fnr_arr).argmin()
            eer = (fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2
            j_scores = tpr_arr - fpr_arr
            best_idx = j_scores.argmax()
            if best_idx < len(thresholds):
                optimal_threshold = float(thresholds[best_idx])
        except Exception:
            pass

    real_probs = [p for l, p in zip(labels, probs) if l == 0]
    fake_probs = [p for l, p in zip(labels, probs) if l == 1]

    return {
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "eer": round(eer, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "specificity": round(spec, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "mcc": round(mcc, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "avg_prob_real": (
            round(sum(real_probs) / len(real_probs), 4) if real_probs else None
        ),
        "avg_prob_fake": (
            round(sum(fake_probs) / len(fake_probs), 4) if fake_probs else None
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_real": len(real_probs),
        "n_fake": len(fake_probs),
        "n_total": n,
    }


def collect_multilingual_files() -> dict[str, list[tuple[Path, int]]]:
    """Collect files grouped by language.

    Returns:
        dict mapping language → list of (filepath, label) tuples
        label: 0 = real, 1 = fake
    """
    by_lang: dict[str, list[tuple[Path, int]]] = defaultdict(list)

    # Real files: real/multilingual/real_{lang}_NNN.wav
    real_dir = DATASET_ROOT / "real" / "multilingual"
    if real_dir.exists():
        for f in sorted(real_dir.iterdir()):
            if not f.is_file() or f.name.startswith("."):
                continue
            # Parse language from filename
            for lang in REAL_LANGS:
                if f.name.startswith(f"real_{lang}_"):
                    by_lang[lang].append((f, 0))
                    break

    # Fake files: fake/multilingual_{lang}/
    for lang, dirname in FAKE_LANGS.items():
        fake_dir = DATASET_ROOT / "fake" / dirname
        if fake_dir.exists():
            for f in sorted(fake_dir.iterdir()):
                if f.is_file() and not f.name.startswith("."):
                    by_lang[lang].append((f, 1))

    return dict(by_lang)


def run_model_on_files(
    model_name: str,
    url: str,
    files: list[tuple[Path, int]],
) -> tuple[list[int], list[float], int, list[dict]]:
    """Run a model on a list of files.

    Returns (labels, probs, error_count, per_file_results).
    """
    labels, probs, errors = [], [], 0
    per_file = []

    def _infer(item):
        fp, lbl = item
        prob, err, inf_time = call_model(url, fp)
        return fp, lbl, prob, err, inf_time

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_infer, item): i for i, item in enumerate(files)}
        done = 0
        for fut in as_completed(futures):
            fp, lbl, prob, err, inf_time = fut.result()
            done += 1
            if prob is None:
                errors += 1
                per_file.append(
                    {
                        "file": fp.name,
                        "label": lbl,
                        "prob": None,
                        "error": err,
                    }
                )
            else:
                labels.append(lbl)
                probs.append(prob)
                per_file.append(
                    {
                        "file": fp.name,
                        "label": lbl,
                        "prob": round(prob, 6),
                        "error": None,
                    }
                )
            if done % 50 == 0 or done == len(files):
                print(f"      {done}/{len(files)}", flush=True)

    return labels, probs, errors, per_file


def main():
    print("=" * 80)
    print("  DeepSafe Cross-Language Audio Evaluation")
    print("=" * 80)

    # Collect files
    by_lang = collect_multilingual_files()
    print("\n  Dataset summary:")
    for lang in sorted(by_lang.keys()):
        files = by_lang[lang]
        n_real = sum(1 for _, l in files if l == 0)
        n_fake = sum(1 for _, l in files if l == 1)
        print(f"    {lang:>10}: {n_real:>3} real + {n_fake:>3} fake = {len(files):>4}")

    total_files = sum(len(f) for f in by_lang.values())
    print(f"    {'TOTAL':>10}: {total_files}")

    # Warmup models
    print("\n  Warming up models...")
    test_file = by_lang[next(iter(by_lang))][0][0]
    live_models = {}
    for name, url in AUDIO_MODELS.items():
        if warmup(name, url, test_file):
            live_models[name] = url

    if not live_models:
        print("\n  No models available. Exiting.")
        return

    # Run evaluation
    results = {}
    all_per_file = {}

    for model_name, url in live_models.items():
        print(f"\n{'─'*80}")
        print(f"  MODEL: {model_name}")
        print(f"{'─'*80}")

        model_results = {}
        model_per_file = {}

        # Evaluate each language
        for lang in sorted(by_lang.keys()):
            files = by_lang[lang]
            n_real = sum(1 for _, l in files if l == 0)
            n_fake = sum(1 for _, l in files if l == 1)
            print(
                f"\n    [{lang}] {len(files)} files "
                f"({n_real} real / {n_fake} fake)...",
                flush=True,
            )

            labels, probs, errors, per_file = run_model_on_files(
                model_name,
                url,
                files,
            )

            if not labels:
                print(f"      All failed ({errors} errors)")
                continue

            metrics = compute_metrics(labels, probs)
            metrics["n_errors"] = errors
            model_results[lang] = metrics
            model_per_file[lang] = per_file

            # Print key metrics inline
            has_both = n_real > 0 and n_fake > 0
            if has_both:
                print(
                    f"      AUC={metrics['auc_roc']:.4f}  "
                    f"FPR={metrics['fpr']:.3f}  "
                    f"FNR={metrics['fnr']:.3f}  "
                    f"Acc={metrics['accuracy']:.3f}  "
                    f"Err={errors}",
                )
            else:
                # FPR-only (real samples only) or FNR-only
                print(
                    f"      FPR={metrics['fpr']:.3f}  "
                    f"AvgProb={metrics.get('avg_prob_real', 'N/A')}  "
                    f"Err={errors}",
                )

        # Aggregate: all languages combined
        all_labels, all_probs = [], []
        for lang in by_lang:
            if lang in model_per_file:
                for pf in model_per_file[lang]:
                    if pf["prob"] is not None:
                        all_labels.append(pf["label"])
                        all_probs.append(pf["prob"])
        if all_labels:
            model_results["_aggregate"] = compute_metrics(all_labels, all_probs)

        results[model_name] = model_results
        all_per_file[model_name] = model_per_file

    # ── Print summary tables ──────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  RESULTS: PER-LANGUAGE AUC-ROC")
    print("=" * 80)

    # Languages with both real + fake (can compute AUC)
    auc_langs = sorted(
        lang
        for lang in by_lang
        if sum(1 for _, l in by_lang[lang] if l == 0) > 0
        and sum(1 for _, l in by_lang[lang] if l == 1) > 0
    )
    # Languages with real only (FPR only)
    fpr_only_langs = sorted(
        lang for lang in by_lang if sum(1 for _, l in by_lang[lang] if l == 1) == 0
    )

    # AUC table
    header = f"  {'Language':<12} {'N':>5}"
    for m in sorted(results.keys()):
        header += f"  {m:>14}"
    print(f"\n{header}")
    print(f"  {'─' * (len(header) - 2)}")

    for lang in auc_langs:
        n = len(by_lang[lang])
        row = f"  {lang:<12} {n:>5}"
        for m in sorted(results.keys()):
            if lang in results[m]:
                auc = results[m][lang]["auc_roc"]
                row += f"  {auc:>14.4f}"
            else:
                row += f"  {'N/A':>14}"
        print(row)

    # Aggregate row
    row = f"  {'AGGREGATE':<12} {total_files:>5}"
    for m in sorted(results.keys()):
        if "_aggregate" in results[m]:
            auc = results[m]["_aggregate"]["auc_roc"]
            row += f"  {auc:>14.4f}"
    print(f"  {'─' * (len(header) - 2)}")
    print(row)

    # FPR table (false positive rate on real speech)
    print(f"\n\n  FALSE POSITIVE RATE ON REAL SPEECH (lower = better)")
    print(f"  {'Language':<12} {'N_real':>6}", end="")
    for m in sorted(results.keys()):
        print(f"  {m:>14}", end="")
    print()
    print(f"  {'─' * 70}")

    for lang in sorted(by_lang.keys()):
        n_real = sum(1 for _, l in by_lang[lang] if l == 0)
        if n_real == 0:
            continue
        row = f"  {lang:<12} {n_real:>6}"
        for m in sorted(results.keys()):
            if lang in results[m]:
                fpr = results[m][lang]["fpr"]
                row += f"  {fpr:>14.3f}"
            else:
                row += f"  {'N/A':>14}"
        print(row)

    # FNR table (false negative rate on fake TTS)
    print(f"\n\n  FALSE NEGATIVE RATE ON FAKE TTS (lower = better)")
    print(f"  {'Language':<12} {'N_fake':>6}", end="")
    for m in sorted(results.keys()):
        print(f"  {m:>14}", end="")
    print()
    print(f"  {'─' * 70}")

    for lang in auc_langs:
        n_fake = sum(1 for _, l in by_lang[lang] if l == 1)
        row = f"  {lang:<12} {n_fake:>6}"
        for m in sorted(results.keys()):
            if lang in results[m]:
                fnr = results[m][lang]["fnr"]
                row += f"  {fnr:>14.3f}"
            else:
                row += f"  {'N/A':>14}"
        print(row)

    # Avg prob table
    print(f"\n\n  AVERAGE PREDICTED PROBABILITY")
    print(f"  {'Language':<12} {'Type':<6}", end="")
    for m in sorted(results.keys()):
        print(f"  {m:>14}", end="")
    print()
    print(f"  {'─' * 70}")

    for lang in sorted(by_lang.keys()):
        n_real = sum(1 for _, l in by_lang[lang] if l == 0)
        n_fake = sum(1 for _, l in by_lang[lang] if l == 1)
        if n_real > 0:
            row = f"  {lang:<12} {'real':<6}"
            for m in sorted(results.keys()):
                if lang in results[m]:
                    val = results[m][lang].get("avg_prob_real")
                    row += f"  {val:>14.4f}" if val is not None else f"  {'N/A':>14}"
                else:
                    row += f"  {'N/A':>14}"
            print(row)
        if n_fake > 0:
            row = f"  {lang:<12} {'fake':<6}"
            for m in sorted(results.keys()):
                if lang in results[m]:
                    val = results[m][lang].get("avg_prob_fake")
                    row += f"  {val:>14.4f}" if val is not None else f"  {'N/A':>14}"
                else:
                    row += f"  {'N/A':>14}"
            print(row)

    # ── Bias analysis ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  BIAS ANALYSIS")
    print(f"{'='*80}")

    for model_name in sorted(results.keys()):
        mr = results[model_name]
        aucs = {lang: mr[lang]["auc_roc"] for lang in auc_langs if lang in mr}
        if not aucs:
            continue

        best_lang = max(aucs, key=aucs.get)
        worst_lang = min(aucs, key=aucs.get)
        spread = aucs[best_lang] - aucs[worst_lang]

        print(f"\n  [{model_name}]")
        print(f"    Best  AUC: {best_lang} ({aucs[best_lang]:.4f})")
        print(f"    Worst AUC: {worst_lang} ({aucs[worst_lang]:.4f})")
        print(f"    Spread: {spread:.4f}")

        if spread > 0.10:
            print(f"    ⚠ SIGNIFICANT language bias detected (spread > 0.10)")
        elif spread > 0.05:
            print(f"    ~ Moderate language variation (spread 0.05-0.10)")
        else:
            print(f"    ✓ Low language variation (spread < 0.05)")

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H-%M")
    out_path = out_dir / f"{timestamp}-multilingual-audio.json"

    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset_root": str(DATASET_ROOT),
            "models": list(live_models.keys()),
            "languages": sorted(by_lang.keys()),
            "file_counts": {
                lang: {
                    "real": sum(1 for _, l in by_lang[lang] if l == 0),
                    "fake": sum(1 for _, l in by_lang[lang] if l == 1),
                }
                for lang in by_lang
            },
        },
        "per_language_metrics": results,
        "per_file_predictions": all_per_file,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
