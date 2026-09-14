#!/usr/bin/env python3
"""Monolith eval: run small dataset through local monolith server.

Sends each file to the running monolith server at localhost:8000,
collects per-model probabilities and latencies, computes comprehensive
metrics (AUC, F1, precision, recall, accuracy, EER) per model and
ensemble per modality.

Usage:
    source .venv/bin/activate
    python eval/run_monolith_eval.py
    python eval/run_monolith_eval.py --workers 8  # test higher concurrency
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_DIR = PROJECT_ROOT / "dataset" / "master_eval"
SERVER_URL = "http://localhost:8000"
DETECT_URL = f"{SERVER_URL}/v1/detect"

MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp4": "video/mp4", ".avi": "video/avi", ".mov": "video/quicktime",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}


def load_manifest():
    """Load metadata.json and return list of sample dicts."""
    meta_path = DATASET_DIR / "metadata.json"
    with open(meta_path) as f:
        return json.load(f)


def detect_file(entry):
    """Send a single file to the monolith server."""
    file_path = DATASET_DIR / entry["path"]
    if not file_path.exists():
        return {**entry, "error": f"File not found: {file_path}", "response": None}

    ext = file_path.suffix.lower()
    mime = MIME_MAP.get(ext, "application/octet-stream")

    t0 = time.perf_counter()
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                DETECT_URL,
                files={"file": (file_path.name, f, mime)},
                timeout=600,
            )
        latency = time.perf_counter() - t0

        if resp.status_code == 200:
            body = resp.json()
            return {**entry, "response": body, "total_latency_s": latency, "error": None}
        else:
            return {**entry, "response": None, "total_latency_s": latency,
                    "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {**entry, "response": None, "total_latency_s": time.perf_counter() - t0,
                "error": str(e)[:200]}


def compute_metrics(y_true, y_scores, threshold=0.5):
    """Compute classification metrics from true labels and scores."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Filter out NaN/None
    valid = ~np.isnan(y_scores)
    y_true = y_true[valid]
    y_scores = y_scores[valid]

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {"n": len(y_true), "error": "insufficient data"}

    y_pred = (y_scores >= threshold).astype(int)

    # AUC
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = None

    # EER
    try:
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except Exception:
        eer = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "n": int(len(y_true)),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(auc) if auc is not None else None,
        "eer": float(eer) if eer is not None else None,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "mean_fake_score": float(y_scores[y_true == 1].mean()) if (y_true == 1).any() else None,
        "mean_real_score": float(y_scores[y_true == 0].mean()) if (y_true == 0).any() else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent request workers (for benchmarking)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    # Verify server is up
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=10).json()
        print(f"Server: {health['status']} | Models loaded: {health['models_loaded']}")
        print(f"GPU: {health.get('gpu_name', 'N/A')} | VRAM: {health.get('vram_used_mb', 0)} MB")
    except Exception as e:
        print(f"Server not responding: {e}")
        sys.exit(1)

    # Load manifest
    manifest = load_manifest()
    print(f"\nDataset: {len(manifest)} samples")

    by_mod = defaultdict(int)
    for e in manifest:
        by_mod[f"{e['modality']}/{e['label']}"] += 1
    for k, v in sorted(by_mod.items()):
        print(f"  {k}: {v}")

    # Run inference
    print(f"\nRunning inference (workers={args.workers})...")
    results = []
    errors = []
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(detect_file, e): e for e in manifest}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result.get("error"):
                errors.append(result)
                status = f"ERR: {result['error'][:60]}"
            else:
                resp = result["response"]
                status = f"{resp['verdict']} ({resp['confidence']:.3f}) {result['total_latency_s']:.1f}s"
            results.append(result)

            if done % 10 == 0 or done == len(manifest):
                elapsed = time.perf_counter() - start_time
                rate = done / elapsed
                print(f"  [{done:>3}/{len(manifest)}] {rate:.1f} samples/s | {status}")

    total_time = time.perf_counter() - start_time
    print(f"\nTotal inference time: {total_time:.1f}s ({len(manifest)/total_time:.2f} samples/s)")
    print(f"Errors: {len(errors)}/{len(manifest)}")

    # Collect per-model probabilities and ensemble scores
    # Group by modality
    modality_map = {"images": "image", "audio": "audio", "video": "video"}
    inv_modality = {v: k for k, v in modality_map.items()}

    # Organize results by modality
    by_modality = defaultdict(list)
    for r in results:
        mod = r["modality"]
        media_type = modality_map.get(mod, mod)
        by_modality[media_type].append(r)

    # Extract per-model and ensemble metrics
    all_metrics = {}
    per_model_latencies = defaultdict(list)

    for media_type, mod_results in sorted(by_modality.items()):
        print(f"\n{'='*70}")
        print(f"  {media_type.upper()} ({len(mod_results)} samples)")
        print(f"{'='*70}")

        # Ground truth labels
        labels = []
        ensemble_scores = []
        model_scores = defaultdict(list)
        model_labels = defaultdict(list)
        latencies = []

        for r in mod_results:
            label = 1 if r["label"] == "fake" else 0
            resp = r.get("response")
            if resp is None:
                continue

            labels.append(label)
            ensemble_scores.append(resp.get("score", float("nan")))
            latencies.append(r.get("total_latency_s", 0))

            # Per-model
            for model_name, model_data in resp.get("model_results", {}).items():
                prob = model_data.get("probability")
                if prob is not None:
                    model_scores[model_name].append(prob)
                    model_labels[model_name].append(label)
                lat = model_data.get("latency_ms")
                if lat is not None:
                    per_model_latencies[model_name].append(lat)

        # Ensemble metrics
        ensemble_m = compute_metrics(labels, ensemble_scores)
        print(f"\n  ENSEMBLE ({ensemble_m.get('n', 0)} samples):")
        print(f"    AUC:       {ensemble_m.get('auc_roc', 'N/A'):.4f}" if ensemble_m.get('auc_roc') else "    AUC:       N/A")
        print(f"    Accuracy:  {ensemble_m.get('accuracy', 0):.4f}")
        print(f"    Precision: {ensemble_m.get('precision', 0):.4f}")
        print(f"    Recall:    {ensemble_m.get('recall', 0):.4f}")
        print(f"    F1:        {ensemble_m.get('f1', 0):.4f}")
        print(f"    EER:       {ensemble_m.get('eer', 'N/A'):.4f}" if ensemble_m.get('eer') else "    EER:       N/A")
        print(f"    TP={ensemble_m.get('tp',0)} FP={ensemble_m.get('fp',0)} "
              f"TN={ensemble_m.get('tn',0)} FN={ensemble_m.get('fn',0)}")
        print(f"    Mean fake score: {ensemble_m.get('mean_fake_score', 'N/A')}")
        print(f"    Mean real score: {ensemble_m.get('mean_real_score', 'N/A')}")
        if latencies:
            print(f"    Avg latency:     {np.mean(latencies):.2f}s (p50={np.percentile(latencies, 50):.2f}s, p95={np.percentile(latencies, 95):.2f}s)")

        all_metrics[media_type] = {"ensemble": ensemble_m, "models": {}}

        # Per-model metrics
        print(f"\n  PER-MODEL BREAKDOWN:")
        print(f"  {'Model':<16} {'AUC':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'EER':>7} {'AvgLat':>8}")
        print(f"  {'-'*74}")

        for model_name in sorted(model_scores.keys()):
            scores = model_scores[model_name]
            m_labels = model_labels[model_name]
            m = compute_metrics(m_labels, scores)

            auc_s = f"{m['auc_roc']:.4f}" if m.get('auc_roc') else "  N/A"
            acc_s = f"{m.get('accuracy', 0):.4f}"
            pre_s = f"{m.get('precision', 0):.4f}"
            rec_s = f"{m.get('recall', 0):.4f}"
            f1_s = f"{m.get('f1', 0):.4f}"
            eer_s = f"{m['eer']:.4f}" if m.get('eer') else "  N/A"

            lats = per_model_latencies.get(model_name, [])
            lat_s = f"{np.mean(lats):.0f}ms" if lats else "N/A"

            print(f"  {model_name:<16} {auc_s:>7} {acc_s:>7} {pre_s:>7} {rec_s:>7} {f1_s:>7} {eer_s:>7} {lat_s:>8}")
            all_metrics[media_type]["models"][model_name] = m

    # Per-generator breakdown
    print(f"\n{'='*70}")
    print(f"  PER-GENERATOR ACCURACY")
    print(f"{'='*70}")

    for media_type, mod_results in sorted(by_modality.items()):
        print(f"\n  [{media_type.upper()}]")
        by_gen = defaultdict(list)
        for r in mod_results:
            if r.get("response"):
                by_gen[f"{r['label']}/{r['generator']}"].append(r)

        for gen_key in sorted(by_gen.keys()):
            gen_results = by_gen[gen_key]
            n = len(gen_results)
            correct = sum(1 for r in gen_results
                         if (r["response"]["verdict"] == "fake" and r["label"] == "fake") or
                            (r["response"]["verdict"] == "real" and r["label"] == "real"))
            acc = correct / n if n > 0 else 0
            avg_score = np.mean([r["response"]["score"] for r in gen_results])
            print(f"    {gen_key:<45} acc={acc:.3f} n={n:>3} avg_score={avg_score:.4f}")

    # Latency analysis
    print(f"\n{'='*70}")
    print(f"  LATENCY ANALYSIS (per-model, ms)")
    print(f"{'='*70}")
    print(f"  {'Model':<16} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8} {'Count':>6}")
    print(f"  {'-'*62}")
    for model_name in sorted(per_model_latencies.keys()):
        lats = np.array(per_model_latencies[model_name])
        print(f"  {model_name:<16} {np.mean(lats):>8.0f} {np.percentile(lats, 50):>8.0f} "
              f"{np.percentile(lats, 95):>8.0f} {np.percentile(lats, 99):>8.0f} "
              f"{np.max(lats):>8.0f} {len(lats):>6}")

    # GPU stats at end
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  GPU VRAM after eval:")
            print(f"    Allocated: {torch.cuda.memory_allocated(0)/1024**2:.0f} MB")
            print(f"    Reserved:  {torch.cuda.memory_reserved(0)/1024**2:.0f} MB")
    except Exception:
        pass

    # Build per-file predictions for meta-learner training
    predictions = []
    for r in results:
        resp = r.get("response")
        if resp is None:
            continue
        entry = {
            "id": r["id"],
            "modality": r["modality"],
            "label": r["label"],
            "generator": r["generator"],
            "ensemble_score": resp.get("score"),
            "ensemble_verdict": resp.get("verdict"),
            "ensemble_method": resp.get("method"),
        }
        # Per-model probabilities (flat keys for easy DataFrame loading)
        for model_name, model_data in resp.get("model_results", {}).items():
            entry[f"prob_{model_name}"] = model_data.get("probability")
            entry[f"lat_{model_name}"] = model_data.get("latency_ms")
        predictions.append(entry)

    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_path = args.output or str(PROJECT_ROOT / "eval" / "results" / f"monolith_eval_{timestamp}.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(manifest),
        "total_time_s": total_time,
        "throughput_samples_per_s": len(manifest) / total_time,
        "workers": args.workers,
        "errors": len(errors),
        "metrics": all_metrics,
        "per_model_latencies": {k: {"mean": float(np.mean(v)), "p50": float(np.percentile(v, 50)),
                                     "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}
                                 for k, v in per_model_latencies.items()},
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nMetrics saved: {output_path}")

    # Save per-file predictions (for meta-learner training)
    pred_path = output_path.replace(".json", "_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved: {pred_path} ({len(predictions)} samples)")


if __name__ == "__main__":
    main()
