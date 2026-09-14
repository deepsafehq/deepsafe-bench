#!/usr/bin/env python3
"""Master eval: run the full DeepSafe eval dataset through production API.

Pure black-box production client. Treats localhost:8000 as an external
service -- no docker access, no internal DB calls, no quota resets.

Supports multiple API keys for quota rotation and automatic checkpointing
for resume on interruption.

Usage:
    DEEPSAFE_EVAL_KEYS="ds_live_key1,ds_live_key2" python3 eval/run_master_eval.py

Env vars:
    DEEPSAFE_EVAL_KEYS  - Comma-separated API keys (required)
    DEEPSAFE_API_URL    - API base (default: http://localhost:8000)
    EVAL_WORKERS        - Concurrent workers (default: 15)
    EVAL_DATASET        - Dataset path override
    EVAL_OUTPUT         - Results output dir (default: eval/results)
"""

import concurrent.futures
import json
import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = os.getenv("DEEPSAFE_API_URL", "http://localhost:8000")
DETECT_URL = f"{API_BASE}/v1/detect"
WORKERS = int(os.getenv("EVAL_WORKERS", "15"))
from deepsafe_eval.config import DATASET_ROOT as _DATASET_ROOT

DATASET_DIR = Path(os.getenv("EVAL_DATASET", str(_DATASET_ROOT / "master_eval")))
OUTPUT_DIR = Path(os.getenv("EVAL_OUTPUT", "eval/results"))

MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".mp4": "video/mp4",
    ".avi": "video/avi",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".m4a": "audio/m4a",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalFile:
    """A single file to evaluate."""

    path: str
    modality: str
    label: str
    generator: str
    mime_type: str


@dataclass
class EvalResult:
    """Result for a single file."""

    path: str
    modality: str
    label: str
    generator: str
    verdict: str
    confidence: float
    correct: bool
    duration_s: float
    error: Optional[str] = None
    detection_id: Optional[str] = None


# ---------------------------------------------------------------------------
# API key manager (thread-safe rotation on quota exhaustion)
# ---------------------------------------------------------------------------


class KeyManager:
    """Manages multiple API keys with automatic rotation on 402."""

    def __init__(self, keys: list[str]):
        self._keys = keys
        self._idx = 0
        self._lock = threading.Lock()
        self._exhausted = set()

    def get_headers(self) -> dict:
        with self._lock:
            if self._idx >= len(self._keys):
                return {}
            return {"Authorization": f"Bearer {self._keys[self._idx]}"}

    def report_quota_hit(self) -> bool:
        """Mark current key as exhausted, rotate to next. Returns False if all exhausted."""
        with self._lock:
            self._exhausted.add(self._idx)
            self._idx += 1
            if self._idx >= len(self._keys):
                return False
            key_num = self._idx + 1
            print(
                f"\n  >> Key {key_num - 1} quota exhausted, "
                f"switching to key {key_num}/{len(self._keys)}"
            )
            return True

    def all_exhausted(self) -> bool:
        with self._lock:
            return len(self._exhausted) >= len(self._keys)

    @property
    def current_key_num(self) -> int:
        with self._lock:
            return self._idx + 1


key_manager: KeyManager


# ---------------------------------------------------------------------------
# Build file manifest
# ---------------------------------------------------------------------------


def build_manifest() -> list[EvalFile]:
    """Scan master_eval/ and build list of all files with metadata."""
    files = []
    for modality_dir_name in ("images", "audio", "video"):
        modality = "image" if modality_dir_name == "images" else modality_dir_name
        modality_path = DATASET_DIR / modality_dir_name

        for label in ("real", "fake"):
            label_path = modality_path / label
            if not label_path.exists():
                continue

            for gen_dir in sorted(label_path.iterdir()):
                if not gen_dir.is_dir():
                    continue
                generator = gen_dir.name

                for fp in sorted(gen_dir.iterdir()):
                    if fp.name.startswith("."):
                        continue
                    real_path = fp.resolve() if fp.is_symlink() else fp
                    if not real_path.exists() or not real_path.is_file():
                        continue

                    ext = fp.suffix.lower()
                    mime = MIME_MAP.get(ext)
                    if not mime:
                        continue

                    files.append(
                        EvalFile(
                            path=str(fp),
                            modality=modality,
                            label=label,
                            generator=generator,
                            mime_type=mime,
                        )
                    )

    return files


# ---------------------------------------------------------------------------
# API call with retry and key rotation
# ---------------------------------------------------------------------------


def detect_file(ef: EvalFile, max_retries: int = 3) -> EvalResult:
    """Send a single file to the production API."""
    if key_manager.all_exhausted():
        return EvalResult(
            path=ef.path,
            modality=ef.modality,
            label=ef.label,
            generator=ef.generator,
            verdict="error",
            confidence=0.0,
            correct=False,
            duration_s=0,
            error="All API keys exhausted",
        )

    for attempt in range(max_retries + 1):
        try:
            headers = key_manager.get_headers()
            if not headers:
                return EvalResult(
                    path=ef.path,
                    modality=ef.modality,
                    label=ef.label,
                    generator=ef.generator,
                    verdict="error",
                    confidence=0.0,
                    correct=False,
                    duration_s=0,
                    error="All API keys exhausted",
                )

            with open(ef.path, "rb") as f:
                t0 = time.time()
                resp = requests.post(
                    DETECT_URL,
                    headers=headers,
                    files={"file": (os.path.basename(ef.path), f, ef.mime_type)},
                    timeout=300,
                )
                duration = time.time() - t0

            if resp.status_code == 200:
                body = resp.json()
                verdict = body.get("verdict", "unknown")
                confidence = body.get("confidence", 0.0)
                correct = (verdict == "fake" and ef.label == "fake") or (
                    verdict == "real" and ef.label == "real"
                )
                return EvalResult(
                    path=ef.path,
                    modality=ef.modality,
                    label=ef.label,
                    generator=ef.generator,
                    verdict=verdict,
                    confidence=confidence,
                    correct=correct,
                    duration_s=duration,
                    detection_id=body.get("id"),
                )

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 15))
                time.sleep(retry_after)
                continue

            if resp.status_code == 402:
                has_more = key_manager.report_quota_hit()
                if has_more:
                    time.sleep(2)
                    continue
                return EvalResult(
                    path=ef.path,
                    modality=ef.modality,
                    label=ef.label,
                    generator=ef.generator,
                    verdict="error",
                    confidence=0.0,
                    correct=False,
                    duration_s=duration,
                    error="All API keys quota exhausted",
                )

            if resp.status_code in (502, 503, 504) and attempt < max_retries:
                time.sleep(2**attempt)
                continue

            try:
                detail = resp.json().get("detail", {}).get("message", resp.text[:200])
            except Exception:
                detail = resp.text[:200]

            return EvalResult(
                path=ef.path,
                modality=ef.modality,
                label=ef.label,
                generator=ef.generator,
                verdict="error",
                confidence=0.0,
                correct=False,
                duration_s=duration,
                error=f"HTTP {resp.status_code}: {detail}",
            )

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(2**attempt)
                continue
            return EvalResult(
                path=ef.path,
                modality=ef.modality,
                label=ef.label,
                generator=ef.generator,
                verdict="error",
                confidence=0.0,
                correct=False,
                duration_s=300,
                error="Timeout after 300s",
            )
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2**attempt)
                continue
            return EvalResult(
                path=ef.path,
                modality=ef.modality,
                label=ef.label,
                generator=ef.generator,
                verdict="error",
                confidence=0.0,
                correct=False,
                duration_s=0,
                error=str(e),
            )

    return EvalResult(
        path=ef.path,
        modality=ef.modality,
        label=ef.label,
        generator=ef.generator,
        verdict="error",
        confidence=0.0,
        correct=False,
        duration_s=0,
        error="Max retries exceeded",
    )


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

_checkpoint_lock = threading.Lock()


def load_checkpoint(path: Path) -> tuple[set[str], list[EvalResult]]:
    """Load already-processed file paths and results from checkpoint."""
    if not path.exists():
        return set(), []
    with open(path) as f:
        data = json.load(f)
    results = [EvalResult(**r) for r in data.get("results", [])]
    paths = {r.path for r in results}
    return paths, results


def save_checkpoint(path: Path, results: list[EvalResult], manifest_size: int):
    """Atomically save results to checkpoint file."""
    with _checkpoint_lock:
        data = {
            "api": API_BASE,
            "total_files": manifest_size,
            "completed": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [asdict(r) for r in results],
        }
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        tmp.rename(path)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(results: list[EvalResult]) -> dict:
    """Compute accuracy, precision, recall, F1 per generator and overall."""

    def _calc(group: list[EvalResult]) -> dict:
        total = len(group)
        errors = sum(1 for r in group if r.verdict == "error")
        valid = [r for r in group if r.verdict != "error"]
        if not valid:
            return {"total": total, "valid": 0, "errors": errors, "accuracy": 0}

        correct = sum(1 for r in valid if r.correct)
        tp = sum(1 for r in valid if r.label == "fake" and r.verdict == "fake")
        fp = sum(1 for r in valid if r.label == "real" and r.verdict == "fake")
        tn = sum(1 for r in valid if r.label == "real" and r.verdict == "real")
        fn = sum(1 for r in valid if r.label == "fake" and r.verdict == "real")

        accuracy = correct / len(valid) if valid else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        avg_conf = sum(r.confidence for r in valid) / len(valid)
        avg_dur = sum(r.duration_s for r in valid) / len(valid)

        fake_rs = [r for r in valid if r.label == "fake"]
        real_rs = [r for r in valid if r.label == "real"]
        avg_fake_conf = (
            sum(r.confidence for r in fake_rs) / len(fake_rs) if fake_rs else 0
        )
        avg_real_conf = (
            sum(r.confidence for r in real_rs) / len(real_rs) if real_rs else 0
        )

        return {
            "total": total,
            "valid": len(valid),
            "errors": errors,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "avg_confidence": round(avg_conf, 4),
            "avg_fake_confidence": round(avg_fake_conf, 4),
            "avg_real_confidence": round(avg_real_conf, 4),
            "avg_duration_s": round(avg_dur, 2),
        }

    metrics = {}
    metrics["overall"] = _calc(results)

    by_modality = defaultdict(list)
    for r in results:
        by_modality[r.modality].append(r)
    metrics["by_modality"] = {m: _calc(rs) for m, rs in sorted(by_modality.items())}

    by_gen = defaultdict(list)
    for r in results:
        by_gen[f"{r.modality}/{r.label}/{r.generator}"].append(r)
    metrics["by_generator"] = {k: _calc(rs) for k, rs in sorted(by_gen.items())}

    return metrics


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Thread-safe progress display."""

    def __init__(self, total: int, already_done: int = 0):
        self.total = total
        self.completed = 0
        self.correct = 0
        self.errors = 0
        self.start_time = time.time()
        self._already_done = already_done
        self._lock = threading.Lock()

    def update(self, result: EvalResult):
        with self._lock:
            self.completed += 1
            if result.verdict == "error":
                self.errors += 1
            elif result.correct:
                self.correct += 1

            if self.completed % 25 == 0 or self.completed == self.total:
                elapsed = time.time() - self.start_time
                rate = self.completed / elapsed if elapsed > 0 else 0
                eta_s = (self.total - self.completed) / rate if rate > 0 else 0
                eta_m = eta_s / 60
                valid = self.completed - self.errors
                acc = self.correct / valid if valid > 0 else 0
                done_total = self._already_done + self.completed
                grand = self._already_done + self.total
                print(
                    f"  [{done_total:>5}/{grand}] "
                    f"acc={acc:.3f} errs={self.errors} "
                    f"rate={rate:.1f}/s ETA={eta_m:.0f}m "
                    f"key={key_manager.current_key_num}",
                    flush=True,
                )


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_report(metrics: dict):
    """Print human-readable eval report."""
    o = metrics["overall"]
    print(f"\n{'=' * 70}")
    print(f"  OVERALL RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total: {o['total']}  Valid: {o['valid']}  Errors: {o['errors']}")
    print(f"  Accuracy:  {o['accuracy']:.4f}")
    print(f"  Precision: {o['precision']:.4f}  (fake detection rate)")
    print(f"  Recall:    {o['recall']:.4f}  (fake catch rate)")
    print(f"  F1 Score:  {o['f1']:.4f}")
    print(f"  TP={o['tp']}  FP={o['fp']}  TN={o['tn']}  FN={o['fn']}")
    print(f"  Avg latency: {o['avg_duration_s']:.1f}s")

    print(f"\n{'=' * 70}")
    print(f"  PER-MODALITY RESULTS")
    print(f"{'=' * 70}")
    for mod, m in metrics.get("by_modality", {}).items():
        print(f"\n  --- {mod.upper()} ---")
        print(f"  Total: {m['total']}  Valid: {m['valid']}  Errors: {m['errors']}")
        print(
            f"  Accuracy:  {m['accuracy']:.4f}  "
            f"Precision: {m['precision']:.4f}  "
            f"Recall: {m['recall']:.4f}  "
            f"F1: {m['f1']:.4f}"
        )
        print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
        print(
            f"  Avg confidence: fake={m['avg_fake_confidence']:.4f}  "
            f"real={m['avg_real_confidence']:.4f}"
        )
        print(f"  Avg latency: {m['avg_duration_s']:.1f}s")

    print(f"\n{'=' * 70}")
    print(f"  PER-GENERATOR BREAKDOWN")
    print(f"{'=' * 70}")

    gen_data = metrics.get("by_generator", {})
    current_mod = ""
    for key in sorted(gen_data.keys()):
        parts = key.split("/")
        mod, label, gen = parts[0], parts[1], parts[2] if len(parts) > 2 else "?"

        if mod != current_mod:
            current_mod = mod
            print(f"\n  [{mod.upper()}]")
            print(
                f"  {'label':<5} {'generator':<35} {'acc':>6} {'n':>5} "
                f"{'errs':>5} {'avg_conf':>9}"
            )
            print(f"  {'-' * 68}")

        g = gen_data[key]
        acc_str = f"{g['accuracy']:.3f}" if g["valid"] > 0 else "  N/A"
        errs = g["errors"]
        print(
            f"  {label:<5} {gen:<35} {acc_str:>6} {g['total']:>5} "
            f"{errs:>5} {g['avg_confidence']:>9.3f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global key_manager

    raw_keys = os.getenv("DEEPSAFE_EVAL_KEYS", "")
    if not raw_keys:
        print("ERROR: Set DEEPSAFE_EVAL_KEYS (comma-separated)")
        sys.exit(1)

    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    key_manager = KeyManager(keys)

    OUTPUT_DIR.mkdir(exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "checkpoint.json"
    report_path = OUTPUT_DIR / "eval_report.json"

    # Build manifest
    print("Scanning dataset...")
    manifest = build_manifest()

    by_mod = defaultdict(int)
    for f in manifest:
        by_mod[f"{f.modality}/{f.label}"] += 1
    for k, v in sorted(by_mod.items()):
        print(f"  {k}: {v}")
    print(f"  TOTAL: {len(manifest)}")

    # Load checkpoint
    done_paths, prev_results = load_checkpoint(checkpoint_path)
    remaining = [f for f in manifest if f.path not in done_paths]

    print(f"\n  Resumed from checkpoint: {len(prev_results)} done")
    print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("\nAll files processed. Computing metrics...")
        metrics = compute_metrics(prev_results)
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print_report(metrics)
        print(f"\nReport: {report_path}")
        return

    # Verify production API health
    print(f"\nVerifying {API_BASE}...")
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=15)
        body = resp.json()
        print(f"  Status: {body.get('status')}")
        for mod, s in body.get("media_types", {}).items():
            print(f"    {mod}: {s}")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Adaptive concurrency per modality to stay under Cloudflare's
    # 100-second origin timeout. Image hits 10 models (~47s solo),
    # so only 1 worker. Audio/video are faster, allow 3 workers.
    MODALITY_WORKERS = {"audio": 3, "video": 3, "image": 1}
    MODALITY_ORDER = ["audio", "video", "image"]  # fastest first

    # Group remaining files by modality
    by_mod_remaining = defaultdict(list)
    for ef in remaining:
        by_mod_remaining[ef.modality].append(ef)

    total_remaining = len(remaining)
    print(f"\n{'=' * 70}")
    print(f"  PRODUCTION EVAL")
    print(f"  API: {API_BASE}")
    print(f"  Files: {total_remaining} remaining")
    print(f"  Keys: {len(keys)} available (pro tier, 10K each)")
    print(
        f"  Concurrency: audio={MODALITY_WORKERS['audio']}, "
        f"video={MODALITY_WORKERS['video']}, "
        f"image={MODALITY_WORKERS['image']}"
    )
    print(f"{'=' * 70}\n")

    all_results = list(prev_results)
    progress = ProgressTracker(total_remaining, len(prev_results))
    batch_count = 0
    stopped = False

    for modality in MODALITY_ORDER:
        mod_files = by_mod_remaining.get(modality, [])
        if not mod_files or stopped:
            continue

        workers = MODALITY_WORKERS.get(modality, 1)
        print(
            f"\n  --- Processing {modality.upper()} "
            f"({len(mod_files)} files, {workers} workers) ---"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(detect_file, ef): ef for ef in mod_files}

            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                all_results.append(result)
                progress.update(result)
                batch_count += 1

                if batch_count % 50 == 0:
                    save_checkpoint(checkpoint_path, all_results, len(manifest))

                if (
                    key_manager.all_exhausted()
                    and result.error
                    and "exhausted" in result.error
                ):
                    print("\n  All API keys exhausted. Stopping eval.")
                    for f in future_map:
                        f.cancel()
                    stopped = True
                    break

        # Checkpoint between modalities
        save_checkpoint(checkpoint_path, all_results, len(manifest))

    # Final checkpoint
    save_checkpoint(checkpoint_path, all_results, len(manifest))

    # Compute and save metrics
    print(f"\n{'=' * 70}")
    print("  Computing metrics...")
    print(f"{'=' * 70}")

    metrics = compute_metrics(all_results)
    metrics["_meta"] = {
        "api": API_BASE,
        "dataset": str(DATASET_DIR),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workers": WORKERS,
        "total_manifest": len(manifest),
        "total_processed": len(all_results),
        "keys_used": key_manager.current_key_num,
    }

    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print_report(metrics)
    print(f"\nFull report: {report_path}")
    print(f"Checkpoint:  {checkpoint_path}")


if __name__ == "__main__":
    main()
