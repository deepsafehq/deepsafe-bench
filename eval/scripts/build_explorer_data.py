#!/usr/bin/env python3
"""Generate the static JSON that powers the benchmark explorer.

Reads the prediction matrix and writes a compact summary to
``apps/web/public/benchmark.json``. Kept dependency-free so it runs in CI.

Usage:
    python eval/scripts/build_explorer_data.py
"""

import collections
import csv
import gzip
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from deepsafe import metrics  # noqa: E402

PREDICTIONS = pathlib.Path("eval/results/predictions_medium.csv.gz")
OUTPUT = pathlib.Path("apps/web/public/benchmark.json")

# Watermark and manifest checks answer a different question than detectors do,
# so they are reported separately rather than ranked alongside them.
PROVENANCE = {"c2pa", "sdxl_watermark", "audioseal", "videoseal", "trustmark"}


def as_float(row, column):
    try:
        return float(row.get(column, ""))
    except (TypeError, ValueError):
        return None


def main():
    with gzip.open(PREDICTIONS, "rt") as handle:
        rows = list(csv.DictReader(handle))

    score_columns = sorted(c for c in rows[0] if c.startswith("prob_"))

    models = []
    for column in score_columns:
        name = column[len("prob_"):]
        scored = [r for r in rows if as_float(r, column) is not None]
        if not scored:
            continue
        modality = collections.Counter(
            r["modality"] for r in scored
        ).most_common(1)[0][0]
        auc = metrics.roc_auc(
            (as_float(r, column), r["label"] == "fake") for r in scored
        )
        models.append({
            "name": name,
            "modality": modality,
            "n": len(scored),
            "auc": round(auc, 4) if auc is not None else None,
            "kind": "provenance" if name in PROVENANCE else "detector",
        })

    generators = []
    grouped = collections.defaultdict(list)
    for row in rows:
        if row["label"] != "fake" or not row.get("ensemble_verdict"):
            continue
        grouped[(row["modality"], row["generator"])].append(row)

    for (modality, generator), group in grouped.items():
        caught = sum(1 for r in group if r["ensemble_verdict"].lower() == "fake")
        per_model = {}
        for column in score_columns:
            name = column[len("prob_"):]
            values = [as_float(r, column) for r in group]
            values = [v for v in values if v is not None]
            if values:
                per_model[name] = round(sum(values) / len(values), 3)
        generators.append({
            "generator": generator,
            "modality": modality,
            "n": len(group),
            "caught": round(caught / len(group), 4),
            "models": per_model,
        })
    generators.sort(key=lambda g: g["caught"])

    ensembles = {}
    for modality in ("images", "audio", "video"):
        subset = [
            r for r in rows
            if r["modality"] == modality and as_float(r, "ensemble_score") is not None
        ]
        auc = metrics.roc_auc(
            (as_float(r, "ensemble_score"), r["label"] == "fake") for r in subset
        )
        ensembles[modality] = {
            "n": len(subset),
            "auc": round(auc, 4) if auc is not None else None,
        }

    all_fakes = [
        r for r in rows if r["label"] == "fake" and r.get("ensemble_verdict")
    ]
    reals = [r for r in rows if r["label"] == "real" and r.get("ensemble_verdict")]
    recall = sum(
        1 for r in all_fakes if r["ensemble_verdict"].lower() == "fake"
    ) / len(all_fakes)
    fpr = sum(
        1 for r in reals if r["ensemble_verdict"].lower() == "fake"
    ) / len(reals)

    payload = {
        "generated_from": PREDICTIONS.name,
        "samples": len(rows),
        "generator_count": len({r["generator"] for r in rows}),
        "summary": {
            "recall": round(recall, 4),
            "fpr": round(fpr, 4),
            "ensembles": ensembles,
        },
        "models": sorted(models, key=lambda m: (m["kind"], m["modality"], -(m["auc"] or 0))),
        "generators": generators,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(models)} models, {len(generators)} generator/modality groups")


if __name__ == "__main__":
    main()
