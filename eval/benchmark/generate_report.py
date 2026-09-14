#!/usr/bin/env python3
"""Generate Markdown benchmark reports from JSON artifacts.

Loads all evaluation artifacts from ``eval/results/`` and produces:
- ``profiling/RTX_PRO_6000_REPORT.md``  -- full technical deep-dive
- ``profiling/RTX_PRO_6000_EXECUTIVE_SUMMARY.md``  -- 2-3 page overview

Usage:
    python eval/benchmark/generate_report.py
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = _PROJECT_ROOT / "eval" / "results"
OUTPUT_DIR = _PROJECT_ROOT / "profiling"

# Expected artifact files.
_ARTIFACTS = {
    "dataset_audit": RESULTS_DIR / "dataset_audit.json",
    "gpu_tier": RESULTS_DIR / "gpu_tier_analysis.json",
    "image_metrics": RESULTS_DIR / "image_profiling.json",
    "audio_metrics": RESULTS_DIR / "audio_profiling.json",
    "video_metrics": RESULTS_DIR / "video_profiling.json",
    "latest_metrics": None,  # Auto-detected below.
}

# Test environment constants (RTX PRO 6000 benchmarking VM).
_TEST_ENV = {
    "gpu": "NVIDIA RTX PRO 6000",
    "vram": "96 GB GDDR7",
    "vcpus": 61,
    "ram": "197 GB",
    "cuda": "13.0",
    "driver": "570.86.16",
    "os": "Ubuntu 22.04 LTS",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_pct(val: Any) -> str:
    """Format a 0-1 float as a percentage string.

    Args:
        val: Numeric value (float or int), or None.

    Returns:
        Formatted string like "92.0%" or "N/A".
    """
    if val is None:
        return "N/A"
    try:
        return f"{float(val) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def fmt_auc(val: Any) -> str:
    """Format an AUC value to 4 decimal places.

    Args:
        val: Numeric AUC value, or None.

    Returns:
        Formatted string like "0.9200" or "N/A".
    """
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _safe_get(data: Dict, *keys: str, default: Any = None) -> Any:
    """Safely navigate nested dicts.

    Args:
        data: Root dict.
        *keys: Sequence of keys to traverse.
        default: Fallback value if any key is missing.

    Returns:
        The value at the nested path, or default.
    """
    current = data
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
        if current is default:
            return default
    return current


# ---------------------------------------------------------------------------
# Artifact loader
# ---------------------------------------------------------------------------


def _find_latest_metrics() -> Optional[Path]:
    """Find the most recent *-metrics.json in results dir.

    Returns:
        Path to the latest metrics file, or None.
    """
    candidates = sorted(
        RESULTS_DIR.glob("*-metrics.json"), reverse=True
    )
    return candidates[0] if candidates else None


def load_artifacts() -> Dict[str, Any]:
    """Load all available JSON artifacts.

    Returns:
        Dict mapping artifact name to parsed JSON content.
        Missing files produce None with a printed warning.
    """
    latest = _find_latest_metrics()
    if latest:
        _ARTIFACTS["latest_metrics"] = latest

    loaded: Dict[str, Any] = {}
    for name, path in _ARTIFACTS.items():
        if path is None:
            loaded[name] = None
            continue
        if not path.is_file():
            print(f"  WARN: artifact not found: {path}")
            loaded[name] = None
            continue
        with open(path, "r") as f:
            loaded[name] = json.load(f)
        print(f"  Loaded: {name} <- {path.name}")
    return loaded


# ---------------------------------------------------------------------------
# Report section builders
# ---------------------------------------------------------------------------


def _build_toc() -> str:
    """Build the table of contents."""
    return (
        "## Table of Contents\n\n"
        "1. [Test Environment](#1-test-environment)\n"
        "2. [Dataset Audit](#2-dataset-audit)\n"
        "3. [Per-Model Scorecards](#3-per-model-scorecards)\n"
        "4. [Ensemble Analysis](#4-ensemble-analysis)\n"
        "5. [GPU Tier Cost Modeling](#5-gpu-tier-cost-modeling)\n"
        "6. [Recommendations](#6-recommendations)\n"
    )


def _build_test_env() -> str:
    """Build the test environment section."""
    lines = [
        "## 1. Test Environment\n",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for key, val in _TEST_ENV.items():
        label = key.upper().replace("_", " ")
        lines.append(f"| {label} | {val} |")
    return "\n".join(lines) + "\n"


def _build_dataset_audit(audit: Optional[Dict]) -> str:
    """Build the dataset audit section.

    Args:
        audit: Parsed dataset_audit.json content.
    """
    if audit is None:
        return "## 2. Dataset Audit\n\nN/A (dataset_audit.json not found)\n"

    lines = ["## 2. Dataset Audit\n"]

    gt = audit.get("grand_totals", {})
    lines.append(
        f"**Total files:** {gt.get('total', 0):,} "
        f"(Real: {gt.get('total_real', 0):,}, "
        f"Fake: {gt.get('total_fake', 0):,})\n"
    )
    lines.append(
        f"**Generators:** {gt.get('total_generators', 0)} "
        f"({gt.get('total_low_n_generators', 0)} with N < 100)\n"
    )

    # Per-modality summary table.
    modalities = audit.get("modalities", {})
    if modalities:
        lines.append(
            "| Modality | Real | Fake | Total | Fake Ratio "
            "| Generators |"
        )
        lines.append(
            "|----------|------|------|-------|------------|"
            "------------|"
        )
        for mod_name in ("images", "audio", "video"):
            m = modalities.get(mod_name, {})
            lines.append(
                f"| {mod_name.capitalize()} "
                f"| {m.get('total_real', 0):,} "
                f"| {m.get('total_fake', 0):,} "
                f"| {m.get('total', 0):,} "
                f"| {fmt_pct(m.get('fake_ratio'))} "
                f"| {len(m.get('generators', {}))} |"
            )
        lines.append("")

    # Era distribution per modality.
    for mod_name in ("images", "audio", "video"):
        m = modalities.get(mod_name, {})
        era = m.get("era_distribution", {})
        if era:
            lines.append(f"**{mod_name.capitalize()} era distribution:**\n")
            for e in ("pre-2024", "2024", "2025+", "unknown"):
                if e in era:
                    lines.append(f"- {e}: {era[e]:,}")
            lines.append("")

    # Low-N warnings.
    all_low_n: List[str] = []
    for mod_name, m in modalities.items():
        low = m.get("low_n_generators", [])
        for g in low:
            all_low_n.append(f"{mod_name}/{g}")
    if all_low_n:
        lines.append(
            f"**Low-N warnings** ({len(all_low_n)} generators "
            f"with < 100 samples):\n"
        )
        for g in all_low_n[:20]:
            lines.append(f"- `{g}`")
        if len(all_low_n) > 20:
            lines.append(f"- ... and {len(all_low_n) - 20} more")
        lines.append("")

    return "\n".join(lines) + "\n"


def _build_model_scorecards(
    metrics: Optional[Dict],
) -> str:
    """Build per-model scorecard tables from the latest metrics.

    Args:
        metrics: Parsed *-metrics.json content.
    """
    if metrics is None:
        return (
            "## 3. Per-Model Scorecards\n\n"
            "N/A (no metrics file found)\n"
        )

    lines = ["## 3. Per-Model Scorecards\n"]

    for modality in ("image", "audio", "video"):
        mod_data = metrics.get(modality, {})
        per_model = mod_data.get("per_model", {})
        if not per_model:
            continue

        lines.append(f"### {modality.capitalize()} Models\n")
        lines.append(
            "| Model | AUC | EER | ECE | Precision | Recall | F1 "
            "| Errors | VRAM | Latency P50 |"
        )
        lines.append(
            "|-------|-----|-----|-----|-----------|--------|----"
            "|--------|------|-------------|"
        )

        for model_name in sorted(per_model.keys()):
            m = per_model[model_name]
            auc = fmt_auc(m.get("auc_roc"))
            eer = fmt_pct(m.get("eer"))
            ece = fmt_pct(m.get("ece"))
            prec = fmt_pct(m.get("precision"))
            rec = fmt_pct(m.get("recall"))
            f1 = fmt_pct(m.get("f1"))
            errors = m.get("errors", "N/A")
            vram = m.get("vram_mb", "N/A")
            if isinstance(vram, (int, float)):
                vram = f"{vram:.0f} MB"
            latency = m.get("latency_p50", "N/A")
            if isinstance(latency, (int, float)):
                latency = f"{latency:.2f}s"

            lines.append(
                f"| {model_name} | {auc} | {eer} | {ece} "
                f"| {prec} | {rec} | {f1} "
                f"| {errors} | {vram} | {latency} |"
            )
        lines.append("")

        # Unique detections from correlation data if available.
        correlation = mod_data.get("correlation", {})
        unique = correlation.get("unique_detections", {})
        if unique:
            lines.append(f"**Unique detections ({modality}):**\n")
            for model_name in sorted(unique.keys()):
                count = unique[model_name]
                lines.append(f"- {model_name}: {count}")
            lines.append("")

    return "\n".join(lines) + "\n"


def _build_ensemble_analysis(
    metrics: Optional[Dict],
) -> str:
    """Build ensemble analysis section.

    Args:
        metrics: Parsed *-metrics.json content.
    """
    if metrics is None:
        return (
            "## 4. Ensemble Analysis\n\n"
            "N/A (no metrics file found)\n"
        )

    lines = ["## 4. Ensemble Analysis\n"]

    for modality in ("image", "audio", "video"):
        mod_data = metrics.get(modality, {})
        ensemble = mod_data.get("ensemble", {})
        if not ensemble:
            continue

        lines.append(f"### {modality.capitalize()} Ensemble\n")

        # Main ensemble metrics.
        auc = fmt_auc(ensemble.get("auc_roc"))
        eer = fmt_pct(ensemble.get("eer"))
        ece = fmt_pct(ensemble.get("ece"))
        f1 = fmt_pct(ensemble.get("f1"))
        ci = ensemble.get("bootstrap_ci", {})
        ci_lower = fmt_auc(ci.get("ci_95_lower"))
        ci_upper = fmt_auc(ci.get("ci_95_upper"))
        n = ensemble.get("n_total", "N/A")

        lines.append(
            f"**AUC:** {auc} | **EER:** {eer} | **ECE:** {ece} "
            f"| **F1:** {f1} | **95% CI:** [{ci_lower}, {ci_upper}] "
            f"| **N:** {n}\n"
        )

        # RF variants if present.
        rf_variants = mod_data.get("rf_variants", {})
        if rf_variants:
            lines.append("#### RF Meta-Learner Variants\n")
            lines.append("| Variant | AUC | EER | F1 |")
            lines.append("|---------|-----|-----|----|")
            for var_name, var_data in rf_variants.items():
                lines.append(
                    f"| {var_name} "
                    f"| {fmt_auc(var_data.get('auc_roc'))} "
                    f"| {fmt_pct(var_data.get('eer'))} "
                    f"| {fmt_pct(var_data.get('f1'))} |"
                )
            lines.append("")

        # Ablation table if present.
        ablation = mod_data.get("ablation", {})
        if ablation:
            lines.append("#### Ablation Study\n")
            lines.append(
                "| Dropped Model | AUC | Delta AUC | Verdict |"
            )
            lines.append(
                "|---------------|-----|-----------|---------|"
            )
            for drop_name, drop_data in sorted(ablation.items()):
                d_auc = fmt_auc(drop_data.get("auc_roc"))
                delta = drop_data.get("delta_auc")
                if delta is not None:
                    delta_str = f"{delta:+.4f}"
                else:
                    delta_str = "N/A"
                verdict = drop_data.get("verdict", "N/A")
                lines.append(
                    f"| {drop_name} | {d_auc} "
                    f"| {delta_str} | {verdict} |"
                )
            lines.append("")

        # Prior-adjusted precision if present.
        prior = mod_data.get("prior_adjusted_precision", {})
        if prior:
            lines.append(
                "#### Prior-Adjusted Precision "
                "(at production prevalence)\n"
            )
            lines.append(
                "| Prevalence | Precision | FPR |"
            )
            lines.append(
                "|------------|-----------|-----|"
            )
            for prev_name, prev_data in sorted(prior.items()):
                lines.append(
                    f"| {prev_name} "
                    f"| {fmt_pct(prev_data.get('precision'))} "
                    f"| {fmt_pct(prev_data.get('fpr'))} |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def _build_gpu_tier(tier_data: Optional[Dict]) -> str:
    """Build the GPU tier cost modeling section.

    Args:
        tier_data: Parsed gpu_tier_analysis.json content.
    """
    if tier_data is None:
        return (
            "## 5. GPU Tier Cost Modeling\n\n"
            "N/A (gpu_tier_analysis.json not found)\n"
        )

    lines = ["## 5. GPU Tier Cost Modeling\n"]

    tiers = tier_data.get("gpu_tiers", [])
    if tiers:
        lines.append(
            "| Tier | VRAM | All Loaded? | Utilization "
            "| Per-Modality? | Monthly Cost | $/Scan |"
        )
        lines.append(
            "|------|------|-------------|-------------|"
            "---------------|--------------|--------|"
        )
        for t in tiers:
            name = t["tier"]
            vram = f"{t['vram_gb']} GB"
            fit_all = _safe_get(t, "fit_analysis", "all_loaded", "fits")
            fit_all_str = "Yes" if fit_all else "No"
            util = _safe_get(
                t, "fit_analysis", "all_loaded", "utilization_pct",
                default=0,
            )
            fit_mod = _safe_get(
                t, "fit_analysis", "per_modality", "fits"
            )
            fit_mod_str = "Yes" if fit_mod else "No"
            monthly = _safe_get(
                t, "unit_economics", "monthly_fixed_cost_usd", default=0
            )
            cps = _safe_get(
                t, "unit_economics", "cost_per_scan_usd", default=0
            )
            lines.append(
                f"| {name} | {vram} | {fit_all_str} "
                f"| {util:.1f}% | {fit_mod_str} "
                f"| ${monthly:,.0f} | ${cps:.4f} |"
            )
        lines.append("")

        # Margin projections for each tier.
        lines.append("### Margin Projections\n")
        for t in tiers:
            name = t["tier"]
            projs = _safe_get(
                t, "unit_economics", "margin_projections", default=[]
            )
            if not projs:
                continue
            lines.append(f"**{name}:**\n")
            lines.append(
                "| Mix | Revenue | Fixed | Variable "
                "| Profit | Margin |"
            )
            lines.append(
                "|-----|---------|-------|----------"
                "|--------|--------|"
            )
            for p in projs:
                lines.append(
                    f"| {p['mix']} "
                    f"| ${p['monthly_revenue']:,.0f} "
                    f"| ${p['fixed_cost']:,.0f} "
                    f"| ${p['variable_cost']:,.0f} "
                    f"| ${p['profit']:,.0f} "
                    f"| {p['margin_pct']:.1f}% |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def _build_recommendations(
    tier_data: Optional[Dict],
    metrics: Optional[Dict],
) -> str:
    """Build the recommendations section.

    Args:
        tier_data: Parsed gpu_tier_analysis.json.
        metrics: Parsed metrics JSON.
    """
    lines = ["## 6. Recommendations\n"]

    # GPU recommendation.
    if tier_data:
        tiers = tier_data.get("gpu_tiers", [])
        fitting = [
            t for t in tiers
            if _safe_get(t, "fit_analysis", "all_loaded", "fits")
        ]
        if fitting:
            cheapest = min(
                fitting,
                key=lambda t: _safe_get(
                    t, "unit_economics", "monthly_fixed_cost_usd",
                    default=float("inf"),
                ),
            )
            lines.append(
                f"1. **Recommended GPU:** {cheapest['tier']} -- "
                f"cheapest tier that fits all models simultaneously "
                f"(${_safe_get(cheapest, 'unit_economics', 'monthly_fixed_cost_usd', default=0):,.0f}/month)."
            )
        else:
            mod_fitting = [
                t for t in tiers
                if _safe_get(
                    t, "fit_analysis", "per_modality", "fits"
                )
            ]
            if mod_fitting:
                cheapest = min(
                    mod_fitting,
                    key=lambda t: _safe_get(
                        t, "unit_economics",
                        "monthly_fixed_cost_usd",
                        default=float("inf"),
                    ),
                )
                lines.append(
                    f"1. **Recommended GPU:** {cheapest['tier']} -- "
                    f"cheapest tier for per-modality loading "
                    f"(${_safe_get(cheapest, 'unit_economics', 'monthly_fixed_cost_usd', default=0):,.0f}/month). "
                    f"No single tier fits all models simultaneously."
                )
            else:
                lines.append(
                    "1. **Recommended GPU:** No evaluated tier "
                    "fits even a single modality. Consider multi-GPU."
                )
    else:
        lines.append(
            "1. **Recommended GPU:** N/A (no tier analysis data)."
        )

    # Model keep/drop.
    if metrics:
        lines.append(
            "\n2. **Model keep/drop:** Review ablation tables above. "
            "Models whose removal improves ensemble AUC should be "
            "candidates for dropping to reduce VRAM and latency."
        )
    else:
        lines.append(
            "\n2. **Model keep/drop:** N/A (no metrics data)."
        )

    lines.append(
        "\n3. **Ensemble optimization:** Retrain RF meta-learners "
        "after any model changes. Consider threshold tuning per "
        "modality to optimize for low FPR at production prevalence."
    )
    lines.append(
        "\n4. **Detection vs. recency:** Verify that 2025+ "
        "generator coverage does not degrade. Prioritize eval "
        "dataset expansion for frontier models."
    )
    lines.append(
        "\n5. **FPR at production prevalence:** At real-world "
        "prevalence (estimated 1-5% fake), prior-adjusted "
        "precision may be significantly lower than eval-set "
        "precision. Monitor deployed FPR closely."
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Full report assembly
# ---------------------------------------------------------------------------


def build_technical_report(artifacts: Dict[str, Any]) -> str:
    """Assemble the full technical report.

    Args:
        artifacts: Loaded artifact data from ``load_artifacts()``.

    Returns:
        Complete Markdown string.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    sections = [
        f"# DeepSafe GPU Benchmark Report: RTX PRO 6000\n",
        f"**Generated:** {timestamp}\n",
        "---\n",
        _build_toc(),
        "---\n",
        _build_test_env(),
        "---\n",
        _build_dataset_audit(artifacts.get("dataset_audit")),
        "---\n",
        _build_model_scorecards(artifacts.get("latest_metrics")),
        "---\n",
        _build_ensemble_analysis(artifacts.get("latest_metrics")),
        "---\n",
        _build_gpu_tier(artifacts.get("gpu_tier")),
        "---\n",
        _build_recommendations(
            artifacts.get("gpu_tier"),
            artifacts.get("latest_metrics"),
        ),
    ]
    return "\n".join(sections)


def build_executive_summary(artifacts: Dict[str, Any]) -> str:
    """Assemble the 2-3 page executive summary.

    Args:
        artifacts: Loaded artifact data from ``load_artifacts()``.

    Returns:
        Complete Markdown string.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    tier_data = artifacts.get("gpu_tier")
    metrics = artifacts.get("latest_metrics")
    audit = artifacts.get("dataset_audit")

    lines = [
        "# DeepSafe GPU Benchmark: Executive Summary\n",
        f"**Generated:** {timestamp}\n",
        "---\n",
        "## Overview\n",
        (
            "This report summarizes the GPU benchmarking evaluation "
            "of DeepSafe's multi-modal deepfake detection platform "
            "on an NVIDIA RTX PRO 6000 (96 GB VRAM). The evaluation "
            "covers 19 detection models across image (7), video (9), "
            "and audio (5) modalities, plus 4 provenance services."
        ),
        "",
    ]

    # Dataset summary.
    if audit:
        gt = audit.get("grand_totals", {})
        lines.append(
            f"The evaluation dataset contains "
            f"**{gt.get('total', 0):,} files** across "
            f"**{gt.get('total_generators', 0)} generators**, "
            f"spanning pre-2024 through 2025+ model eras.\n"
        )

    lines.append("---\n")
    lines.append("## Key Findings\n")

    # 1. Recommended GPU tier.
    if tier_data:
        tiers = tier_data.get("gpu_tiers", [])
        fitting = [
            t for t in tiers
            if _safe_get(t, "fit_analysis", "all_loaded", "fits")
        ]
        if fitting:
            cheapest = min(
                fitting,
                key=lambda t: _safe_get(
                    t, "unit_economics", "monthly_fixed_cost_usd",
                    default=float("inf"),
                ),
            )
            util = _safe_get(
                cheapest, "fit_analysis", "all_loaded",
                "utilization_pct", default=0,
            )
            monthly = _safe_get(
                cheapest, "unit_economics",
                "monthly_fixed_cost_usd", default=0,
            )
            lines.append(
                f"**1. Recommended GPU tier:** {cheapest['tier']} -- "
                f"{cheapest['vram_gb']} GB VRAM, {util:.0f}% utilization "
                f"with all models loaded, ${monthly:,.0f}/month "
                f"total cost.\n"
            )
        else:
            lines.append(
                "**1. Recommended GPU tier:** No single GPU fits all "
                "models. Per-modality loading or multi-GPU required.\n"
            )
    else:
        lines.append(
            "**1. Recommended GPU tier:** Data unavailable.\n"
        )

    # 2. Model keep/drop.
    lines.append(
        "**2. Model keep/drop list:** See ablation study in the "
        "technical report. Models whose removal improves or "
        "negligibly impacts ensemble AUC are candidates for "
        "dropping to save VRAM and reduce latency.\n"
    )

    # 3. Ensemble optimization.
    if metrics:
        for modality in ("image", "audio", "video"):
            mod_data = metrics.get(modality, {})
            ensemble = mod_data.get("ensemble", {})
            if ensemble:
                auc = fmt_auc(ensemble.get("auc_roc"))
                eer = fmt_pct(ensemble.get("eer"))
                lines.append(
                    f"- **{modality.capitalize()} ensemble:** "
                    f"AUC={auc}, EER={eer}"
                )
        lines.append("")

    lines.append(
        "**3. Ensemble optimization:** RF meta-learners outperform "
        "simple averaging. Threshold tuning per modality recommended "
        "for production deployment.\n"
    )

    # 4. Detection vs. recency.
    lines.append(
        "**4. Detection vs. recency:** 2025+ frontier generators "
        "(GPT Image 1, Midjourney 7, Ideogram 3.0) require "
        "continued dataset expansion to maintain detection rates.\n"
    )

    # 5. FPR at production prevalence.
    lines.append(
        "**5. FPR at production prevalence:** At real-world "
        "prevalence (1-5% fake content), Bayesian base-rate "
        "adjustment significantly reduces effective precision. "
        "Deploying with a higher detection threshold is recommended "
        "to minimize false positives.\n"
    )

    lines.append("---\n")
    lines.append("## Next Steps\n")
    lines.append(
        "1. Finalize GPU tier selection and provision cloud VM.\n"
        "2. Run full production load test with selected tier.\n"
        "3. Retrain ensemble meta-learners with expanded 2025+ data.\n"
        "4. Implement per-modality threshold tuning for production.\n"
        "5. Establish continuous evaluation pipeline for new "
        "generator families.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Load artifacts and generate both reports."""
    print("Generating benchmark reports...")
    artifacts = load_artifacts()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Technical report.
    tech_path = OUTPUT_DIR / "RTX_PRO_6000_REPORT.md"
    tech_report = build_technical_report(artifacts)
    with open(tech_path, "w") as f:
        f.write(tech_report)
    print(f"\n  Technical report -> {tech_path}")

    # Executive summary.
    exec_path = OUTPUT_DIR / "RTX_PRO_6000_EXECUTIVE_SUMMARY.md"
    exec_summary = build_executive_summary(artifacts)
    with open(exec_path, "w") as f:
        f.write(exec_summary)
    print(f"  Executive summary -> {exec_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
