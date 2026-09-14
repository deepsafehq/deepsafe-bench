"""Markdown report generator for DeepSafe evaluation results.

Transforms a structured metrics dict into a comprehensive Markdown
report with summary tables, per-modality breakdowns, and optional
robustness sections.

Usage:
    from deepsafe_eval.report import generate_report
    generate_report(results, Path("eval/results/report.md"))
"""

from pathlib import Path

from deepsafe_eval.config import get_family

# Modality keys in preferred display order.
_MODALITIES = ("images", "audio", "video")

# Modality display labels.
_MODALITY_LABELS = {
    "images": "Image",
    "audio": "Audio",
    "video": "Video",
}


def _fmt(value, decimals: int = 4, fallback: str = "N/A") -> str:
    """Format a numeric value to fixed decimals, or return fallback.

    Args:
        value: Numeric value or None.
        decimals: Number of decimal places.
        fallback: String to return if value is None.

    Returns:
        Formatted string.
    """
    if value is None:
        return fallback
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return fallback


def _pct(value, fallback: str = "N/A") -> str:
    """Format a 0-1 float as a percentage string.

    Args:
        value: Float between 0 and 1, or None.
        fallback: Fallback string.

    Returns:
        Percentage string like '95.23%'.
    """
    if value is None:
        return fallback
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return fallback


def _build_summary_table(results: dict) -> str:
    """Build the top-level summary table across all modalities.

    Args:
        results: Full results dict with modality keys.

    Returns:
        Markdown table string.
    """
    lines = [
        "## Summary",
        "",
        "| Modality | AUC | EER | ECE | FPR@1%FNR | Samples | " "95% CI |",
        "|----------|-----|-----|-----|-----------|---------|" "-------|",
    ]

    for mod in _MODALITIES:
        mod_data = results.get(mod)
        if not mod_data:
            continue
        ens = mod_data.get("ensemble", {})
        if "error" in ens:
            lines.append(
                f"| {_MODALITY_LABELS.get(mod, mod)} | -- | -- | -- "
                f"| -- | -- | -- |"
            )
            continue

        ci = ens.get("bootstrap_ci", {})
        ci_str = (
            f"[{_fmt(ci.get('ci_95_lower'))}, " f"{_fmt(ci.get('ci_95_upper'))}]"
            if ci
            else "N/A"
        )
        n = ens.get("n_total", 0)

        lines.append(
            f"| {_MODALITY_LABELS.get(mod, mod)} "
            f"| {_fmt(ens.get('auc_roc'))} "
            f"| {_fmt(ens.get('eer'))} "
            f"| {_fmt(ens.get('ece'))} "
            f"| {_fmt(ens.get('fpr_at_1pct_fnr'))} "
            f"| {n} "
            f"| {ci_str} |"
        )

    lines.append("")
    return "\n".join(lines)


def _build_per_model_table(mod_data: dict) -> str:
    """Build the per-model metrics table for a single modality.

    Args:
        mod_data: Dict with 'per_model' key.

    Returns:
        Markdown table string.
    """
    per_model = mod_data.get("per_model", {})
    if not per_model:
        return "_No per-model data available._\n"

    lines = [
        "#### Per-Model Metrics",
        "",
        "| Model | AUC | EER | ECE | Precision | Recall | F1 |",
        "|-------|-----|-----|-----|-----------|--------|----|",
    ]

    for name in sorted(per_model.keys()):
        m = per_model[name]
        if "error" in m:
            lines.append(f"| {name} | -- | -- | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| {name} "
            f"| {_fmt(m.get('auc_roc'))} "
            f"| {_fmt(m.get('eer'))} "
            f"| {_fmt(m.get('ece'))} "
            f"| {_fmt(m.get('precision'))} "
            f"| {_fmt(m.get('recall'))} "
            f"| {_fmt(m.get('f1'))} |"
        )

    lines.append("")
    return "\n".join(lines)


def _build_per_generator_table(mod_data: dict) -> str:
    """Build the per-generator AUC table sorted worst to best.

    Args:
        mod_data: Dict with 'per_generator_auc' key.

    Returns:
        Markdown table string.
    """
    gen_auc = mod_data.get("per_generator_auc", {})
    if not gen_auc:
        return "_No per-generator data available._\n"

    # Sort by AUC ascending (worst first).
    sorted_gens = sorted(
        gen_auc.items(),
        key=lambda x: x[1] if x[1] is not None else -1,
    )

    lines = [
        "#### Per-Generator AUC (sorted worst to best)",
        "",
        "| Generator | AUC |",
        "|-----------|-----|",
    ]

    for gen, auc_val in sorted_gens:
        lines.append(f"| {gen} | {_fmt(auc_val)} |")

    lines.append("")
    return "\n".join(lines)


def _build_per_family_table(mod_data: dict) -> str:
    """Build the per-generator-family AUC summary table.

    Args:
        mod_data: Dict with 'per_family_auc' key.

    Returns:
        Markdown table string.
    """
    fam_auc = mod_data.get("per_family_auc", {})
    if not fam_auc:
        return ""

    # Sort by AUC ascending (weakest family first).
    sorted_fams = sorted(
        fam_auc.items(),
        key=lambda x: x[1] if x[1] is not None else -1,
    )

    lines = [
        "#### Per-Family AUC",
        "",
        "| Family | AUC |",
        "|--------|-----|",
    ]

    for fam, auc_val in sorted_fams:
        lines.append(f"| {fam} | {_fmt(auc_val)} |")

    lines.append("")
    return "\n".join(lines)


def _build_modality_section(modality: str, mod_data: dict) -> str:
    """Build the full section for one modality.

    Args:
        modality: Modality key (e.g. 'images').
        mod_data: Metrics dict for this modality.

    Returns:
        Markdown section string.
    """
    label = _MODALITY_LABELS.get(modality, modality)
    parts = [f"### {label}"]

    # Ensemble headline
    ens = mod_data.get("ensemble", {})
    if "error" not in ens:
        ci = ens.get("bootstrap_ci", {})
        ci_str = (
            f"[{_fmt(ci.get('ci_95_lower'))}, " f"{_fmt(ci.get('ci_95_upper'))}]"
            if ci
            else "N/A"
        )
        parts.append("")
        parts.append(
            f"**Ensemble**: AUC={_fmt(ens.get('auc_roc'))} | "
            f"EER={_fmt(ens.get('eer'))} | "
            f"ECE={_fmt(ens.get('ece'))} | "
            f"F1={_fmt(ens.get('f1'))} | "
            f"95% CI={ci_str} | "
            f"N={ens.get('n_total', 0)}"
        )
    else:
        parts.append(f"\n_Ensemble: {ens.get('error', 'No data')}_")

    parts.append("")

    # Sub-tables
    parts.append(_build_per_model_table(mod_data))
    parts.append(_build_per_generator_table(mod_data))

    family_table = _build_per_family_table(mod_data)
    if family_table:
        parts.append(family_table)

    return "\n".join(parts)


def _build_robustness_section(results: dict) -> str:
    """Build the robustness section if robustness results are present.

    Args:
        results: Full results dict (may contain 'robustness' key).

    Returns:
        Markdown section string, or empty string if no data.
    """
    robustness = results.get("robustness")
    if not robustness:
        return ""

    lines = [
        "## Robustness",
        "",
        "| Attack | Detection Rate | Mean Prob |",
        "|--------|----------------|-----------|",
    ]

    for attack, data in sorted(robustness.items()):
        det_rate = _fmt(data.get("detection_rate"), decimals=4)
        mean_prob = _fmt(data.get("mean_prob"), decimals=4)
        lines.append(f"| {attack} | {det_rate} | {mean_prob} |")

    lines.append("")
    return "\n".join(lines)


def generate_report(results: dict, output_path: Path) -> Path:
    """Generate a comprehensive Markdown evaluation report.

    Produces a report with a summary table, per-modality sections
    (models, generators, families), and an optional robustness section.

    Args:
        results: Structured metrics dict with keys like 'timestamp',
            'source', 'images', 'audio', 'video', and optionally
            'robustness'.
        output_path: Filesystem path for the output .md file.

    Returns:
        The path the report was written to.
    """
    parts = []

    # Title and metadata
    timestamp = results.get("timestamp", "unknown")
    source = results.get("source", "unknown")
    metadata = results.get("metadata", {})

    # Prefer metadata block if present (from run_metrics_only).
    if metadata:
        timestamp = metadata.get("timestamp", timestamp)
        source = metadata.get("predictions_file", source)

    parts.append("# DeepSafe Evaluation Report")
    parts.append("")
    parts.append(f"- **Generated**: {timestamp}")
    parts.append(f"- **Source**: `{source}`")

    threshold = metadata.get("threshold") if metadata else None
    if threshold is not None:
        parts.append(f"- **Threshold**: {threshold}")

    parts.append("")

    # Summary table
    parts.append(_build_summary_table(results))

    # Per-modality sections
    parts.append("## Modality Details")
    parts.append("")

    for mod in _MODALITIES:
        mod_data = results.get(mod)
        if not mod_data:
            continue
        parts.append(_build_modality_section(mod, mod_data))

    # Robustness section (optional)
    robustness_md = _build_robustness_section(results)
    if robustness_md:
        parts.append(robustness_md)

    # Footer
    parts.append("---")
    parts.append(
        "_Report generated by `deepsafe_eval.report`. " "Do not edit manually._"
    )
    parts.append("")

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(parts)
    output_path.write_text(report_text, encoding="utf-8")

    return output_path
