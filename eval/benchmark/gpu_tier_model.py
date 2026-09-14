#!/usr/bin/env python3
"""GPU tier cost modeling for DeepSafe deployment.

Loads per-model profiling data (VRAM, latency) from
``eval/results/{image,audio,video}_profiling.json`` and projects
deployment feasibility and unit economics across GPU tiers.

Usage:
    python eval/benchmark/gpu_tier_model.py

Produces ``eval/results/gpu_tier_analysis.json``.
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = _PROJECT_ROOT / "eval" / "results"

_PROFILING_FILES = {
    "image": RESULTS_DIR / "image_profiling.json",
    "audio": RESULTS_DIR / "audio_profiling.json",
    "video": RESULTS_DIR / "video_profiling.json",
}

# ---------------------------------------------------------------------------
# GPU tier definitions
# ---------------------------------------------------------------------------

ELECTRICITY_RATE_KWH = 0.12   # USD per kWh
AVERAGE_LOAD_FACTOR = 0.50    # assume 50% average GPU utilization
HOURS_PER_MONTH = 730         # ~365.25 * 24 / 12


@dataclass(frozen=True)
class GpuTier:
    """Specification for a GPU rental tier."""

    name: str
    vram_gb: int
    hourly_rate: float   # USD
    monthly_rate: float  # USD
    tdp_watts: int       # thermal design power

    @property
    def vram_mb(self) -> int:
        """VRAM capacity in megabytes."""
        return self.vram_gb * 1024

    @property
    def usable_vram_mb(self) -> float:
        """90% of total VRAM (safety margin for driver/OS overhead)."""
        return self.vram_mb * 0.90

    @property
    def monthly_electricity(self) -> float:
        """Estimated monthly electricity cost at average load.

        Returns:
            USD per month.
        """
        avg_watts = self.tdp_watts * AVERAGE_LOAD_FACTOR
        kwh_per_month = (avg_watts / 1000.0) * HOURS_PER_MONTH
        return round(kwh_per_month * ELECTRICITY_RATE_KWH, 2)

    @property
    def monthly_fixed_cost(self) -> float:
        """Total monthly fixed cost (rental + electricity).

        Returns:
            USD per month.
        """
        return round(self.monthly_rate + self.monthly_electricity, 2)


GPU_TIERS: List[GpuTier] = [
    GpuTier("RTX_4090_24GB", 24, 0.35, 250, 450),
    GpuTier("L40S_48GB", 48, 0.80, 575, 350),
    GpuTier("A100_80GB", 80, 2.00, 1440, 300),
    GpuTier("RTX_PRO_6000_96GB", 96, 2.50, 1800, 600),
    GpuTier("H100_80GB", 80, 3.50, 2520, 700),
    GpuTier("2x_RTX_4090_48GB", 48, 0.70, 500, 900),
]

# ---------------------------------------------------------------------------
# Pricing tiers (for break-even analysis)
# ---------------------------------------------------------------------------

PRICING_TIERS = {
    "Free": {"monthly_revenue": 0.0, "scans_per_month": 0},
    "Starter": {"monthly_revenue": 29.0, "scans_per_month": 2000},
    "Pro": {"monthly_revenue": 99.0, "scans_per_month": 10000},
}

# Customer mix scenarios: (starter_count, pro_count)
CUSTOMER_MIXES = [
    (50, 10),
    (100, 25),
    (200, 50),
]

# ---------------------------------------------------------------------------
# Model -> modality mapping
# ---------------------------------------------------------------------------

_IMAGE_MODELS = {
    "npr", "yermandy", "universal", "aide", "fsd", "effort", "cospy",
}
_VIDEO_MODELS = {
    "fakestormer", "sbi", "dfd_fcg", "pwtf_dvd", "lipfd", "recce",
    "mintime",
}
_AUDIO_MODELS = {
    "shiftyspeech", "safeear", "sonics", "nes2net",
}


def _model_to_modality(name: str) -> str:
    """Map a model short name to its modality.

    Args:
        name: Model short name (e.g. "npr", "fakestormer").

    Returns:
        One of "image", "video", "audio", or "unknown".
    """
    if name in _IMAGE_MODELS:
        return "image"
    if name in _VIDEO_MODELS:
        return "video"
    if name in _AUDIO_MODELS:
        return "audio"
    return "unknown"


# ---------------------------------------------------------------------------
# Profiling data loader
# ---------------------------------------------------------------------------


def _load_profiling() -> Dict[str, List[Dict[str, Any]]]:
    """Load profiling JSON files for each modality.

    Returns:
        Dict mapping modality name to list of per-model profiling
        records.  Missing files produce an empty list with a warning.
    """
    data: Dict[str, List[Dict[str, Any]]] = {}
    for modality, path in _PROFILING_FILES.items():
        if not path.is_file():
            print(f"  WARN: profiling file not found: {path}")
            data[modality] = []
            continue
        with open(path, "r") as f:
            raw = json.load(f)
        # Handle both list-of-dicts and dict-with-models key.
        if isinstance(raw, list):
            data[modality] = raw
        elif isinstance(raw, dict) and "models" in raw:
            data[modality] = raw["models"]
        elif isinstance(raw, dict):
            # Flat dict keyed by model name.
            data[modality] = [
                {"name": k, **v} for k, v in raw.items()
            ]
        else:
            print(f"  WARN: unexpected format in {path}")
            data[modality] = []
    return data


def _extract_model_stats(
    profiling: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Flatten all modality profiling data into a unified model list.

    Each entry gets ``modality``, ``name``, ``delta_vram_mb``, and
    ``latency_p50_s`` fields.

    Args:
        profiling: Output of ``_load_profiling()``.

    Returns:
        List of per-model stat dicts.
    """
    models = []
    for modality, records in profiling.items():
        for rec in records:
            name = rec.get("name", rec.get("service_name", "unknown"))
            # Accept various key names for VRAM delta.
            vram = (
                rec.get("delta_vram_mb")
                or rec.get("vram_delta_mb")
                or rec.get("vram_mb", 0)
            )
            # Accept various key names for latency.
            latency = (
                rec.get("latency_p50_s")
                or rec.get("latency_p50")
                or rec.get("p50")
                or rec.get("latency_seconds")
                or 0
            )
            models.append({
                "name": name,
                "modality": _model_to_modality(name),
                "delta_vram_mb": float(vram),
                "latency_p50_s": float(latency),
            })
    return models


# ---------------------------------------------------------------------------
# Fit analysis
# ---------------------------------------------------------------------------


def _fit_analysis(
    tier: GpuTier,
    models: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Determine whether all models fit on a given GPU tier.

    Checks two strategies:
    - **all_loaded**: every model loaded simultaneously.
    - **per_modality**: only one modality loaded at a time (the worst
      case modality must fit).

    Args:
        tier: GPU tier to evaluate.
        models: Unified model stats from ``_extract_model_stats()``.

    Returns:
        Dict with fit booleans, VRAM totals, and utilization percentages.
    """
    total_vram = sum(m["delta_vram_mb"] for m in models)
    usable = tier.usable_vram_mb

    # Per-modality VRAM totals.
    modality_vram: Dict[str, float] = {}
    for m in models:
        mod = m["modality"]
        modality_vram[mod] = modality_vram.get(mod, 0) + m["delta_vram_mb"]

    max_modality_vram = max(modality_vram.values()) if modality_vram else 0

    return {
        "all_loaded": {
            "total_vram_mb": round(total_vram, 1),
            "usable_vram_mb": round(usable, 1),
            "fits": total_vram <= usable,
            "utilization_pct": round(
                (total_vram / usable) * 100, 1
            ) if usable > 0 else 0.0,
        },
        "per_modality": {
            "max_modality": max(
                modality_vram, key=modality_vram.get, default="N/A"
            ),
            "max_modality_vram_mb": round(max_modality_vram, 1),
            "usable_vram_mb": round(usable, 1),
            "fits": max_modality_vram <= usable,
            "utilization_pct": round(
                (max_modality_vram / usable) * 100, 1
            ) if usable > 0 else 0.0,
            "modality_breakdown": {
                k: round(v, 1) for k, v in modality_vram.items()
            },
        },
    }


# ---------------------------------------------------------------------------
# Unit economics
# ---------------------------------------------------------------------------


def _avg_latency(models: List[Dict[str, Any]]) -> float:
    """Compute mean latency across all models.

    Args:
        models: Unified model stats.

    Returns:
        Mean latency in seconds, or 0 if no data.
    """
    latencies = [m["latency_p50_s"] for m in models if m["latency_p50_s"] > 0]
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies)


def _unit_economics(
    tier: GpuTier,
    models: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute cost-per-scan and break-even projections.

    Args:
        tier: GPU tier to evaluate.
        models: Unified model stats.

    Returns:
        Dict with cost_per_scan, break_even per pricing tier,
        and margin projections for customer mixes.
    """
    avg_lat = _avg_latency(models)
    cost_per_scan = round(
        tier.hourly_rate * avg_lat / 3600.0, 6
    ) if avg_lat > 0 else 0.0

    # Break-even: how many customers per tier to cover monthly fixed cost.
    break_even: Dict[str, Optional[float]] = {}
    for tier_name, tier_info in PRICING_TIERS.items():
        revenue = tier_info["monthly_revenue"]
        if revenue <= 0:
            break_even[tier_name] = None  # Free tier has no revenue.
        else:
            scans = tier_info["scans_per_month"]
            variable_cost = cost_per_scan * scans
            net_per_customer = revenue - variable_cost
            if net_per_customer > 0:
                break_even[tier_name] = round(
                    tier.monthly_fixed_cost / net_per_customer, 1
                )
            else:
                break_even[tier_name] = None  # Variable cost exceeds price.

    # Margin projections at different customer mixes.
    margin_projections: List[Dict[str, Any]] = []
    for n_starter, n_pro in CUSTOMER_MIXES:
        label = f"{n_starter}S+{n_pro}P"
        revenue = (
            n_starter * PRICING_TIERS["Starter"]["monthly_revenue"]
            + n_pro * PRICING_TIERS["Pro"]["monthly_revenue"]
        )
        scans = (
            n_starter * PRICING_TIERS["Starter"]["scans_per_month"]
            + n_pro * PRICING_TIERS["Pro"]["scans_per_month"]
        )
        variable = cost_per_scan * scans
        profit = revenue - tier.monthly_fixed_cost - variable
        margin_pct = round(
            (profit / revenue) * 100, 1
        ) if revenue > 0 else 0.0

        margin_projections.append({
            "mix": label,
            "n_starter": n_starter,
            "n_pro": n_pro,
            "monthly_revenue": round(revenue, 2),
            "monthly_scans": scans,
            "variable_cost": round(variable, 2),
            "fixed_cost": tier.monthly_fixed_cost,
            "profit": round(profit, 2),
            "margin_pct": margin_pct,
        })

    return {
        "avg_latency_s": round(avg_lat, 3),
        "cost_per_scan_usd": cost_per_scan,
        "monthly_fixed_cost_usd": tier.monthly_fixed_cost,
        "monthly_electricity_usd": tier.monthly_electricity,
        "break_even_customers": break_even,
        "margin_projections": margin_projections,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis() -> Dict[str, Any]:
    """Run the full GPU tier analysis.

    Returns:
        Complete analysis dict ready for JSON serialization.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    profiling = _load_profiling()
    models = _extract_model_stats(profiling)

    if not models:
        print("  WARN: no model profiling data loaded.")

    tiers_analysis: List[Dict[str, Any]] = []
    for tier in GPU_TIERS:
        fit = _fit_analysis(tier, models)
        econ = _unit_economics(tier, models)
        tiers_analysis.append({
            "tier": tier.name,
            "vram_gb": tier.vram_gb,
            "hourly_rate_usd": tier.hourly_rate,
            "monthly_rate_usd": tier.monthly_rate,
            "tdp_watts": tier.tdp_watts,
            "fit_analysis": fit,
            "unit_economics": econ,
        })

    return {
        "metadata": {
            "timestamp": timestamp,
            "profiling_files": {
                k: str(v) for k, v in _PROFILING_FILES.items()
            },
            "n_models_loaded": len(models),
            "electricity_rate_kwh": ELECTRICITY_RATE_KWH,
            "average_load_factor": AVERAGE_LOAD_FACTOR,
        },
        "models_summary": models,
        "gpu_tiers": tiers_analysis,
    }


def _print_summary(analysis: Dict[str, Any]) -> None:
    """Print a concise tier comparison to stdout.

    Args:
        analysis: The full analysis dict from ``run_analysis()``.
    """
    print("\n" + "=" * 72)
    print("  GPU Tier Cost Model Summary")
    print("=" * 72)
    print(
        f"  Models loaded: "
        f"{analysis['metadata']['n_models_loaded']}"
    )
    print(
        f"  {'Tier':<24} {'VRAM':>6} {'All Fit?':>9} "
        f"{'Mod Fit?':>9} {'$/scan':>10} {'$/month':>10}"
    )
    print("  " + "-" * 70)

    for t in analysis["gpu_tiers"]:
        name = t["tier"]
        vram = f"{t['vram_gb']}GB"
        all_fit = "Yes" if t["fit_analysis"]["all_loaded"]["fits"] else "No"
        mod_fit = (
            "Yes"
            if t["fit_analysis"]["per_modality"]["fits"]
            else "No"
        )
        cps = f"${t['unit_economics']['cost_per_scan_usd']:.4f}"
        monthly = f"${t['unit_economics']['monthly_fixed_cost_usd']:.0f}"
        print(
            f"  {name:<24} {vram:>6} {all_fit:>9} "
            f"{mod_fit:>9} {cps:>10} {monthly:>10}"
        )

    # Margin projections for best-value tier.
    print("\n  Margin projections (best tier by monthly cost):")
    cheapest = min(
        analysis["gpu_tiers"],
        key=lambda t: t["unit_economics"]["monthly_fixed_cost_usd"],
    )
    for proj in cheapest["unit_economics"]["margin_projections"]:
        print(
            f"    {proj['mix']}: "
            f"Rev ${proj['monthly_revenue']:,.0f} | "
            f"Profit ${proj['profit']:,.0f} | "
            f"Margin {proj['margin_pct']:.1f}%"
        )

    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run GPU tier analysis and save results."""
    print("GPU tier cost model analysis")
    analysis = run_analysis()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "gpu_tier_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Saved -> {out_path}")

    _print_summary(analysis)


if __name__ == "__main__":
    main()
