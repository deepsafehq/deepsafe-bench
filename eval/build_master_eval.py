"""Build a stratified master evaluation set from the DeepSafe dataset.

Walks the dataset directory structure, performs stratified sampling across
generators for each modality/label combination, copies selected files into
a tiered output directory, and writes a ``metadata.json`` manifest.

Three tiers are available:
  - **small**: Quick smoke tests (~200 files).
  - **medium**: Standard evaluation (~15K files).
  - **full**: Exhaustive benchmark (~46K files, uses symlinks).

Usage:
    uv run python -m eval.build_master_eval --tier small --dry-run
    uv run python -m eval.build_master_eval --tier medium
    uv run python -m eval.build_master_eval --tier all
"""

import argparse
import json
import logging
import os
import random
import shutil
from typing import Any

from deepsafe_eval.config import DATASET_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default dataset root resolved from config (env var or project symlink).
_DEFAULT_ROOT = str(DATASET_ROOT)

# ID prefixes by modality directory name.
_ID_PREFIX = {
    "images": "img",
    "audio": "aud",
    "video": "vid",
}

# Per-tier sampling targets for each modality/label.
TIER_CONFIGS = {
    "small": {
        "images": {"real": 50, "fake": 50},
        "audio": {"real": 20, "fake": 30},
        "video": {"real": 20, "fake": 30},
    },
    "medium": {
        "images": {"real": 5000, "fake": 5000},
        "audio": {"real": 1000, "fake": 2500},
        "video": {"real": 1000, "fake": 1000},
    },
    "full": {
        "images": {"real": 10000, "fake": 15000},
        "audio": {"real": 3000, "fake": 10000},
        "video": {"real": 3000, "fake": 5000},
    },
}

# Output directory name for each tier.
TIER_OUTPUT_DIRS = {
    "small": "master_eval_small",
    "medium": "master_eval",
    "full": "master_eval_full",
}


def stratified_sample(
    files_by_generator: dict[str, list[str]],
    total: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Sample *total* files with balanced representation across generators.

    Uses round-robin allocation: each generator gets ``total // n_generators``
    files.  Generators with fewer files than their quota contribute everything
    they have; the surplus quota is redistributed among the remaining
    generators until the target is met or all files are exhausted.

    Args:
        files_by_generator: Mapping of generator name to list of file paths.
        total: Desired number of samples.
        seed: Random seed for reproducibility.

    Returns:
        List of ``(file_path, generator_name)`` tuples.
    """
    if total <= 0 or not files_by_generator:
        return []

    rng = random.Random(seed)

    # Shuffle each generator's file list independently.
    shuffled: dict[str, list[str]] = {}
    for gen, paths in files_by_generator.items():
        copy = list(paths)
        rng.shuffle(copy)
        shuffled[gen] = copy

    result: list[tuple[str, str]] = []
    remaining_gens: dict[str, list[str]] = dict(shuffled)
    remaining_total = total

    while remaining_total > 0 and remaining_gens:
        n_gens = len(remaining_gens)
        per_gen = remaining_total // n_gens
        # Ensure at least 1 per generator when there's remaining quota.
        if per_gen == 0:
            per_gen = 1

        exhausted: list[str] = []
        allocated_this_round = 0

        for gen in sorted(remaining_gens.keys()):
            available = remaining_gens[gen]
            quota = min(per_gen, len(available), remaining_total - allocated_this_round)
            if quota <= 0:
                continue

            selected = available[:quota]
            result.extend((f, gen) for f in selected)
            remaining_gens[gen] = available[quota:]
            allocated_this_round += quota

            if not remaining_gens[gen]:
                exhausted.append(gen)

        for gen in exhausted:
            del remaining_gens[gen]

        remaining_total -= allocated_this_round

        # Safety: if nothing was allocated this round, break to avoid
        # infinite loop (shouldn't happen, but defensive).
        if allocated_this_round == 0:
            break

    return result


def gather_files(
    root: str,
    modality: str,
    label: str,
) -> dict[str, list[str]]:
    """Walk ``{root}/{modality}/{label}/`` and collect files by source.

    Each immediate subdirectory under ``{modality}/{label}/`` is treated as
    a distinct source (generator for fakes, corpus for reals).

    Args:
        root: Dataset root directory.
        modality: One of ``"images"``, ``"audio"``, ``"video"``.
        label: One of ``"real"``, ``"fake"``.

    Returns:
        Dict mapping source_name to list of absolute file paths.
    """
    base = os.path.join(root, modality, label)
    if not os.path.isdir(base):
        return {}

    files_by_source: dict[str, list[str]] = {}

    for source_name in sorted(os.listdir(base)):
        source_dir = os.path.join(base, source_name)
        if not os.path.isdir(source_dir):
            continue

        file_paths: list[str] = []
        for fname in sorted(os.listdir(source_dir)):
            # Skip hidden/system files.
            if fname.startswith("."):
                continue
            fpath = os.path.join(source_dir, fname)
            if os.path.isfile(fpath):
                file_paths.append(os.path.abspath(fpath))

        if file_paths:
            files_by_source[source_name] = file_paths

    return files_by_source


def build_master_eval(
    root: str,
    tier: str,
    seed: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Orchestrate stratified sampling and copy/link files.

    Args:
        root: Dataset root directory.
        tier: One of ``"small"``, ``"medium"``, ``"full"``.
        seed: Random seed for reproducibility.
        dry_run: If True, log what would happen but don't copy.

    Returns:
        The full metadata list (useful for dry-run inspection).

    Raises:
        ValueError: If *tier* is not a recognised tier name.
    """
    if tier not in TIER_CONFIGS:
        raise ValueError(
            f"Unknown tier {tier!r}. " f"Choose from {sorted(TIER_CONFIGS)}."
        )

    cfg = TIER_CONFIGS[tier]
    out_name = TIER_OUTPUT_DIRS[tier]
    master_dir = os.path.join(root, out_name)
    use_symlinks = tier == "full"

    # Define all (modality, label, count) combinations.
    slices = [
        ("images", "real", cfg["images"]["real"]),
        ("images", "fake", cfg["images"]["fake"]),
        ("audio", "real", cfg["audio"]["real"]),
        ("audio", "fake", cfg["audio"]["fake"]),
        ("video", "real", cfg["video"]["real"]),
        ("video", "fake", cfg["video"]["fake"]),
    ]

    metadata: list[dict[str, Any]] = []
    global_counter: dict[str, int] = {}  # per-modality counter for IDs

    for modality, label, count in slices:
        logger.info(
            "Gathering %s/%s (target: %d) ...",
            modality,
            label,
            count,
        )
        files_by_source = gather_files(root, modality, label)

        total_available = sum(len(v) for v in files_by_source.values())
        logger.info(
            "  Found %d files across %d sources.",
            total_available,
            len(files_by_source),
        )

        sampled = stratified_sample(files_by_source, total=count, seed=seed)
        logger.info("  Sampled %d files.", len(sampled))

        # Log per-generator breakdown.
        gen_counts: dict[str, int] = {}
        for _, gen in sampled:
            gen_counts[gen] = gen_counts.get(gen, 0) + 1
        for gen in sorted(gen_counts):
            logger.info("    %s: %d", gen, gen_counts[gen])

        prefix = _ID_PREFIX.get(modality, modality[:3])

        for src_path, generator in sampled:
            # Increment global counter for this modality.
            global_counter[modality] = global_counter.get(modality, 0) + 1
            idx = global_counter[modality]
            sample_id = f"{prefix}_{idx:05d}"

            original_name = os.path.basename(src_path)
            ext = original_name.rsplit(".", 1)[-1] if "." in original_name else ""

            # Build destination path preserving generator structure.
            dest_filename = f"{sample_id}.{ext}" if ext else sample_id
            rel_dest = os.path.join(modality, label, generator, dest_filename)

            # Source file relative path (for metadata).
            # Compute relative to dataset root.
            try:
                rel_source = os.path.relpath(src_path, root)
            except ValueError:
                rel_source = src_path

            entry = {
                "id": sample_id,
                "path": rel_dest,
                "modality": modality,
                "label": label,
                "generator": generator,
                "source_file": rel_source,
                "format": ext,
            }
            metadata.append(entry)

            if not dry_run:
                dest_full = os.path.join(master_dir, rel_dest)
                os.makedirs(
                    os.path.dirname(dest_full),
                    exist_ok=True,
                )
                if use_symlinks:
                    os.symlink(
                        os.path.abspath(src_path),
                        dest_full,
                    )
                else:
                    shutil.copy2(src_path, dest_full)

    # Copy provenance eval data into the tier.
    provenance_src = os.path.join(root, "provenance", "eval")
    provenance_dst = os.path.join(master_dir, "provenance")
    if os.path.isdir(provenance_src) and not dry_run:
        if use_symlinks:
            os.makedirs(provenance_dst, exist_ok=True)
            # Copy labels.csv, symlink data dirs.
            labels_src = os.path.join(provenance_src, "labels.csv")
            if os.path.exists(labels_src):
                shutil.copy2(labels_src, provenance_dst)
            for entry in sorted(os.listdir(provenance_src)):
                src_entry = os.path.join(provenance_src, entry)
                dst_entry = os.path.join(provenance_dst, entry)
                if os.path.isdir(src_entry) and not os.path.exists(dst_entry):
                    os.symlink(os.path.abspath(src_entry), dst_entry)
        else:
            if os.path.exists(provenance_dst):
                shutil.rmtree(provenance_dst)
            shutil.copytree(provenance_src, provenance_dst)
        logger.info("Included provenance eval data in %s.", out_name)

    # Write metadata.
    if not dry_run:
        metadata_path = os.path.join(master_dir, "metadata.json")
        os.makedirs(master_dir, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Wrote metadata.json with %d entries.", len(metadata))
    else:
        logger.info(
            "[DRY RUN] Would write %d entries to metadata.json.",
            len(metadata),
        )

    # Summary.
    logger.info("=== Summary ===")
    for modality_name in ("images", "audio", "video"):
        for lbl in ("real", "fake"):
            n = sum(
                1
                for e in metadata
                if e["modality"] == modality_name and e["label"] == lbl
            )
            logger.info("  %s/%s: %d", modality_name, lbl, n)
    logger.info("  Total: %d", len(metadata))

    return metadata


def main() -> None:
    """CLI entry point for building the master evaluation set."""
    parser = argparse.ArgumentParser(
        description="Build a stratified master evaluation set.",
    )
    parser.add_argument(
        "--root",
        default=_DEFAULT_ROOT,
        help="Dataset root directory (default: %(default)s).",
    )
    parser.add_argument(
        "--tier",
        choices=["small", "medium", "full", "all"],
        default="medium",
        help=(
            "Evaluation tier to build. 'all' builds every "
            "tier sequentially (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would happen without copying files.",
    )

    args = parser.parse_args()

    tiers = list(TIER_CONFIGS.keys()) if args.tier == "all" else [args.tier]

    for tier in tiers:
        logger.info("=== Building tier: %s ===", tier)
        build_master_eval(
            root=args.root,
            tier=tier,
            seed=args.seed,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
