"""Command-line interface for DeepSafe."""

from __future__ import annotations

import argparse
import pathlib
import sys

from deepsafe import __version__, citation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepsafe",
        description="Benchmark, model zoo, and adaptation toolkit for "
                    "deepfake detection.",
        epilog="Free for non-commercial use (PolyForm Noncommercial 1.0.0). "
               "Detection does not generalize; see BENCHMARK.md.",
    )
    parser.add_argument("--version", action="version", version=f"deepsafe {__version__}")
    parser.add_argument(
        "--cite", action="store_true", help="print the BibTeX citation and exit"
    )
    sub = parser.add_subparsers(dest="command")

    p_detect = sub.add_parser("detect", help="run the ensemble over a file")
    p_detect.add_argument("path", type=pathlib.Path)
    p_detect.add_argument("--server", default="http://localhost:8000")
    p_detect.add_argument(
        "--modality", choices=["images", "audio", "video"], default=None
    )

    p_eval = sub.add_parser("eval", help="score a detector against the benchmark")
    p_eval.add_argument(
        "--model", type=pathlib.Path, default=None,
        help="path to a .py file exposing predict(path) -> float",
    )
    p_eval.add_argument(
        "--baseline", default=None,
        help="score a stored model instead (e.g. npr, ensemble); needs no media",
    )
    p_eval.add_argument("--tier", choices=["small", "medium", "full"], default="small")
    p_eval.add_argument("--data", type=pathlib.Path, default=pathlib.Path("./dataset"))
    p_eval.add_argument("--modality", choices=["images", "audio", "video"], default=None)
    p_eval.add_argument("--threshold", type=float, default=0.5)
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--out", type=pathlib.Path, default=None,
                        help="write report card and CITATIONS.bib here")
    p_eval.add_argument("--download", action="store_true",
                        help="fetch the tier from HuggingFace first")
    p_eval.add_argument("--list-baselines", action="store_true")

    p_fit = sub.add_parser("fit", help="adapt the ensemble to your own data")
    p_fit.add_argument(
        "--predictions", type=pathlib.Path, default=None,
        help="CSV(.gz) of detector scores with label and generator columns",
    )
    p_fit.add_argument("--tier", type=int, choices=[1, 2, 3], default=1)
    p_fit.add_argument("--holdout", type=float, default=0.3)
    p_fit.add_argument("--seed", type=int, default=42)
    return parser


def _cmd_detect(args) -> int:
    from deepsafe.detect import InferenceUnavailable, detect

    try:
        print(detect(args.path, server=args.server, modality=args.modality).render())
    except (InferenceUnavailable, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_eval(args) -> int:
    from deepsafe import data
    from deepsafe.detectors import DetectorLoadError, load_detector
    from deepsafe.evaluate import evaluate_baseline, evaluate_detector

    if args.list_baselines:
        rows = data.load_predictions()
        names = sorted(c[len("prob_"):] for c in rows[0] if c.startswith("prob_"))
        print("Stored baselines (scoreable with no media):")
        for name in names:
            print(f"  {name}")
        print("  ensemble")
        return 0

    if args.download:
        try:
            data.download_tier(args.tier, args.data)
        except data.DatasetNotFound as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    try:
        if args.baseline:
            card = evaluate_baseline(
                args.baseline, modality=args.modality, threshold=args.threshold
            )
        elif args.model:
            predict = load_detector(args.model)
            card = evaluate_detector(
                predict,
                name=args.model.stem,
                tier=args.tier,
                root=args.data,
                threshold=args.threshold,
                limit=args.limit,
                on_progress=_progress,
            )
            print(file=sys.stderr)
        else:
            print(
                "error: pass --model <file.py>, --baseline <name>, or "
                "--list-baselines",
                file=sys.stderr,
            )
            return 1
    except (KeyError, DetectorLoadError, data.DatasetNotFound) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(card.render())
    if args.out:
        report, bib = card.write(args.out)
        print(f"\nwrote {report}\nwrote {bib}")
    return 0


def _cmd_fit(args) -> int:
    from deepsafe import data
    from deepsafe.fit import fit_ensemble

    if args.tier in (2, 3):
        print(
            f"error: Tier {args.tier} needs the inference stack and a GPU.\n"
            f"Tier 2 (linear probe) and Tier 3 (full fine-tuning) are "
            f"implemented for a subset of models in `deepsafe.fit_gpu`, which "
            f"requires torch.\n"
            f"Tier 1 (ensemble refit) runs on CPU and is usually the "
            f"adaptation that matters:\n"
            f"    deepsafe fit --tier 1",
            file=sys.stderr,
        )
        return 1

    try:
        rows = data.load_predictions(args.predictions)
    except data.DatasetNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    columns = [c for c in rows[0] if c.startswith("prob_")]
    complete = [c for c in columns
                if sum(1 for r in rows if data.as_float(r, c) is not None) > len(rows) * 0.5]
    if not complete:
        print("error: no score column is populated for most rows", file=sys.stderr)
        return 1

    try:
        model, report = fit_ensemble(
            rows, complete, holdout_fraction=args.holdout, seed=args.seed
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Tier 1 ensemble refit over {len(complete)} detector scores.")
    print(f"Features: {', '.join(c[len('prob_'):] for c in complete)}\n")
    print(report.render())
    return 0 if report.transferred else 2


def _progress(done: int, total: int) -> None:
    print(f"\r  scoring {done}/{total}", end="", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cite:
        print(citation())
        return 0
    if not args.command:
        parser.print_help()
        return 1

    return {"detect": _cmd_detect, "eval": _cmd_eval, "fit": _cmd_fit}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
