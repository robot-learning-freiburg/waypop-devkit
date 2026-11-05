"""Command-line entry points for the public WayPOP devkit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import SPLITS, WaypopRelease
from .submissions import TASKS, SubmissionError, package_submission, validate_submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="waypop", description="WayPOP dataset and benchmark tools")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="List released samples and sequences")
    inspect.add_argument("--data-root", type=Path, required=True)
    inspect.add_argument("--split", choices=SPLITS, default="val")

    evaluate = commands.add_parser("evaluate", help="Evaluate validation predictions locally")
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--pred-root", type=Path, required=True)
    evaluate.add_argument("--task", choices=TASKS, required=True)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate.add_argument("--engine", choices=("auto", "optimized", "reference"), default="auto")
    evaluate.add_argument("--output-json", type=Path)

    check = commands.add_parser("check-submission", help="Check a submission directory or ZIP")
    check.add_argument("--data-root", type=Path, required=True)
    check.add_argument("--input", type=Path, required=True)
    check.add_argument("--task", choices=TASKS, required=True)
    check.add_argument("--split", choices=("test", "val"), default="test")

    package = commands.add_parser("package-submission", help="Check then build a test submission ZIP")
    package.add_argument("--data-root", type=Path, required=True)
    package.add_argument("--pred-root", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--task", choices=TASKS, required=True)

    viewer = commands.add_parser("visualize", help="Launch the five-camera browser viewer")
    viewer.add_argument("--data-root", type=Path, required=True)
    viewer.add_argument("--split", choices=SPLITS, default="val")
    viewer.add_argument("--pred-root", type=Path)
    viewer.add_argument("--host", default="127.0.0.1")
    viewer.add_argument("--port", type=int, default=8080)
    viewer.add_argument("--open-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            release = WaypopRelease(args.data_root)
            samples = release.samples(args.split)
            print(json.dumps({"split": args.split, "samples": len(samples), "sequences": len(release.sequence_ids(args.split))}, indent=2))
        elif args.command == "evaluate":
            from .evaluate import evaluate_validation

            report = evaluate_validation(
                args.data_root, args.pred_root, args.task, device=args.device,
                engine=args.engine, output_json=args.output_json
            )
            print(json.dumps(report, indent=2, allow_nan=False))
        elif args.command == "check-submission":
            report = validate_submission(args.input, args.data_root, args.task, split=args.split)
            print(f"Valid {args.task} {args.split} bundle: {report.annotations} annotations, {report.required_samples} required samples")
            for warning in report.warnings:
                print(f"Warning: {warning}", file=sys.stderr)
        elif args.command == "package-submission":
            report = package_submission(args.pred_root, args.output, args.data_root, args.task)
            print(f"Wrote {args.output}: {report.annotations} annotations, {report.required_samples} required samples")
            for warning in report.warnings:
                print(f"Warning: {warning}", file=sys.stderr)
        elif args.command == "visualize":
            from .visualize import serve

            serve(
                args.data_root, args.split, prediction_root=args.pred_root,
                host=args.host, port=args.port, open_browser=args.open_browser,
            )
    except (SubmissionError, ValueError, KeyError, FileNotFoundError, ImportError, OSError) as error:
        parser.exit(2, f"waypop: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
