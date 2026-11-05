"""Public validation evaluation using the pinned benchmark scoring core."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .submissions import TASKS, validate_submission

METRICS = {"mv-ospt": ("wstq", "segment", None), "fv-ospt": ("stq", "segment_camera", "FRONT"), "vi-osps": ("pq", "segment", None)}


def evaluate_validation(
    data_root: Path | str,
    prediction_root: Path | str,
    task: str,
    *,
    device: str = "cpu",
    engine: str = "auto",
    output_json: Path | str | None = None,
) -> dict[str, Any]:
    """Score val only. Test GT is withheld and never loaded by this entry point."""
    task = task.lower()
    if task not in TASKS:
        raise ValueError(f"Expected task in {TASKS}")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    if engine not in ("auto", "optimized", "reference"):
        raise ValueError("engine must be 'auto', 'optimized', or 'reference'")
    data_root = Path(data_root).expanduser().resolve()
    prediction_root = Path(prediction_root).expanduser().resolve()
    checked = validate_submission(prediction_root, data_root, task, split="val")

    # Import the heavy evaluator only when requested. These modules are copied
    # without changes from the same scoring core used by the hosted benchmark.
    from waypop_eval.benchmark import (
        BenchmarkPredictionResolver,
        BenchmarkRegistry,
        build_scope_samples,
        evaluate_task_scopes,
        load_collection_data,
        sequence_key,
    )

    config = Path(__file__).resolve().parent / "config"
    specs = BenchmarkRegistry.from_json(data_root).select(collection_ids=("val",))
    meta, collection_data = load_collection_data(specs, config / "classes.yaml")
    metric, grouping, camera = METRICS[task]
    from waypop_eval.optimized.runner import supports_optimized

    resolved_engine = (
        "optimized" if engine != "reference" and supports_optimized(metric, device)
        else "reference"
    )
    if engine == "optimized" and resolved_engine != "optimized":
        raise ValueError(f"No optimized {task} evaluator for {device}")
    samples = build_scope_samples(collection_data, collection_ids=("val",), camera_name=camera)
    if len(samples) != checked.required_samples:
        raise ValueError("Evaluator and release manifest disagree on required sample count")
    predictions = BenchmarkPredictionResolver(
        specs, prediction_root, meta, "coco", config / "format"
    )
    predictions.validate(samples)
    report: dict[str, Any] = {
        "task": task,
        "metric": metric,
        "engine": resolved_engine,
        "prediction_format": "coco",
        "collections": ["val"],
        "num_samples": len(samples),
        "num_sequences": len({sequence_key(sample, grouping) for sample in samples}),
        "scopes": evaluate_task_scopes(
            samples=samples,
            meta=meta,
            predictions=predictions,
            metric_name=metric,
            device=device,
            sequence_grouping=grouping,
            include_unknown_breakdowns=False,
            engine=engine,
        ),
    }
    if output_json is not None:
        destination = Path(output_json).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
