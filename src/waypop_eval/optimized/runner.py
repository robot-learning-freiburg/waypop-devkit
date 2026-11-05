"""Evaluate validated release samples with the compact scoring counters."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

import numpy as np
import torch

from ..dataset import ClassMetadata, Sample
from .compact_counter import FIELDS, CPUCounter, GPUCounter, gpu_batch
from .compact_stq import CompactSTQ
from .compact_wstq import CompactWSTQ
from .fast_loading import prediction, target


def supports_optimized(metric_name: str, device: str) -> bool:
    """PQ supports CPU and CUDA; compact tracking counters require CUDA."""
    if metric_name not in {"pq", "stq", "wstq"} or device not in {"cpu", "cuda"}:
        return False
    return metric_name == "pq" or device == "cuda"


class _PQAccumulator:
    def __init__(self, meta: ClassMetadata, device: str) -> None:
        self.meta = meta
        self.device = device
        count = len(meta.order)
        things = sorted(meta.eval_order[name] for name in meta.things - meta.ignored)
        stuff = sorted(meta.eval_order[name] for name in meta.stuff - meta.ignored)
        self.counter = (
            GPUCounter(count, things, stuff)
            if device == "cuda"
            else CPUCounter(count, things, stuff)
        )
        self.statistics = {
            field: np.zeros(
                count, dtype=np.int64 if field.startswith("num_") else np.float64
            )
            for field in FIELDS
        }
        self.images = 0

    def update(self, pred: np.ndarray, truth: np.ndarray) -> None:
        if self.device == "cuda":
            pred_batch, truth_batch = gpu_batch([(pred, truth)], self.counter.void)
            values = self.counter.statistics(pred_batch, truth_batch)
        else:
            values = self.counter.statistics(pred, truth)
        for field in FIELDS:
            self.statistics[field] += values[field]
        self.images += 1

    def compute(self) -> dict[str, Any]:
        from ..benchmark import METRIC_REGISTRY

        metric = METRIC_REGISTRY["pq"](self.meta)
        for field in FIELDS:
            getattr(metric.base, field).copy_(torch.from_numpy(self.statistics[field]))
        metric._update_count = metric.base._update_count = self.images
        return metric.compute()


def _load_sample(
    sample: Sample, predictions: Any, needs_wstq_aux: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    pred = prediction(predictions[sample])
    truth = target(sample)
    if pred.shape != truth.shape:
        raise ValueError(
            f"Prediction/label shape mismatch for {sample.prediction_key}: "
            f"{pred.shape} != {truth.shape}"
        )
    if not needs_wstq_aux:
        return pred, truth, None, None
    coverage, tracked_mask = sample.load_wstq_metadata()
    return pred, truth, coverage.numpy(), tracked_mask.numpy()


def _prefetched_samples(
    samples: list[Sample], predictions: Any, needs_wstq_aux: bool
) -> Iterable[
    tuple[Sample, tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]]
]:
    if not samples:
        return
    workers = min(8, len(samples))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        remaining = iter(samples)
        pending = deque()
        for _ in range(workers):
            sample = next(remaining)
            pending.append(
                (
                    sample,
                    executor.submit(_load_sample, sample, predictions, needs_wstq_aux),
                )
            )
        while pending:
            sample, future = pending.popleft()
            arrays = future.result()
            following = next(remaining, None)
            if following is not None:
                pending.append(
                    (
                        following,
                        executor.submit(
                            _load_sample, following, predictions, needs_wstq_aux
                        ),
                    )
                )
            yield sample, arrays


def evaluate_task_scopes_optimized(
    samples: list[Sample],
    meta: ClassMetadata,
    predictions: Any,
    metric_name: str,
    device: str,
    sequence_grouping: str,
) -> dict[str, Any]:
    """Score the same all/domain/collection scopes as the reference evaluator."""
    from ..benchmark import build_metric_report, sequence_key

    if not supports_optimized(metric_name, device):
        raise ValueError(f"No optimized {metric_name} evaluator for {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested, but no CUDA device is available")

    source_domains = sorted({sample.source_domain for sample in samples})
    collection_ids = sorted({sample.set_id for sample in samples})
    scope_members: dict[str, tuple[int, ...]] = {"all": tuple(range(len(samples)))}
    scope_members.update(
        {
            f"source_domain::{domain}": tuple(
                index
                for index, sample in enumerate(samples)
                if sample.source_domain == domain
            )
            for domain in source_domains
        }
    )
    scope_members.update(
        {
            f"collection::{collection}": tuple(
                index
                for index, sample in enumerate(samples)
                if sample.set_id == collection
            )
            for collection in collection_ids
        }
    )

    unique_counters: dict[tuple[int, ...], Any] = {}
    scope_counters: dict[str, Any] = {}
    for scope, members in scope_members.items():
        if not members:
            continue
        if members not in unique_counters:
            if metric_name == "pq":
                unique_counters[members] = _PQAccumulator(meta, device)
            elif metric_name == "stq":
                unique_counters[members] = CompactSTQ(meta)
            else:
                unique_counters[members] = CompactWSTQ(meta)
        scope_counters[scope] = unique_counters[members]

    for sample, (pred, truth, coverage, tracked_mask) in _prefetched_samples(
        samples, predictions, metric_name == "wstq"
    ):
        counters = {
            id(scope_counters[scope]): scope_counters[scope]
            for scope in (
                "all",
                f"source_domain::{sample.source_domain}",
                f"collection::{sample.set_id}",
            )
        }
        if metric_name == "pq":
            for counter in counters.values():
                counter.update(pred, truth)
            continue

        pred_gpu = torch.from_numpy(pred).to(device=device)
        truth_gpu = torch.from_numpy(truth).to(device=device)
        sequence = sequence_key(sample, sequence_grouping)
        if metric_name == "stq":
            for counter in counters.values():
                counter.update(pred_gpu, truth_gpu, sequence)
        else:
            coverage_gpu = torch.from_numpy(coverage).to(device=device)
            tracked_gpu = torch.from_numpy(tracked_mask).to(device=device)
            for counter in counters.values():
                counter.update(pred_gpu, truth_gpu, coverage_gpu, tracked_gpu, sequence)

    computed = {
        members: build_metric_report(counter.compute(), metric_name, meta)
        for members, counter in unique_counters.items()
    }
    report: dict[str, Any] = {"all": {}, "source_domain": {}, "collection": {}}
    for scope, members in scope_members.items():
        if members not in computed:
            continue
        result = computed[members]
        if scope == "all":
            report["all"] = result
        elif scope.startswith("source_domain::"):
            report["source_domain"][scope.split("::", 1)[1]] = result
        else:
            report["collection"][scope.split("::", 1)[1]] = result
    return report
