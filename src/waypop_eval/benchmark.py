from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torchmetrics import Metric

from . import metrics
from .dataset import ClassMetadata, Sample, WayPoP
from .predictions import Predictions
from .release import release_registry

DEFAULT_FRONT_CAMERA = "FRONT"
DEFAULT_DEPTH_BINS = (
    (0.0, 10.0, "0-10"),
    (10.0, 20.0, "10-20"),
    (20.0, 30.0, "20-30"),
    (30.0, 50.0, "30-50"),
    (50.0, float("inf"), "50+"),
)
DEFAULT_UNKNOWN_BREAKDOWN_KINDS = ("size", "placement")

METRIC_REGISTRY: dict[str, Callable[[ClassMetadata], Metric]] = {
    "pq": lambda meta: metrics.PanopticQuality(meta=meta),
    "stq": lambda meta: metrics.SegmentationTrackingQuality(
        meta=meta, allow_invalid_instances="restrict"
    ),
    "wstq": lambda meta: metrics.SegmentationTrackingQuality(
        meta=meta, allow_invalid_instances="restrict", weighted=True
    ),
}


@dataclass(frozen=True)
class BenchmarkCollectionSpec:
    id: str
    root: Path
    source_domain: str
    split: str
    segments: frozenset[str] | None = None
    annotations_root: Path | None = None


@dataclass
class BenchmarkCollectionData:
    spec: BenchmarkCollectionSpec
    dataset: WayPoP
    samples: list[Sample]


class BenchmarkPredictionResolver:
    def __init__(
        self,
        collection_specs: Iterable[BenchmarkCollectionSpec],
        predictions_root: Path,
        meta: ClassMetadata,
        pred_format: str,
        format_root: Path,
    ) -> None:
        self._predictions_by_set: dict[str, Predictions] = {}
        for spec in collection_specs:
            pred_dir = predictions_root / spec.id
            if not pred_dir.exists():
                raise FileNotFoundError(
                    f"Missing prediction directory for collection {spec.id}: {pred_dir}"
                )
            self._predictions_by_set[spec.id] = Predictions(
                pred_dir, meta, format_root / f"{pred_format}.yaml"
            )

    def __getitem__(self, sample: Sample):
        set_id = sample.set_id
        if not set_id:
            raise KeyError(
                "Sample is missing set_id; benchmark evaluation requires it."
            )
        return self._predictions_by_set[set_id][sample]

    def validate(self, samples: Iterable[Sample]) -> None:
        missing = []
        for sample in samples:
            try:
                prediction = self[sample]
            except KeyError:
                missing.append(sample.sample_uid or sample.prediction_key)
                continue
            if not prediction.panoptic_path.is_file():
                missing.append(str(prediction.panoptic_path))
        if missing:
            raise ValueError(
                f"Missing predictions for {len(missing)} samples; first: {missing[0]}"
            )


class BenchmarkRegistry:
    def __init__(self, collections: list[BenchmarkCollectionSpec]) -> None:
        self.collections = collections

    @classmethod
    def from_json(
        cls,
        data_root: Path,
        registry_path: Path | None = None,
        annotations_root: Path | None = None,
    ) -> BenchmarkRegistry:
        if registry_path is None and (data_root / "release.json").is_file():
            payload = release_registry(data_root)
        else:
            registry_path = registry_path or (data_root / "benchmark_registry.json")
            with registry_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

        collection_specs: list[BenchmarkCollectionSpec] = []
        for item in payload["collections"]:
            root_value = Path(item["root"])
            root_path = (
                root_value if root_value.is_absolute() else (data_root / root_value)
            )
            segments = item.get("segments")
            reference_root = None
            if annotations_root is not None and item["split"] == "test":
                reference_root = annotations_root / item["id"]
            collection_specs.append(
                BenchmarkCollectionSpec(
                    id=item["id"],
                    root=root_path,
                    source_domain=item["source_domain"],
                    split=item["split"],
                    segments=frozenset(segments) if segments is not None else None,
                    annotations_root=reference_root,
                )
            )
        return cls(collection_specs)

    def select(
        self,
        collection_ids: Iterable[str] | None = None,
        split: str | None = None,
        source_domain: str | None = None,
    ) -> list[BenchmarkCollectionSpec]:
        selected = []
        allowed = set(collection_ids) if collection_ids else None
        if allowed is not None and allowed - {spec.id for spec in self.collections}:
            raise ValueError(
                f"Unknown collections: {sorted(allowed - {spec.id for spec in self.collections})}"
            )
        for spec in self.collections:
            if allowed is not None and spec.id not in allowed:
                continue
            if split is not None and (
                spec.id != split if split in {"v1", "xl"} else spec.split != split
            ):
                continue
            if source_domain is not None and spec.source_domain not in {
                source_domain,
                "mixed",
            }:
                continue
            selected.append(spec)
        return selected


def _annotate_sample(sample: Sample, spec: BenchmarkCollectionSpec) -> Sample:
    segment_uid = sample.segment_uid or f"{spec.id}::{sample.segment}"
    sample_uid = sample.sample_uid or (
        f"{spec.id}::{sample.segment}::{sample.camera}::{sample.frame_index}"
    )
    return replace(
        sample,
        set_id=spec.id,
        source_domain=sample.source_domain or spec.source_domain,
        split=sample.split or spec.split,
        segment_uid=segment_uid,
        sample_uid=sample_uid,
    )


def load_collection_data(
    collection_specs: Iterable[BenchmarkCollectionSpec],
    classes_config: Path,
) -> tuple[ClassMetadata, list[BenchmarkCollectionData]]:
    all_data: list[BenchmarkCollectionData] = []
    meta: ClassMetadata | None = None

    for spec in collection_specs:
        dataset = WayPoP(
            spec.root, classes_config, annotations_root=spec.annotations_root
        )
        if meta is None:
            meta = dataset.meta
        elif dataset.meta.order != meta.order or dataset.meta.classes != meta.classes:
            raise ValueError(f"Class metadata mismatch for collection {spec.id}")

        samples = []
        for sample in dataset:
            if spec.segments is not None and sample.segment not in spec.segments:
                continue
            samples.append(_annotate_sample(sample, spec))

        if spec.segments is not None:
            missing = spec.segments - {sample.segment for sample in samples}
            if missing:
                raise ValueError(
                    f"Missing sequences in {spec.id} manifest: {sorted(missing)}"
                )

        all_data.append(
            BenchmarkCollectionData(spec=spec, dataset=dataset, samples=samples)
        )

    if meta is None:
        raise ValueError("No collections selected.")
    return meta, all_data


def sequence_key(sample: Sample, grouping: str) -> str:
    if grouping == "segment":
        return sample.segment_uid or f"{sample.set_id}::{sample.segment}"
    if grouping == "segment_camera":
        base = sample.segment_uid or f"{sample.set_id}::{sample.segment}"
        return f"{base}::{sample.camera}"
    raise ValueError(f"Unknown grouping: {grouping}")


def build_scope_samples(
    collection_data: Iterable[BenchmarkCollectionData],
    collection_ids: Iterable[str] | None = None,
    source_domain: str | None = None,
    camera_name: str | None = None,
) -> list[Sample]:
    allowed = set(collection_ids) if collection_ids else None
    samples: list[Sample] = []
    for item in collection_data:
        if allowed is not None and item.spec.id not in allowed:
            continue
        for sample in item.samples:
            if source_domain is not None and sample.source_domain != source_domain:
                continue
            if camera_name is not None and sample.camera != camera_name:
                continue
            samples.append(sample)
    return samples


def evaluate_metric(
    samples: Iterable[Sample],
    meta: ClassMetadata,
    predictions: BenchmarkPredictionResolver,
    metric_name: str,
    device: str,
    sequence_grouping: str,
    mask_factory: Callable[[Sample], torch.Tensor | None] | None = None,
) -> dict[str, Any]:
    metric = METRIC_REGISTRY[metric_name](meta).to(device=device)
    needs_wstq_aux = bool(getattr(metric, "requires_wstq_aux", False))

    for sample_gt in samples:
        sample_pred = predictions[sample_gt]

        gt = sample_gt.load_panoptic()[None, ...].to(device=device)
        pred = sample_pred.load_panoptic()[None, ...].to(device=device)
        if gt.shape != pred.shape:
            raise ValueError(
                f"Prediction/label shape mismatch for {sample_gt.prediction_key}: {pred.shape} != {gt.shape}"
            )

        mask = None
        if mask_factory is not None:
            sample_mask = mask_factory(sample_gt)
            if sample_mask is None:
                continue
            if not bool(sample_mask.any().item()):
                continue
            mask = sample_mask[None, ...].to(device=device)

        coverage = None
        tracked_mask = None
        if needs_wstq_aux:
            coverage, tracked_mask = sample_gt.load_wstq_metadata()
            coverage = coverage[None, ...].to(device=device)
            tracked_mask = tracked_mask[None, ...].to(device=device)

        metric.update(
            pred=pred,
            target=gt,
            sequence_id=[sequence_key(sample_gt, sequence_grouping)],
            frame_id=[sample_gt.sequence_position],
            coverage=coverage,
            tracked_mask=tracked_mask,
            mask=mask,
        )

    return metric.compute()


def normalize_breakdown_key(kind: str, detail: dict) -> str | None:
    if kind == "size":
        value = str(detail.get("size_bucket", "")).strip().lower()
        if value in {"small", "medium", "large"}:
            return value
        return None

    if kind == "placement":
        surface_type = str(detail.get("placement_surface_type", "")).strip().lower()
        dominant = (
            str(detail.get("placement_dominant_surface_label", "")).strip().lower()
        )
        if surface_type == "road" or dominant == "road":
            return "road"
        if surface_type == "sidewalk" or dominant in {"sidewalk", "curb"}:
            return "sidewalk"
        return "other_ground"

    if kind == "depth":
        try:
            distance = float(detail.get("distance_to_ego_m"))
        except (TypeError, ValueError):
            return None
        for lower, upper, label in DEFAULT_DEPTH_BINS:
            if lower <= distance < upper:
                return label
        return None

    raise ValueError(f"Unknown breakdown kind: {kind}")


def make_unknown_breakdown_mask_factory(kind: str, value: str):
    def _mask_factory(sample: Sample) -> torch.Tensor | None:
        selected_ids = []
        for detail in sample.unknown_object_details_visible:
            if normalize_breakdown_key(kind, detail) == value:
                raw_panoptic_id = detail.get("raw_panoptic_id")
                if raw_panoptic_id is not None:
                    selected_ids.append(int(raw_panoptic_id))

        if not selected_ids:
            return None

        raw_panoptic = sample.load_raw_panoptic_ids()
        mask = torch.zeros_like(raw_panoptic, dtype=torch.bool)
        for raw_panoptic_id in selected_ids:
            mask |= raw_panoptic == int(raw_panoptic_id)
        return mask

    return _mask_factory


def collect_breakdown_values(samples: Iterable[Sample], kind: str) -> list[str]:
    values = set()
    for sample in samples:
        for detail in sample.unknown_object_details_visible:
            normalized = normalize_breakdown_key(kind, detail)
            if normalized is not None:
                values.add(normalized)
    if kind == "size":
        order = ["small", "medium", "large"]
    elif kind == "placement":
        order = ["road", "sidewalk", "other_ground"]
    elif kind == "depth":
        order = [label for _, _, label in DEFAULT_DEPTH_BINS]
    else:
        raise ValueError(f"Unknown breakdown kind: {kind}")
    return [value for value in order if value in values]


def flatten_metric_section(result: dict[str, Any], prefix: str) -> dict[str, float]:
    section = {}
    for key, value in result.items():
        if not key.startswith(prefix):
            continue
        tail = key[len(prefix) :].lstrip("/")
        section[tail.replace("/", "_")] = float(value)
    return section


def build_metric_report(
    result: dict[str, Any],
    metric_name: str,
    meta: ClassMetadata,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "known": flatten_metric_section(result, f"summary/known/{metric_name}/"),
        "unknown": flatten_metric_section(result, f"summary/unknown/{metric_name}/"),
        "per_class": {},
    }
    for class_name in meta.order:
        if class_name in meta.ignored:
            continue
        section = flatten_metric_section(result, f"class/{class_name}/{metric_name}/")
        if section:
            report["per_class"][class_name] = section
    return report


def build_unknown_breakdown_report(
    samples: list[Sample],
    meta: ClassMetadata,
    predictions: BenchmarkPredictionResolver,
    device: str,
    breakdown_kinds: tuple[str, ...] = DEFAULT_UNKNOWN_BREAKDOWN_KINDS,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for kind in breakdown_kinds:
        values = collect_breakdown_values(samples, kind)
        bins: dict[str, Any] = {}
        for value in values:
            result = evaluate_metric(
                samples=samples,
                meta=meta,
                predictions=predictions,
                metric_name="pq",
                device=device,
                sequence_grouping="segment",
                mask_factory=make_unknown_breakdown_mask_factory(kind, value),
            )
            bins[value] = {
                "unknown": flatten_metric_section(result, "summary/unknown/pq/"),
                "ood": flatten_metric_section(result, "class/ood/pq/"),
            }
        report[kind] = bins
    return report


def _build_scope_metric(metric_name: str, meta: ClassMetadata, device: str) -> Metric:
    return METRIC_REGISTRY[metric_name](meta).to(device=device)


def _relevant_scope_keys(sample: Sample) -> tuple[str, str, str]:
    return (
        "all",
        f"source_domain::{sample.source_domain}",
        f"collection::{sample.set_id}",
    )


def evaluate_task_scopes(
    samples: list[Sample],
    meta: ClassMetadata,
    predictions: BenchmarkPredictionResolver,
    metric_name: str,
    device: str,
    sequence_grouping: str,
    include_unknown_breakdowns: bool = False,
    unknown_breakdown_kinds: tuple[str, ...] = DEFAULT_UNKNOWN_BREAKDOWN_KINDS,
    engine: str = "auto",
) -> dict[str, Any]:
    if engine not in {"auto", "optimized", "reference"}:
        raise ValueError("engine must be auto, optimized, or reference")
    if engine != "reference":
        from .optimized.runner import evaluate_task_scopes_optimized, supports_optimized

        if include_unknown_breakdowns and engine == "optimized":
            raise ValueError("Unknown-object breakdowns require the reference engine")
        if not include_unknown_breakdowns and supports_optimized(metric_name, device):
            return evaluate_task_scopes_optimized(
                samples, meta, predictions, metric_name, device, sequence_grouping
            )
        if engine == "optimized":
            raise ValueError(f"No optimized {metric_name} evaluator for {device}")

    source_domains = sorted({sample.source_domain for sample in samples})
    collection_ids = sorted({sample.set_id for sample in samples})

    scope_metrics: dict[str, Metric] = {
        "all": _build_scope_metric(metric_name, meta, device)
    }
    scope_update_counts: dict[str, int] = defaultdict(int)
    for domain in source_domains:
        scope_metrics[f"source_domain::{domain}"] = _build_scope_metric(
            metric_name, meta, device
        )
    for collection_id in collection_ids:
        scope_metrics[f"collection::{collection_id}"] = _build_scope_metric(
            metric_name, meta, device
        )

    needs_wstq_aux = any(
        getattr(metric, "requires_wstq_aux", False) for metric in scope_metrics.values()
    )

    breakdown_metrics: dict[str, dict[str, dict[str, Metric]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    breakdown_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    breakdown_values: dict[str, list[str]] = {}
    if include_unknown_breakdowns:
        for kind in unknown_breakdown_kinds:
            breakdown_values[kind] = collect_breakdown_values(samples, kind)
        breakdown_scope_keys = [
            scope_key
            for scope_key in scope_metrics
            if scope_key == "all" or scope_key.startswith("source_domain::")
        ]
        for scope_key in breakdown_scope_keys:
            for kind, values in breakdown_values.items():
                for value in values:
                    breakdown_metrics[scope_key][kind][value] = _build_scope_metric(
                        "pq", meta, device
                    )

    for sample_gt in samples:
        sample_pred = predictions[sample_gt]
        gt = sample_gt.load_panoptic()[None, ...].to(device=device)
        pred = sample_pred.load_panoptic()[None, ...].to(device=device)
        if gt.shape != pred.shape:
            raise ValueError(
                f"Prediction/label shape mismatch for {sample_gt.prediction_key}: {pred.shape} != {gt.shape}"
            )

        coverage = None
        tracked_mask = None
        if needs_wstq_aux:
            coverage, tracked_mask = sample_gt.load_wstq_metadata()
            coverage = coverage[None, ...].to(device=device)
            tracked_mask = tracked_mask[None, ...].to(device=device)

        scope_keys = _relevant_scope_keys(sample_gt)
        sequence_id = [sequence_key(sample_gt, sequence_grouping)]
        frame_id = [sample_gt.sequence_position]

        for scope_key in scope_keys:
            scope_metrics[scope_key].update(
                pred=pred,
                target=gt,
                sequence_id=sequence_id,
                frame_id=frame_id,
                coverage=coverage,
                tracked_mask=tracked_mask,
                mask=None,
            )
            scope_update_counts[scope_key] += 1

        if include_unknown_breakdowns:
            raw_panoptic = sample_gt.load_raw_panoptic_ids()
            masks_by_kind: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
            for kind, values in breakdown_values.items():
                for value in values:
                    mask = make_unknown_breakdown_mask_factory(kind, value)(sample_gt)
                    if mask is None or not bool(mask.any().item()):
                        continue
                    masks_by_kind[kind][value] = mask[None, ...].to(device=device)

            for scope_key in scope_keys:
                if scope_key not in breakdown_metrics:
                    continue
                for kind, value_masks in masks_by_kind.items():
                    for value, mask in value_masks.items():
                        breakdown_metrics[scope_key][kind][value].update(
                            pred=pred,
                            target=gt,
                            sequence_id=sequence_id,
                            frame_id=frame_id,
                            mask=mask,
                        )
                        breakdown_counts[scope_key][kind][value] += 1

    report: dict[str, Any] = {"all": {}, "source_domain": {}, "collection": {}}
    for scope_key, metric in scope_metrics.items():
        if scope_update_counts[scope_key] == 0:
            continue
        scope_result = build_metric_report(metric.compute(), metric_name, meta)
        if include_unknown_breakdowns:
            unknown_breakdowns = {}
            for kind, value_metrics in breakdown_metrics[scope_key].items():
                kind_payload = {}
                for value, breakdown_metric in value_metrics.items():
                    if breakdown_counts[scope_key][kind][value] == 0:
                        continue
                    result = breakdown_metric.compute()
                    kind_payload[value] = {
                        "unknown": flatten_metric_section(
                            result, "summary/unknown/pq/"
                        ),
                        "ood": flatten_metric_section(result, "class/ood/pq/"),
                    }
                if kind_payload:
                    unknown_breakdowns[kind] = kind_payload
            scope_result["unknown_breakdowns"] = unknown_breakdowns

        if scope_key == "all":
            report["all"] = scope_result
        elif scope_key.startswith("source_domain::"):
            report["source_domain"][scope_key.split("::", 1)[1]] = scope_result
        elif scope_key.startswith("collection::"):
            report["collection"][scope_key.split("::", 1)[1]] = scope_result

    return report
