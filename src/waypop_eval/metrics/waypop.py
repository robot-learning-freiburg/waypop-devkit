"""
Higher-level metric wrappers for WayPoP evaluation.

These classes wrap the low-level base metrics and handle the class-name-to-index
mapping via ``ClassMetadata``, so callers do not have to build integer class sets
manually.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Sequence

import torch
from torchmetrics import Metric

from ..dataset import ClassMetadata
from .base import AssociationQuality as AssociationQualityMetric
from .base import PanopticOspa as PanopticOspaMetric
from .base import PanopticQuality as PanopticQualityMetric
from .base import PanopticTrackingMetric
from .base import SegmentationTrackingQuality as SegmentationTrackingQualityMetric
from .base import SemanticQuality as SemanticQualityMetric


def _build_subsets(meta: ClassMetadata) -> list[tuple[str, list[int] | None]]:
    """Build the standard evaluation subsets from ``ClassMetadata``.

    Returns a list of ``(name, class_indices)`` pairs suitable for iterating
    over when computing per-subset summary metrics.  ``class_indices`` is
    ``None`` for the "all" subset (meaning no filtering), and a sorted list of
    eval indices otherwise.
    """
    order_idx = meta.eval_order
    return [
        ("known", sorted(order_idx[n] for n in meta.known - meta.ignored)),
        (
            "known_thing",
            sorted(order_idx[n] for n in (meta.things & meta.known) - meta.ignored),
        ),
        ("stuff", sorted(order_idx[n] for n in meta.stuff - meta.ignored)),
        ("unknown", sorted(order_idx[n] for n in meta.unknown - meta.ignored)),
        ("all", None),
    ]


class PanopticQuality(Metric):
    """
    Panoptic Quality (PQ) wrapper for WayPoP evaluation.

    Accepts ``ClassMetadata`` and computes PQ, SQ, RQ and the modified PQ†
    (aIoU-based) for all classes.  ``compute()`` returns a flat dict of metric
    values, broken down per class and per predefined subset (all, known,
    known_thing, stuff, unknown).
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(self, meta: ClassMetadata) -> None:
        super().__init__()

        self.meta = meta
        order_idx = meta.eval_order
        num_classes = len(meta.order)
        classes_thing = sorted(order_idx[n] for n in meta.things - meta.ignored)
        classes_stuff = sorted(order_idx[n] for n in meta.stuff - meta.ignored)

        self.base = PanopticQualityMetric(
            num_classes=num_classes,
            classes_thing=classes_thing,
            classes_stuff=classes_stuff,
        )

        self._subsets = _build_subsets(meta)

    def reset(self) -> None:
        super().reset()
        self.base.reset()

    # pylint: disable-next=arguments-differ
    def update(  # pylint: disable=unused-argument
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sequence_id: Sequence[Hashable] | None = None,
        frame_id: Sequence[int] | None = None,
        mask: torch.Tensor | None = None,
        **_: Any,
    ) -> None:
        """
        Update the metric state with new predictions and targets.

        Args:
            pred (torch.Tensor [B, *D, 2]):
                Predicted panoptic segmentation.  Channel 0 is the eval class
                index, channel 1 the instance ID (0 or negative = no instance).
            target (torch.Tensor [B, *D, 2]):
                Ground-truth panoptic segmentation in the same format.
            sequence_id: Ignored (not needed by this metric).
            frame_id: Ignored (not needed by this metric).
            mask (torch.Tensor [B, *D], optional):
                Boolean mask of valid pixels/voxels.  Defaults to None.
        """
        self.base.update(pred, target, mask)

    def compute(self) -> dict[str, Any]:
        result = self.base.compute()

        metrics: dict[str, Any] = {}

        for eval_idx, name in enumerate(self.meta.order):
            if name in self.meta.ignored:
                continue
            metrics[f"class/{name}/pq/PQ"] = result.pq[eval_idx].item()
            metrics[f"class/{name}/pq/SQ"] = result.sq[eval_idx].item()
            metrics[f"class/{name}/pq/RQ"] = result.rq[eval_idx].item()
            metrics[f"class/{name}/pq/PQ†"] = result.mod_pq[eval_idx].item()
            metrics[f"class/{name}/pq/aIoU"] = result.aiou[eval_idx].item()
            metrics[f"class/{name}/pq/GT"] = result.gt[eval_idx].item()
            metrics[f"class/{name}/pq/TP"] = result.tp[eval_idx].item()
            metrics[f"class/{name}/pq/FP"] = result.fp[eval_idx].item()
            metrics[f"class/{name}/pq/FN"] = result.fn[eval_idx].item()

        for subset_name, subset_classes in self._subsets:
            pq, sq, rq = result.mean_pq(subset_classes)
            mod_pq, aiou = result.mean_mod_pq(subset_classes)
            metrics[f"summary/{subset_name}/pq/PQ"] = pq.item()
            metrics[f"summary/{subset_name}/pq/SQ"] = sq.item()
            metrics[f"summary/{subset_name}/pq/RQ"] = rq.item()
            metrics[f"summary/{subset_name}/pq/PQ†"] = mod_pq.item()
            metrics[f"summary/{subset_name}/pq/aIoU"] = aiou.item()

        return metrics


class PanopticTracking(Metric):
    """
    Panoptic Tracking (PAT) wrapper for WayPoP evaluation.

    Accepts ``ClassMetadata`` and computes PQ, TQ (association score and track
    fragmentation), and PAT for all relevant classes.  ``compute()`` returns a
    flat dict of metric values, broken down per class and per predefined subset
    (all, known, known_thing, stuff, unknown).
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        meta: ClassMetadata,
        allow_invalid_instances: str = "restrict",
    ) -> None:
        super().__init__()

        self.meta = meta
        order_idx = meta.eval_order
        num_classes = len(meta.order)
        classes_thing = sorted(order_idx[n] for n in meta.things - meta.ignored)
        classes_stuff = sorted(order_idx[n] for n in meta.stuff - meta.ignored)

        self.base = PanopticTrackingMetric(
            num_classes=num_classes,
            classes_thing=classes_thing,
            classes_stuff=classes_stuff,
            allow_invalid_instances=allow_invalid_instances,
        )

        self._subsets = _build_subsets(meta)

    def reset(self) -> None:
        super().reset()
        self.base.reset()

    # pylint: disable-next=arguments-differ
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sequence_id: Sequence[Hashable] | None = None,
        frame_id: Sequence[int] | None = None,
        mask: torch.Tensor | None = None,
        **_: Any,
    ) -> None:
        """
        Update the metric state with new predictions and targets.

        Args:
            pred (torch.Tensor [B, *D, 2]):
                Predicted panoptic segmentation.  Channel 0 is the eval class
                index, channel 1 the instance ID (0 or negative = no instance).
            target (torch.Tensor [B, *D, 2]):
                Ground-truth panoptic segmentation in the same format.
            sequence_id (Sequence[Hashable]):
                Sequence identifier for each sample in the batch.
            frame_id (Sequence[int]):
                Temporal frame index for each sample.  Must be unique within a
                sequence across all batches and processes and sort into temporal
                order.
            mask (torch.Tensor [B, *D], optional):
                Boolean mask of valid pixels/voxels.  Defaults to None.
        """
        assert sequence_id is not None, "PanopticTracking requires sequence_id"
        assert frame_id is not None, "PanopticTracking requires frame_id"
        self.base.update(pred, target, sequence_id, frame_id, mask)

    def compute(self) -> dict[str, Any]:  # pylint: disable=too-many-locals
        result = self.base.compute()

        pq_per_class = result.pq_per_class
        sq_per_class = {int(c): float(result.panoptic.sq[int(c)]) for c in pq_per_class}
        tq_per_class = result.tq_per_class
        as_per_class = result.tracking.association_score_per_class
        tf_per_class = result.tracking.track_fragmentation_per_class
        pat_per_class = result.pat_per_class

        metrics: dict[str, Any] = {}

        for eval_idx, name in enumerate(self.meta.order):
            if name in self.meta.ignored:
                continue
            if eval_idx in pq_per_class:
                metrics[f"class/{name}/pat/PQ"] = pq_per_class[eval_idx]
                metrics[f"class/{name}/pat/SQ"] = sq_per_class[eval_idx]
            if eval_idx in tq_per_class:
                metrics[f"class/{name}/pat/TQ"] = tq_per_class[eval_idx]
                metrics[f"class/{name}/pat/association_score"] = as_per_class[eval_idx]
                metrics[f"class/{name}/pat/track_fragmentation"] = tf_per_class[
                    eval_idx
                ]
            if eval_idx in pat_per_class:
                metrics[f"class/{name}/pat/PAT"] = pat_per_class[eval_idx]

        for subset_name, subset_classes in self._subsets:
            pq, sq, _ = result.panoptic.mean_pq(subset_classes)
            tq = result.mean_tq(subset_classes)
            assoc = result.tracking.mean_association_score(subset_classes)
            frag = result.tracking.mean_track_fragmentation(subset_classes)
            pat = result.mean_pat(subset_classes, subset_classes)

            metrics[f"summary/{subset_name}/pat/PQ"] = pq.item()
            metrics[f"summary/{subset_name}/pat/SQ"] = sq.item()
            metrics[f"summary/{subset_name}/pat/TQ"] = tq
            metrics[f"summary/{subset_name}/pat/association_score"] = assoc
            metrics[f"summary/{subset_name}/pat/track_fragmentation"] = frag
            metrics[f"summary/{subset_name}/pat/PAT"] = pat

        return metrics


class SemanticQuality(Metric):
    """
    Semantic Quality (SQ / mIoU) wrapper for WayPoP evaluation.

    Computes per-class IoU, precision, and recall from a confusion matrix, plus
    mean IoU over configurable class subsets.
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(self, meta: ClassMetadata) -> None:
        super().__init__()

        self.meta = meta
        order_idx = meta.eval_order
        num_classes = len(meta.order)
        ignored_classes = sorted(order_idx[n] for n in meta.ignored)

        self.base = SemanticQualityMetric(
            num_classes=num_classes,
            ignored_classes=ignored_classes,
        )

        self._subsets = _build_subsets(meta)

    def reset(self) -> None:
        super().reset()
        self.base.reset()

    # pylint: disable-next=arguments-differ
    def update(  # pylint: disable=unused-argument
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sequence_id: Sequence[Hashable] | None = None,
        frame_id: Sequence[int] | None = None,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        **_: Any,
    ) -> None:
        """
        Update the metric state with new predictions and targets.

        Args:
            pred (torch.Tensor [B, *D, 2]):
                Predicted panoptic segmentation.  Only the semantic channel
                (index 0) is used; the instance channel is ignored.
            target (torch.Tensor [B, *D, 2]):
                Ground-truth panoptic segmentation in the same format.
            sequence_id: Ignored (not needed by this metric).
            frame_id: Ignored (not needed by this metric).
            mask (torch.Tensor [B, *D], optional):
                Boolean mask of valid pixels/voxels.  Defaults to None.
        """
        self.base.update(pred[..., 0], target[..., 0], mask, weights=weights)

    def compute(self) -> dict[str, Any]:
        result = self.base.compute()

        metrics: dict[str, Any] = {}

        for eval_idx, name in enumerate(self.meta.order):
            if name in self.meta.ignored:
                continue
            metrics[f"class/{name}/sq/IoU"] = result.iou[eval_idx].item()
            metrics[f"class/{name}/sq/precision"] = result.precision[eval_idx].item()
            metrics[f"class/{name}/sq/recall"] = result.recall[eval_idx].item()
            metrics[f"class/{name}/sq/TP"] = result.tp[eval_idx].item()
            metrics[f"class/{name}/sq/FP"] = result.fp[eval_idx].item()
            metrics[f"class/{name}/sq/FN"] = result.fn[eval_idx].item()

        for subset_name, subset_classes in self._subsets:
            miou, prec, recall = result.mean_iou(subset_classes)
            metrics[f"summary/{subset_name}/sq/mIoU"] = miou.item()
            metrics[f"summary/{subset_name}/sq/precision"] = prec.item()
            metrics[f"summary/{subset_name}/sq/recall"] = recall.item()

        return metrics


class AssociationQuality(Metric):
    """
    Association Quality (AQ) wrapper for WayPoP evaluation.

    Computes AQ, FPR, and FNR for thing/instance classes.
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        meta: ClassMetadata,
        allow_invalid_instances: str = "restrict",
        per_frame: bool = False,
    ) -> None:
        super().__init__()

        self.meta = meta
        order_idx = meta.eval_order
        num_classes = len(meta.order)
        ignored_classes = sorted(order_idx[n] for n in meta.ignored)
        classes_thing = sorted(order_idx[n] for n in meta.things - meta.ignored)

        self.base = AssociationQualityMetric(
            num_classes=num_classes,
            class_subset=classes_thing,
            ignored_classes=ignored_classes,
            allow_invalid_instances=allow_invalid_instances,
            per_frame=per_frame,
        )

        self._subsets = _build_subsets(meta)
        self._ns = "aq1" if per_frame else "aq"

    def reset(self) -> None:
        super().reset()
        self.base.reset()

    # pylint: disable-next=arguments-differ
    def update(  # pylint: disable=unused-argument
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sequence_id: Sequence[Hashable] | None = None,
        frame_id: Sequence[int] | None = None,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        **_: Any,
    ) -> None:
        """
        Update the metric state with new predictions and targets.

        Args:
            pred (torch.Tensor [B, *D, 2]):
                Predicted panoptic segmentation.  Channel 0 is the eval class
                index, channel 1 the instance ID (0 or negative = no instance).
            target (torch.Tensor [B, *D, 2]):
                Ground-truth panoptic segmentation in the same format.
            sequence_id (Sequence[Hashable]):
                Sequence identifier for each sample in the batch.
            frame_id: Ignored (not needed by this metric).
            mask (torch.Tensor [B, *D], optional):
                Boolean mask of valid pixels/voxels.  Defaults to None.
        """
        assert sequence_id is not None, "AssociationQuality requires sequence_id"
        self.base.update(pred, target, sequence_id, mask, weights=weights)

    def compute(self) -> dict[str, Any]:
        result = self.base.compute()
        ns = self._ns

        metrics: dict[str, Any] = {}

        for eval_idx, name in enumerate(self.meta.order):
            if name in self.meta.ignored:
                continue
            if eval_idx not in result.per_class:
                continue
            res = result.per_class[eval_idx]
            metrics[f"class/{name}/{ns}/AQ"] = res.aq
            metrics[f"class/{name}/{ns}/FPR"] = res.fpr
            metrics[f"class/{name}/{ns}/FNR"] = res.fnr

        metrics[f"summary/total/{ns}/AQ"] = result.total.aq
        metrics[f"summary/total/{ns}/FPR"] = result.total.fpr
        metrics[f"summary/total/{ns}/FNR"] = result.total.fnr

        return metrics


class SegmentationTrackingQuality(Metric):
    """
    Segmentation and Tracking Quality (STQ) wrapper for WayPoP evaluation.

    Computes STQ = sqrt(SQ * AQ), with SQ (semantic mIoU) and AQ (association
    quality) as sub-components.
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        meta: ClassMetadata,
        allow_invalid_instances: str = "restrict",
        per_frame: bool = False,
        weighted: bool = False,
    ) -> None:
        super().__init__()

        self.meta = meta
        order_idx = meta.eval_order
        num_classes = len(meta.order)
        ignored_classes = sorted(order_idx[n] for n in meta.ignored)
        classes_semantic = sorted(
            order_idx[n] for n in (meta.things | meta.stuff) - meta.ignored
        )
        classes_thing = sorted(order_idx[n] for n in meta.things - meta.ignored)

        self.base = SegmentationTrackingQualityMetric(
            num_classes=num_classes,
            semantic_classes=classes_semantic,
            track_classes=classes_thing,
            ignored_classes=ignored_classes,
            allow_invalid_instances=allow_invalid_instances,
            per_frame=per_frame,
        )

        self._subsets = _build_subsets(meta)
        self._weighted = weighted
        self.requires_wstq_aux = weighted
        if per_frame:
            self._ns = "wstq1" if weighted else "stq1"
        else:
            self._ns = "wstq" if weighted else "stq"

    def reset(self) -> None:
        super().reset()
        self.base.reset()

    # pylint: disable-next=arguments-differ
    def update(  # pylint: disable=unused-argument
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sequence_id: Sequence[Hashable] | None = None,
        frame_id: Sequence[int] | None = None,
        mask: torch.Tensor | None = None,
        coverage: torch.Tensor | None = None,
        tracked_mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        **_: Any,
    ) -> None:
        """
        Update the metric state with new predictions and targets.

        Args:
            pred (torch.Tensor [B, *D, 2]):
                Predicted panoptic segmentation.  Channel 0 is the eval class
                index, channel 1 the instance ID (0 or negative = no instance).
            target (torch.Tensor [B, *D, 2]):
                Ground-truth panoptic segmentation in the same format.
            sequence_id (Sequence[Hashable]):
                Sequence identifier for each sample in the batch.
            frame_id: Ignored (not needed by this metric).
            mask (torch.Tensor [B, *D], optional):
                Boolean mask of valid pixels/voxels.  Defaults to None.
        """
        assert (
            sequence_id is not None
        ), "SegmentationTrackingQuality requires sequence_id"
        if self._weighted:
            if tracked_mask is None:
                raise ValueError(
                    "wSTQ requires per-pixel tracked_mask metadata in the ground truth bundle."
                )
            if weights is None:
                if coverage is None:
                    raise ValueError(
                        "wSTQ requires per-pixel coverage metadata in the ground truth bundle."
                    )
                weights = 1.0 / torch.clamp(
                    coverage.to(device=pred.device, dtype=torch.float32), min=1e-8
                )

            target = target.clone()
            tracked_mask = tracked_mask.to(device=target.device, dtype=torch.bool)
            target[..., 1] = torch.where(
                tracked_mask,
                target[..., 1],
                torch.full_like(target[..., 1], -1),
            )

        self.base.update(pred, target, sequence_id, mask, weights=weights)

    def compute(self) -> dict[str, Any]:  # pylint: disable=too-many-locals
        result = self.base.compute()
        ns = self._ns

        metrics: dict[str, Any] = {}

        sq = result.semantic
        aq = result.association

        for eval_idx, name in enumerate(self.meta.order):
            if name in self.meta.ignored:
                continue

            metrics[f"class/{name}/{ns}/STQ"] = result.stq_per_class.get(eval_idx, 0.0)

            # SQ sub-components (same fields as SemanticQuality)
            metrics[f"class/{name}/{ns}/IoU"] = sq.iou[eval_idx].item()
            metrics[f"class/{name}/{ns}/sq/precision"] = sq.precision[eval_idx].item()
            metrics[f"class/{name}/{ns}/sq/recall"] = sq.recall[eval_idx].item()
            metrics[f"class/{name}/{ns}/sq/TP"] = sq.tp[eval_idx].item()
            metrics[f"class/{name}/{ns}/sq/FP"] = sq.fp[eval_idx].item()
            metrics[f"class/{name}/{ns}/sq/FN"] = sq.fn[eval_idx].item()

            # AQ sub-components (same fields as AssociationQuality), only for tracked classes
            if eval_idx in aq.per_class:
                res = aq.per_class[eval_idx]
                metrics[f"class/{name}/{ns}/AQ"] = res.aq
                metrics[f"class/{name}/{ns}/aq/FPR"] = res.fpr
                metrics[f"class/{name}/{ns}/aq/FNR"] = res.fnr

        for subset_name, subset_classes in self._subsets:
            miou, prec, recall = sq.mean_iou(subset_classes)
            metrics[f"summary/{subset_name}/{ns}/mIoU"] = miou.item()
            metrics[f"summary/{subset_name}/{ns}/sq/precision"] = prec.item()
            metrics[f"summary/{subset_name}/{ns}/sq/recall"] = recall.item()

            # AQ averaged over the tracked classes present in this subset
            tracked = [
                c
                for c in (
                    subset_classes if subset_classes is not None else aq.per_class
                )
                if c in aq.per_class
            ]
            if tracked:
                subset_aq = sum(aq.per_class[c].aq for c in tracked) / len(tracked)
                subset_fpr = sum(aq.per_class[c].fpr for c in tracked) / len(tracked)
                subset_fnr = sum(aq.per_class[c].fnr for c in tracked) / len(tracked)
            else:
                subset_aq = subset_fpr = subset_fnr = 0.0

            metrics[f"summary/{subset_name}/{ns}/AQ"] = subset_aq
            metrics[f"summary/{subset_name}/{ns}/aq/FPR"] = subset_fpr
            metrics[f"summary/{subset_name}/{ns}/aq/FNR"] = subset_fnr
            metrics[f"summary/{subset_name}/{ns}/STQ"] = (
                miou.item() * subset_aq
            ) ** 0.5

        metrics[f"summary/total/{ns}/STQ"] = result.stq
        metrics[f"summary/total/{ns}/mIoU"] = result.sq
        metrics[f"summary/total/{ns}/AQ"] = result.aq

        return metrics


class PanopticOspa(Metric):
    """
    Panoptic OSPA wrapper for WayPoP evaluation.

    Computes the OSPA metric for panoptic segmentation/tracking with its
    localization (LOC) and cardinality (CARD) sub-components.
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        meta: ClassMetadata,
        per_frame: bool = False,
        c: float = 1.0,
        p: float = 1.0,
    ) -> None:
        super().__init__()

        self.meta = meta
        order_idx = meta.eval_order
        num_classes = len(meta.order)
        ignored_classes = sorted(order_idx[n] for n in meta.ignored)

        self.base = PanopticOspaMetric(
            num_classes=num_classes,
            ignored_classes=ignored_classes,
            per_frame=per_frame,
            c=c,
            p=p,
        )

        self._subsets = _build_subsets(meta)
        self._ns = "ospa1" if per_frame else "ospa"

    def reset(self) -> None:
        super().reset()
        self.base.reset()

    # pylint: disable-next=arguments-differ
    def update(  # pylint: disable=unused-argument
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sequence_id: Sequence[Hashable] | None = None,
        frame_id: Sequence[int] | None = None,
        mask: torch.Tensor | None = None,
        **_: Any,
    ) -> None:
        """
        Update the metric state with new predictions and targets.

        Args:
            pred (torch.Tensor [B, *D, 2]):
                Predicted panoptic segmentation.  Channel 0 is the eval class
                index, channel 1 the instance ID (0 or negative = no instance).
            target (torch.Tensor [B, *D, 2]):
                Ground-truth panoptic segmentation in the same format.
            sequence_id (Sequence[Hashable]):
                Sequence identifier for each sample in the batch.
            frame_id: Ignored (not needed by this metric).
            mask (torch.Tensor [B, *D], optional):
                Boolean mask of valid pixels/voxels.  Defaults to None.
        """
        assert sequence_id is not None, "PanopticOspa requires sequence_id"
        self.base.update(pred, target, sequence_id, mask)

    def compute(self) -> dict[str, Any]:
        result = self.base.compute()
        ns = self._ns

        metrics: dict[str, Any] = {}

        for eval_idx, name in enumerate(self.meta.order):
            if name in self.meta.ignored:
                continue
            if eval_idx not in result.per_class:
                continue
            res = result.per_class[eval_idx]
            metrics[f"class/{name}/{ns}/OSPA"] = res.ospa
            metrics[f"class/{name}/{ns}/LOC"] = res.loc
            metrics[f"class/{name}/{ns}/CARD"] = res.card
            metrics[f"class/{name}/{ns}/GT"] = res.gt
            metrics[f"class/{name}/{ns}/TP"] = res.tp
            metrics[f"class/{name}/{ns}/FP"] = res.fp
            metrics[f"class/{name}/{ns}/FN"] = res.fn

        for subset_name, subset_classes in self._subsets:
            ospa, loc, card = result.mean_ospa(subset_classes)
            metrics[f"summary/{subset_name}/{ns}/OSPA"] = ospa
            metrics[f"summary/{subset_name}/{ns}/LOC"] = loc
            metrics[f"summary/{subset_name}/{ns}/CARD"] = card

        return metrics
