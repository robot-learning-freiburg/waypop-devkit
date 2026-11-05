from collections.abc import Hashable
from typing import Collection, Literal, Sequence

import torch
from torchmetrics import Metric

from .association_quality import AssociationQuality
from .association_quality import Result as AqResult
from .semantic_quality import Result as SqResult
from .semantic_quality import SemanticQuality


class Result:
    def __init__(
        self,
        num_classes: int,
        semantic_classes: Collection[int],
        track_classes: Collection[int],
        sq: SqResult,
        aq: AqResult,
    ) -> None:
        self._num_classes = num_classes
        self._semantic_classes = set(semantic_classes)
        self._track_classes = set(track_classes)

        self.semantic = sq
        self.association = aq

    @property
    def sq(self) -> float:
        miou, _prec, _rec = self.semantic.mean_iou(self._semantic_classes)
        return miou.item()

    @property
    def aq(self) -> float:
        return self.association.total.aq

    @property
    def stq(self) -> float:
        return (self.aq * self.sq) ** 0.5

    @property
    def aq_per_class(self) -> dict[int, float]:
        return {c: res.aq for c, res in self.association.per_class.items()}

    @property
    def sq_per_class(self) -> dict[int, float]:
        return {c: self.semantic.iou[c].item() for c in self._semantic_classes}

    @property
    def stq_per_class(self) -> dict[int, float]:
        sq = self.sq_per_class
        aq = self.aq_per_class

        return {c: (aq[c] * sq[c]) ** 0.5 for c in self._track_classes}


class SegmentationTrackingQuality(Metric):
    """
    Segmentation and Tracking Quality (STQ) metric.

    Refer to https://arxiv.org/abs/2102.11859 for more details. Supports wSTQ with
    per-element (e.g., pixel/voxel) weights (https://arxiv.org/abs/2206.07704).
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        semantic_classes: Collection[int],
        track_classes: Collection[int],
        ignored_classes: Collection[int] | None = None,
        min_instance_count: int = 0,
        per_frame: bool = False,
        allow_invalid_instances: Literal["disallow", "restrict", "ignore"] = "disallow",
    ):
        """
        Initialize the metric."

        Args:
            num_classes (int):
                The number of classes in the dataset.
            semantic_classes (Collection[int]):
                The classes to consider for the semantic quality metric.
            track_classes (Collection[int]):
                The classes to consider for the association quality metric
                (e.g., "thing" classes). All other classes are ignored. The
                classes are expected to be in the range [0, num_classes).
            per_frame (bool, optional):
                Whether to compute the association quality per frame/sample or
                over temporal sequences. Meaning, whether to comptue the AQ
                independently for each frame for classical instance
                segmentation (per_sample=True) or for the entire sequences for
                tracking (per_sample=False). Defaults to False.
            min_instance_count (int, optional):
                The minimum number of points/voxels per frame for a
                ground-truth instance to be considered valid. Instances with
                fewer points/voxels are ignored for the Association Quality
                metric. Defaults to 0.
            ignored_classes (Collection[int], optional):
                The classes to ignore in the evaluation. Regions with
                ground-truth labels of the respective classes will be treated
                as unlabeled. Meaning, predictions of non-ignored classes for
                these regions will not be considered as false positives.
                Defaults to None
            allow_invalid_instances (str, optional):
                How to handle invalid instance IDs in the ground-truth data.
                Options are:
                - "disallow": Raise an error if invalid instance IDs are found
                  for instance/thing classes.
                - "restrict": Treat invalid instance IDs for instance/thing
                  classes as non-instance (i.e., stuff).
                - "ignore": Ignore invalid instance IDs for instance/thing
                  classes, meaning the respective points/voxels are masked out
                  and not considered for evaluation.
        """
        super().__init__()

        self.num_classes = num_classes
        self.track_classes = set(track_classes)
        self.semantic_classes = set(semantic_classes)
        self.ignored_classes = set(ignored_classes) if ignored_classes else set()

        self.sq = SemanticQuality(
            num_classes=num_classes,
            ignored_classes=ignored_classes,
        )

        self.aq = AssociationQuality(
            num_classes=num_classes,
            class_subset=track_classes,
            ignored_classes=ignored_classes,
            min_count=min_instance_count,
            per_frame=per_frame,
            allow_invalid_instances=allow_invalid_instances,
        )

    def reset(self) -> None:
        super().reset()

        self.sq.reset()
        self.aq.reset()

    # pylint: disable-next=arguments-differ
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sequence_id: torch.Tensor | Sequence[Hashable],
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> None:
        """
        Update the metric with new predictions and targets.

        Args:
            preds (torch.Tensor [B, *D, 2])
                The predicted panoptic segmentation, where preds[..., 0] is the
                class ID and preds[..., 1] the instance ID.
            target (torch.Tensor [B, *D, 2])
                The target panoptic segmentation, where target[..., 0] is the
                class ID and target[..., 1] the instance ID.
            sequence_id (torch.Tensor [B] | Sequence[Hashable])
                The sequence ID for each sample in the batch.
            mask (torch.Tensor [B, *D], optional)
                An optional mask to filter valid samples in the batch. Only samples
                where mask is True are considered for evaluation.
            weights (torch.Tensor [B, *D], optional)
                An optional tensor of weights for each element (pixel/voxel) in
                the batch.
        """
        self.sq.update(
            preds=preds[..., 0],
            target=target[..., 0],
            mask=mask,
            weights=weights,
        )

        self.aq.update(
            preds=preds,
            target=target,
            sequence_id=sequence_id,
            mask=mask,
            weights=weights,
        )

    def compute(self) -> Result:
        return Result(
            self.num_classes,
            self.semantic_classes,
            self.track_classes,
            sq=self.sq.compute(),
            aq=self.aq.compute(),
        )
