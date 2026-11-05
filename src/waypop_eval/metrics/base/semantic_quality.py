from typing import Collection, Self

import torch
from torchmetrics import Metric


class Result:
    """
    Proxy object to compute various semantic quality metrics from a confusion
    matrix.
    """

    def __init__(self, confusion: torch.Tensor, ignored: set[int]) -> None:
        # remove any false-positives inside the ground-truth ignored classes
        confusion = confusion.clone()
        confusion[list(ignored), :] = 0

        self._confusion = confusion
        self._ignored = ignored

        # pre-compute common values
        self._tp = torch.diag(confusion)
        self._fp = confusion.sum(dim=0) - self.tp
        self._fn = confusion.sum(dim=1) - self.tp

        self._gt = confusion.sum(dim=1)
        self._pred = confusion.sum(dim=0)

    @property
    def num_gt(self) -> torch.Tensor:
        """
        Ground truth counts per class.
        """
        return self._gt

    @property
    def num_pred(self) -> torch.Tensor:
        """
        Predicted counts per class.
        """
        return self._pred

    @property
    def confusion(self) -> torch.Tensor:
        """
        Confusion matrix [gt, pred].
        """
        return self._confusion

    @property
    def tp(self) -> torch.Tensor:
        """
        True positives per class.
        """
        return self._tp

    @property
    def fp(self) -> torch.Tensor:
        """
        False positives per class.
        """
        return self._fp

    @property
    def fn(self) -> torch.Tensor:
        """
        False negatives per class.
        """
        return self._fn

    @property
    def iou(self) -> torch.Tensor:
        """
        Intersection over Union (IoU) per class.
        """
        return self.tp / (self.tp + self.fp + self.fn).clamp(min=1)

    @property
    def precision(self) -> torch.Tensor:
        """
        Precision per class.
        """
        return self.tp / (self.tp + self.fp).clamp(min=1)

    @property
    def recall(self) -> torch.Tensor:
        """
        Recall per class.
        """
        return self.tp / (self.tp + self.fn).clamp(min=1)

    def mean_iou(
        self, classes: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean IoU, precision, and recall over the specified classes.

        Args:
            classes (torch.Tensor, optional):
                The classes to consider. If None, all classes are considered.
                Can either be a boolean tensor of shape [num_classes] or an
                integer tensor of class indices. If a boolean tensor, True
                indicates that the class should be included in the computation.
                If a tensor of class indices, only those classes will be
                included. If None, all classes are included. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                The mean IoU, precision, and recall.
        """
        num_classes = self._confusion.shape[0]
        classes = _class_mask(classes, num_classes, self._confusion.device)

        # remove classes with no ground truth
        # Note: This will automatically remove any ignored classes, as we have
        # already set their GT count to 0 in the confusion matrix. So num_gt
        # will only ever be nonzero for classes that are not ignored.
        classes = classes & (self.num_gt > 0)

        mean_iou = self.iou[classes].mean()
        mean_precision = self.precision[classes].mean()
        mean_recall = self.recall[classes].mean()

        return mean_iou, mean_precision, mean_recall

    def binary_counts(
        self, classes: torch.Tensor | Collection[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute true positives, false positives, and false negatives for binary classification.

        Args:
            classes (torch.Tensor | Collection[int]):
                The classes to consider as the "True" class. All other classes
                are considered as the "False" class. Can be a boolean tensor of
                shape [num_classes], an integer tensor of class indices, or a
                collection of integer class indices.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                The true positives, false positives, and false negatives for the
                specified binary setting.
        """
        num_classes = self._confusion.shape[0]
        classes = _class_mask(classes, num_classes, self._confusion.device)

        tp = self.confusion[classes][:, classes].sum()
        fp = self.confusion[~classes][:, classes].sum()
        fn = self.confusion[classes][:, ~classes].sum()

        return tp, fp, fn

    def binary_iou(
        self, classes: torch.Tensor | Collection[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute IoU, precision, and recall for binary classification.

        Args:
            classes (torch.Tensor | Collection[int]):
                The classes to consider as the "True" class. All other classes
                are considered as the "False" class. Can be a boolean tensor of
                shape [num_classes], an integer tensor of class indices, or a
                collection of integer class indices.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                The IoU, precision, and recall for the specified
                binary setting.
        """
        tp, fp, fn = self.binary_counts(classes)

        iou = tp / (tp + fp + fn).clamp(min=1)
        precision = tp / (tp + fp).clamp(min=1)
        recall = tp / (tp + fn).clamp(min=1)

        return iou, precision, recall

    def to(self, *args, **kwargs) -> Self:
        return Result(self._confusion.to(*args, **kwargs), self._ignored)


class SemanticQuality(Metric):
    """
    Semantic quality metrics for semantic segmentation.

    Computes the confusion matrix for semantic segmentation, from which various
    metrics can be computed, such as (per-class) IoU, precision, and recall.
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    confusion: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        ignored_classes: Collection[int] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """
        Initialize the metric.

        Args:
            num_classes (int):
                The number of classes in the segmentation task.
            ignored_classes (Collection[int], optional):
                The classes to ignore in the metric computation. Any classes
                specified here will be treated as "void" regions. Meaning any
                false-positives of other classes inside the ignored classes will be
                ignored. This is useful for datasets with "void" classes, such as
                "unlabeled" or "ignore" classes. The classes are expected to be
                in the range [0, num_classes). Defaults to None.
            dtype (torch.dtype, optional):
                The data type to use for the confusion matrix. Defaults to torch.float64.
        """
        super().__init__()

        self.num_classes = num_classes
        self.ignored_classes = set(ignored_classes) if ignored_classes else set()

        # confusion matrix [gt, pred]
        confusion = torch.zeros(self.num_classes, self.num_classes, dtype=dtype)
        self.add_state("confusion", default=confusion, dist_reduce_fx="sum")

    # pylint: disable-next=arguments-differ
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> None:
        """
        Update the metric with new predictions and targets."

        Args:
            preds (torch.Tensor [...]):
                The predicted segmentation map with integer class labels from 0
                to self.num_classes.
            target (torch.Tensor [...]):
                The target segmentation map with integer class labels from 0
                to self.num_classes.
            mask (torch.Tensor [...], optional):
                A mask to filter the predictions and targets. If None, no
                filtering is applied. Defaults to None.
            weights (torch.Tensor [...], optional):
                An optional tensor of weights for each element in the predictions
                and targets. If provided, the confusion matrix will be updated
                with the respective weights. Defaults to None.
        """
        mask = mask if mask is not None else True

        preds = preds[mask].flatten()
        target = target[mask].flatten()
        if weights is not None:
            weights = weights[mask].flatten()

        assert preds.numel() == 0 or preds.max() < self.num_classes, (
            f"Prediction labels must be in [0, {self.num_classes}), "
            f"got max={preds.max().item()}"
        )
        assert target.numel() == 0 or target.max() < self.num_classes, (
            f"Target labels must be in [0, {self.num_classes}), "
            f"got max={target.max().item()}"
        )

        # compute confusion matrix
        bins = target.to(torch.long) * self.num_classes + preds.to(torch.long)
        w = weights.to(torch.float64) if weights is not None else None
        bins = torch.bincount(bins, weights=w, minlength=self.num_classes**2)
        bins = bins.view(self.num_classes, self.num_classes)

        # pylint: disable-next=no-member
        self.confusion += bins

    def compute(self) -> Result:
        """
        Compute the semantic quality metrics.

        Returns:
            Result:
                A proxy object that allows to derive various metrics from the
                computed confusion matrix as well as the confusion matrix itself.
        """
        # pylint: disable=no-member

        return Result(self.confusion.clone(), self.ignored_classes)


def _class_mask(
    classes: torch.Tensor | None,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    if classes is None:
        return torch.ones(num_classes, dtype=torch.bool, device=device)

    if not torch.is_tensor(classes):
        classes = list(classes)
        classes = torch.tensor(classes, device=device, dtype=torch.int)

    if classes.dtype == torch.bool:
        assert classes.shape[0] == num_classes
        return classes

    mask = torch.arange(num_classes, device=device)
    mask = torch.isin(mask, classes)

    return mask
