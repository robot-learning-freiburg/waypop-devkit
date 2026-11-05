from typing import Collection, Self

import torch
from torchmetrics import Metric

# Segment "color" (batch, class, and instance IDs)
_Color = tuple[int, int, int]


class Result:
    """
    Proxy object to compute panoptic quality metrics, such as PQ and PQ†,
    from intermediate results.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        gt: torch.Tensor,
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
        iou_gt: torch.Tensor,
        iou_tp: torch.Tensor,
        thing_mask: torch.Tensor,
    ) -> None:
        """
        Initialize a new PanopticQualityResult instance.

        Args:
            gt (torch.Tensor [num_classes]):
                The number of ground truth segments per class.
            tp (torch.Tensor [num_classes]):
                The number of true positives per class.
            fp (torch.Tensor [num_classes]):
                The number of false positives per class.
            fn (torch.Tensor [num_classes]):
                The number of false negatives per class.
            iou_gt (torch.Tensor [num_classes]):
                The sum of all IoU scores of all segments, per class.
            iou_tp (torch.Tensor [num_classes]):
                The sum of all IoU scores of all true positives, per class.
        """

        self._gt = gt
        self._tp = tp
        self._fp = fp
        self._fn = fn
        self._iou_gt = iou_gt
        self._iou_tp = iou_tp
        self._thing_mask = thing_mask

        self._pq_valid = None
        self._pq = None
        self._sq = None
        self._rq = None

        self._mod_pq_valid = None
        self._mod_pq = None
        self._aiou = None

    def _compute_pq(self) -> None:
        """
        Compute Panoptic Quality (PQ) metrics, including SQ (Segmentation Quality)
        and RQ (Recognition Quality).
        """
        if self._pq_valid is not None:
            return

        denom = self._tp + 0.5 * self._fp + 0.5 * self._fn
        sq = torch.where(self._tp > 0.0, self._iou_tp / self._tp, 0.0)
        rq = torch.where(denom > 0.0, self._tp / denom, 0.0)
        pq = sq * rq

        self._pq_valid = denom > 0.0
        self._pq = pq
        self._sq = sq
        self._rq = rq

    def _compute_mod_pq(self) -> None:
        """
        Compute modified Panoptic Quality (PQ†) metrics, including aIoU (average IoU).
        """

        if self._mod_pq_valid is not None:
            return

        self._compute_pq()

        aiou = torch.where(self.gt > 0.0, self._iou_gt / self._gt, 0.0)
        mod_pq = torch.where(self._thing_mask, self._pq, aiou)
        valid = torch.where(self._thing_mask, self._pq_valid, self._gt > 0.0)

        self._mod_pq_valid = valid
        self._mod_pq = mod_pq
        self._aiou = aiou

    @property
    def gt(self) -> torch.Tensor:
        """
        Number of ground truth segments per class.
        """
        return self._gt

    @property
    def tp(self) -> torch.Tensor:
        """
        Number of true positives per class.
        """
        return self._tp

    @property
    def fp(self) -> torch.Tensor:
        """
        Number of false positives per class.
        """
        return self._fp

    @property
    def fn(self) -> torch.Tensor:
        """
        Number of false negatives per class.
        """
        return self._fn

    @property
    def pq(self) -> torch.Tensor:
        """
        Per-class Panoptic Quality (PQ).
        """
        self._compute_pq()
        return self._pq

    @property
    def sq(self) -> torch.Tensor:
        """
        Per-class Segmentation Quality (SQ).
        """
        self._compute_pq()
        return self._sq

    @property
    def rq(self) -> torch.Tensor:
        """
        Per-class Recognition Quality (RQ).
        """
        self._compute_pq()
        return self._rq

    @property
    def mod_pq(self) -> torch.Tensor:
        """
        Per-class modified Panoptic Quality (PQ†).
        """
        self._compute_mod_pq()
        return self._mod_pq

    @property
    def aiou(self) -> torch.Tensor:
        """
        Per-class average IoU (aIoU).
        """
        self._compute_mod_pq()
        return self._aiou

    def mean_pq(
        self, classes: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mean Panoptic Quality (SQ) over all classes.

        Args:
            classes (torch.Tensor [num_classes] optional):
                The classes to consider for the mean. If None, all classes are
                considered. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                The mean Panoptic Quality (PQ), Segmentation Quality (SQ), and
                Recognition Quality (RQ).
        """
        self._compute_pq()

        mask = self._pq_valid
        mask = _class_mask(mask, classes)

        pq = torch.mean(self._pq[mask])
        sq = torch.mean(self._sq[mask])
        rq = torch.mean(self._rq[mask])

        return pq, sq, rq

    def mean_mod_pq(
        self, classes: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mean modified Panoptic Quality (PQ†) over all classes.

        Args:
            classes (torch.Tensor [num_classes] optional):
                The classes to consider for the mean. If None, all classes are
                considered. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                The mean modified Panoptic Quality (PQ†) and average IoU (aIoU).
        """
        self._compute_mod_pq()

        mask = self._mod_pq_valid
        mask = _class_mask(mask, classes)

        pqm = torch.mean(self._mod_pq[mask])
        aiou = torch.mean(self._aiou[mask])

        return pqm, aiou

    def to(self, *args, **kwargs) -> Self:
        return Result(
            gt=self._gt.to(*args, **kwargs),
            tp=self._tp.to(*args, **kwargs),
            fp=self._fp.to(*args, **kwargs),
            fn=self._fn.to(*args, **kwargs),
            iou_gt=self._iou_gt.to(*args, **kwargs),
            iou_tp=self._iou_tp.to(*args, **kwargs),
            thing_mask=self._thing_mask.to(*args, **kwargs),
        )


class PanopticQuality(Metric):
    """
    Panoptic Quality (PQ) metric for panoptic segmentation.

    Adapted from the torchmetrics implementation. Compared to the torchmetrics
    implementation, this version computes both standard PQ and modified PQ
    (PQ†) metrics and provides extended statistics.

    Refer to the following papers for more information:
    - PQ, SQ, RQ: https://arxiv.org/abs/1801.00868
    - PQ†: https://arxiv.org/abs/1905.01220
    """

    # pylint: disable=too-many-instance-attributes

    is_differentiable: bool = False
    full_state_update: bool = False

    # buffers
    classes_thing: torch.Tensor
    classes_stuff: torch.Tensor

    # state
    iou_sum_gt: torch.Tensor
    iou_sum_tp: torch.Tensor
    num_gt: torch.Tensor
    num_tp: torch.Tensor
    num_fp: torch.Tensor
    num_fn: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        classes_thing: Collection[int],
        classes_stuff: Collection[int],
        allow_unknown_preds: bool = True,
    ):
        """
        Initialize a new PanopticQuality instance.

        Note: Any class that is not in classes_thing or classes_stuff is
        considered unlabelled/invalid and ignored for evaluation.

        Args:
            num_classes (int):
                The total number of classes in the dataset.
            classes_thing (Collection[int]):
                The class IDs of "thing" classes (countable objects).
            classes_stuff (Collection[int]):
                The class IDs of "stuff" classes (amorphous regions).
            allow_unknown (bool):
                Whether to allow unknown classes (classes that are not in
                classes_thing or classes_stuff). If False, an error is raised
                if any class ID is not in either set. Defaults to True.
        """
        super().__init__()

        self.num_classes = num_classes
        self.allow_unknown_preds = allow_unknown_preds

        classes_thing = set(classes_thing)
        classes_stuff = set(classes_stuff)
        classes = classes_thing | classes_stuff

        self.void_category = _get_void_category(classes_thing, classes_stuff)

        # make sure that thing and stuff classes are disjoint
        assert not classes_thing & classes_stuff

        # make sure that we have at least one class
        assert self.num_classes > 0

        # make sure that all classes are within the valid range
        assert max(classes_thing) < self.num_classes
        assert max(classes_stuff) < self.num_classes

        # set up class tensors
        classes_thing = torch.tensor(sorted(classes_thing), dtype=torch.long)
        self.register_buffer("classes_thing", classes_thing, persistent=False)

        classes_stuff = torch.tensor(sorted(classes_stuff), dtype=torch.long)
        self.register_buffer("classes_stuff", classes_stuff, persistent=False)

        classes = torch.tensor(sorted(classes), dtype=torch.long)
        self.register_buffer("classes", classes, persistent=False)

        # set up state tensors
        iou = torch.zeros(self.num_classes, dtype=torch.double)
        self.add_state("iou_sum_gt", default=iou.clone(), dist_reduce_fx="sum")
        self.add_state("iou_sum_tp", default=iou.clone(), dist_reduce_fx="sum")

        count = torch.zeros(self.num_classes, dtype=torch.long)
        self.add_state("num_gt", default=count.clone(), dist_reduce_fx="sum")
        self.add_state("num_tp", default=count.clone(), dist_reduce_fx="sum")
        self.add_state("num_fp", default=count.clone(), dist_reduce_fx="sum")
        self.add_state("num_fn", default=count.clone(), dist_reduce_fx="sum")

    def _prepare_input(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        allow_unknown: bool,
    ) -> torch.Tensor:
        """
        Prepare input tensor for processing.

        Args:
            x (torch.Tensor [B, *D, 2]):
                The input tensor, where x[..., 0] is the class ID and x[..., 1] the
                instance ID.
            mask (torch.Tensor [B, *D] optional):
                An optional mask to filter valid samples in the batch. Only samples
                where mask is True are considered for evaluation.
            allow_unknown (bool):
                Whether to allow unknown classes (classes that are not in
                classes_thing or classes_stuff). If False, an error is raised
                if any class ID is not in either set.

        Returns:
            torch.Tensor [N, 3]:
                The prepared input tensor, where x[..., 0] is the batch ID, x[..., 1]
                is the class ID, and x[..., 2] is the instance ID.
        """
        batch_size, *_, c = x.shape
        assert c == 2

        # detach input to avoid side effects
        x = x.detach().flatten(1, -2)

        # add batch ID to all elements
        batch_id = torch.arange(batch_size, device=x.device)
        batch_id = batch_id.view(batch_size, 1, 1).expand(-1, x.shape[1], 1)
        x = torch.cat([batch_id, x], dim=-1)

        # apply mask and flatten input to [N, 3]
        if mask is not None:
            x = x[mask.flatten(1, -1), :]
        else:
            x = x.flatten(0, -2)

        # generate class masks
        is_thing = torch.isin(x[..., 1], self.classes_thing)
        is_stuff = torch.isin(x[..., 1], self.classes_stuff)

        # make sure that we do not have any unknown classes
        if allow_unknown:
            x[~(is_thing | is_stuff), 1] = self.void_category
            x[~(is_thing | is_stuff), 2] = 0
        else:
            assert torch.all(is_thing | is_stuff)

        # reset/sanitize instance IDs for stuff classes
        x[is_stuff, 2] = 0

        return x

    def _get_segments(self, x: torch.Tensor) -> dict[_Color, int]:
        """
        Get unique segments and their area.

        Args:
            x (torch.Tensor [N, *D]):
                The input tensor, where unique items are determined along the first
                dimension.

        Returns:
            dict[_Color, int]:
                The unique segments and their area, meaning segments with same batch
                and class IDs (but not necessarily same instance ID).
        """
        # get unique segments and their area
        segments, area = torch.unique(x, dim=0, return_counts=True)

        segments = segments.cpu()
        area = area.cpu()

        # convert into dict: (batch, class, instance) -> area
        segments = _to_tuple(segments)
        segments = dict(zip(segments, area))

        return segments

    def _get_intersections(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[tuple[_Color, _Color], int]:
        """
        Get intersections between predictions and targets.

        Args:
            preds (torch.Tensor [N, 3]):
                The predicted segments, where preds[..., 0] is the batch ID,
                preds[..., 1] the class ID, and preds[..., 2] the instance ID.
            target (torch.Tensor [M, 3]):
                The target segments, where target[..., 0] is the batch ID,
                target[..., 1] the class ID, and target[..., 2] the instance ID.

        Returns:
            dict[tuple[_Color, _Color], int]:
                The intersections between predictions and targets, meaning pairs of
                segments with same batch and class IDs (but not necessarily same
                instance ID).
        """
        # overlay predictions and targets
        pairs = torch.stack((preds, target), dim=-2)  # [N, 2, 3]

        # get matching pairs (same batch and class, but not necessarily same instance)
        mask = torch.all(pairs[:, 0, :2] == pairs[:, 1, :2], dim=-1)

        # filter out invalid/ignored classes
        mask &= pairs[:, 0, 1] != self.void_category  # ignore void category

        pairs = pairs[mask, :, :]  # [K, 2, 3]

        # get unique matching segments and their area
        return self._get_segments(pairs)

    def _get_pred_void_areas(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[_Color, int]:
        """
        Get the area of each predicted segment that falls on void target pixels.

        Args:
            preds (torch.Tensor [N, 3]):
                The predicted segments, where preds[..., 1] is the class ID.
            target (torch.Tensor [N, 3]):
                The target segments aligned pixel-wise with preds.

        Returns:
            dict[_Color, int]:
                For each predicted segment that overlaps any void target pixel,
                the number of such pixels.
        """
        void_mask = target[:, 1] == self.void_category
        if not void_mask.any():
            return {}
        return self._get_segments(preds[void_mask])

    # pylint: disable-next=arguments-differ
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """
        Update the metric state with new predictions and targets.

        Args:
            preds (torch.Tensor [B, *D 2]):
                The predicted panoptic segmentation, where preds[..., 0] is the
                class ID and preds[..., 1] the instance ID.
            target (torch.Tensor [B, *D, 2]):
                The target panoptic segmentation, where target[..., 0] is the
                class ID and target[..., 1] the instance ID.
            mask (torch.Tensor [B, *D] optional):
                An optional mask to filter valid samples in the batch. Only samples
                where mask is True are considered for evaluation.
        """
        # pylint: disable=too-many-locals, too-many-statements

        # sanitize and flatten inputs to [N, 2]
        preds = self._prepare_input(preds, mask, self.allow_unknown_preds)
        target = self._prepare_input(target, mask, True)

        # compute predicted and target segments
        pred_area = self._get_segments(preds)
        target_area = self._get_segments(target)

        # compute intersection between predictions and targets
        intersections = self._get_intersections(preds, target)

        # compute void-target overlap per predicted segment (for FP exemption and IoU)
        pred_void_areas = self._get_pred_void_areas(preds, target)

        # alloate local tensors on CPU
        num_gt = torch.zeros(self.num_classes, dtype=torch.long)
        num_tp = torch.zeros(self.num_classes, dtype=torch.long)
        num_fp = torch.zeros(self.num_classes, dtype=torch.long)
        num_fn = torch.zeros(self.num_classes, dtype=torch.long)
        iou_sum_gt = torch.zeros(self.num_classes, dtype=torch.double)
        iou_sum_tp = torch.zeros(self.num_classes, dtype=torch.double)

        # compute true positives / find matches
        pred_matched, target_matched = set(), set()
        for (pred_color, target_color), intersect_area in intersections.items():
            # batch and class IDs should match
            assert pred_color[:2] == target_color[:2]

            class_id = target_color[1]

            # compute union, excluding pixels of the pred segment that fall on
            # void/unlabelled target regions (following panopticapi / EOPSN):
            #   union = pred_area - pred_void_area + target_area - intersection
            pred_void_area = pred_void_areas.get(pred_color, 0)
            union = (
                pred_area[pred_color]
                - pred_void_area
                + target_area[target_color]
                - intersect_area
            )

            assert union > 0
            assert union >= intersect_area

            # compute IoU
            iou = intersect_area / union

            assert iou > 0.0

            # update state tensors
            iou_sum_gt[class_id] += iou

            if iou > 0.5:  # threshold for true positive match
                num_tp[class_id] += 1
                iou_sum_tp[class_id] += iou

                pred_matched.add(pred_color)
                target_matched.add(target_color)

        # count ground truth segments, skipping void-class targets
        for _batch, class_id, _instance_id in target_area.keys():
            if class_id != self.void_category:
                num_gt[class_id] += 1

        # count false positives, skipping void-class predictions and predictions
        # that fall mostly on void target regions (treated as ignored, not a model error)
        false_positives = pred_area.keys() - pred_matched
        for pred_color in false_positives:
            _batch, class_id, _instance_id = pred_color
            if class_id == self.void_category:
                continue

            void_area = pred_void_areas.get(pred_color, 0)
            if void_area / pred_area[pred_color] <= 0.5:
                num_fp[class_id] += 1

        # count false negatives, skipping void-class targets
        false_negatives = target_area.keys() - target_matched
        for _batch, class_id, _instance_id in false_negatives:
            if class_id != self.void_category:
                num_fn[class_id] += 1

        # update state
        # pylint: disable=no-member
        self.num_gt += num_gt.to(self.num_gt.device)
        self.num_tp += num_tp.to(self.num_tp.device)
        self.num_fp += num_fp.to(self.num_fp.device)
        self.num_fn += num_fn.to(self.num_fn.device)
        self.iou_sum_gt += iou_sum_gt.to(self.iou_sum_gt.device)
        self.iou_sum_tp += iou_sum_tp.to(self.iou_sum_tp.device)

    def compute(self) -> Result:
        """
        Compute the panoptic quality metrics.

        Returns:
            Result:
                A proxy object that allows to derive various metrics from the
                computed intermediate results as well as the Panoptic Quality
                (PQ) itself.
        """
        # pylint: disable=no-member
        device = self.iou_sum_gt.device

        thing_mask = torch.arange(self.num_classes, device=device)
        thing_mask = torch.isin(thing_mask, self.classes_thing)

        return Result(
            gt=self.num_gt.clone(),
            tp=self.num_tp.clone(),
            fp=self.num_fp.clone(),
            fn=self.num_fn.clone(),
            iou_gt=self.iou_sum_gt.clone(),
            iou_tp=self.iou_sum_tp.clone(),
            thing_mask=thing_mask,
        )


def _to_tuple(x: torch.Tensor) -> tuple:
    """
    Convert a tensor to a tuple.

    Args:
        x (torch.Tensor):
            The input tensor.

    Returns:
        tuple:
            The input tensor as a tuple.
    """

    def _to_tuple_recursive(x: list) -> tuple:
        if isinstance(x, list):
            return tuple(map(_to_tuple_recursive, x))

        return x

    return _to_tuple_recursive(x.tolist())


def _class_mask(valid: torch.Tensor, classes: torch.Tensor | None) -> torch.Tensor:
    """
    Generate a class mask.

    Args:
        valid (torch.Tensor [num_classes]):
            The valid mask.

        classes (torch.Tensor [num_classes or N] optional):
            The classes to consider for the mask. Either a bool tensor or an
            index tensor. If None, all classes are considered. Defaults to
            None.

    Returns:
        torch.Tensor [num_classes]:
            The class mask.
    """
    if classes is None:
        return valid

    if not torch.is_tensor(classes):
        classes = list(classes)
        classes = torch.tensor(classes, device=valid.device, dtype=torch.int)

    if classes.dtype == torch.bool:
        return valid & classes

    num_classes = valid.shape[0]
    mask = torch.arange(num_classes, device=valid.device)
    mask = torch.isin(mask, classes)

    return valid & mask


def _get_void_category(things: set[int], stuffs: set[int]) -> tuple[int, int]:
    """Get an unused category ID.

    Args:
        things: The set of category IDs for things.
        stuffs: The set of category IDs for stuffs.

    Returns:
        A new category ID that does not belong to things nor stuffs.

    """
    return 1 + max([0, *list(things), *list(stuffs)])
