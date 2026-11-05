import copy
import functools
import warnings
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Callable, Collection, Iterable, Literal, Sequence, TypeVar

import torch
import torch.distributed as dist
from torchmetrics import Metric
from torchmetrics.utilities.distributed import gather_all_tensors

NO_INSTANCE = -1

_K = TypeVar("_K")


@dataclass(unsafe_hash=True)
class _SegmentKey:
    frame: int
    sequence: Hashable
    category: int
    instance: int

    def is_instance(self) -> bool:
        return self.instance != NO_INSTANCE


@dataclass
class _Segment:
    key: _SegmentKey
    area: int | float


@dataclass(unsafe_hash=True)
class _IntersectionKey:
    pred: _SegmentKey
    target: _SegmentKey

    def is_instance(self) -> bool:
        return NO_INSTANCE not in {self.pred.instance, self.target.instance}


@dataclass
class _Intersection:
    key: _IntersectionKey
    area: int | float


@dataclass(frozen=True)
class _MergeKey:
    sequence: Hashable
    instance: int
    keys: tuple[Any, ...]

    def __init__(self, sequence: Hashable, instance: int, *keys: Any):
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "instance", instance)
        object.__setattr__(self, "keys", keys)


class Result:
    """
    Proxy object to compute association quality and related metrics from
    intermediate results.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        preds: list[_Segment],
        targets: list[_Segment],
        intersects: list[_Intersection],
        per_frame: bool,
        classes: Collection[int],
    ):
        self._preds = preds
        self._targets = targets
        self._intersects = intersects
        self._per_frame = per_frame

        key = _key_aq_per_frame if per_frame else _key_aq

        self.total = SubsetResult(
            preds,
            targets,
            intersects,
            subset=lambda _: True,
            key=key,
        )

        def _filter_class(x: _SegmentKey, c: int) -> bool:
            return x.category == c

        self.per_class = {}
        for c in classes:
            self.per_class[c] = SubsetResult(
                preds,
                targets,
                intersects,
                subset=functools.partial(_filter_class, c=c),
                key=key,
            )


class SubsetResult:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        preds: list[_Segment],
        targets: list[_Segment],
        intersects: list[_Intersection],
        subset: Callable[[_SegmentKey], bool],
        key: Callable[[_SegmentKey], _MergeKey],
    ):
        self._preds = preds
        self._targets = targets
        self._intersects = intersects
        self._filter = subset
        self._key = key

        self._aq = None
        self._fpr = None
        self._fnr = None
        self._aq_per_seq = None
        self._fpr_per_seq = None
        self._fnr_per_seq = None

        self._num_gt_tracks = None
        self._num_gt_tracks_per_seq = None

        self._num_gt_elems = None
        self._num_gt_elems_per_seq = None

        self._num_pred_tracks = None
        self._num_pred_tracks_per_seq = None

        self._num_pred_elems = None
        self._num_pred_elems_per_seq = None

    def _compute_aq(self) -> None:
        if self._aq is not None:
            return

        # filter the segments
        preds = self._preds
        targets = [seg for seg in self._targets if self._filter(seg.key)]
        intersects = [seg for seg in self._intersects if self._filter(seg.key.target)]

        # compute the association quality
        result = _compute_aq(preds, targets, intersects, key=self._key)

        # store the results
        self._aq = result.aq
        self._fpr = result.fpr
        self._fnr = result.fnr
        self._aq_per_seq = result.aq_per_seq
        self._fpr_per_seq = result.fpr_per_seq
        self._fnr_per_seq = result.fnr_per_seq

    def _compute_num_gt_per_seq(self):
        if self._num_gt_tracks_per_seq is not None:
            return

        num_gt_tracks_per_seq = defaultdict(int)
        num_gt_elems_per_seq = defaultdict(int)
        for seg in self._targets:
            if not self._filter(seg.key):
                continue

            num_gt_tracks_per_seq[seg.key.sequence] += 1
            num_gt_elems_per_seq[seg.key.sequence] += seg.area

        self._num_gt_tracks_per_seq = num_gt_tracks_per_seq
        self._num_gt_elems_per_seq = num_gt_elems_per_seq

    def _compute_num_pred_per_seq(self):
        if self._num_pred_tracks_per_seq is not None:
            return

        num_pred_tracks_per_seq = defaultdict(int)
        num_pred_elems_per_seq = defaultdict(int)

        for seg in self._preds:
            if not self._filter(seg.key):
                continue

            num_pred_tracks_per_seq[seg.key.sequence] += 1
            num_pred_elems_per_seq[seg.key.sequence] += seg.area

        self._num_pred_tracks_per_seq = num_pred_tracks_per_seq
        self._num_pred_elems_per_seq = num_pred_elems_per_seq

    @property
    def num_gt_tracks(self) -> int:
        """
        The number of ground-truth tracks.
        """
        if self._num_gt_tracks is None:
            self._num_gt_tracks = sum(
                1 for seg in self._targets if self._filter(seg.key)
            )

        return self._num_gt_tracks

    @property
    def num_pred_tracks(self) -> int:
        """
        The number of predicted tracks.
        """
        if self._num_pred_tracks is None:
            self._num_pred_tracks = sum(
                1 for seg in self._preds if self._filter(seg.key)
            )

        return self._num_pred_tracks

    @property
    def num_gt_elems(self) -> int:
        """
        The number of ground-truth elements.
        """
        if self._num_gt_elems is None:
            targets = (seg for seg in self._targets if self._filter(seg.key))
            self._num_gt_elems = sum(seg.area for seg in targets)

        return self._num_gt_elems

    @property
    def num_pred_elems(self) -> int:
        """
        The number of predicted elements.
        """
        if self._num_pred_elems is None:
            preds = (seg for seg in self._preds if self._filter(seg.key))
            self._num_pred_elems = sum(seg.area for seg in preds)

        return self._num_pred_elems

    @property
    def num_gt_tracks_per_seq(self) -> dict[Any, int]:
        """
        The number of ground-truth tracks per sequence.
        """
        self._compute_num_gt_per_seq()
        return self._num_gt_tracks_per_seq

    @property
    def num_gt_elems_per_seq(self) -> dict[Any, int]:
        """
        The number of ground-truth elements per sequence.
        """
        self._compute_num_gt_per_seq()
        return self._num_gt_elems_per_seq

    @property
    def num_pred_tracks_per_seq(self) -> dict[Any, int]:
        """
        The number of predicted tracks per sequence.
        """
        self._compute_num_pred_per_seq()
        return self._num_pred_tracks_per_seq

    @property
    def num_pred_elems_per_seq(self) -> dict[Any, int]:
        """
        The number of predicted elements per sequence.
        """
        self._compute_num_pred_per_seq()
        return self._num_pred_elems_per_seq

    @property
    def aq(self) -> float:
        """
        The association quality (AQ).
        """
        self._compute_aq()
        return self._aq

    @property
    def fpr(self) -> float:
        """
        The false positive (FP) rate.

        Computed similar to the association quality (AQ) metric, but whereas
        the AQ represents the true positive rate (TP), the FP rate is computed
        as the IoU weighted average of false positives over all ground truth
        tracks.
        """
        self._compute_aq()
        return self._fpr

    @property
    def fnr(self) -> float:
        """
        The false negative (FN) rate.

        Computed similar to the association quality (AQ) metric, but whereas
        the AQ represents the true positive rate (TP), the FN rate is computed
        as the IoU weighted average of false negatives over all ground truth
        tracks.
        """
        self._compute_aq()
        return self._fnr

    @property
    def aq_per_seq(self) -> dict[Any, float]:
        """
        The association quality (AQ) per sequence.
        """
        self._compute_aq()
        return self._aq_per_seq

    @property
    def fpr_per_seq(self) -> dict[Any, float]:
        """
        The false positive (FP) rate per sequence.
        """
        self._compute_aq()
        return self._fpr_per_seq

    @property
    def fnr_per_seq(self) -> dict[Any, float]:
        """
        The false negative (FN) rate per sequence.
        """
        self._compute_aq()
        return self._fnr_per_seq


class _ComputeAqResult:
    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        aq: float,
        fpr: float,
        fnr: float,
        aq_per_seq: dict[Any, float],
        fpr_per_seq: dict[Any, float],
        fnr_per_seq: dict[Any, float],
    ):
        self._aq = aq
        self._fpr = fpr
        self._fnr = fnr
        self._aq_per_seq = aq_per_seq
        self._fpr_per_seq = fpr_per_seq
        self._fnr_per_seq = fnr_per_seq

    @property
    def aq(self) -> float:
        """
        The association quality (AQ).
        """
        return self._aq

    @property
    def fpr(self) -> float:
        """
        The false positive (FP) rate.

        Computed similar to the association quality (AQ) metric, but whereas
        the AQ represents the true positive rate (TP), the FP rate is computed
        as the IoU weighted average of false positives over all ground truth
        tracks.
        """
        return self._fpr

    @property
    def fnr(self) -> float:
        """
        The false negative (FN) rate.

        Computed similar to the association quality (AQ) metric, but whereas
        the AQ represents the true positive rate (TP), the FN rate is computed
        as the IoU weighted average of false negatives over all ground truth
        tracks.
        """
        return self._fnr

    @property
    def aq_per_seq(self) -> dict[Any, float]:
        """
        The association quality (AQ) per sequence.
        """
        return self._aq_per_seq

    @property
    def fpr_per_seq(self) -> dict[Any, float]:
        """
        The false positive (FP) rate per sequence.
        """
        return self._fpr_per_seq

    @property
    def fnr_per_seq(self) -> dict[Any, float]:
        """
        The false negative (FN) rate per sequence.
        """
        return self._fnr_per_seq


class AssociationQuality(Metric):
    """
    Association quality (AQ) metric for segmentation and tracking, as part of
    the Segmentation and Tracking Quality (STQ).

    Refer to https://arxiv.org/abs/2102.11859 for more details.
    """

    # pylint: disable=too-many-instance-attributes

    is_differentiable: bool = False
    full_state_update: bool = False

    # buffers
    class_subset: torch.Tensor

    # state
    segments_pred: list[_Segment]
    segments_target: list[_Segment]
    segments_intersect: list[_Intersection]

    def __init__(
        self,
        num_classes: int,
        class_subset: Collection[int],
        ignored_classes: Collection[int] | None = None,
        min_count: int = 0,
        per_frame: bool = False,
        allow_invalid_instances: Literal["disallow", "restrict", "ignore"] = "disallow",
    ):
        """
        Initialize the metric."

        Args:
            num_classes (int):
                The number of classes in the dataset.
            class_subset (Collection[int]):
                The classes to consider for the association quality metric
                (e.g., "thing" classes). All other classes are ignored. Note
                that this is different from ignored_classes in that
                intersections between the subset and other non-ignored classes
                (e.g., cars and street) will be treated as false positives.
                The classes are expected to be in the range [0, num_classes).
            ignored_classes (Collection[int], optional):
                The classes to ignore in the metric computation. Regions with
                these classes as ground-truth labels are masked out completely
                and treated as unlabeled areas. Meaning predictions inside
                these regions are not counted as false positives. The classes
                are expected to be in the range [0, num_classes). Defaults to
                None.
            min_count (int, optional):
                The minimum number of points/voxels per frame for a
                ground-truth instance to be considered valid. Instances with
                fewer points/voxels are ignored. Defaults to 0.
            per_frame (bool, optional):
                Whether to compute the association quality per frame/sample or
                over temporal sequences. Meaning, whether to comptue the AQ
                independently for each frame for classical instance
                segmentation (per_sample=True) or for the entire sequences for
                tracking (per_sample=False). Defaults to False.
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
        self.min_count = min_count
        self.per_frame = per_frame
        self.allow_invalid_instances = allow_invalid_instances

        assert allow_invalid_instances in {"disallow", "restrict", "ignore"}

        # counter to get unique (process local) frame IDs
        self.frame_count = 0

        # set up class tensors
        class_subset = torch.tensor(sorted(class_subset), dtype=torch.long)
        self.register_buffer("class_subset", class_subset, persistent=False)

        ignored = set(ignored_classes) if ignored_classes else set()
        ignored = torch.tensor(sorted(ignored), dtype=torch.long)
        self.register_buffer("ignored_classes", ignored, persistent=False)

        # set up state
        self.add_state("segments_pred", default=[], dist_reduce_fx=None)
        self.add_state("segments_target", default=[], dist_reduce_fx=None)
        self.add_state("segments_intersect", default=[], dist_reduce_fx=None)

    def reset(self) -> None:
        super().reset()

        self.frame_count = 0

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Any | None = None,
    ) -> None:
        # pylint: disable=too-many-locals

        # Note: Torchmetrics currently does not support states other than
        #       tensors or lists of tensors. So we need to do things manually
        #       here.

        # deduplicate the segments before syncing
        if not self.per_frame:
            self.segments_pred = _deduplicate_segments(self.segments_pred)
            self.segments_target = _deduplicate_segments(self.segments_target)
            self.segments_intersect = _deduplicate_intersect(self.segments_intersect)

        # collect the current state
        state = (self.segments_pred, self.segments_target, self.segments_intersect)

        # gather across processes
        world_size = dist.get_world_size(process_group)
        synced = [([], [], []) for _ in range(world_size)]

        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            # Note: This will emit a warning that pickle is unsafe... we take
            # not of that and ignore it.
            dist.all_gather_object(synced, state, group=process_group)

        # if we evaluate AQ per frame, we need to update the frame ID for each
        # sample
        if self.per_frame:
            n = 0

            for pred, target, intersect in synced:
                # get the number of frames for the current process
                # NOTE: We do not need to check intersect, as it is derived
                #       from preds and target.
                max_frame_p = max(x.key.frame for x in pred) if pred else -1
                max_frame_t = max(x.key.frame for x in target) if target else -1
                max_frame = max(max_frame_p, max_frame_t)

                # update the frame ID for each segment
                for p in pred:
                    p.key.frame += n

                for t in target:
                    t.key.frame += n

                for i in intersect:
                    i.key.pred.frame += n
                    i.key.target.frame += n

                # update the frame count
                n = n + max_frame + 1

        # reduce the synced states
        preds, targets, intersects = zip(*synced)
        preds = [x for process in preds for x in process]
        targets = [x for process in targets for x in process]
        intersects = [x for process in intersects for x in process]

        # update the state, deduplicate the segments
        self.segments_pred = _deduplicate_segments(preds)
        self.segments_target = _deduplicate_segments(targets)
        self.segments_intersect = _deduplicate_intersect(intersects)

    def _get_local_sequence_index(
        self, sequence_id: torch.Tensor | Sequence[Hashable]
    ) -> Sequence[int] | dict[int, Hashable]:
        """
        Convert sequence IDs to batch-local sequence indices.

        Note:
            This method is used to convert sequence IDs to batch-local indices
            and vice versa. The conversion is necessary to handle arbitrary
            sequence IDs (e.g., NuScenes scene tokens) in the metric
            computation.

            The returned sequence IDs are unique integers _local to the current
            batch_. They cannot be used to match sequences across batches.

        Args:
            sequence_id (torch.Tensor [B] | Sequence[Hashable]):
                The sequence ID for each sample in the batch.

        Returns:
            sequence_id (Sequence[int]):
                The sequence ID for each sample in the batch as unique integers
                local to the current batch.
            index_to_id (dict[int, Hashable]):
                A mapping from the unique integers to the original sequence IDs
                to facilitate global sequence ID matching.
        """
        # pylint: disable=unnecessary-comprehension

        if torch.is_tensor(sequence_id):
            sequence_id = sequence_id.tolist()

        unique_ids = set(sequence_id)

        id_to_index = {seq_id: i for i, seq_id in enumerate(unique_ids)}
        index_to_id = {i: seq_id for seq_id, i in id_to_index.items()}

        sequence_id = [id_to_index[seq_id] for seq_id in sequence_id]
        sequence_id = torch.tensor(sequence_id, dtype=torch.long)

        return sequence_id, index_to_id

    def _prepare_input(
        self,
        x: torch.Tensor,
        sequence_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare the input for the metric calculation.

        Args:
            x (torch.Tensor [B, *D, 2]):
                The input tensor, where x[..., 0] is the class ID and x[..., 1]
                the instance ID.
            sequence_index (torch.Tensor [B]):
                The sequence index for each sample in the batch.
            mask (torch.Tensor [B, *D]):
                A mask to filter valid samples in the batch. Only samples where
                mask is True are considered for evaluation.

        Returns:
            x (torch.Tensor [N, 4]):
                The prepared input tensor, where x[..., 0] is the batch ID,
                x[..., 1] the sequence ID, x[..., 2] the class ID, and
                x[..., 3] the instance ID.
        """

        batch_size, *_ = x.shape

        # detach and flatten the input
        x = x.detach().flatten(1, -2)
        x = x.to(dtype=torch.long)

        # add the sequence index to the input to prevent matching across
        # sequences
        seq = sequence_index.to(device=x.device)
        seq = seq.view(batch_size, 1, 1).expand(-1, x.shape[1], 1)

        # if we compute AQ per frame, prefix the (batch-local) frame index to
        # the area keys to prevent matching across frames
        if self.per_frame:
            batch_id = torch.arange(batch_size, device=x.device)
            batch_id = batch_id.view(batch_size, 1, 1).expand(-1, x.shape[1], 1)
        else:
            batch_id = torch.zeros_like(seq)

        x = torch.cat((batch_id, seq, x), dim=-1)

        # apply mask and flatten input to [N, 4]
        return x[mask.flatten(1, -1), :]

    def _prepare_weights(
        self,
        weights: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Prepare weights by applying the same mask used in _prepare_input.

        Args:
            weights (torch.Tensor [B, *D]):
                Per-element weights.
            mask (torch.Tensor [B, *D]):
                The mask of valid elements.

        Returns:
            weights (torch.Tensor [N]):
                The flattened, masked weights as float64.
        """
        if weights is None:
            return None

        return weights.detach().flatten(1, -1)[mask.flatten(1, -1)].to(torch.float64)

    def _sanitize_instance_ids(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # sanitize predictions
        is_instance = preds[..., 3] >= 0

        # set instance IDs for stuff/non-thing classes to NO_INSTANCE so that
        # we can easily filter them out
        preds[~is_instance, 2:] = NO_INSTANCE

        # get mask for instance/thing classes
        is_instance = torch.isin(targets[..., 2], self.class_subset)

        if self.allow_invalid_instances == "disallow":
            # make sure all instance IDs for instance/thing classes are valid
            assert torch.all(
                targets[is_instance, 3] >= 0
            ), "Instance IDs must be non-negative."

        elif self.allow_invalid_instances == "restrict":
            # treat invalid instance IDs for instance/thing classes as non-instance
            is_instance = is_instance & (targets[..., 3] >= 0)

        elif self.allow_invalid_instances == "ignore":
            # filter out invalid instance IDs for instance/thing classes and treat
            # them as ignored/masked
            is_invalid = is_instance & (targets[..., 3] < 0)

            is_instance = is_instance[~is_invalid]
            preds = preds[~is_invalid, :]
            targets = targets[~is_invalid, :]
            if weights is not None:
                weights = weights[~is_invalid]

        # set instance IDs for stuff/non-thing classes to NO_INSTANCE so that
        # we can easily filter them out
        targets[~is_instance, 2:] = NO_INSTANCE

        return preds, targets, weights

    def _get_segments(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> list[_Segment]:
        """
        Get unique segments and their area.

        Args:
            x (torch.Tensor [N, 4]):
                The input tensor, where unique items are determined along the first
                dimension.
            weights (torch.Tensor [N], optional):
                Per-element weights. When provided, segment areas are the sum of
                weights rather than pixel counts. Defaults to None.

        Returns:
            list[_Segment]:
                The unique segments and their area, meaning segments with same
                segment key.
        """
        if weights is None:
            segments, area = torch.unique(x, dim=0, return_counts=True)
            area = area.tolist()
        else:
            segments, inverse = torch.unique(x, dim=0, return_inverse=True)
            area = torch.zeros(segments.shape[0], dtype=torch.float64, device=x.device)
            area.scatter_add_(0, inverse, weights.to(torch.float64))
            area = area.tolist()

        segments = segments.tolist()
        segments = (_SegmentKey(*k) for k in segments)
        segments = [_Segment(k, a) for k, a in zip(segments, area)]

        return segments

    def _get_intersections(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> list[_Intersection]:
        """
        Get intersections between predictions and targets.

        Args:
            preds (torch.Tensor [N, 4]):
                The predicted segments, where preds[..., 0] is the batch/frame
                ID, preds[..., 1] the sequence ID, preds[..., 2] the class ID, and
                preds[..., 3] the instance ID.
            target (torch.Tensor [M, 4]):
                The target/gt segments, where target[..., 0] is the batch/frame
                ID, target[..., 1] the sequence ID, target[..., 2] the class
                ID, and target[..., 3] the instance ID.
            weights (torch.Tensor [N], optional):
                Per-element weights. When provided, intersection areas are the
                sum of weights rather than pixel counts. Defaults to None.

        Returns:
            list[_Intersection]:
                The intersections between predictions and targets, meaning
                pairs of segments with same batch and sequence IDs (but not
                necessarily same class and instance ID).
        """
        # overlay predictions and targets
        pairs = torch.stack((preds, target), dim=-2)  # [N, 2, 4]

        # ensure that the batch/frame ID and sequence ID match
        mask = torch.all(pairs[:, 0, :2] == pairs[:, 1, :2], dim=-1)
        assert torch.all(mask)

        # get unique matching segments and their area
        if weights is None:
            segments, area = torch.unique(pairs, dim=0, return_counts=True)
            area = area.tolist()
        else:
            segments, inverse = torch.unique(pairs, dim=0, return_inverse=True)
            area = torch.zeros(
                segments.shape[0], dtype=torch.float64, device=pairs.device
            )
            area.scatter_add_(0, inverse, weights.to(torch.float64))
            area = area.tolist()

        segments = segments.tolist()
        segments = ((_SegmentKey(*kp), _SegmentKey(*kt)) for (kp, kt) in segments)
        segments = (_IntersectionKey(*k) for k in segments)
        segments = [_Intersection(k, a) for k, a in zip(segments, area)]

        return segments

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
                The target panoptic segmentation, where preds[..., 0] is the
                class ID and preds[..., 1] the instance ID.
            sequence_id (torch.Tensor [B] | Sequence[Hashable])
                The sequence ID for each sample in the batch.
            mask (torch.Tensor [B, *D], optional)
                An optional mask to filter valid samples in the batch. Only samples
                where mask is True are considered for evaluation.
            weights (torch.Tensor [B, *D], optional)
                An optional tensor of weights for each element (pixel/voxel) in
                the batch.
        """
        # pylint: disable=too-many-locals
        batch_size, *_ = preds.shape

        # convert sequence IDs to batch-local indices
        sequence_id, sequence_id_map = self._get_local_sequence_index(sequence_id)

        # filter out ignored classes
        is_ignored = torch.isin(target[..., 0], self.ignored_classes)
        mask = (mask & ~is_ignored) if mask is not None else ~is_ignored

        # sanitize/prepare the input
        w = self._prepare_weights(weights, mask)
        preds = self._prepare_input(preds, sequence_id, mask)
        target = self._prepare_input(target, sequence_id, mask)
        preds, target, w = self._sanitize_instance_ids(preds, target, w)

        # compute predicted and target segments
        pr_area = self._get_segments(preds, w)
        gt_area = self._get_segments(target, w)

        # compute intersection between predictions and targets
        isect = self._get_intersections(preds, target, w)

        # filter out stuff/non-thing segments
        pr_area = [seg for seg in pr_area if seg.key.is_instance()]
        gt_area = [seg for seg in gt_area if seg.key.is_instance()]
        isect = [seg for seg in isect if seg.key.is_instance()]

        # filter out GT instances with too few points/voxels
        invalid_gt = set(seg.key for seg in gt_area if seg.area <= self.min_count)
        gt_area = [seg for seg in gt_area if seg.key not in invalid_gt]
        isect = [seg for seg in isect if seg.key.target not in invalid_gt]

        # map keys back to the original sequence IDs (and global frame IDs if we
        # compute AQ per frame)
        frame_offset = self.frame_count if self.per_frame else 0

        for seg in pr_area:
            seg.key.frame = frame_offset + seg.key.frame
            seg.key.sequence = sequence_id_map[seg.key.sequence]

        for seg in gt_area:
            seg.key.frame = frame_offset + seg.key.frame
            seg.key.sequence = sequence_id_map[seg.key.sequence]

        for seg in isect:
            seg.key.pred.frame = frame_offset + seg.key.pred.frame
            seg.key.target.frame = frame_offset + seg.key.target.frame

            seg.key.pred.sequence = sequence_id_map[seg.key.pred.sequence]
            seg.key.target.sequence = sequence_id_map[seg.key.target.sequence]

        # update the frame count
        self.frame_count += batch_size

        # update the state
        self.segments_pred += pr_area
        self.segments_target += gt_area
        self.segments_intersect += isect

        # deduplicate the segments every now and then to conserve memory
        if not self.per_frame and self.frame_count % 100 == 0:
            self.segments_pred = _deduplicate_segments(self.segments_pred)
            self.segments_target = _deduplicate_segments(self.segments_target)
            self.segments_intersect = _deduplicate_intersect(self.segments_intersect)

    def compute(self) -> Result:
        """
        Compute the association quality metric.

        Returns:
            Result:
                A proxy object that allows retrieving the association quality and
                per-sequence association quality.
        """
        return Result(
            preds=copy.deepcopy(self.segments_pred),
            targets=copy.deepcopy(self.segments_target),
            intersects=copy.deepcopy(self.segments_intersect),
            per_frame=self.per_frame,
            classes=self.class_subset.tolist(),
        )


def _compute_aq(
    preds: Sequence[_Segment],
    targets: Sequence[_Segment],
    intersects: Sequence[_Intersection],
    key: Callable[[_SegmentKey], _MergeKey],
) -> _ComputeAqResult:
    """
    Compute the association quality metric.

    Args:
        preds (Sequence[_Segment]):
            The predicted segments, including the area.
        targets (Sequence[_Segment]):
            The target segments, including the area.
        intersects (Sequence[_Intersection]):
            The intersections between predictions and targets, including the
            intersection area.
        key (Callable[[_SegmentKey], _MergeKey]):
            A function that returns key properties used to determine whether
            segments are the same. The function is called with the segment key
            and should return a new key that is used to merge similer segments.

    Returns:
        tuple[float, dict[Any, float]]:
            The association quality and per-sequence association quality.
    """
    # pylint: disable=too-many-locals

    # merge / deduplicate the segments
    preds = [(key(x.key), x.area) for x in preds]
    targets = [(key(x.key), x.area) for x in targets]
    intersects = [((key(x.key.pred), key(x.key.target)), x.area) for x in intersects]

    preds = _deduplicate(preds)
    targets = _deduplicate(targets)
    intersects = _deduplicate(intersects)

    # compute the number of segments
    num_gt_tracks = len(targets)

    # compute the number of ground-truth tracks per sequence
    num_gt_tracks_per_seq = defaultdict(int)
    for key_target, _ in targets.items():
        num_gt_tracks_per_seq[key_target.sequence] += 1

    # reorganize the segments
    # Note: We could compute the AQ in a single loop, but this more closely
    # follows the reference implementation and is numerically more stable.
    intersects_for_target = defaultdict(list)
    for (key_pred, key_target), area in intersects.items():
        intersects_for_target[key_target].append((key_pred, area))

    # compute the association quality per sequence
    aq_per_seq = defaultdict(float)
    fpr_per_seq = defaultdict(float)
    fnr_per_seq = defaultdict(float)

    for key_target, num_target in targets.items():
        sequence_id = key_target.sequence

        aq_per_track = 0.0
        fpr_per_track = 0.0
        fnr_per_track = 0.0
        for key_pred, tpa in intersects_for_target[key_target]:
            assert key_pred.sequence == sequence_id, "Sequence IDs must match."

            num_pred = preds[key_pred]

            fpa = num_pred - tpa
            fna = num_target - tpa
            iou = tpa / (fpa + fna + tpa)

            aq_per_track += tpa * iou
            fpr_per_track += fpa * iou
            fnr_per_track += fna * iou

        aq_per_seq[sequence_id] += aq_per_track / num_target
        fpr_per_seq[sequence_id] += fpr_per_track / num_target
        fnr_per_seq[sequence_id] += fnr_per_track / num_target

    # compute the overall association quality
    aq = sum(aq_per_seq.values()) / max(num_gt_tracks, 1)
    fpr = sum(fpr_per_seq.values()) / max(num_gt_tracks, 1)
    fnr = sum(fnr_per_seq.values()) / max(num_gt_tracks, 1)

    aq_per_seq = {k: v / num_gt_tracks_per_seq[k] for k, v in aq_per_seq.items()}
    fpr_per_seq = {k: v / num_gt_tracks_per_seq[k] for k, v in fpr_per_seq.items()}
    fnr_per_seq = {k: v / num_gt_tracks_per_seq[k] for k, v in fnr_per_seq.items()}

    return _ComputeAqResult(
        aq=aq,
        fpr=fpr,
        fnr=fnr,
        aq_per_seq=aq_per_seq,
        fpr_per_seq=fpr_per_seq,
        fnr_per_seq=fnr_per_seq,
    )


def _key_aq(key: _SegmentKey) -> _MergeKey:
    return _MergeKey(key.sequence, key.instance)


def _key_aq_per_frame(key: _SegmentKey) -> _MergeKey:
    return _MergeKey(key.sequence, key.instance, key.frame)


def _deduplicate(sequence: Iterable[tuple[_K, int]]) -> dict[_K, int]:
    """
    Deduplicate sequences by summing the values for each unique key.

    Args:
        segments (list[tuple[_K, int]]):
            The segments to deduplicate, where each segment is a tuple of
            (batch, class, instance) and the area is the second element of
            the tuple.

    Returns:
        dict[_Color, int]:
            The deduplicated segments.
    """
    result = defaultdict(int)

    for key, value in sequence:
        result[key] += value

    return dict(result)


def _deduplicate_segments(seg: Iterable[_Segment]) -> list[_Segment]:
    seg = ((x.key, x.area) for x in seg)
    seg = _deduplicate(seg)
    seg = [_Segment(k, v) for k, v in seg.items()]
    return seg


def _deduplicate_intersect(seg: Iterable[_Intersection]) -> list[_Intersection]:
    seg = ((x.key, x.area) for x in seg)
    seg = _deduplicate(seg)
    seg = [_Intersection(k, v) for k, v in seg.items()]
    return seg
