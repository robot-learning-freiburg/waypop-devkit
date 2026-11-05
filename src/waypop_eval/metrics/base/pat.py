"""
Panoptic Tracking (PAT) metric and corresponding Tracking Quality (TQ) metric.

PAT is the harmonic mean of Panoptic Quality (PQ) and Tracking Quality (TQ). TQ
itself combines an association score and a track fragmentation component.

References:
- PAT and TQ metrics: https://arxiv.org/abs/2109.03805
"""

import math
import warnings
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Callable, Collection, Literal, Sequence

import torch
import torch.distributed as dist
from torchmetrics import Metric
from torchmetrics.utilities.distributed import gather_all_tensors

from .panoptic_quality import PanopticQuality
from .panoptic_quality import Result as PanopticQualityResult

NO_INSTANCE = -1  # sentinel for "not a valid instance" in input (class, instance) pairs
NO_MATCH = -1  # sentinel for "GT unmatched in this frame" in _GtMatch records

# Segment "color" key (batch, class, instance)
_Color = tuple[int, int, int]


@dataclass(unsafe_hash=True)
class _GtMatchKey:
    sequence: Hashable
    category: int
    gt_instance: int
    frame: int


@dataclass
class _GtMatch:
    key: _GtMatchKey
    pred_instance: int  # NO_MATCH if unmatched


@dataclass(unsafe_hash=True)
class _PredOccKey:
    sequence: Hashable
    pred_instance: int
    frame: int


@dataclass
class _PredOcc:
    key: _PredOccKey


class TrackingQualityResult:
    """
    Proxy object to compute Tracking Quality (TQ) and related per-track and
    per-class statistics from accumulated GT matches and pred occurrences.
    """

    def __init__(
        self,
        gt_matches: list[_GtMatch],
        pred_occurrences: list[_PredOcc],
        classes: Collection[int],
    ) -> None:
        """
        Initialize a new TrackingQualityResult instance.

        Args:
            gt_matches (list[_GtMatch]):
                One entry per (sequence, class, GT instance, frame): the
                predicted instance matched to the GT in that frame, or
                NO_MATCH if unmatched.
            pred_occurrences (list[_PredOcc]):
                One entry per (sequence, predicted instance, frame):
                represents that this predicted instance appeared in that
                frame. Pred occurrences are class-agnostic - the total
                occurrence count for a (sequence, predicted instance) pair
                spans all classes in which that pred ID was predicted.
            classes (Collection[int]):
                The set of tracked classes. Only GT tracks in these classes
                contribute to the aggregate TQ.
        """
        self._gt_matches = gt_matches
        self._pred_occurrences = pred_occurrences
        self._classes = set(classes)

        self._per_track: dict[tuple[Hashable, int, int], float] | None = None
        self._per_track_as: dict[tuple[Hashable, int, int], float] | None = None
        self._per_track_tf: dict[tuple[Hashable, int, int], float] | None = None

    def _compute(self) -> None:
        # pylint: disable=too-many-locals
        if self._per_track is not None:
            return

        # group GT matches by track (sequence, class, gt_instance) -> list of (frame, pred_inst)
        tracks: dict[tuple[Hashable, int, int], list[tuple[int, int]]] = defaultdict(
            list
        )
        for m in self._gt_matches:
            tracks[(m.key.sequence, m.key.category, m.key.gt_instance)].append(
                (m.key.frame, m.pred_instance)
            )

        # count pred occurrences per (sequence, pred_instance) — class-agnostic, matching the
        # nuScenes reference: the same pred ID's total frame count spans all classes in a sequence
        pred_totals: dict[tuple[Hashable, int], int] = defaultdict(int)
        for occ in self._pred_occurrences:
            pred_totals[(occ.key.sequence, occ.key.pred_instance)] += 1

        per_track: dict[tuple[Hashable, int, int], float] = {}
        per_track_as: dict[tuple[Hashable, int, int], float] = {}
        per_track_tf: dict[tuple[Hashable, int, int], float] = {}

        for track_key, entries in tracks.items():
            sequence, _, _ = track_key

            # sort pr_ids by frame to get the temporal order
            entries.sort(key=lambda x: x[0])
            pr_ids = [pi for _, pi in entries]
            length = len(pr_ids)

            # association score
            matched = [p for p in pr_ids if p != NO_MATCH]
            if not matched:
                assoc = 0.0
            else:
                counts: dict[int, int] = defaultdict(int)
                for p in matched:
                    counts[p] += 1

                assoc = 0.0
                for pred_inst, count in counts.items():
                    total = pred_totals.get((sequence, pred_inst), count)
                    fp = total - count
                    assoc += (count * count) / (length + fp)
                assoc /= length

            # track fragmentation (1 - IDS / (L-1); higher = fewer identity switches)
            if length <= 1:
                frag = 1.0
            else:
                ids = 0
                prev: int | None = None
                for p in pr_ids:
                    if prev is not None and (p != prev or prev == NO_MATCH):
                        ids += 1
                    prev = p
                frag = 1.0 - ids / (length - 1)

            per_track_as[track_key] = assoc
            per_track_tf[track_key] = frag
            per_track[track_key] = math.sqrt(assoc * frag)

        self._per_track = per_track
        self._per_track_as = per_track_as
        self._per_track_tf = per_track_tf

    def _aggregate(
        self, values: dict[tuple[Hashable, int, int], float], classes: set[int] | None
    ) -> float:
        if not values:
            return 0.0

        if classes is None:
            selected = list(values.values())
        else:
            selected = [v for (_s, c, _i), v in values.items() if c in classes]

        if not selected:
            return 0.0

        return sum(selected) / len(selected)

    @property
    def tq(self) -> float:
        """
        Mean Tracking Quality (TQ) over all GT tracks in the configured
        tracked classes.
        """
        return self.mean_tq()

    @property
    def association_score(self) -> float:
        """
        Mean per-track association score over all GT tracks in the configured
        tracked classes.
        """
        return self.mean_association_score()

    @property
    def track_fragmentation(self) -> float:
        """
        Mean per-track track fragmentation score over all GT tracks in the
        configured tracked classes (1 - IDS/(L-1); higher = fewer identity
        switches).
        """
        return self.mean_track_fragmentation()

    def mean_tq(self, classes: Collection[int] | None = None) -> float:
        """
        Mean Tracking Quality (TQ) over all GT tracks in the given class
        subset.

        Args:
            classes (Collection[int], optional):
                The classes to aggregate over. If None, the classes configured
                on the metric are used. Defaults to None.
        """
        self._compute()
        selected = self._classes if classes is None else set(classes)
        return self._aggregate(self._per_track, selected)

    def mean_association_score(self, classes: Collection[int] | None = None) -> float:
        """
        Mean per-track association score over all GT tracks in the given
        class subset.

        Args:
            classes (Collection[int], optional):
                The classes to aggregate over. If None, the classes configured
                on the metric are used. Defaults to None.
        """
        self._compute()
        selected = self._classes if classes is None else set(classes)
        return self._aggregate(self._per_track_as, selected)

    def mean_track_fragmentation(self, classes: Collection[int] | None = None) -> float:
        """
        Mean per-track track fragmentation score over all GT tracks in the
        given class subset.

        Args:
            classes (Collection[int], optional):
                The classes to aggregate over. If None, the classes configured
                on the metric are used. Defaults to None.
        """
        self._compute()
        selected = self._classes if classes is None else set(classes)
        return self._aggregate(self._per_track_tf, selected)

    @property
    def tq_per_class(self) -> dict[int, float]:
        """
        Per-class mean TQ. Classes with no GT tracks are omitted.
        """
        return self._per_class_mean(self._track_values)

    @property
    def association_score_per_class(self) -> dict[int, float]:
        """
        Per-class mean per-track association score. Classes with no GT tracks
        are omitted.
        """
        return self._per_class_mean(self._track_as_values)

    @property
    def track_fragmentation_per_class(self) -> dict[int, float]:
        """
        Per-class mean per-track track fragmentation score. Classes with no
        GT tracks are omitted.
        """
        return self._per_class_mean(self._track_tf_values)

    @property
    def _track_values(self) -> dict[tuple[Hashable, int, int], float]:
        self._compute()
        return self._per_track

    @property
    def _track_as_values(self) -> dict[tuple[Hashable, int, int], float]:
        self._compute()
        return self._per_track_as

    @property
    def _track_tf_values(self) -> dict[tuple[Hashable, int, int], float]:
        self._compute()
        return self._per_track_tf

    def _per_class_mean(
        self, values: dict[tuple[Hashable, int, int], float]
    ) -> dict[int, float]:
        per_class: dict[int, list[float]] = defaultdict(list)
        for (_s, c, _i), v in values.items():
            if c in self._classes:
                per_class[c].append(v)

        return {c: sum(vs) / len(vs) for c, vs in per_class.items()}

    @property
    def num_tracks(self) -> int:
        """
        Number of GT tracks contributing to the aggregate TQ.
        """
        self._compute()
        return sum(1 for (_s, c, _i) in self._per_track if c in self._classes)


class TrackingQuality(Metric):
    """
    Tracking Quality (TQ) metric for panoptic tracking.

    Computes a per-track association score and track fragmentation score and
    combines them as TQ = sqrt(association_score * track_fragmentation), then
    takes the mean over all GT tracks in the tracked classes.

    The caller supplies an explicit `frame_id` for each sample in `update()`.
    Frame IDs must be unique within each sequence (across all batches and all
    processes) and sort into temporal order. Given this, sequences may be
    split across processes freely.

    Refer to https://arxiv.org/abs/2109.03805 for more details.
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    # buffers
    track_classes: torch.Tensor
    ignored_classes: torch.Tensor

    # state (list of _GtMatch / _PredOcc, custom _sync_dist)
    gt_matches: list[_GtMatch]
    pred_occurrences: list[_PredOcc]

    def __init__(
        self,
        num_classes: int,
        track_classes: Collection[int],
        ignored_classes: Collection[int] | None = None,
        min_instance_count: int = 0,
        allow_invalid_instances: Literal["disallow", "restrict", "ignore"] = "disallow",
    ):
        """
        Initialize a new TrackingQuality instance.

        Input sanitization assumes methods only assign instance IDs to
        instance (thing) classes - on both sides, pixels whose class is not
        in `track_classes` are treated as non-instance regardless of their
        instance ID. A negative pred instance ID is also treated as
        non-instance as a safeguard.

        Args:
            num_classes (int):
                The total number of classes in the dataset.
            track_classes (Collection[int]):
                The class IDs to compute tracking quality for. Typically the
                "thing" classes.
            ignored_classes (Collection[int], optional):
                Classes to ignore. Regions with these GT labels are masked
                out, so predictions there do not count as false positives.
                Defaults to None.
            min_instance_count (int, optional):
                Minimum per-frame element count for a GT or predicted instance
                to be considered valid. Instances with fewer elements in a
                given frame are ignored for that frame (on both the GT and
                pred sides, matching the nuScenes reference). Defaults to 0.
            allow_invalid_instances (str, optional):
                How to handle target pixels whose class is in `track_classes`
                but whose instance ID is negative. "disallow" (default) asserts
                no such pixel exists. "restrict" treats them as non-instance.
                "ignore" drops them entirely (as if masked out). Defaults to
                "disallow".
        """
        super().__init__()

        self.num_classes = num_classes
        self.min_instance_count = min_instance_count
        self.allow_invalid_instances = allow_invalid_instances
        assert allow_invalid_instances in {"disallow", "restrict", "ignore"}

        track = torch.tensor(sorted(set(track_classes)), dtype=torch.long)
        self.register_buffer("track_classes", track, persistent=False)

        ignored = set(ignored_classes) if ignored_classes else set()
        ignored = torch.tensor(sorted(ignored), dtype=torch.long)
        self.register_buffer("ignored_classes", ignored, persistent=False)

        self.add_state("gt_matches", default=[], dist_reduce_fx=None)
        self.add_state("pred_occurrences", default=[], dist_reduce_fx=None)

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Any | None = None,
    ) -> None:
        # Callers supply globally-unique frame IDs (unique within each sequence
        # across all batches and all ranks), so record keys never collide and
        # we just concatenate the gathered lists.

        state = (self.gt_matches, self.pred_occurrences)

        world_size = dist.get_world_size(process_group)
        synced: list[tuple[list[_GtMatch], list[_PredOcc]]] = [
            ([], []) for _ in range(world_size)
        ]

        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            # dist.all_gather_object emits a pickle-is-unsafe warning; accept it
            # (mirrors association_quality.AssociationQuality).
            dist.all_gather_object(synced, state, group=process_group)

        gt_all: list[_GtMatch] = []
        pred_all: list[_PredOcc] = []
        for gts, preds in synced:
            gt_all.extend(gts)
            pred_all.extend(preds)

        self.gt_matches = gt_all
        self.pred_occurrences = pred_all

    # pylint: disable-next=arguments-differ,too-many-locals,too-many-branches
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sequence_id: torch.Tensor | Sequence[Hashable],
        frame_id: torch.Tensor | Sequence[int],
        mask: torch.Tensor | None = None,
    ) -> None:
        """
        Update the metric state with new predictions and targets.

        Args:
            preds (torch.Tensor [B, *D, 2]):
                The predicted panoptic segmentation, where preds[..., 0] is
                the class ID and preds[..., 1] the instance ID.
            target (torch.Tensor [B, *D, 2]):
                The target panoptic segmentation, where target[..., 0] is the
                class ID and target[..., 1] the instance ID.
            sequence_id (torch.Tensor [B] | Sequence[Hashable]):
                The sequence ID for each sample in the batch.
            frame_id (torch.Tensor [B] | Sequence[int]):
                The temporal frame index for each sample. Must be unique
                within a sequence across all batches and all ranks, and sort
                into the intended temporal order.
            mask (torch.Tensor [B, *D], optional):
                An optional mask to filter valid samples in the batch. Only
                samples where mask is True are considered for evaluation.
        """
        batch_size = preds.shape[0]

        # materialize sequence_id as a Python list of hashables, one per sample
        if torch.is_tensor(sequence_id):
            sequence_ids = sequence_id.tolist()
        else:
            sequence_ids = list(sequence_id)
        assert len(sequence_ids) == batch_size

        # materialize frame_id as a Python list of ints, one per sample
        if torch.is_tensor(frame_id):
            frame_ids = frame_id.tolist()
        else:
            frame_ids = list(frame_id)
        assert len(frame_ids) == batch_size

        # filter out ignored GT classes (treat as unlabeled/masked)
        is_ignored = torch.isin(target[..., 0], self.ignored_classes)
        mask = (mask & ~is_ignored) if mask is not None else ~is_ignored

        # flatten inputs to [N, 3] = (batch, class, instance)
        preds_flat = self._prepare_input(preds, mask)
        target_flat = self._prepare_input(target, mask)

        # sanitize (class, instance) pairs so non-instance pixels are uniformly marked as
        # NO_INSTANCE on both sides, and can then be filtered by a single check downstream
        preds_flat, target_flat = self._sanitize_instance_ids(preds_flat, target_flat)

        # target segments keyed by (batch, class, instance)
        target_area = _get_segments(target_flat)

        # pred segments keyed by (batch, pred_instance), class-agnostic: pred pixels across
        # different predicted classes for the same instance contribute to the same pred segment
        pred_area = _get_segments_pred_agnostic(preds_flat)

        # intersections keyed by ((batch, pred_inst), (batch, gt_class, gt_inst)); pred's own
        # predicted class is discarded, so a pred that spans multiple classes can still match a
        # GT of any (tracked) class via overall IoU
        intersections = _get_intersections_pred_agnostic(preds_flat, target_flat)

        # determine matches: for each target, the pred (if any) with IoU > 0.5
        target_matches: dict[_Color, int] = {}
        for (pred_key, target_key), inter_area in intersections.items():
            _bp, pi = pred_key
            _bt, ct, ti = target_key

            if NO_INSTANCE in (pi, ct, ti):
                continue

            union = pred_area[pred_key] + target_area[target_key] - inter_area
            assert union > 0
            iou = inter_area / union

            if iou > 0.5:
                target_matches[target_key] = pi

        # emit _GtMatch records for every valid GT segment in a tracked class
        new_gt: list[_GtMatch] = []
        for target_key, area in target_area.items():
            b, c, i = target_key
            if NO_INSTANCE in (c, i):
                continue
            if area <= self.min_instance_count:
                continue

            matched_pred = target_matches.get(target_key, NO_MATCH)
            new_gt.append(
                _GtMatch(
                    key=_GtMatchKey(
                        sequence=sequence_ids[b],
                        category=c,
                        gt_instance=i,
                        frame=frame_ids[b],
                    ),
                    pred_instance=matched_pred,
                )
            )

        # emit _PredOcc records for every pred segment (class-agnostic); one per (batch, pred)
        # observed in the frame, regardless of the pred's class. Small preds are excluded so
        # they do not inflate the FP denominator for matched preds in other frames
        new_pred: list[_PredOcc] = []
        for (b, pi), area in pred_area.items():
            if pi == NO_INSTANCE:
                continue
            if area <= self.min_instance_count:
                continue

            new_pred.append(
                _PredOcc(
                    key=_PredOccKey(
                        sequence=sequence_ids[b],
                        pred_instance=pi,
                        frame=frame_ids[b],
                    ),
                )
            )

        self.gt_matches += new_gt
        self.pred_occurrences += new_pred

    def _prepare_input(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, *_, c = x.shape
        assert c == 2

        x = x.detach().flatten(1, -2).to(dtype=torch.long)

        batch_id = torch.arange(batch_size, device=x.device)
        batch_id = batch_id.view(batch_size, 1, 1).expand(-1, x.shape[1], 1)
        x = torch.cat([batch_id, x], dim=-1)

        return x[mask.flatten(1, -1), :]

    def _sanitize_instance_ids(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mark non-instance pixels as NO_INSTANCE on both sides so they are
        uniformly filterable downstream.

        Inputs are the flattened `[N, 3] = (batch, class, instance)` tensors
        returned by `_prepare_input`. They are already freshly allocated, so
        in-place modifications do not leak back to the caller.

        Sanitization rules (assumes evaluated methods only assign instance IDs
        to instance classes, i.e. `track_classes`):
        - Pred pixels whose class is not in `track_classes` → (class,
          instance) = NO_INSTANCE. Additionally, a negative instance ID is
          treated as NO_INSTANCE as a safeguard.
        - Target pixels whose class is not in `track_classes` → (class,
          instance) = NO_INSTANCE.
        - Target pixels with class in `track_classes` but `instance < 0`:
          handled per `allow_invalid_instances` (assert, treat as
          non-instance, or drop entirely).
        """
        # sanitize predictions: class ∉ track_classes → NO_INSTANCE (also negative instance)
        preds_invalid = ~torch.isin(preds[..., 1], self.track_classes)
        preds_invalid = preds_invalid | (preds[..., 2] < 0)
        preds[preds_invalid, 1:] = NO_INSTANCE

        # sanitize target: class-based filter first
        is_instance = torch.isin(target[..., 1], self.track_classes)

        if self.allow_invalid_instances == "disallow":
            assert torch.all(
                target[is_instance, 2] >= 0
            ), "Target instance IDs for track classes must be non-negative."

        elif self.allow_invalid_instances == "restrict":
            is_instance = is_instance & (target[..., 2] >= 0)

        elif self.allow_invalid_instances == "ignore":
            is_invalid = is_instance & (target[..., 2] < 0)
            is_instance = is_instance[~is_invalid]
            preds = preds[~is_invalid, :]
            target = target[~is_invalid, :]

        target[~is_instance, 1:] = NO_INSTANCE

        return preds, target

    def compute(self) -> TrackingQualityResult:
        """
        Compute the tracking quality metric.

        Returns:
            TrackingQualityResult:
                Proxy object exposing TQ and its per-class / per-track
                breakdowns.
        """
        return TrackingQualityResult(
            gt_matches=list(self.gt_matches),
            pred_occurrences=list(self.pred_occurrences),
            classes={int(c) for c in self.track_classes.tolist()},
        )


class Result:
    """
    Result object for Panoptic Tracking (PAT) metric.
    """

    def __init__(
        self,
        panoptic: PanopticQualityResult,
        tracking: TrackingQualityResult,
        panoptic_classes: torch.Tensor,
        use_mod_pq: bool = False,
    ) -> None:
        """
        Initialize a new PAT Result instance.

        Args:
            panoptic (PanopticQualityResult):
                The result of the PanopticQuality metric.
            tracking (TrackingQualityResult):
                The result of the TrackingQuality metric.
            panoptic_classes (torch.Tensor):
                The class subset over which the PQ component of PAT is
                averaged by default.
            use_mod_pq (bool, optional):
                If True, use the modified PQ (PQ†) instead of PQ when
                computing PAT. Defaults to False.
        """
        self.panoptic = panoptic
        self.tracking = tracking
        self._panoptic_classes = panoptic_classes
        self._use_mod_pq = use_mod_pq

    @property
    def pq(self) -> float:
        """
        Mean PQ (or PQ†, if configured) over the default panoptic class
        subset.
        """
        return self.mean_pq()

    @property
    def tq(self) -> float:
        """
        Mean Tracking Quality (TQ) over the default tracked class subset.
        """
        return self.mean_tq()

    @property
    def pat(self) -> float:
        """
        The Panoptic Tracking (PAT) metric as harmonic mean of PQ and TQ over
        the default class subsets.
        """
        return self.mean_pat()

    def mean_pq(self, classes: Collection[int] | None = None) -> float:
        """
        Mean PQ (or PQ†, if configured) over a class subset.

        Args:
            classes (Collection[int], optional):
                The classes to aggregate over. If None, the panoptic class
                subset configured on the metric is used. Defaults to None.
        """
        cls = self._panoptic_classes if classes is None else classes

        if self._use_mod_pq:
            pqm, _ = self.panoptic.mean_mod_pq(cls)
            return pqm.item()

        pq, _, _ = self.panoptic.mean_pq(cls)
        return pq.item()

    def mean_tq(self, classes: Collection[int] | None = None) -> float:
        """
        Mean TQ over a class subset.

        Args:
            classes (Collection[int], optional):
                The classes to aggregate over. If None, the tracked class
                subset configured on the metric is used. Defaults to None.
        """
        return self.tracking.mean_tq(classes)

    def mean_pat(
        self,
        panoptic_classes: Collection[int] | None = None,
        track_classes: Collection[int] | None = None,
    ) -> float:
        """
        Panoptic Tracking (PAT) metric over configurable class subsets.

        Args:
            panoptic_classes (Collection[int], optional):
                The classes to average PQ over. If None, the configured
                panoptic class subset is used. Defaults to None.
            track_classes (Collection[int], optional):
                The classes to average TQ over. If None, the configured
                tracked class subset is used. Defaults to None.
        """
        return _harmonic_mean(
            self.mean_pq(panoptic_classes), self.mean_tq(track_classes)
        )

    @property
    def pq_per_class(self) -> dict[int, float]:
        """
        Per-class PQ (or PQ†, if configured) for the configured panoptic
        class subset.
        """
        pq_tensor = self.panoptic.mod_pq if self._use_mod_pq else self.panoptic.pq
        return {
            int(c): float(pq_tensor[int(c)]) for c in self._panoptic_classes.tolist()
        }

    @property
    def tq_per_class(self) -> dict[int, float]:
        """
        Per-class TQ for the configured tracked class subset. Classes with no
        GT tracks are omitted.
        """
        return self.tracking.tq_per_class

    @property
    def pat_per_class(self) -> dict[int, float]:
        """
        Per-class PAT, defined as the harmonic mean of per-class PQ and
        per-class TQ. Restricted to classes that appear in both the panoptic
        and tracked class subsets (and have GT tracks on the TQ side).
        """
        pq_pc = self.pq_per_class
        tq_pc = self.tq_per_class

        return {
            c: _harmonic_mean(pq_pc[c], tq_pc[c]) for c in pq_pc.keys() & tq_pc.keys()
        }


class PanopticTrackingMetric(Metric):
    """
    Panoptic Tracking (PAT) metric.

    Composes a PanopticQuality and a TrackingQuality instance and combines
    their results as PAT = 2 * PQ * TQ / (PQ + TQ).

    The caller supplies an explicit `frame_id` for each sample in `update()`.
    Frame IDs must be unique within each sequence (across all batches and all
    processes) and sort into temporal order. Given this, sequences may be
    split across processes freely.

    Refer to https://arxiv.org/abs/2109.03805 for more details.
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    _panoptic_classes: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        classes_thing: Collection[int],
        classes_stuff: Collection[int],
        track_classes: Collection[int] | None = None,
        panoptic_classes: Collection[int] | None = None,
        min_instance_count: int = 0,
        allow_unknown_preds: bool = True,
        allow_invalid_instances: Literal["disallow", "restrict", "ignore"] = "disallow",
        use_mod_pq: bool = False,
    ):
        """
        Initialize a new PanopticTrackingMetric instance.

        Any class in `range(num_classes)` not in `classes_thing ∪
        classes_stuff` is treated as an ignored/void class. PanopticQuality
        handles this internally via its `void_category` logic, and here we
        forward the same set of ignored classes to TrackingQuality so pred
        pixels over ignored GT regions do not inflate pred occurrence totals.

        Args:
            num_classes (int):
                Total number of classes in the dataset.
            classes_thing (Collection[int]):
                Class IDs of "thing" classes (countable objects).
            classes_stuff (Collection[int]):
                Class IDs of "stuff" classes (amorphous regions).
            track_classes (Collection[int], optional):
                Classes used for tracking quality. Defaults to classes_thing.
            panoptic_classes (Collection[int], optional):
                Classes over which PQ is averaged when computing PAT. Defaults
                to classes_thing ∪ classes_stuff (matches nuScenes' "include"
                convention).
            min_instance_count (int, optional):
                Minimum per-frame element count for a GT or predicted instance
                to be considered valid (passed to TrackingQuality). Defaults
                to 0.
            allow_unknown_preds (bool, optional):
                Forwarded to PanopticQuality. Defaults to True.
            allow_invalid_instances (str, optional):
                Forwarded to TrackingQuality. Defaults to "disallow".
            use_mod_pq (bool, optional):
                If True, PAT uses PQ† instead of PQ. Defaults to False.
        """
        super().__init__()

        self.num_classes = num_classes
        self._use_mod_pq = use_mod_pq

        self.pq = PanopticQuality(
            num_classes=num_classes,
            classes_thing=classes_thing,
            classes_stuff=classes_stuff,
            allow_unknown_preds=allow_unknown_preds,
        )

        ignored_classes = set(range(num_classes)) - (
            set(classes_thing) | set(classes_stuff)
        )

        self.tq = TrackingQuality(
            num_classes=num_classes,
            track_classes=track_classes if track_classes is not None else classes_thing,
            ignored_classes=ignored_classes,
            min_instance_count=min_instance_count,
            allow_invalid_instances=allow_invalid_instances,
        )

        if panoptic_classes is None:
            panoptic_classes = set(classes_thing) | set(classes_stuff)

        pc = torch.tensor(sorted(set(panoptic_classes)), dtype=torch.long)
        self.register_buffer("_panoptic_classes", pc, persistent=False)

    def reset(self) -> None:
        super().reset()
        self.pq.reset()
        self.tq.reset()

    # pylint: disable-next=arguments-differ
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sequence_id: torch.Tensor | Sequence[Hashable],
        frame_id: torch.Tensor | Sequence[int],
        mask: torch.Tensor | None = None,
    ) -> None:
        """
        Update the metric state with new predictions and targets.

        Args:
            preds (torch.Tensor [B, *D, 2]):
                Predicted panoptic segmentation.
            target (torch.Tensor [B, *D, 2]):
                Target panoptic segmentation.
            sequence_id (torch.Tensor [B] | Sequence[Hashable]):
                Sequence ID for each sample in the batch.
            frame_id (torch.Tensor [B] | Sequence[int]):
                Temporal frame index for each sample. Must be unique within a
                sequence across all batches and all ranks, and sort into the
                intended temporal order.
            mask (torch.Tensor [B, *D], optional):
                Optional mask to filter valid samples. Defaults to None.
        """
        self.pq.update(preds=preds, target=target, mask=mask)
        self.tq.update(
            preds=preds,
            target=target,
            sequence_id=sequence_id,
            frame_id=frame_id,
            mask=mask,
        )

    def compute(self) -> Result:
        """
        Compute the Panoptic Tracking (PAT) metric.
        """
        return Result(
            panoptic=self.pq.compute(),
            tracking=self.tq.compute(),
            panoptic_classes=self._panoptic_classes,
            use_mod_pq=self._use_mod_pq,
        )


def _get_segments(x: torch.Tensor) -> dict[_Color, int]:
    """
    Get unique segments and their area.

    Args:
        x (torch.Tensor [N, 3]):
            Input tensor where each row is (batch, class, instance).

    Returns:
        dict[_Color, int]:
            Mapping from (batch, class, instance) to area (element count).
    """
    segments, area = torch.unique(x, dim=0, return_counts=True)
    segments = _to_tuple(segments.cpu())
    area = area.cpu().tolist()
    return dict(zip(segments, area))


def _get_segments_pred_agnostic(x: torch.Tensor) -> dict[tuple[int, int], int]:
    """
    Get unique (batch, instance) pred segments and their area, collapsing across classes.

    Args:
        x (torch.Tensor [N, 3]):
            Input tensor where each row is (batch, class, instance). The class
            column is discarded.

    Returns:
        dict[tuple[int, int], int]:
            Mapping from (batch, instance) to area (element count).
    """
    # drop class column: keep (batch, instance)
    x2 = x[:, [0, 2]]
    segments, area = torch.unique(x2, dim=0, return_counts=True)
    segments_list = [tuple(s) for s in segments.cpu().tolist()]
    area_list = area.cpu().tolist()
    return dict(zip(segments_list, area_list))


def _get_intersections_pred_agnostic(
    preds: torch.Tensor, target: torch.Tensor
) -> dict[tuple[tuple[int, int], _Color], int]:
    """
    Get pairwise pred-target intersections with class-agnostic pred matching.

    Args:
        preds (torch.Tensor [N, 3]):
            Pred rows (batch, class, instance); the class column is discarded.
        target (torch.Tensor [N, 3]):
            Target rows aligned pixel-wise with preds.

    Returns:
        dict[tuple[tuple[int, int], _Color], int]:
            Intersection areas keyed by pairs ((batch, pred_inst),
            (batch, gt_class, gt_inst)). Pred and target share the batch index
            by pixel alignment.
    """
    # [N, 4] = (batch, pred_inst, gt_class, gt_inst)
    pairs = torch.cat([preds[:, [0, 2]], target[:, 1:]], dim=-1)

    segments, area = torch.unique(pairs, dim=0, return_counts=True)
    segments = segments.cpu().tolist()
    area = area.cpu().tolist()

    return {((b, pi), (b, cl, gi)): a for (b, pi, cl, gi), a in zip(segments, area)}


def _to_tuple(x: torch.Tensor) -> list[tuple]:
    def _rec(v):
        if isinstance(v, list):
            return tuple(_rec(e) for e in v)
        return v

    return [_rec(row) for row in x.tolist()]


def _harmonic_mean(a: float, b: float) -> float:
    if (a + b) == 0:
        return 0.0
    return (2 * a * b) / (a + b)
