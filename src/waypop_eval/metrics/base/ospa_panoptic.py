"""
Optimal Subpattern Assignment (OSPA) Metric for Panoptic Segmentation and Tracking

References:
- Original OSPA metric: https://ieeexplore.ieee.org/abstract/document/4567674
- Adaptation for panoptic tracking (JRDB dataset): https://arxiv.org/abs/2404.01686
"""

import warnings
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Callable, Collection, Sequence

import numpy as np
import torch
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric
from torchmetrics.utilities.distributed import gather_all_tensors


@dataclass(unsafe_hash=True)
class _SegmentKey:
    frame: int
    sequence: Hashable
    category: int
    instance: int


@dataclass
class _Segment:
    key: _SegmentKey
    area: int


@dataclass(unsafe_hash=True)
class _IntersectionKey:
    pred: _SegmentKey
    target: _SegmentKey


@dataclass
class _Intersection:
    key: _IntersectionKey
    area: int


class SubsetResult:
    """
    OSPA result for a single class (or the mean across classes for ``total``),
    with a per-sequence breakdown.

    The OSPA score decomposes as OSPA^p = OSPA_LOC^p + OSPA_CARD^p, where:
    - OSPA_LOC reflects the quality of matched pairs (localization error)
    - OSPA_CARD reflects the mismatch in instance counts (cardinality error)

    With the default c=1, p=1 this simplifies to OSPA = OSPA_LOC + OSPA_CARD.

    TP, FP, and FN are defined at the track level (no IoU threshold):
    - TP = min(m, n): matched GT-prediction pairs
    - FP = max(0, n - m): unmatched predicted tracks
    - FN = max(0, m - n): unmatched GT tracks
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        ospa_per_seq: dict[Any, float],
        loc_per_seq: dict[Any, float],
        card_per_seq: dict[Any, float],
        tp_per_seq: dict[Any, int],
        fp_per_seq: dict[Any, int],
        fn_per_seq: dict[Any, int],
        gt_per_seq: dict[Any, int],
    ) -> None:
        self._ospa_per_seq = ospa_per_seq
        self._loc_per_seq = loc_per_seq
        self._card_per_seq = card_per_seq
        self._tp_per_seq = tp_per_seq
        self._fp_per_seq = fp_per_seq
        self._fn_per_seq = fn_per_seq
        self._gt_per_seq = gt_per_seq

        # cached scalar values (lazy)
        self._ospa: float | None = None
        self._loc: float | None = None
        self._card: float | None = None

    def _compute_scalars(self) -> None:
        if self._ospa is not None:
            return

        n = len(self._ospa_per_seq)
        if n == 0:
            self._ospa = 0.0
            self._loc = 0.0
            self._card = 0.0
        else:
            self._ospa = sum(self._ospa_per_seq.values()) / n
            self._loc = sum(self._loc_per_seq.values()) / n
            self._card = sum(self._card_per_seq.values()) / n

    @property
    def ospa(self) -> float:
        """Mean OSPA score over evaluated sequences. Returns 0.0 if none exist."""
        self._compute_scalars()
        return self._ospa

    @property
    def loc(self) -> float:
        """Mean localization (LOC) component over evaluated sequences."""
        self._compute_scalars()
        return self._loc

    @property
    def card(self) -> float:
        """Mean cardinality (CARD) component over evaluated sequences."""
        self._compute_scalars()
        return self._card

    @property
    def ospa_per_seq(self) -> dict[Any, float]:
        """Per-sequence OSPA. Sequences with no GT and no predictions are excluded."""
        return self._ospa_per_seq

    @property
    def loc_per_seq(self) -> dict[Any, float]:
        """Per-sequence LOC. Sequences with no GT and no predictions are excluded."""
        return self._loc_per_seq

    @property
    def card_per_seq(self) -> dict[Any, float]:
        """Per-sequence CARD. Sequences with no GT and no predictions are excluded."""
        return self._card_per_seq

    @property
    def tp(self) -> int:
        """Total matched GT–prediction track pairs."""
        return sum(self._tp_per_seq.values())

    @property
    def fp(self) -> int:
        """Total unmatched predicted tracks."""
        return sum(self._fp_per_seq.values())

    @property
    def fn(self) -> int:
        """Total unmatched GT tracks."""
        return sum(self._fn_per_seq.values())

    @property
    def gt(self) -> int:
        """Total GT instance/track count."""
        return sum(self._gt_per_seq.values())

    @property
    def tp_per_seq(self) -> dict[Any, int]:
        """Matched GT–prediction pairs per sequence."""
        return self._tp_per_seq

    @property
    def fp_per_seq(self) -> dict[Any, int]:
        """Unmatched predicted tracks per sequence."""
        return self._fp_per_seq

    @property
    def fn_per_seq(self) -> dict[Any, int]:
        """Unmatched GT tracks per sequence."""
        return self._fn_per_seq

    @property
    def gt_per_seq(self) -> dict[Any, int]:
        """GT instance/track count per sequence."""
        return self._gt_per_seq


class Result:
    """
    Proxy object for Panoptic OSPA metrics.

    Exposes two views of the computed scores:

    - ``total``: a :class:`SubsetResult` that averages over all non-ignored
      classes. For each sequence the value is the mean over classes with any
      GT or any prediction in that sequence; the overall scalar is then the
      mean across sequences (equal weight per sequence).

    - ``per_class``: a ``dict[int, SubsetResult]`` mapping each non-ignored
      class ID to its per-sequence SubsetResult. Each scalar is the mean of
      the per-sequence values for that class.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        per_class: dict[int, SubsetResult],
        total: SubsetResult,
    ) -> None:
        self.per_class = per_class
        self.total = total

    def mean_ospa(
        self, classes: Collection[int] | None = None
    ) -> tuple[float, float, float]:
        """
        Compute mean OSPA, LOC, and CARD over a selected subset of classes.

        For each sequence, the mean is taken over the selected classes that
        have at least one GT or one prediction in that sequence; the overall
        scalar is then the mean across sequences (equal weight per sequence).

        Args:
            classes (Collection[int] | None):
                Classes to include. If None, all non-ignored classes are
                considered. Class IDs that are ignored or absent from
                ``per_class`` are silently skipped.

        Returns:
            tuple[float, float, float]:
                Mean OSPA, LOC, CARD. All zero if no sequence contributes.
        """
        if classes is None:
            cls_list = list(self.per_class.keys())
        else:
            cls_list = [c for c in classes if c in self.per_class]

        # collect every sequence touched by any selected class
        all_seqs: set = set()
        for cls in cls_list:
            all_seqs.update(self.per_class[cls].ospa_per_seq.keys())

        seq_ospas: list[float] = []
        seq_locs: list[float] = []
        seq_cards: list[float] = []
        for seq in all_seqs:
            present = [c for c in cls_list if seq in self.per_class[c].ospa_per_seq]
            if not present:
                continue
            n = len(present)
            seq_ospas.append(
                sum(self.per_class[c].ospa_per_seq[seq] for c in present) / n
            )
            seq_locs.append(
                sum(self.per_class[c].loc_per_seq[seq] for c in present) / n
            )
            seq_cards.append(
                sum(self.per_class[c].card_per_seq[seq] for c in present) / n
            )

        if not seq_ospas:
            return 0.0, 0.0, 0.0

        n = len(seq_ospas)
        return (
            sum(seq_ospas) / n,
            sum(seq_locs) / n,
            sum(seq_cards) / n,
        )


class PanopticOspa(Metric):
    """
    Panoptic OSPA metric for panoptic segmentation and tracking.

    Computes the Optimal Subpattern Assignment (OSPA) metric adapted for
    panoptic segmentation, supporting both single-frame and tracking evaluation.

    In single-frame mode (per_frame=True), each frame is evaluated independently:
    segments within a frame are matched to minimize the total assignment cost.

    In tracking mode (per_frame=False), tracks are matched globally across all
    frames within each sequence. The cost between two tracks is the average
    IoU-based distance over all frames where at least one track exists
    (OSPA2 accumulation strategy).

    The OSPA score for a set of m GT tracks and n predicted tracks is:

        OSPA = ((optimal_cost + c^p * |m - n|) / max(m, n))^(1/p)

    where the cost matrix D[i, j] = min(avg_d[i, j], c)^p and
    avg_d[i, j] = 1 - iou_sum[i,j] / union_frame_count[i,j].

    Refer to https://arxiv.org/abs/2404.01686 for more details.
    """

    # pylint: disable=too-many-instance-attributes

    is_differentiable: bool = False
    full_state_update: bool = False

    segments_pred: list[_Segment]
    segments_target: list[_Segment]
    segments_intersect: list[_Intersection]

    def __init__(
        self,
        num_classes: int,
        ignored_classes: Collection[int] | None = None,
        per_frame: bool = False,
        c: float = 1.0,
        p: float = 1.0,
    ) -> None:
        """
        Initialize the metric.

        Args:
            num_classes (int):
                The number of classes in the dataset.
            ignored_classes (Collection[int] | None):
                Classes to ignore. GT segments of ignored classes are excluded
                (no FN), and GT pixels of ignored classes act as void for
                predicted segments (pred area in ignored regions is masked out).
                Defaults to None.
            per_frame (bool):
                If True, evaluate each frame independently (panoptic segmentation
                mode). If False, match tracks across frames within each sequence
                (panoptic tracking mode). Defaults to False.
            c (float):
                OSPA cutoff parameter. Distances are capped at c. Must be > 0.
                Defaults to 1.0.
            p (float):
                OSPA order parameter. Higher values penalize large errors more
                heavily. Defaults to 1.0 (linear).
        """
        super().__init__()

        assert c > 0.0, "Cutoff c must be positive."
        assert p > 0.0, "Order p must be positive."

        self.num_classes = num_classes
        self.ignored_classes = set(ignored_classes) if ignored_classes else set()
        self.per_frame = per_frame
        self.c = c
        self.p = p

        self.frame_count = 0

        ignored = torch.tensor(sorted(self.ignored_classes), dtype=torch.long)
        self.register_buffer("ignored_classes_tensor", ignored, persistent=False)

        self.add_state("segments_pred", default=[], dist_reduce_fx=None)
        self.add_state("segments_target", default=[], dist_reduce_fx=None)
        self.add_state("segments_intersect", default=[], dist_reduce_fx=None)

    def reset(self) -> None:
        super().reset()

        self.frame_count = 0

    def _get_local_sequence_index(
        self, sequence_id: torch.Tensor | Sequence[Hashable]
    ) -> tuple[torch.Tensor, dict[int, Hashable]]:
        """
        Convert arbitrary sequence IDs to batch-local integer indices.

        Args:
            sequence_id (torch.Tensor [B] | Sequence[Hashable]):
                Sequence identifier for each sample in the batch.

        Returns:
            tuple[torch.Tensor, dict[int, Hashable]]:
                - local_indices [B]: integer index per sample, unique within
                  this batch.
                - index_to_id: mapping from local index back to original
                  sequence ID.
        """
        if torch.is_tensor(sequence_id):
            sequence_id = sequence_id.tolist()

        # dict.fromkeys preserves order and deduplicates
        unique_ids = list(dict.fromkeys(sequence_id))
        id_to_index = {sid: i for i, sid in enumerate(unique_ids)}
        index_to_id = {i: sid for sid, i in id_to_index.items()}

        local = torch.tensor(
            [id_to_index[sid] for sid in sequence_id], dtype=torch.long
        )
        return local, index_to_id

    def _prepare_input(
        self,
        x: torch.Tensor,
        sequence_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flatten input and prepend (frame_id, seq_id) to each pixel entry.

        Args:
            x (torch.Tensor [B, *D, 2]):
                Input tensor with (class, instance) per pixel.
            sequence_index (torch.Tensor [B]):
                Batch-local sequence index per sample.
            mask (torch.Tensor [B, *D]):
                Boolean validity mask.

        Returns:
            torch.Tensor [N, 4]:
                Flattened valid pixels: (frame_id, seq_id, class, instance).
        """
        batch_size = x.shape[0]

        x = x.detach().flatten(1, -2).to(dtype=torch.long)

        # always assign a unique frame ID per batch sample
        frame_id = torch.arange(batch_size, device=x.device)
        frame_id = frame_id.view(batch_size, 1, 1).expand(-1, x.shape[1], 1)

        if self.per_frame:
            # in per_frame mode each frame is its own sequence
            seq = frame_id
        else:
            seq = sequence_index.to(device=x.device)
            seq = seq.view(batch_size, 1, 1).expand(-1, x.shape[1], 1)

        x = torch.cat((frame_id, seq, x), dim=-1)
        return x[mask.flatten(1, -1), :]

    def _get_segments(self, x: torch.Tensor) -> list[_Segment]:
        """
        Compute unique segments and their pixel areas from a [N, 4] tensor.

        Args:
            x (torch.Tensor [N, 4]):
                Pixels as (frame_id, seq_id, class, instance).

        Returns:
            list[_Segment]:
                One entry per unique (frame, seq, class, instance) combination.
        """
        if x.shape[0] == 0:
            return []

        segments, area = torch.unique(x, dim=0, return_counts=True)
        return [
            _Segment(_SegmentKey(*k), a)
            for k, a in zip(segments.tolist(), area.tolist())
        ]

    def _get_intersections(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> list[_Intersection]:
        """
        Compute pixel-level intersections between predicted and target segments.

        Each pixel contributes to exactly one (pred_segment, target_segment) pair.

        Args:
            preds (torch.Tensor [N, 4]):
                Predicted segments: (frame_id, seq_id, class, instance) per pixel.
            target (torch.Tensor [N, 4]):
                Target segments, pixel-aligned with preds.

        Returns:
            list[_Intersection]:
                One entry per unique (pred_segment, target_segment) pair, with
                area = number of pixels where that pair co-occurs.
        """
        if preds.shape[0] == 0:
            return []

        # preds and target are pixel-aligned; (frame_id, seq_id) must match
        assert torch.all(preds[:, :2] == target[:, :2])

        pairs = torch.stack((preds, target), dim=-2)  # [N, 2, 4]
        segments, area = torch.unique(pairs, dim=0, return_counts=True)

        result = []
        for (kp, kt), a in zip(segments.tolist(), area.tolist()):
            key = _IntersectionKey(_SegmentKey(*kp), _SegmentKey(*kt))
            result.append(_Intersection(key, a))

        return result

    # pylint: disable-next=arguments-differ
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sequence_id: torch.Tensor | Sequence[Hashable],
        mask: torch.Tensor | None = None,
    ) -> None:
        """
        Update the metric with new predictions and targets.

        Args:
            preds (torch.Tensor [B, *D, 2]):
                Predicted panoptic segmentation. preds[..., 0] is the class ID,
                preds[..., 1] the instance ID. All classes (stuff and thing) are
                included with their natural instance IDs (stuff: 0).
            target (torch.Tensor [B, *D, 2]):
                Target panoptic segmentation, same structure as preds.
            sequence_id (torch.Tensor [B] | Sequence[Hashable]):
                Sequence identifier for each sample in the batch. In per_frame
                mode this is only used for reporting; each sample is evaluated
                independently regardless of its sequence ID.
            mask (torch.Tensor [B, *D] | None):
                Optional boolean mask. Pixels where mask is False are excluded
                from all computations. Defaults to None (all pixels valid).
        """
        # pylint: disable=too-many-locals
        batch_size = preds.shape[0]

        local_seq_idx, seq_idx_to_id = self._get_local_sequence_index(sequence_id)

        # void mask: GT pixels of ignored classes are masked out for pred areas too
        if self.ignored_classes_tensor.numel() > 0:
            is_void = torch.isin(target[..., 0], self.ignored_classes_tensor)
        else:
            is_void = torch.zeros(
                target.shape[:-1], dtype=torch.bool, device=target.device
            )

        valid = (~is_void) if mask is None else (mask & ~is_void)

        preds_flat = self._prepare_input(preds, local_seq_idx, valid)
        target_flat = self._prepare_input(target, local_seq_idx, valid)

        pr_area = self._get_segments(preds_flat)
        gt_area = self._get_segments(target_flat)
        isect = self._get_intersections(preds_flat, target_flat)

        # only same-class intersections are used in per-class OSPA computation
        isect = [s for s in isect if s.key.pred.category == s.key.target.category]

        # remap to global frame IDs and original sequence IDs
        for seg in pr_area:
            seg.key.frame += self.frame_count
            if self.per_frame:
                seg.key.sequence = seg.key.frame
            else:
                seg.key.sequence = seq_idx_to_id[seg.key.sequence]

        for seg in gt_area:
            seg.key.frame += self.frame_count
            if self.per_frame:
                seg.key.sequence = seg.key.frame
            else:
                seg.key.sequence = seq_idx_to_id[seg.key.sequence]

        for seg in isect:
            seg.key.pred.frame += self.frame_count
            seg.key.target.frame += self.frame_count
            if self.per_frame:
                seg.key.pred.sequence = seg.key.pred.frame
                seg.key.target.sequence = seg.key.target.frame
            else:
                seg.key.pred.sequence = seq_idx_to_id[seg.key.pred.sequence]
                seg.key.target.sequence = seq_idx_to_id[seg.key.target.sequence]

        self.frame_count += batch_size

        # pylint: disable=no-member
        self.segments_pred += pr_area
        self.segments_target += gt_area
        self.segments_intersect += isect

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Any | None = None,
    ) -> None:
        # pylint: disable=too-many-locals

        state = (self.segments_pred, self.segments_target, self.segments_intersect)
        world_size = dist.get_world_size(process_group)
        synced = [([], [], []) for _ in range(world_size)]

        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            dist.all_gather_object(synced, state, group=process_group)

        # always reindex frame IDs to avoid collisions between processes;
        # OSPA always stores per-frame data (unlike AQ which uses frame=0 in
        # tracking mode)
        n = 0
        for pred, target, intersect in synced:
            max_frame_p = max((x.key.frame for x in pred), default=-1)
            max_frame_t = max((x.key.frame for x in target), default=-1)
            max_frame = max(max_frame_p, max_frame_t)

            for p in pred:
                p.key.frame += n
                if self.per_frame:
                    p.key.sequence = p.key.frame

            for t in target:
                t.key.frame += n
                if self.per_frame:
                    t.key.sequence = t.key.frame

            for i in intersect:
                i.key.pred.frame += n
                i.key.target.frame += n
                if self.per_frame:
                    i.key.pred.sequence = i.key.pred.frame
                    i.key.target.sequence = i.key.target.frame

            n = n + max_frame + 1

        preds, targets, intersects = zip(*synced)
        preds = [x for proc in preds for x in proc]
        targets = [x for proc in targets for x in proc]
        intersects = [x for proc in intersects for x in proc]

        self.segments_pred = preds
        self.segments_target = targets
        self.segments_intersect = intersects

    def _build_index(
        self,
    ) -> tuple[
        dict[tuple, dict[int, dict[int, float]]],
        dict[tuple, dict[int, dict[int, float]]],
        dict[tuple, dict[tuple, float]],
        list[Any],
    ]:
        """
        Group accumulated state into per-(seq, cls) lookup structures.

        Returns:
            gt_segments:
                (seq, cls) → {instance: {frame: pixel_area}}
            pred_segments:
                (seq, cls) → {instance: {frame: pixel_area}}
            iou_sums:
                (seq, cls) → {(gt_inst, pred_inst): accumulated_iou}
                Only pairs with at least one frame of pixel overlap are present.
                Frames where both tracks appear but don't overlap contribute 0 to
                the numerator while still growing the union-frame denominator via
                the per-instance frame sets in gt_segments / pred_segments.
            all_sequences:
                All sequence IDs seen, in insertion order.
        """
        # pylint: disable=no-member, too-many-locals

        gt_segments = defaultdict(lambda: defaultdict(dict))
        seen_seqs: dict[Any, None] = {}

        for seg in self.segments_target:
            k, area = seg.key, seg.area

            gt_segments[(k.sequence, k.category)][k.instance][k.frame] = area
            seen_seqs[k.sequence] = None

        pred_segments = defaultdict(lambda: defaultdict(dict))

        for seg in self.segments_pred:
            k, area = seg.key, seg.area

            pred_segments[(k.sequence, k.category)][k.instance][k.frame] = area
            seen_seqs[k.sequence] = None

        iou_sums = defaultdict(lambda: defaultdict(float))
        for seg in self.segments_intersect:
            pk = seg.key.pred
            tk = seg.key.target

            assert tk.sequence == pk.sequence
            assert tk.category == pk.category
            assert tk.frame == pk.frame

            key = (tk.sequence, tk.category)
            frame = tk.frame

            gt_area = gt_segments[key][tk.instance][frame]
            pred_area = pred_segments[key][pk.instance][frame]

            inter_area = seg.area
            union = gt_area + pred_area - inter_area
            iou_sums[key][(tk.instance, pk.instance)] += inter_area / union

        return gt_segments, pred_segments, iou_sums, list(seen_seqs)

    def _compute_seq_cls(
        self,
        gt_seg: dict[int, dict[int, float]],
        pred_seg: dict[int, dict[int, float]],
        iou_sums: dict[tuple, float],
    ) -> tuple[float, float, float, int, int, int]:
        """
        Compute OSPA components for one (seq, cls) pair.

        Caller guarantees both ``gt_seg`` and ``pred_seg`` are non-empty.

        Args:
            gt_seg: instance → {frame: pixel_area} for GT tracks.
            pred_seg: instance → {frame: pixel_area} for predicted tracks.
            iou_sums: (gt_inst, pred_inst) → accumulated per-frame IoU.

        Returns:
            (ospa, loc, card, tp, fp, fn)
        """
        # pylint: disable=too-many-locals

        c = self.c
        p = self.p
        c_p = c**p

        gt_ids = sorted(gt_seg.keys())
        pred_ids = sorted(pred_seg.keys())
        m = len(gt_ids)
        n = len(pred_ids)

        # collect every frame where any track of this (seq, cls) appears
        all_frames: set[int] = set()
        for inst in gt_ids:
            all_frames.update(gt_seg[inst].keys())
        for inst in pred_ids:
            all_frames.update(pred_seg[inst].keys())
        frame_to_idx = {f: i for i, f in enumerate(sorted(all_frames))}
        f_count = len(frame_to_idx)

        # binary presence: gt_mask[f, i] == 1 iff GT track i appears in frame f
        gt_mask = np.zeros((f_count, m), dtype=np.int64)
        for i, inst in enumerate(gt_ids):
            for f in gt_seg[inst]:
                gt_mask[frame_to_idx[f], i] = 1

        pred_mask = np.zeros((f_count, n), dtype=np.int64)
        for j, inst in enumerate(pred_ids):
            for f in pred_seg[inst]:
                pred_mask[frame_to_idx[f], j] = 1

        # |frames_i ∪ frames_j| via inclusion-exclusion
        cooccur = gt_mask.T @ pred_mask  # [m, n]
        size_gt = gt_mask.sum(axis=0)  # [m]
        size_pred = pred_mask.sum(axis=0)  # [n]
        union_count = size_gt[:, None] + size_pred[None, :] - cooccur  # [m, n]

        # accumulated per-frame IoU per (gt, pred) pair
        gt_id_to_idx = {gid: i for i, gid in enumerate(gt_ids)}
        pred_id_to_idx = {pid: j for j, pid in enumerate(pred_ids)}
        iou_total = np.zeros((m, n), dtype=np.float64)
        for (gt_inst, pred_inst), v in iou_sums.items():
            iou_total[gt_id_to_idx[gt_inst], pred_id_to_idx[pred_inst]] = v

        # cost matrix D[i, j] = min(avg_d[i, j], c)^p; union_count >= 1 always
        avg_d = 1.0 - iou_total / union_count
        distance = np.minimum(avg_d, c) ** p

        match_gt, match_pred = linear_sum_assignment(distance)
        optimal_cost = distance[match_gt, match_pred].sum()

        max_mn = max(m, n)
        loc_raw = optimal_cost / max_mn
        card_raw = c_p * abs(m - n) / max_mn
        ospa_raw = loc_raw + card_raw

        # apply p-th root: OSPA_LOC^p + OSPA_CARD^p = OSPA^p
        return (
            ospa_raw ** (1.0 / p),
            loc_raw ** (1.0 / p),
            card_raw ** (1.0 / p),
            min(m, n),
            max(0, n - m),
            max(0, m - n),
        )

    def compute(self) -> Result:
        """
        Compute the Panoptic OSPA metric from accumulated state.

        Returns:
            Result:
                Per-(sequence, class) OSPA values with ``total`` and
                ``per_class`` SubsetResults. (Sequence, class) pairs with no
                GT and no predictions are excluded from averages.
        """
        # pylint: disable=too-many-locals, too-many-statements

        gt_segments, pred_segments, iou_sums, all_sequences = self._build_index()

        valid_cls = [
            cls for cls in range(self.num_classes) if cls not in self.ignored_classes
        ]

        # per-(class, seq) accumulators
        per_class_ospa: dict[int, dict[Any, float]] = {cls: {} for cls in valid_cls}
        per_class_loc: dict[int, dict[Any, float]] = {cls: {} for cls in valid_cls}
        per_class_card: dict[int, dict[Any, float]] = {cls: {} for cls in valid_cls}
        per_class_tp: dict[int, dict[Any, int]] = {cls: {} for cls in valid_cls}
        per_class_fp: dict[int, dict[Any, int]] = {cls: {} for cls in valid_cls}
        per_class_fn: dict[int, dict[Any, int]] = {cls: {} for cls in valid_cls}
        per_class_gt: dict[int, dict[Any, int]] = {cls: {} for cls in valid_cls}

        # total (mean across valid classes within each sequence) accumulators
        total_ospa: dict[Any, float] = {}
        total_loc: dict[Any, float] = {}
        total_card: dict[Any, float] = {}
        total_tp: dict[Any, int] = {}
        total_fp: dict[Any, int] = {}
        total_fn: dict[Any, int] = {}
        total_gt: dict[Any, int] = {}

        c = self.c

        for seq in all_sequences:
            seq_ospas: list[float] = []
            seq_locs: list[float] = []
            seq_cards: list[float] = []
            seq_tp = seq_fp = seq_fn = seq_gt = 0

            for cls in valid_cls:
                sc = (seq, cls)
                gt_seg = gt_segments.get(sc, {})
                pred_seg = pred_segments.get(sc, {})
                m = len(gt_seg)
                n = len(pred_seg)

                if m == 0 and n == 0:
                    # nothing to evaluate
                    continue

                if n == 0:
                    # all GT unmatched: card_raw = c^p * m/m = c^p,
                    # so card = (c^p)^(1/p) = c regardless of p
                    ospa, loc, card = c, 0.0, c
                    tp, fp, fn = 0, 0, m
                elif m == 0:
                    # all preds are false positives: card_raw = c^p * n/n = c^p,
                    # so card = (c^p)^(1/p) = c regardless of p
                    ospa, loc, card = c, 0.0, c
                    tp, fp, fn = 0, n, 0
                else:
                    ospa, loc, card, tp, fp, fn = self._compute_seq_cls(
                        gt_seg,
                        pred_seg,
                        iou_sums.get(sc, {}),
                    )

                per_class_ospa[cls][seq] = ospa
                per_class_loc[cls][seq] = loc
                per_class_card[cls][seq] = card
                per_class_tp[cls][seq] = tp
                per_class_fp[cls][seq] = fp
                per_class_fn[cls][seq] = fn
                per_class_gt[cls][seq] = m

                seq_ospas.append(ospa)
                seq_locs.append(loc)
                seq_cards.append(card)
                seq_tp += tp
                seq_fp += fp
                seq_fn += fn
                seq_gt += m

            if seq_ospas:
                k = len(seq_ospas)
                total_ospa[seq] = sum(seq_ospas) / k
                total_loc[seq] = sum(seq_locs) / k
                total_card[seq] = sum(seq_cards) / k
                total_tp[seq] = seq_tp
                total_fp[seq] = seq_fp
                total_fn[seq] = seq_fn
                total_gt[seq] = seq_gt

        per_class = {
            cls: SubsetResult(
                per_class_ospa[cls],
                per_class_loc[cls],
                per_class_card[cls],
                per_class_tp[cls],
                per_class_fp[cls],
                per_class_fn[cls],
                per_class_gt[cls],
            )
            for cls in valid_cls
        }
        total = SubsetResult(
            total_ospa,
            total_loc,
            total_card,
            total_tp,
            total_fp,
            total_fn,
            total_gt,
        )

        return Result(per_class=per_class, total=total)
