"""Weighted STQ with compact GPU histograms and sequence-preserving state.

The unchanged evaluator owns the final semantic/association reduction. In
particular, predicted classes are NOT used to restrict association matches.
"""

from collections import defaultdict
import time

import numpy as np
import torch

INSTANCE_BITS = 25
INSTANCE_MASK = (1 << INSTANCE_BITS) - 1
PAIR_MASK = (1 << 32) - 1


class CompactWSTQ:
    def __init__(self, meta):
        self.meta = meta
        self.count = len(meta.order)
        assert self.count < 32
        self.things = torch.tensor(
            sorted(meta.eval_order[c] for c in meta.things - meta.ignored),
            dtype=torch.long,
            device="cuda",
        )
        self.ignored = torch.tensor(
            sorted(meta.eval_order[c] for c in meta.ignored),
            dtype=torch.long,
            device="cuda",
        )
        self.confusion = np.zeros((self.count, self.count), dtype=np.float64)
        self.predicted = defaultdict(float)
        self.target = defaultdict(float)
        self.intersections = defaultdict(float)
        self.images = 0
        self.last_timings = {}

    @staticmethod
    def histogram(keys, weights):
        unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        areas = torch.zeros(unique.shape[0], device="cuda", dtype=torch.float64)
        areas.scatter_add_(0, inverse, weights)
        return unique, areas

    def update(self, pred, target, coverage, tracked, sequence, mask=None):
        start = time.perf_counter()
        pc = pred[..., 0].reshape(-1).long()
        pi = pred[..., 1].reshape(-1).long()
        tc = target[..., 0].reshape(-1).long()
        ti = target[..., 1].reshape(-1).long()
        tracked = tracked.reshape(-1).bool()
        # Match the reference metric's float32 reciprocal and float64 sums.
        weights = (1.0 / torch.clamp(coverage.reshape(-1).float(), min=1e-8)).double()
        semantic_mask = (
            torch.ones_like(tc, dtype=torch.bool)
            if mask is None
            else mask.reshape(-1).bool()
        )
        confusion = torch.bincount(
            (tc * self.count + pc)[semantic_mask],
            weights=weights[semantic_mask],
            minlength=self.count**2,
        )
        allowed = semantic_mask & ~torch.isin(tc, self.ignored)
        pred_valid = allowed & (pi >= 0)
        target_valid = allowed & torch.isin(tc, self.things) & (ti >= 0) & tracked
        pair_valid = target_valid & (pi >= 0)
        # All predicted classes, including void/stuff, retain their submitted
        # instance IDs. The reference's tracked mask restricts only GT tracks.
        pk = (pc << INSTANCE_BITS) | (pi + 1)
        tk = (tc << INSTANCE_BITS) | (ti + 1)
        groups = [
            self.histogram(pk[pred_valid], weights[pred_valid]),
            self.histogram(tk[target_valid], weights[target_valid]),
            self.histogram(
                (pk[pair_valid] << 32) | tk[pair_valid], weights[pair_valid]
            ),
        ]
        lengths = [keys.shape[0] for keys, _ in groups]
        size = sum(lengths)
        keys = torch.cat([keys for keys, _ in groups])
        areas = torch.cat([*[area for _, area in groups], confusion])
        # Transfer bytes rather than casting int64 pair IDs to float64: packed
        # pair IDs can exceed the exact integer range of a floating-point key.
        payload = (
            torch.cat((keys.view(torch.uint8), areas.view(torch.uint8))).cpu().numpy()
        )
        key_values = np.frombuffer(payload, dtype=np.int64, count=size)
        area_values = np.frombuffer(
            payload, dtype=np.float64, count=size + self.count**2, offset=size * 8
        )
        histogram_end = time.perf_counter()
        self.confusion += area_values[size:].reshape(self.count, self.count)
        offset = 0
        for group, length in enumerate(lengths):
            for key, area in zip(
                key_values[offset : offset + length].tolist(),
                area_values[offset : offset + length].tolist(),
            ):
                if group < 2:
                    identity = (
                        sequence,
                        key >> INSTANCE_BITS,
                        (key & INSTANCE_MASK) - 1,
                    )
                    (self.predicted if group == 0 else self.target)[identity] += area
                else:
                    p, t = key >> 32, key & PAIR_MASK
                    identity = (
                        sequence,
                        p >> INSTANCE_BITS,
                        (p & INSTANCE_MASK) - 1,
                        t >> INSTANCE_BITS,
                        (t & INSTANCE_MASK) - 1,
                    )
                    self.intersections[identity] += area
            offset += length
        self.images += 1
        self.last_timings = {
            "gpu_histograms_and_transfer_seconds": histogram_end - start,
            "cpu_state_merge_seconds": time.perf_counter() - histogram_end,
        }

    def metric(self):
        from waypop_eval.benchmark import METRIC_REGISTRY
        from waypop_eval.metrics.base.association_quality import (
            _Segment,
            _SegmentKey,
            _Intersection,
            _IntersectionKey,
        )

        metric = METRIC_REGISTRY["wstq"](self.meta)
        metric.base.sq.confusion.copy_(torch.from_numpy(self.confusion))
        association = metric.base.aq
        association.segments_pred = [
            _Segment(_SegmentKey(0, seq, cls, instance), area)
            for (seq, cls, instance), area in self.predicted.items()
        ]
        association.segments_target = [
            _Segment(_SegmentKey(0, seq, cls, instance), area)
            for (seq, cls, instance), area in self.target.items()
        ]
        association.segments_intersect = [
            _Intersection(
                _IntersectionKey(
                    _SegmentKey(0, seq, pc, pi), _SegmentKey(0, seq, tc, ti)
                ),
                area,
            )
            for (seq, pc, pi, tc, ti), area in self.intersections.items()
        ]
        association.frame_count = self.images
        metric._update_count = metric.base._update_count = (
            metric.base.sq._update_count
        ) = association._update_count = self.images
        return metric

    def compute(self):
        return self.metric().compute()
