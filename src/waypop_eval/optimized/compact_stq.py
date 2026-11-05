"""Unweighted FRONT STQ with compact integer GPU histograms.

Sequence-camera identities and the original class-agnostic association policy
are retained. The official evaluator performs the final STQ/AQ/mIoU reduction.
"""

from collections import defaultdict
import time
import numpy as np
import torch

INSTANCE_BITS = 25
INSTANCE_MASK = (1 << INSTANCE_BITS) - 1
PAIR_MASK = (1 << 32) - 1


class CompactSTQ:
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
        self.confusion = np.zeros((self.count, self.count), dtype=np.int64)
        self.predicted = defaultdict(int)
        self.target = defaultdict(int)
        self.intersections = defaultdict(int)
        self.images = 0
        self.last_timings = {}

    @staticmethod
    def histogram(keys):
        return torch.unique(keys, sorted=True, return_counts=True)

    def update(self, pred, target, sequence, mask=None):
        start = time.perf_counter()
        pc = pred[..., 0].reshape(-1).long()
        pi = pred[..., 1].reshape(-1).long()
        tc = target[..., 0].reshape(-1).long()
        ti = target[..., 1].reshape(-1).long()
        semantic_mask = (
            torch.ones_like(tc, dtype=torch.bool)
            if mask is None
            else mask.reshape(-1).bool()
        )
        confusion = torch.bincount(
            (tc * self.count + pc)[semantic_mask], minlength=self.count**2
        )
        allowed = semantic_mask & ~torch.isin(tc, self.ignored)
        pred_valid = allowed & (pi >= 0)
        target_valid = allowed & torch.isin(tc, self.things) & (ti >= 0)
        pair_valid = target_valid & (pi >= 0)
        pk = (pc << INSTANCE_BITS) | (pi + 1)
        tk = (tc << INSTANCE_BITS) | (ti + 1)
        groups = [
            self.histogram(pk[pred_valid]),
            self.histogram(tk[target_valid]),
            self.histogram((pk[pair_valid] << 32) | tk[pair_valid]),
        ]
        lengths = [keys.shape[0] for keys, _ in groups]
        size = sum(lengths)
        keys = torch.cat([keys for keys, _ in groups])
        counts = torch.cat([*[counts for _, counts in groups], confusion])
        payload = (
            torch.cat((keys.view(torch.uint8), counts.view(torch.uint8))).cpu().numpy()
        )
        key_values = np.frombuffer(payload, dtype=np.int64, count=size)
        area_values = np.frombuffer(
            payload, dtype=np.int64, count=size + self.count**2, offset=size * 8
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

        metric = METRIC_REGISTRY["stq"](self.meta)
        metric.base.sq.confusion.copy_(torch.from_numpy(self.confusion))
        aq = metric.base.aq
        aq.segments_pred = [
            _Segment(_SegmentKey(0, seq, cls, instance), area)
            for (seq, cls, instance), area in self.predicted.items()
        ]
        aq.segments_target = [
            _Segment(_SegmentKey(0, seq, cls, instance), area)
            for (seq, cls, instance), area in self.target.items()
        ]
        aq.segments_intersect = [
            _Intersection(
                _IntersectionKey(
                    _SegmentKey(0, seq, pc, pi), _SegmentKey(0, seq, tc, ti)
                ),
                area,
            )
            for (seq, pc, pi, tc, ti), area in self.intersections.items()
        ]
        aq.frame_count = self.images
        metric._update_count = metric.base._update_count = (
            metric.base.sq._update_count
        ) = aq._update_count = self.images
        return metric

    def compute(self):
        return self.metric().compute()
