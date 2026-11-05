"""Compact PQ histograms on NumPy/CPU and PyTorch/CUDA.

Both backends share the reference metric's IoU arithmetic and void policy.
Batch IDs prevent images from sharing a segment namespace.
"""

from __future__ import annotations

import time
import numpy as np

FIELDS = ("num_gt", "num_tp", "num_fp", "num_fn", "iou_sum_gt", "iou_sum_tp")
INSTANCE_BITS = 25
IMAGE_BITS = 30
IMAGE_MASK = (1 << IMAGE_BITS) - 1
PAIR_MASK = (1 << 32) - 1


def histogram(array):
    values, counts = np.unique(array, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def finish(pred_area, target_area, void_area, intersections, count, void):
    result = {
        name: np.zeros(count, dtype=np.int64 if name.startswith("num_") else np.float64)
        for name in FIELDS
    }
    pred_matched, target_matched = set(), set()
    for pair, area in intersections.items():
        pred, target = pair >> 32, pair & PAIR_MASK
        cls = (target & IMAGE_MASK) >> INSTANCE_BITS
        union = pred_area[pred] - void_area.get(pred, 0) + target_area[target] - area
        iou = np.float32(area) / np.float32(union)
        result["iou_sum_gt"][cls] += float(iou)
        if iou > 0.5:
            result["num_tp"][cls] += 1
            result["iou_sum_tp"][cls] += float(iou)
            pred_matched.add(pred)
            target_matched.add(target)
    for key in target_area:
        cls = (key & IMAGE_MASK) >> INSTANCE_BITS
        if cls != void:
            result["num_gt"][cls] += 1
            if key not in target_matched:
                result["num_fn"][cls] += 1
    for key, area in pred_area.items():
        cls = (key & IMAGE_MASK) >> INSTANCE_BITS
        if cls != void and key not in pred_matched:
            if np.float32(void_area.get(key, 0)) / np.float32(area) <= 0.5:
                result["num_fp"][cls] += 1
    return result


class CPUCounter:
    def __init__(self, count, things, stuff):
        self.count = count
        self.things = list(things)
        self.classes = sorted(set(things) | set(stuff))
        self.void = 1 + max(self.classes)
        assert self.void < 32

    def keys(self, value):
        if value.ndim == 3:
            value = value[None]
        batch = value.shape[0]
        assert 1 <= batch <= 4
        semantic = value[..., 0].reshape(batch, -1)
        instance = value[..., 1].reshape(batch, -1)
        semantic = np.where(np.isin(semantic, self.classes), semantic, self.void)
        instance = np.where(np.isin(semantic, self.things), instance, 0)
        if instance.min() < -1 or instance.max() >= (1 << 24):
            raise ValueError("Instance identity outside RGB24 contract")
        keys = (semantic.astype(np.uint64) << np.uint64(INSTANCE_BITS)) | (
            instance + 1
        ).astype(np.uint64)
        keys |= np.arange(batch, dtype=np.uint64)[:, None] << np.uint64(IMAGE_BITS)
        return keys.reshape(-1)

    def statistics(self, pred, target):
        pred_keys, target_keys = self.keys(pred), self.keys(target)
        pred_class = (pred_keys & np.uint64(IMAGE_MASK)) >> np.uint64(INSTANCE_BITS)
        target_class = (target_keys & np.uint64(IMAGE_MASK)) >> np.uint64(INSTANCE_BITS)
        valid = (pred_class == target_class) & (target_class != self.void)
        return finish(
            histogram(pred_keys),
            histogram(target_keys),
            histogram(pred_keys[target_class == self.void]),
            histogram((pred_keys[valid] << np.uint64(32)) | target_keys[valid]),
            self.count,
            self.void,
        )


class GPUCounter:
    def __init__(self, count, things, stuff):
        import torch

        self.torch = torch
        self.count = count
        self.void = 1 + max(set(things) | set(stuff))
        assert self.void < 32
        self.classes = torch.tensor(
            sorted(set(things) | set(stuff)), device="cuda", dtype=torch.long
        )
        self.things = torch.tensor(list(things), device="cuda", dtype=torch.long)
        self.last_timings = {}

    def keys(self, value):
        torch = self.torch
        batch = value.shape[0]
        assert 1 <= batch <= 4
        semantic = value[..., 0].reshape(batch, -1).to(dtype=torch.long)
        instance = value[..., 1].reshape(batch, -1).to(dtype=torch.long)
        semantic = torch.where(torch.isin(semantic, self.classes), semantic, self.void)
        instance = torch.where(torch.isin(semantic, self.things), instance, 0)
        keys = (semantic << INSTANCE_BITS) | (instance + 1)
        keys |= (
            torch.arange(batch, dtype=torch.long, device="cuda")[:, None] << IMAGE_BITS
        )
        return keys.reshape(-1)

    def statistics(self, pred, target):
        torch = self.torch
        start = time.perf_counter()
        pred_keys, target_keys = self.keys(pred), self.keys(target)
        pred_class = (pred_keys & IMAGE_MASK) >> INSTANCE_BITS
        target_class = (target_keys & IMAGE_MASK) >> INSTANCE_BITS
        valid = (pred_class == target_class) & (target_class != self.void)
        groups = []
        for keys in (
            pred_keys,
            target_keys,
            pred_keys[target_class == self.void],
            (pred_keys[valid] << 32) | target_keys[valid],
        ):
            keys, counts = torch.unique(keys, sorted=True, return_counts=True)
            groups.append(torch.stack((keys, counts), dim=1))
        lengths = [group.shape[0] for group in groups]
        # One small device-to-host transfer for all histogram tables.
        values = torch.cat(groups, dim=0).cpu().numpy()
        hist_end = time.perf_counter()
        dictionaries = []
        offset = 0
        for length in lengths:
            part = values[offset : offset + length]
            offset += length
            keys = part[:, 0].astype(np.uint64)
            order = np.argsort(keys)
            dictionaries.append(
                dict(zip(keys[order].tolist(), part[order, 1].tolist()))
            )
        result = finish(*dictionaries, self.count, self.void)
        self.last_timings = {
            "gpu_histograms_and_small_transfer_seconds": hist_end - start,
            "cpu_matching_seconds": time.perf_counter() - hist_end,
        }
        return result


def gpu_batch(arrays, void):
    """Pad both maps with void; padding contributes no scored segments."""
    import torch

    height = max(pair[0].shape[0] for pair in arrays)
    width = max(pair[0].shape[1] for pair in arrays)
    dtype = torch.int32 if arrays[0][0].dtype == np.int32 else torch.long
    pred = torch.empty((len(arrays), height, width, 2), dtype=dtype, device="cuda")
    target = torch.empty_like(pred)
    pred[..., 0] = void
    target[..., 0] = void
    pred[..., 1] = 0
    target[..., 1] = 0
    for index, (p, t) in enumerate(arrays):
        h, w = p.shape[:2]
        pred[index, :h, :w].copy_(torch.from_numpy(p))
        target[index, :h, :w].copy_(torch.from_numpy(t))
    return pred, target
