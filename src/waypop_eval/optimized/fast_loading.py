"""Decode prediction and reference PNGs into compact evaluation maps."""

from __future__ import annotations

import numpy as np
from PIL import Image


def read_ids(path, require_rgb=False):
    with Image.open(path) as image:
        pixels = np.array(image)
    if pixels.ndim == 3 and pixels.shape[-1] == 3 and pixels.dtype == np.uint8:
        ids = pixels[..., 0].astype(np.uint32)
        ids |= pixels[..., 1].astype(np.uint32) << np.uint32(8)
        ids |= pixels[..., 2].astype(np.uint32) << np.uint32(16)
        return ids
    if not require_rgb and pixels.ndim == 2 and np.issubdtype(pixels.dtype, np.integer):
        if np.all((pixels >= 0) & (pixels < (1 << 24))):
            return pixels.astype(np.uint32)
    raise ValueError("Expected RGB24 IDs or an allowed integer ground-truth map")


def prediction(sample):
    ids = read_ids(sample.panoptic_path, require_rgb=True)
    segments = sample.meta.get("segments_info", {})
    maximum = max([0, *segments])
    if (
        maximum >= (1 << 24)
        or maximum < 0
        or any(identity < 0 for identity in segments)
    ):
        raise ValueError("Segment metadata IDs outside RGB24 contract")
    # At most 32 MiB even for sparse IDs near 2^24; sentinel detects unlisted IDs.
    lookup = np.full(maximum + 2, -1, dtype=np.int16)
    lookup[0] = sample.void_idx
    for identity, category in segments.items():
        lookup[identity] = sample.remap.get(category, sample.void_idx)
    semantic = np.take(lookup, ids, mode="clip")
    if np.any(semantic < 0):
        raise ValueError("Prediction IDs missing from segments_info")
    output = np.empty((*ids.shape, 2), dtype=np.int32)
    output[..., 0] = semantic
    output[..., 1] = ids
    return output


def target(sample):
    ids = read_ids(sample.panoptic_path)
    if ids.shape != (sample.height, sample.width):
        raise ValueError("Ground-truth dimensions differ from sample manifest")
    if sample.label_divisor <= 0:
        raise ValueError("Invalid ground-truth divisor")
    lookup = np.full(
        max(sample.semantic_remap, default=0) + 2, sample.void_idx, dtype=np.int16
    )
    for category, index in sample.semantic_remap.items():
        lookup[category] = index
    output = np.empty((*ids.shape, 2), dtype=np.int32)
    output[..., 0] = np.take(lookup, ids // sample.label_divisor, mode="clip")
    output[..., 1] = (ids % sample.label_divisor).astype(np.int32) - 1
    return output
