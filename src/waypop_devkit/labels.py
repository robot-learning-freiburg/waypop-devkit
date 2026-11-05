"""RGB24 panoptic decoding and stable, transparent visualization colors."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_ids(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("Expected an H x W x 3 uint8 RGB array")
    values = array.astype(np.uint32)
    return values[:, :, 0] | (values[:, :, 1] << 8) | (values[:, :, 2] << 16)


def ids_to_rgb(ids: np.ndarray) -> np.ndarray:
    values = np.asarray(ids)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Expected a two-dimensional integer ID map")
    if np.any(values < 0) or np.any(values >= 1 << 24):
        raise ValueError("Segment IDs must be in [0, 2^24)")
    values = values.astype(np.uint32)
    return np.stack((values & 255, (values >> 8) & 255, (values >> 16) & 255), axis=-1).astype(np.uint8)


def load_ids(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError(f"Expected an RGB PNG: {path}")
        return rgb_to_ids(np.asarray(image))


def decode_ground_truth(ids: np.ndarray, divisor: int) -> tuple[np.ndarray, np.ndarray]:
    if divisor <= 0:
        raise ValueError("panoptic_label_divisor must be positive")
    return ids // divisor, ids % divisor


def stable_instance_color(sequence_id: str, instance_id: int) -> tuple[int, int, int]:
    digest = hashlib.blake2b(f"{sequence_id}:{instance_id}".encode(), digest_size=3).digest()
    return tuple(64 + (value % 160) for value in digest)


def ground_truth_mask(ids: np.ndarray, divisor: int, sequence_id: str, categories: dict[int, dict]) -> np.ndarray:
    """Return a transparent RGBA overlay; the browser applies opacity 0.5."""
    result = np.zeros((*ids.shape, 4), dtype=np.uint8)
    for raw_id in np.unique(ids):
        if raw_id == 0:
            continue
        category_id, instance_id = divmod(int(raw_id), divisor)
        category = categories.get(category_id)
        if category and (category.get("isthing") or category_id == 29):
            color = stable_instance_color(sequence_id, instance_id)
        elif category:
            color = tuple(category["color"])
        else:
            color = (140, 140, 140)
        result[ids == raw_id] = (*color, 255)
    return result


def prediction_mask(ids: np.ndarray, sequence_id: str, segments_info: list[dict], categories: dict[int, dict]) -> np.ndarray:
    segments = {int(item["id"]): int(item["category_id"]) for item in segments_info}
    result = np.zeros((*ids.shape, 4), dtype=np.uint8)
    for segment_id in np.unique(ids):
        if segment_id == 0:
            continue
        category_id = segments.get(int(segment_id))
        if category_id is None:
            raise ValueError(f"Prediction segment {segment_id} is missing from segments_info")
        category = categories.get(category_id)
        if category and (category.get("isthing") or category_id == 29):
            color = stable_instance_color(sequence_id, int(segment_id))
        elif category:
            color = tuple(category["color"])
        else:
            color = (140, 140, 140)
        result[ids == segment_id] = (*color, 255)
    return result
