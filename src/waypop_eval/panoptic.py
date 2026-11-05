"""Lossless panoptic image decoding and manifest-relative prediction keys."""

from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


def relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Expected a relative dataset path: {value!r}")
    return Path(*path.parts)


def load_panoptic_ids(path: Path, *, require_rgb: bool = False) -> np.ndarray:
    """Read RGB24 IDs, or the single-channel integers used by older releases."""
    with Image.open(path) as image:
        pixels = np.array(image)
    if pixels.ndim == 3 and pixels.shape[-1] == 3 and pixels.dtype == np.uint8:
        rgb = pixels.astype(np.int64)
        return rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)
    if not require_rgb and pixels.ndim == 2 and np.issubdtype(pixels.dtype, np.integer):
        if np.all((pixels >= 0) & (pixels < 2**24)):
            return pixels.astype(np.int64)
    expected = "RGB24" if require_rgb else "RGB24 or single-channel integer"
    raise ValueError(
        f"Expected {expected} panoptic labels at {path}; got {pixels.shape} {pixels.dtype}"
    )


def ids_to_rgb(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids)
    if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Panoptic IDs must be a two-dimensional integer array")
    if not np.all((ids >= 0) & (ids < 2**24)):
        raise ValueError("Panoptic IDs must fit in unsigned RGB24")
    return np.stack((ids & 255, (ids >> 8) & 255, (ids >> 16) & 255), axis=-1).astype(
        np.uint8
    )
