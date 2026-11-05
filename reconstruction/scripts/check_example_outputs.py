"""Validate the completed three-frame, five-camera reconstruction example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from check_retry_state import CAMERA_NAMES, FRAME_TOKENS, validate_saved_render_state

SUFFIXES = (".jpg", "_mask.png", "_sem_label.png", "_inst_label.png",
            "_panoptic_label.png", "_panoptic_id.png")


def _image(path: Path) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing output image: {path}")
    try:
        with Image.open(path) as image:
            return np.asarray(image).copy()
    except Exception as exc:
        raise ValueError(f"Unreadable output image: {path}") from exc


def validate_example_outputs(output_root: Path, objects_path: Path, segment: str) -> None:
    validate_saved_render_state(output_root, objects_path, segment)
    segment_root = output_root / "outputs" / segment
    progress_path = segment_root / "run_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("stage") != "go" or progress.get("status") != "completed":
        raise ValueError(f"Renderer did not complete: {progress_path}")

    merged_root = (output_root / "images" / segment /
                   "merged_directly_shadow1_resized" / "camera")
    visible_ids = set()
    checked_frame_count = 0
    for camera in CAMERA_NAMES:
        for token in FRAME_TOKENS:
            stem = f"final_frame_00{token}"
            images = {
                suffix: _image(merged_root / camera / f"{stem}{suffix}")
                for suffix in SUFFIXES
            }
            shapes = {array.shape[:2] for array in images.values()}
            if len(shapes) != 1:
                raise ValueError(f"Mismatched image dimensions for {camera} token {token}")
            mask = images["_mask.png"] > 0
            semantic = images["_sem_label.png"]
            instance = images["_inst_label.png"]
            panoptic = images["_panoptic_label.png"].astype(np.uint32)
            rgb_id = images["_panoptic_id.png"].astype(np.uint32)
            decoded = rgb_id[..., 0] | (rgb_id[..., 1] << 8) | (rgb_id[..., 2] << 16)
            if not np.array_equal(mask, semantic == 29):
                raise ValueError(f"Unknown mask/semantic mismatch for {camera} token {token}")
            if not np.array_equal(panoptic, decoded):
                raise ValueError(f"Panoptic integer/RGB mismatch for {camera} token {token}")
            if not np.array_equal(panoptic[mask], 29000 + instance[mask].astype(np.uint32)):
                raise ValueError(f"Unknown instance/panoptic mismatch for {camera} token {token}")
            ids = {int(value) for value in np.unique(instance[mask])}
            if not ids.issubset(set(range(1, 9))):
                raise ValueError(f"Unexpected unknown object ID in {camera} token {token}: {ids}")
            visible_ids.update(ids)
            checked_frame_count += 1

    if checked_frame_count != 15 or visible_ids != set(range(1, 9)):
        raise ValueError(f"Expected 15 camera frames with all eight objects visible; got {checked_frame_count}, IDs {visible_ids}")
    print(f"Checked {checked_frame_count} merged camera frames and label sets; all eight object IDs are visible.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--segment", required=True)
    args = parser.parse_args()
    validate_example_outputs(args.output_root, args.objects, args.segment)


if __name__ == "__main__":
    main()
