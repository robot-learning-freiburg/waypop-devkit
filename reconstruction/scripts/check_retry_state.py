"""Reject incomplete or stale render layers before --skip-blender-render reuse."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from placement_checks import assigned_point, check_sampled_position  # noqa: E402

CAMERA_NAMES = ("front", "front_left", "front_right", "side_left", "side_right")
FRAME_TOKENS = (2, 4, 6)


def validate_saved_render_state(output_root: Path, objects_path: Path, segment: str) -> None:
    objects = json.loads(objects_path.read_text(encoding="utf-8"))
    segment_root = output_root / "outputs" / segment
    details_path = segment_root / "ood_details.json"
    if not details_path.is_file():
        raise FileNotFoundError(f"No saved object placements for --retry: {details_path}")
    details = json.loads(details_path.read_text(encoding="utf-8"))
    if not isinstance(details, dict) or len(details) != len(objects):
        raise ValueError("Saved object count differs from the input assignment")

    for ordinal, obj in enumerate(objects, 1):
        saved = details.get(str(ordinal))
        if not isinstance(saved, dict):
            raise ValueError(f"No saved state for OOD {ordinal}")
        if saved.get("file_identifier") != obj.get("fileIdentifier"):
            raise ValueError(f"Saved asset differs for OOD {ordinal}")
        check_sampled_position(
            saved.get("sampled_position"),
            assigned_point(objects, ordinal, required=True),
            ordinal,
        )
        for camera in CAMERA_NAMES:
            layer_root = segment_root / "Renders" / f"ood_{ordinal}" / "camera" / camera
            for token in FRAME_TOKENS:
                for prefix in ("", "no_shadow_"):
                    path = layer_root / f"{prefix}frame_00{token}.png"
                    if not path.is_file() or path.stat().st_size == 0:
                        raise FileNotFoundError(f"Missing saved Blender layer: {path}")
                    try:
                        with Image.open(path) as image:
                            image.verify()
                    except Exception as exc:
                        raise ValueError(f"Unreadable saved Blender layer: {path}") from exc
    print(f"Checked {len(objects)} saved placements and {len(objects) * len(CAMERA_NAMES) * len(FRAME_TOKENS) * 2} Blender layers.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--segment", required=True)
    args = parser.parse_args()
    validate_saved_render_state(args.output_root, args.objects, args.segment)


if __name__ == "__main__":
    main()
