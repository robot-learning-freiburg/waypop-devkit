"""Generate a complete, all-void prediction bundle to exercise the format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from waypop_devkit.data import WaypopRelease
from waypop_devkit.submissions import TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()

    release = WaypopRelease(args.data_root)
    for split in (("val",) if args.split == "val" else ("v1", "xl")):
        annotations = []
        for sample in release.samples(split):
            if args.task == "fv-ospt" and sample.camera != "FRONT":
                continue
            destination = args.output_root / split / "predictions" / sample.prediction_file_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            width, height = sample.size
            Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(destination)
            annotations.append({"file_name": sample.prediction_file_name, "segments_info": []})
        manifest = args.output_root / split / "predictions.json"
        manifest.write_text(json.dumps({"annotations": annotations}, indent=2) + "\n", encoding="utf-8")
        print(split, len(annotations), "all-void predictions")


if __name__ == "__main__":
    main()
