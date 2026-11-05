"""Read one public validation sample and decode its labels and XYZ point cloud."""

from __future__ import annotations

import argparse

import numpy as np

from waypop_devkit.data import WaypopRelease
from waypop_devkit.labels import decode_ground_truth, load_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    sample = WaypopRelease(args.data_root).samples("val")[0]
    labels = load_ids(sample.path("panoptic_path"))
    semantic, instance = decode_ground_truth(labels, int(sample.row["panoptic_label_divisor"]))
    with np.load(sample.path("pointcloud_path"), allow_pickle=False) as archive:
        xyz = archive["xyz"]
    print("sample:", sample.sample_id)
    print("image:", sample.path("image_path"))
    print("semantic/instance shape:", semantic.shape, instance.shape)
    print("vehicle-frame xyz shape:", xyz.shape)


if __name__ == "__main__":
    main()
