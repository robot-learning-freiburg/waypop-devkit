"""Manifest-driven access to the public WayPOP release."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator

SPLITS = ("val", "v1", "xl")
CAMERAS = ("SIDE_LEFT", "FRONT_LEFT", "FRONT", "FRONT_RIGHT", "SIDE_RIGHT")


def relative_path(value: str) -> Path:
    """Reject absolute, parent-traversing, and platform-dependent manifest paths."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid release-relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in value.split("/")):
        raise ValueError(f"Invalid release-relative path: {value!r}")
    if not path.parts or path.parts[0].endswith(":"):
        raise ValueError(f"Invalid release-relative path: {value!r}")
    return Path(*path.parts)


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{number}")
            yield value


@dataclass(frozen=True)
class Sample:
    root: Path
    split: str
    index: int
    row: dict

    @property
    def sample_id(self) -> str:
        return self.row["sample_id"]

    @property
    def sequence_id(self) -> str:
        return self.row["sequence_id"]

    @property
    def camera(self) -> str:
        return self.row["camera_name"]

    @property
    def position(self) -> int:
        return self.row["sequence_position"]

    @property
    def size(self) -> tuple[int, int]:
        return int(self.row["width"]), int(self.row["height"])

    @property
    def prediction_file_name(self) -> str:
        image = PurePosixPath(self.row["image_path"])
        return str(PurePosixPath(*image.parts[1:]).with_suffix(".png"))

    def path(self, field: str) -> Path:
        return self.root / self.split / relative_path(self.row[field])


class WaypopRelease:
    """Index released samples without requiring withheld test annotations."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()
        with (self.root / "release.json").open(encoding="utf-8") as handle:
            self.release = json.load(handle)
        with (self.root / "categories.json").open(encoding="utf-8") as handle:
            self.categories = json.load(handle)
        if self.release.get("dataset") != "WayPOP":
            raise ValueError("release.json does not describe WayPOP")
        self._sequences = list(read_jsonl(self.root / "sequences.jsonl"))
        membership = {row["sequence_id"]: row for row in self._sequences}
        if len(membership) != len(self._sequences):
            raise ValueError("Duplicate sequence_id in sequences.jsonl")
        self._membership = membership
        self._samples: dict[str, list[Sample]] = {}

    @property
    def category_map(self) -> dict[int, dict]:
        return {int(row["id"]): row for row in self.categories["categories"]}

    def samples(self, split: str) -> list[Sample]:
        if split not in SPLITS:
            raise ValueError(f"Expected one of {SPLITS}, got {split!r}")
        if split not in self._samples:
            path = self.root / split / "meta/samples.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows = []
            seen = set()
            for index, row in enumerate(read_jsonl(path)):
                identity = row["sample_id"]
                sequence = row["sequence_id"]
                if identity in seen:
                    raise ValueError(f"Duplicate sample_id in {path}: {identity}")
                if sequence not in self._membership or self._membership[sequence]["split"] != split:
                    raise ValueError(f"Sample {identity} disagrees with sequences.jsonl")
                if row["split"] != split or row["camera_name"] not in CAMERAS:
                    raise ValueError(f"Invalid split or camera for {identity}")
                if int(row["width"]) <= 0 or int(row["height"]) <= 0:
                    raise ValueError(f"Invalid image dimensions for {identity}")
                image = relative_path(row["image_path"])
                if image.parts[0] != "images" or image.suffix.lower() != ".png":
                    raise ValueError(f"Invalid image path for {identity}")
                expected_identity = str(PurePosixPath(*image.parts[1:]).with_suffix(""))
                if identity != expected_identity:
                    raise ValueError(f"sample_id and image_path disagree for {identity}")
                relative_path(row["pointcloud_path"])
                if split == "val":
                    relative_path(row["panoptic_path"])
                    relative_path(row["wstq_meta_path"])
                    if int(row["panoptic_label_divisor"]) <= 0:
                        raise ValueError(f"Invalid label divisor for {identity}")
                seen.add(identity)
                rows.append(Sample(self.root, split, index, row))
            if not rows:
                raise ValueError(f"No samples in {path}")
            self._samples[split] = rows
        return self._samples[split]

    def sequence_ids(self, split: str) -> list[str]:
        return sorted({sample.sequence_id for sample in self.samples(split)})

    def frames(self, split: str, sequence_id: str) -> list[dict[str, Sample]]:
        grouped: dict[int, dict[str, Sample]] = {}
        for sample in self.samples(split):
            if sample.sequence_id != sequence_id:
                continue
            cameras = grouped.setdefault(sample.position, {})
            if sample.camera in cameras:
                raise ValueError(f"Duplicate {sequence_id} position {sample.position} camera {sample.camera}")
            cameras[sample.camera] = sample
        if not grouped:
            raise ValueError(f"Unknown sequence in {split}: {sequence_id}")
        frames = [grouped[position] for position in sorted(grouped)]
        for position, cameras in zip(sorted(grouped), frames):
            if set(cameras) != set(CAMERAS):
                raise ValueError(f"Incomplete five-camera frame {sequence_id} position {position}")
            if len({s.row["timestamp_micros"] for s in cameras.values()}) != 1:
                raise ValueError(f"Camera timestamps disagree in {sequence_id} position {position}")
        return frames
