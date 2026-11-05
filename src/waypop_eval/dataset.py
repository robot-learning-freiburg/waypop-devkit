from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import yaml
from PIL import Image

from .panoptic import load_panoptic_ids, relative_path

LABEL_DIVISOR = 1000


@dataclass
class ClassMetadata:
    # pylint: disable=too-many-instance-attributes

    classes: dict[int, str]  # category_id -> name
    order: list[str]  # category names in evaluation order
    ignored: frozenset[str]  # class names to ignore in evaluation
    things: frozenset[str]  # thing class names (panoptic semantics)
    stuff: frozenset[str]  # stuff class names (panoptic semantics)
    known: frozenset[str]  # in-distribution class names
    unknown: frozenset[str]  # OOD class names
    colors: dict[str, tuple[int, int, int]]  # name -> (R, G, B)

    @property
    def names(self) -> dict[str, int]:
        """Reverse lookup: name -> category_id."""
        return {name: id for id, name in self.classes.items()}

    @property
    def eval_order(self) -> dict[str, int]:
        """Map class name -> eval index (0-based position in ``order``)."""
        return {name: i for i, name in enumerate(self.order)}

    @property
    def ignored_eval_idx(self) -> int:
        """Eval index of the first ignored class (used as void/fallback)."""
        order_map = self.eval_order
        return next(order_map[name] for name in self.order if name in self.ignored)

    def build_eval_remap(self) -> dict[int, int]:
        """Build a mapping from dataset category_id -> contiguous eval index.

        Category IDs not present in ``classes`` (e.g. void = 0) are not
        included; callers should use ``ignored_eval_idx`` as the fallback.
        """
        order_map = self.eval_order
        ignored_idx = self.ignored_eval_idx

        return {
            cat_id: order_map.get(name, ignored_idx)
            for cat_id, name in self.classes.items()
        }

    @classmethod
    def from_yaml(cls, path: Path | str) -> ClassMetadata:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        return cls(
            classes=cfg["classes"],
            order=cfg["order"],
            ignored=frozenset(cfg["ignored"]),
            things=frozenset(cfg["things"]),
            stuff=frozenset(cfg["stuff"]),
            known=frozenset(cfg["known"]),
            unknown=frozenset(cfg["unknown"]),
            colors={name: tuple(rgb) for name, rgb in cfg["colors"].items()},
        )


@dataclass
class Sample:
    # pylint: disable=too-many-instance-attributes

    # Identity
    segment: str
    camera: str
    frame_index: int
    timestamp_micros: int

    # Sequence grouping
    sequence_id: str
    sequence_index: int
    sequence_position: int

    # Paths (absolute)
    image_path: Path
    panoptic_path: Path
    wstq_meta_path: Path | None

    # Panoptic encoding
    label_divisor: int
    semantic_remap: dict[int, int]  # dataset category_id -> eval index
    void_idx: int  # eval index for unmapped pixels

    # Poses in local segment coordinates (4×4, float64)
    vehicle_pose: np.ndarray
    image_pose: np.ndarray

    # Optional benchmark metadata
    set_id: str = ""
    source_domain: str = ""
    split: str = ""
    segment_uid: str = ""
    sample_uid: str = ""
    unknown_object_details_visible: tuple[dict, ...] = ()
    prediction_key: str = ""
    width: int = 0
    height: int = 0

    def load_panoptic(self) -> torch.Tensor:
        """Load the panoptic label map as an (H, W, 2) int64 tensor.

        Channel 0 — semantic[y, x]: eval class index
        Channel 1 — instance[y, x]: instance_id (0 for stuff classes)
        """
        panoptic = load_panoptic_ids(self.panoptic_path)
        self._check_shape(panoptic.shape, self.panoptic_path)
        raw_semantic, instance = _decode_panoptic(panoptic, self.label_divisor)

        # We use -1 for no-instance. The raw instance IDs are 1-based, so this
        # is just a shift.
        instance = instance - 1

        semantic = np.full_like(raw_semantic, self.void_idx)
        for cat_id, eval_idx in self.semantic_remap.items():
            semantic[raw_semantic == cat_id] = eval_idx

        return torch.from_numpy(np.stack((semantic, instance), axis=-1))  # (H, W, 2)

    def load_image(self) -> torch.Tensor:
        """Load the RGB image as a uint8 tensor of shape (H, W, 3)."""
        with Image.open(self.image_path) as image:
            return torch.from_numpy(np.array(image.convert("RGB")))

    def load_raw_panoptic_ids(self) -> torch.Tensor:
        """Load the raw panoptic PNG ids as an int64 tensor of shape (H, W)."""
        return torch.from_numpy(load_panoptic_ids(self.panoptic_path))

    def _check_shape(self, shape: tuple[int, ...], path: Path) -> None:
        if self.width and self.height and shape != (self.height, self.width):
            raise ValueError(
                f"Incorrect label shape at {path}: {shape}; expected {(self.height, self.width)}"
            )

    def load_wstq_metadata(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load Waymo wSTQ side metadata.

        Returns:
            coverage: ``(H, W)`` float32 tensor with ``num_cameras_covered``.
            tracked_mask: ``(H, W)`` uint8 tensor where 1 marks tracked pixels.
        """
        if self.wstq_meta_path is None or not self.wstq_meta_path.exists():
            raise FileNotFoundError(
                "Missing wSTQ metadata for sample: "
                f"{self.segment} {self.camera} frame={self.frame_index} "
                f"expected at {self.wstq_meta_path}"
            )

        with np.load(self.wstq_meta_path, allow_pickle=False) as data:
            coverage = np.asarray(data["coverage"], dtype=np.float32)
            tracked_mask = np.asarray(data["is_tracked_mask"], dtype=np.uint8)
        self._check_shape(coverage.shape, self.wstq_meta_path)
        if tracked_mask.shape != coverage.shape or not np.all(
            (tracked_mask == 0) | (tracked_mask == 1)
        ):
            raise ValueError(f"Invalid tracking mask at {self.wstq_meta_path}")
        if not np.all(np.isfinite(coverage) & (coverage >= 0) & (coverage <= 5)):
            raise ValueError(f"Invalid camera coverage at {self.wstq_meta_path}")
        return torch.from_numpy(coverage), torch.from_numpy(tracked_mask)


def _decode_panoptic(
    panoptic: np.ndarray, label_divisor: int = LABEL_DIVISOR
) -> tuple[np.ndarray, np.ndarray]:
    """Split an integer panoptic map using its sequence's native divisor."""
    if label_divisor <= 0:
        raise ValueError("panoptic_label_divisor must be positive")
    return panoptic // label_divisor, panoptic % label_divisor


class WayPoP:
    def __init__(
        self,
        path: Path | str,
        config: Path | str,
        annotations_root: Path | str | None = None,
    ):
        self.path = Path(path)
        self.meta = ClassMetadata.from_yaml(config)
        self._samples = WayPoP._load_samples(
            self.path,
            self.meta,
            Path(annotations_root) if annotations_root is not None else None,
        )
        divisors = {sample.label_divisor for sample in self._samples}
        self.label_divisor = next(iter(divisors)) if len(divisors) == 1 else None

    @classmethod
    def _load_samples(
        cls, path: Path, meta: ClassMetadata, annotations_root: Path | None = None
    ) -> list[Sample]:
        semantic_remap = meta.build_eval_remap()
        void_idx = meta.ignored_eval_idx

        samples_path = path / "meta" / "samples.jsonl"
        references: dict[str, dict] = {}
        if annotations_root is not None:
            with (annotations_root / "meta/samples.jsonl").open(
                encoding="utf-8"
            ) as handle:
                for line in handle:
                    reference = json.loads(line)
                    identity = reference["sample_id"]
                    if identity in references:
                        raise ValueError(
                            f"Duplicate private annotation sample_id: {identity}"
                        )
                    references[identity] = reference

        samples: list[Sample] = []
        seen = set()
        sequence_indices: dict[str, int] = {}
        sequence_metadata: dict[str, dict] = {}
        with open(samples_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sequence_id = row["sequence_id"]
                image_rel = relative_path(row["image_path"])
                prediction_key = str(image_rel.relative_to("images").with_suffix(""))
                identity = row.get("sample_id", row.get("sample_uid", prediction_key))
                if identity in seen:
                    raise ValueError(f"Duplicate sample identity: {identity}")
                seen.add(identity)
                label_row, label_root = row, path
                if "panoptic_path" not in row:
                    if identity not in references:
                        raise ValueError(
                            f"Annotations are withheld for {path.name}/{identity}. "
                            "Use validation for local evaluation, or provide organizer test references "
                            "with --annotations-root."
                        )
                    label_row, label_root = references[identity], annotations_root
                    for field in (
                        "sequence_id",
                        "source_sequence",
                        "clip_id",
                        "asset_source",
                        "split",
                        "camera_name",
                        "frame_index",
                        "timestamp_micros",
                        "sequence_position",
                        "width",
                        "height",
                        "image_path",
                        "pointcloud_path",
                    ):
                        if field in row and row[field] != label_row.get(field):
                            raise ValueError(
                                f"Private annotation mismatch for {identity}: {field}"
                            )
                panoptic_rel = relative_path(label_row["panoptic_path"])
                if "wstq_meta_path" in label_row:
                    wstq_meta_path = label_root / relative_path(
                        label_row["wstq_meta_path"]
                    )
                else:
                    wstq_meta_path = (
                        label_root / "wstq_meta" / panoptic_rel.relative_to("panoptic")
                    ).with_suffix(".npz")
                divisor = label_row.get(
                    "panoptic_label_divisor",
                    label_row.get("panoptic_label_divisor_remapped"),
                )
                if divisor is None:
                    if sequence_id not in sequence_metadata:
                        with (
                            label_root / "meta/sequences" / f"{sequence_id}.json"
                        ).open(encoding="utf-8") as handle:
                            sequence_metadata[sequence_id] = json.load(handle)
                    divisor = sequence_metadata[sequence_id]["panoptic_label_divisor"]
                divisor = int(divisor)
                if divisor <= 0:
                    raise ValueError(
                        f"Invalid panoptic_label_divisor for {identity}: {divisor}"
                    )
                if panoptic_rel.relative_to("panoptic").with_suffix("") != Path(
                    prediction_key
                ):
                    raise ValueError(f"Image/annotation path mismatch for {identity}")
                sequence_indices.setdefault(sequence_id, len(sequence_indices))

                samples.append(
                    Sample(
                        segment=row.get("segment", sequence_id),
                        camera=row["camera_name"],
                        frame_index=row["frame_index"],
                        timestamp_micros=row["timestamp_micros"],
                        sequence_id=sequence_id,
                        sequence_index=row.get(
                            "sequence_index", sequence_indices[sequence_id]
                        ),
                        sequence_position=row["sequence_position"],
                        image_path=path / image_rel,
                        panoptic_path=label_root / panoptic_rel,
                        wstq_meta_path=wstq_meta_path,
                        label_divisor=divisor,
                        semantic_remap=semantic_remap,
                        void_idx=void_idx,
                        vehicle_pose=np.array(
                            row["vehicle_pose_local_4x4"], dtype=np.float64
                        ),
                        image_pose=np.array(
                            row["image_pose_local_4x4"], dtype=np.float64
                        ),
                        set_id=row.get("set_id", ""),
                        source_domain=row.get(
                            "source_domain", row.get("asset_source", "")
                        ),
                        split=row.get("split", ""),
                        segment_uid=row.get("segment_uid", ""),
                        sample_uid=row.get("sample_id", row.get("sample_uid", "")),
                        unknown_object_details_visible=tuple(
                            row.get("unknown_object_details_visible", ())
                        ),
                        prediction_key=prediction_key,
                        width=int(row.get("width", 0)),
                        height=int(row.get("height", 0)),
                    )
                )

        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)
