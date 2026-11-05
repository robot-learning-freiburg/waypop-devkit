# pylint: disable=too-few-public-methods

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from .dataset import ClassMetadata, Sample
from .panoptic import load_panoptic_ids, relative_path

# ---------------------------------------------------------------------------
# Format configuration
# ---------------------------------------------------------------------------


@dataclass
class FormatConfig:
    """Parsed format/<name>.yaml — drives loader and decoder selection."""

    format: str
    encoding: str
    classes: dict[int, str]
    raw: dict = field(repr=False)  # full YAML dict, forwarded to loaders

    @property
    def names(self) -> dict[str, int]:
        """Reverse lookup: name -> category_id."""
        return {name: id for id, name in self.classes.items()}

    @classmethod
    def from_yaml(cls, path: Path | str) -> FormatConfig:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        data = cfg["data"]
        return cls(
            format=data["format"],
            encoding=data.get("encoding", "rgb_packed"),
            classes=cfg["classes"],
            raw=cfg,
        )


# ---------------------------------------------------------------------------
# Panoptic decoders — convert a prediction file to a raw segment-ID map
# ---------------------------------------------------------------------------


class PanopticDecoder(ABC):
    """Decode a panoptic prediction file into (semantic, instance) tensors.

    Each decoder owns both the pixel-level encoding (how segment IDs are stored
    in the image) and the segment-to-category mapping scheme (how segment IDs
    are resolved to class labels).  Format-specific per-sample metadata (e.g.
    COCO ``segments_info``) is passed in via ``meta``.
    """

    @abstractmethod
    def decode(
        self,
        path: Path,
        meta: dict,
        remap: dict[int, int],
        void_idx: int,
    ) -> torch.Tensor:
        """Return an (H, W, 2) int64 tensor: channel 0 = semantic, channel 1 = instance.

        Args:
            path:     Path to the panoptic prediction image.
            meta:     Decoder-specific per-sample metadata stored by the loader.
            remap:    pred category_id -> eval index.
            void_idx: Eval index for pred category IDs not present in remap.
        """


class CocoRgbPackedDecoder(PanopticDecoder):
    """COCO panoptic encoding: segment_id = R + G*256 + B*65536.

    Expects ``meta["segments_info"]`` — a ``{segment_id: pred_category_id}``
    dict populated by :class:`CocoPredictionLoader`.
    """

    def decode(
        self,
        path: Path,
        meta: dict,
        remap: dict[int, int],
        void_idx: int,
    ) -> torch.Tensor:
        instance = load_panoptic_ids(path, require_rgb=True)
        unique_ids, inverse = np.unique(instance, return_inverse=True)
        unlisted = set(unique_ids.tolist()) - set(meta.get("segments_info", {})) - {0}
        if unlisted:
            raise ValueError(
                f"Prediction IDs missing from segments_info at {path}: {sorted(unlisted)}"
            )

        segments = meta.get("segments_info", {})
        category_lookup = np.array(
            [
                remap.get(segments.get(int(identity)), void_idx)
                for identity in unique_ids
            ],
            dtype=np.int64,
        )
        semantic = category_lookup[inverse].reshape(instance.shape)

        return torch.from_numpy(np.stack((semantic, instance), axis=-1))


_DECODERS: dict[str, type[PanopticDecoder]] = {
    "coco_rgb_packed": CocoRgbPackedDecoder,
}


# ---------------------------------------------------------------------------
# Prediction sample
# ---------------------------------------------------------------------------


@dataclass
class PredictionSample:
    segment: str
    camera: str
    panoptic_path: Path
    remap: dict[int, int]  # pred category_id -> eval index
    void_idx: int  # eval index for pred category IDs not in remap
    decoder: PanopticDecoder
    meta: dict = field(default_factory=dict)  # decoder-specific per-sample data

    def load_panoptic(self) -> torch.Tensor:
        """Load the panoptic prediction as an (H, W, 2) int64 tensor.

        Channel 0 — semantic[y, x]: eval class index
        Channel 1 — instance[y, x]: panoptic segment ID (unique per segment; 0 = void)
        """
        return self.decoder.decode(
            self.panoptic_path, self.meta, self.remap, self.void_idx
        )


# ---------------------------------------------------------------------------
# Prediction loaders — format-specific index building
# ---------------------------------------------------------------------------


class PredictionLoader(ABC):
    """Parse a prediction directory into a keyed index of PredictionSamples."""

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: dict) -> PredictionLoader:
        """Construct from the raw YAML config dict."""

    @abstractmethod
    def build_index(
        self, path: Path, remap: dict[int, int], void_idx: int, decoder: PanopticDecoder
    ) -> dict[str, PredictionSample]:
        """Return a key -> PredictionSample mapping for all predictions at path."""


class CocoPredictionLoader(PredictionLoader):
    """Loads predictions stored in COCO panoptic format.

    Expects <path>/<index_file> (default: predictions.json) with COCO
    panoptic structure, and panoptic PNGs under <path>/<panoptic_dir>/.

    Configurable via the ``loader:`` section of the format YAML:

    .. code-block:: yaml

        loader:
          index_file: predictions.json   # default
          panoptic_dir: predictions      # default
    """

    def __init__(
        self,
        index_file: str = "predictions.json",
        panoptic_dir: str = "predictions",
    ):
        self.index_file = index_file
        self.panoptic_dir = panoptic_dir

    @classmethod
    def from_config(cls, cfg: dict) -> CocoPredictionLoader:
        loader_cfg = cfg.get("loader", {})
        return cls(
            index_file=loader_cfg.get("index_file", "predictions.json"),
            panoptic_dir=loader_cfg.get("panoptic_dir", "predictions"),
        )

    def build_index(
        self, path: Path, remap: dict[int, int], void_idx: int, decoder: PanopticDecoder
    ) -> dict[str, PredictionSample]:
        with open(path / self.index_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        index = {}
        for ann in data["annotations"]:
            file_name = relative_path(ann["file_name"])
            if len(file_name.parts) < 3 or file_name.suffix.lower() != ".png":
                raise ValueError(
                    f"Expected SEQUENCE/CAMERA/FRAME.png prediction path: {file_name}"
                )
            parts = file_name.parts  # (segment, camera, seq_NNN, filename)
            key = str(file_name.with_suffix(""))
            if key in index:
                raise ValueError(f"Duplicate prediction annotation: {ann['file_name']}")
            segment_ids = [int(s["id"]) for s in ann["segments_info"]]
            if len(segment_ids) != len(set(segment_ids)):
                raise ValueError(f"Duplicate segments_info id for {ann['file_name']}")
            if any(value < 0 or value >= 2**24 for value in segment_ids):
                raise ValueError(
                    f"segments_info ids exceed RGB24 range for {ann['file_name']}"
                )
            index[key] = PredictionSample(
                segment=parts[0],
                camera=parts[1],
                panoptic_path=path / self.panoptic_dir / file_name,
                remap=remap,
                void_idx=void_idx,
                decoder=decoder,
                meta={
                    "segments_info": {
                        int(s["id"]): int(s["category_id"])
                        for s in ann["segments_info"]
                    }
                },
            )

        return index


_LOADERS: dict[str, type[PredictionLoader]] = {
    "coco": CocoPredictionLoader,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_remap(config: FormatConfig, dataset_meta: ClassMetadata) -> dict[int, int]:
    """Build a pred category_id -> eval index mapping by joining on class name."""
    order_map = dataset_meta.eval_order
    ignored_idx = dataset_meta.ignored_eval_idx

    return {
        pred_id: order_map.get(name, ignored_idx)
        for pred_id, name in config.classes.items()
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Predictions:
    def __init__(
        self,
        path: Path | str,
        meta: ClassMetadata,
        config: Path | str,
    ):
        self.path = Path(path)

        cfg = FormatConfig.from_yaml(config)
        remap = _build_remap(cfg, meta)

        loader_cls = _LOADERS.get(cfg.format)
        if loader_cls is None:
            raise ValueError(
                f"Unknown prediction format {cfg.format!r}. "
                f"Supported: {sorted(_LOADERS)}"
            )

        decoder_cls = _DECODERS.get(cfg.encoding)
        if decoder_cls is None:
            raise ValueError(
                f"Unknown panoptic encoding {cfg.encoding!r}. "
                f"Supported: {sorted(_DECODERS)}"
            )

        loader = loader_cls.from_config(cfg.raw)
        decoder = decoder_cls()
        void_idx = meta.ignored_eval_idx
        self._index = loader.build_index(self.path, remap, void_idx, decoder)

    def __getitem__(self, sample: Sample) -> PredictionSample:
        # Reconstruct the relative path key: segment/camera/seq_NNN/stem
        rel = sample.panoptic_path.parts
        # panoptic_path is absolute: .../panoptic/segment/camera/seq_NNN/file.png
        # key is segment/camera/seq_NNN/stem (last 4 parts, no extension)
        key = sample.prediction_key or str(Path(*rel[-4:]).with_suffix(""))
        return self._index[key]

    def __len__(self) -> int:
        return len(self._index)
