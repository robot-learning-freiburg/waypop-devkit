"""Validate and package participant prediction bundles without test labels."""

from __future__ import annotations

import io
import json
import stat
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
from PIL import Image

from .data import Sample, WaypopRelease, relative_path
from .labels import rgb_to_ids

TASKS = ("mv-ospt", "fv-ospt", "vi-osps")
MAX_INDEX_BYTES = 512 * 1024 * 1024


class SubmissionError(ValueError):
    """A participant-facing prediction format error."""


def _integer(value: object, description: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise SubmissionError(f"{description} must be an integer from {minimum} to {maximum}")
    return value


class _DirectorySource:
    def __init__(self, root: Path):
        self.root = root
        if root.is_symlink() or not root.is_dir():
            raise SubmissionError(f"Expected a prediction directory: {root}")
        self.files: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise SubmissionError(f"Symlink is not allowed: {path}")
            if not path.is_file() and not path.is_dir():
                raise SubmissionError(f"Unexpected non-regular entry: {path}")
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                relative_path(relative)
                self.files.add(relative)

    def size(self, name: str) -> int:
        return (self.root / name).stat().st_size

    def open(self, name: str) -> BinaryIO:
        return (self.root / name).open("rb")


class _ZipSource:
    def __init__(self, path: Path):
        try:
            self.archive = zipfile.ZipFile(path)
        except (OSError, zipfile.BadZipFile) as error:
            raise SubmissionError(f"Cannot open submission ZIP: {path}") from error
        self.info: dict[str, zipfile.ZipInfo] = {}
        for item in self.archive.infolist():
            name = item.filename
            kind = stat.S_IFMT(item.external_attr >> 16) if item.create_system == 3 else 0
            if item.is_dir():
                name = name.rstrip("/") + "/placeholder"
                relative_path(name)
                if kind not in (0, stat.S_IFDIR):
                    raise SubmissionError(f"Non-directory ZIP entry has a directory name: {item.filename}")
                continue
            relative_path(name)
            if item.flag_bits & 1:
                raise SubmissionError(f"Encrypted ZIP entry is not allowed: {name}")
            if kind not in (0, stat.S_IFREG):
                raise SubmissionError(f"Non-regular ZIP entry is not allowed: {name}")
            if name in self.info:
                raise SubmissionError(f"Duplicate ZIP entry: {name}")
            self.info[name] = item
        self.files = set(self.info)

    def size(self, name: str) -> int:
        return self.info[name].file_size

    def open(self, name: str) -> BinaryIO:
        return self.archive.open(self.info[name])

    def close(self) -> None:
        self.archive.close()


@contextmanager
def _open_source(path: Path) -> Iterator[_DirectorySource | _ZipSource]:
    if path.is_symlink():
        raise SubmissionError(f"Symlink is not allowed: {path}")
    if path.is_dir():
        yield _DirectorySource(path)
    else:
        source = _ZipSource(path)
        try:
            yield source
        finally:
            source.close()


@dataclass(frozen=True)
class ValidationReport:
    task: str
    splits: tuple[str, ...]
    annotations: int
    required_samples: int
    warnings: tuple[str, ...]


def _read_json(source: _DirectorySource | _ZipSource, name: str) -> dict:
    if name not in source.files:
        raise SubmissionError(f"Missing {name}")
    if source.size(name) > MAX_INDEX_BYTES:
        raise SubmissionError(f"Oversized {name}")
    try:
        with source.open(name) as handle:
            payload = json.loads(handle.read().decode("utf-8"))
    except (UnicodeError, ValueError, OSError, zipfile.BadZipFile) as error:
        raise SubmissionError(f"Invalid JSON in {name}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("annotations"), list):
        raise SubmissionError(f"{name} must contain an annotations list")
    return payload


def _load_png(
    source: _DirectorySource | _ZipSource, name: str, sample: Sample
) -> np.ndarray:
    if name not in source.files:
        raise SubmissionError(f"Missing prediction PNG: {name}")
    width, height = sample.size
    if source.size(name) > width * height * 4 + 1024 * 1024:
        raise SubmissionError(f"Oversized prediction PNG: {name}")
    try:
        with source.open(name) as handle:
            with Image.open(io.BytesIO(handle.read())) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (width, height):
                    raise SubmissionError(f"Expected an RGB PNG of {width}x{height}: {name}")
                return rgb_to_ids(np.asarray(image))
    except SubmissionError:
        raise
    except Exception as error:
        raise SubmissionError(f"Cannot decode prediction PNG: {name}") from error


def validate_submission(
    input_path: Path | str,
    data_root: Path | str,
    task: str,
    *,
    split: str = "test",
) -> ValidationReport:
    """Check exact coverage and files for one task, with no access to test GT."""
    task = task.lower()
    if task not in TASKS:
        raise ValueError(f"Expected task in {TASKS}")
    if split not in ("test", "val"):
        raise ValueError("split must be 'test' or 'val'")
    splits = ("v1", "xl") if split == "test" else ("val",)
    release = WaypopRelease(data_root)
    categories = set(release.category_map) | {255}  # 255 is a scorer-supported legacy OOD alias.
    expected_files: set[str] = set()
    warnings: set[str] = set()
    total_annotations = 0
    required_samples = 0
    category_history: dict[tuple[str, str, int], int] = {}

    with _open_source(Path(input_path).expanduser().absolute()) as source:
        for collection in splits:
            samples = release.samples(collection)
            allowed = {sample.prediction_file_name: sample for sample in samples}
            required = {
                key for key, sample in allowed.items() if task != "fv-ospt" or sample.camera == "FRONT"
            }
            required_samples += len(required)
            index_name = f"{collection}/predictions.json"
            payload = _read_json(source, index_name)
            expected_files.add(index_name)
            seen: set[str] = set()
            for annotation in payload["annotations"]:
                if not isinstance(annotation, dict) or not isinstance(annotation.get("file_name"), str):
                    raise SubmissionError(f"Invalid annotation in {index_name}")
                file_name = annotation["file_name"]
                try:
                    relative = relative_path(file_name)
                except ValueError as error:
                    raise SubmissionError(f"Invalid prediction file_name: {file_name!r}") from error
                if len(relative.parts) != 3 or relative.suffix != ".png" or file_name not in allowed:
                    raise SubmissionError(f"Prediction does not match a released {collection} sample: {file_name}")
                if file_name in seen:
                    raise SubmissionError(f"Duplicate prediction annotation: {collection}/{file_name}")
                seen.add(file_name)
                segments = annotation.get("segments_info")
                if not isinstance(segments, list):
                    raise SubmissionError(f"Missing segments_info: {collection}/{file_name}")
                segment_categories: dict[int, int] = {}
                for segment in segments:
                    if not isinstance(segment, dict):
                        raise SubmissionError(f"Invalid segments_info: {collection}/{file_name}")
                    segment_id = _integer(segment.get("id"), "segment ID", minimum=1, maximum=(1 << 24) - 1)
                    category_id = _integer(segment.get("category_id"), "category ID", minimum=1, maximum=255)
                    if segment_id in segment_categories:
                        raise SubmissionError(f"Duplicate segment ID in {collection}/{file_name}: {segment_id}")
                    if category_id not in categories:
                        raise SubmissionError(f"Unknown category ID in {collection}/{file_name}: {category_id}")
                    if category_id == 255:
                        warnings.add("Category 255 is accepted as a legacy unknown alias; category 29 is canonical.")
                    segment_categories[segment_id] = category_id

                sample = allowed[file_name]
                png_name = f"{collection}/predictions/{file_name}"
                expected_files.add(png_name)
                ids = _load_png(source, png_name, sample)
                used = {int(value) for value in np.unique(ids) if value != 0}
                missing_metadata = used - set(segment_categories)
                if missing_metadata:
                    raise SubmissionError(f"PNG IDs missing from segments_info in {png_name}: {sorted(missing_metadata)[:5]}")
                unused_metadata = set(segment_categories) - used
                if unused_metadata:
                    warnings.add(f"Some segments_info IDs have no pixels (first: {png_name}).")
                if task != "vi-osps":
                    for segment_id in used:
                        history_key = (collection, sample.sequence_id, segment_id)
                        category_id = segment_categories[segment_id]
                        previous = category_history.setdefault(history_key, category_id)
                        if previous != category_id:
                            raise SubmissionError(
                                f"Tracking ID {segment_id} changes category within {sample.sequence_id}"
                            )
                total_annotations += 1
            missing = required - seen
            if missing:
                raise SubmissionError(
                    f"Missing {len(missing)} required {collection} predictions; first: {sorted(missing)[0]}"
                )
        extra = source.files - expected_files
        if extra:
            raise SubmissionError(f"Unexpected file in submission: {sorted(extra)[0]}")
    return ValidationReport(task, splits, total_annotations, required_samples, tuple(sorted(warnings)))


def package_submission(
    prediction_root: Path | str,
    output_zip: Path | str,
    data_root: Path | str,
    task: str,
) -> ValidationReport:
    """Validate a directory, then write an archive with only indexed files."""
    prediction_root = Path(prediction_root).expanduser().absolute()
    output_zip = Path(output_zip).expanduser().resolve()
    if output_zip.is_relative_to(prediction_root):
        raise SubmissionError("Write the ZIP outside the prediction directory")
    report = validate_submission(prediction_root, data_root, task)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_zip.parent, suffix=".zip", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            with _open_source(prediction_root) as source:
                for name in sorted(source.files):
                    archive.write(prediction_root / name, arcname=name)
        temporary_path.replace(output_zip)
    finally:
        temporary_path.unlink(missing_ok=True)
    return report
