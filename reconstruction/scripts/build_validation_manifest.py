"""Map public validation samples to their original Waymo TFRecord frames."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CAMERAS = {"FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"}


def build_manifest(release_root: Path) -> dict:
    path = release_root / "val" / "meta" / "samples.jsonl"
    groups: dict[str, dict[int, dict]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("split") != "val":
                raise ValueError(f"Non-validation sample at {path}:{line_number}")
            sequence_id = str(sample["sequence_id"])
            frame_index = int(sample["frame_index"])
            camera = str(sample["camera_name"])
            frame = groups[sequence_id].setdefault(
                frame_index,
                {
                    "source_sequence": str(sample["source_sequence"]),
                    "timestamp_micros": int(sample["timestamp_micros"]),
                    "cameras": set(),
                },
            )
            if frame["source_sequence"] != sample["source_sequence"]:
                raise ValueError(f"Source sequence mismatch at {path}:{line_number}")
            if frame["timestamp_micros"] != int(sample["timestamp_micros"]):
                raise ValueError(f"Timestamp mismatch at {path}:{line_number}")
            if camera in frame["cameras"]:
                raise ValueError(f"Duplicate camera at {path}:{line_number}")
            frame["cameras"].add(camera)

    sequences = []
    for sequence_id, frames in sorted(groups.items()):
        indexes = sorted(frames)
        if not indexes:
            raise ValueError(f"Empty validation sequence: {sequence_id}")
        if any(frames[index]["cameras"] != CAMERAS for index in indexes):
            raise ValueError(f"Incomplete five-camera timestamp in {sequence_id}")
        sources = {frames[index]["source_sequence"] for index in indexes}
        if len(sources) != 1:
            raise ValueError(f"Multiple source sequences in {sequence_id}")
        sequences.append(
            {
                "sequence_id": sequence_id,
                "source_sequence": sources.pop(),
                "release_frame_indices": indexes,
                "source_tfrecord_frame_tokens": [index + 1 for index in indexes],
                "timestamps_micros": [frames[index]["timestamp_micros"] for index in indexes],
                "camera_views_per_timestamp": 5,
            }
        )

    if len(sequences) != 6 or sum(len(s["release_frame_indices"]) * 5 for s in sequences) != 1635:
        raise ValueError("Validation membership differs from the six-clip, 1,635-image release")
    return {
        "schema_version": 1,
        "description": "Public WayPOP validation clips mapped to 1-based Waymo TFRecord tokens.",
        "sequences": sequences,
    }


def verify_source_timestamps(manifest: dict, waymo_directory: Path) -> None:
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset

    expected: dict[str, dict[int, int]] = defaultdict(dict)
    for sequence in manifest["sequences"]:
        source = sequence["source_sequence"]
        for token, timestamp in zip(
            sequence["source_tfrecord_frame_tokens"], sequence["timestamps_micros"]
        ):
            previous = expected[source].setdefault(token, timestamp)
            if previous != timestamp:
                raise ValueError(f"Conflicting timestamp for {source} token {token}")

    for source, token_map in sorted(expected.items()):
        path = waymo_directory / f"{source}.tfrecord"
        if not path.is_file():
            raise FileNotFoundError(path)
        seen = set()
        for token, record in enumerate(tf.data.TFRecordDataset(str(path)), 1):
            if token not in token_map:
                continue
            frame = open_dataset.Frame()
            frame.ParseFromString(bytes(record.numpy()))
            if int(frame.timestamp_micros) != token_map[token]:
                raise ValueError(f"Waymo timestamp mismatch for {source} token {token}")
            seen.add(token)
        if seen != set(token_map):
            raise ValueError(f"Waymo source {source} is missing {len(set(token_map) - seen)} frames")
        print(f"Verified {len(seen)} source frames in {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-source-root", type=Path,
        help="Optional directory containing original Waymo validation TFRecords.",
    )
    args = parser.parse_args()
    manifest = build_manifest(args.release_root)
    if args.verify_source_root:
        verify_source_timestamps(manifest, args.verify_source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['sequences'])} validation clips to {args.output}")


if __name__ == "__main__":
    main()
