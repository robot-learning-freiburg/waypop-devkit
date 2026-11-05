import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_tfrecord", required=True)
    parser.add_argument("--output_tfrecord", required=True)
    parser.add_argument("--frame_tokens",
                        required=True,
                        help="Comma-separated 1-based frame ids.")
    parser.add_argument("--expected_timestamps",
                        help="Optional comma-separated Waymo timestamps in microseconds.")
    return parser.parse_args()


def parse_tokens(spec: str):
    out = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError(f"Frame token must be >= 1, got {value}")
        out.append(value)
    if not out:
        raise ValueError("No frame tokens provided.")
    return out


def main():
    args = parse_args()
    source_arg = Path(args.source_tfrecord)
    output_arg = Path(args.output_tfrecord)
    source = source_arg.resolve()
    output = output_arg.resolve()
    if source == output:
        raise ValueError("Source and output TFRecord paths must differ.")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_tokens = parse_tokens(args.frame_tokens)
    expected_timestamps = (parse_tokens(args.expected_timestamps)
                           if args.expected_timestamps else None)
    if len(frame_tokens) != len(set(frame_tokens)) or frame_tokens != sorted(frame_tokens):
        raise ValueError("Frame tokens must be unique and in increasing order.")
    if expected_timestamps and len(expected_timestamps) != len(frame_tokens):
        raise ValueError("Expected timestamps must match the frame-token count.")
    token_set = set(frame_tokens)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)

    written = []
    observed_timestamps = []
    total = 0
    with tf.io.TFRecordWriter(str(partial)) as writer:
        dataset = tf.data.TFRecordDataset(str(source), compression_type="")
        for idx, record in enumerate(dataset, start=1):
            total += 1
            if idx not in token_set:
                continue
            frame = open_dataset.Frame()
            frame.ParseFromString(bytes(record.numpy()))
            observed_timestamps.append(int(frame.timestamp_micros))
            writer.write(record.numpy())
            written.append(idx)
            if idx == frame_tokens[-1]:
                break

    if written != frame_tokens:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Mini TFRecord mismatch: requested={frame_tokens} written={written} source_frames_scanned={total}"
        )
    if expected_timestamps and observed_timestamps != expected_timestamps:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Timestamp mismatch: expected={expected_timestamps} observed={observed_timestamps}"
        )
    os.replace(partial, output)

    manifest = {
        "source_tfrecord_name": source.name,
        "output_tfrecord_name": output.name,
        "frame_tokens": frame_tokens,
        "timestamps_micros": observed_timestamps,
        "frames_written": len(written),
        "source_frames_scanned": total,
    }
    manifest_path = output.with_suffix(".manifest.json")
    partial_manifest = manifest_path.with_name(manifest_path.name + ".partial")
    with open(partial_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(partial_manifest, manifest_path)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
