"""Check the three-frame example's saved assignments before rendering."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def validate_example_inputs(
    objects_path: Path, placement_path: Path, segment: str, expected_object_count: int
) -> None:
    objects = _load_json(objects_path)
    placement = _load_json(placement_path)
    if not isinstance(objects, list) or len(objects) != expected_object_count:
        raise ValueError(f"Expected {expected_object_count} object assignments in {objects_path}")
    if not isinstance(placement, dict) or placement.get("segment") != segment:
        raise ValueError(f"Placement segment must be {segment}")
    primary = placement.get("primary_candidate_ids")
    table = placement.get("candidate_table")
    if not isinstance(primary, list) or not isinstance(table, list) or not table:
        raise ValueError("Placement must contain primary_candidate_ids and candidate_table")
    if placement.get("candidate_count") != len(table):
        raise ValueError("Placement candidate_count differs from candidate_table length")
    by_id = {}
    for row in table:
        if not isinstance(row, dict):
            raise ValueError("Placement candidate_table has a non-object row")
        candidate_id = int(row["candidate_id"])
        if candidate_id in by_id:
            raise ValueError(f"Duplicate placement candidate ID {candidate_id}")
        by_id[candidate_id] = row

    slots = set()
    slot_order = []
    assigned_candidates = set()
    for ordinal, obj in enumerate(objects, 1):
        if not isinstance(obj, dict):
            raise ValueError(f"Object {ordinal} is not a JSON object")
        slot = int(obj["placement_slot_index"])
        candidate_id = int(obj["placement_candidate_id"])
        if slot < 0 or slot >= len(primary) or slot in slots:
            raise ValueError(f"Object {ordinal} has invalid or duplicate placement slot {slot}")
        if candidate_id in assigned_candidates:
            raise ValueError(f"Object {ordinal} repeats candidate ID {candidate_id}")
        if int(primary[slot]) != candidate_id or candidate_id not in by_id:
            raise ValueError(f"Object {ordinal} candidate ID does not match placement slot/table")
        slots.add(slot)
        slot_order.append(slot)
        assigned_candidates.add(candidate_id)

        point = obj.get("placement_point_xyz")
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError(f"Object {ordinal} has no 3D placement point")
        expected = [float(by_id[candidate_id][axis]) for axis in ("x", "y", "z")]
        actual = [float(value) for value in point]
        if not all(math.isfinite(value) for value in actual + expected):
            raise ValueError(f"Object {ordinal} has a non-finite placement point")
        if math.dist(actual, expected) > 1e-5:
            raise ValueError(f"Object {ordinal} point differs from candidate {candidate_id}")

        raw_path = obj.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Object {ordinal} has no asset path")
        expanded = os.path.expanduser(os.path.expandvars(raw_path))
        if "$" in expanded:
            raise ValueError(f"Object {ordinal} has an unresolved asset path variable: {raw_path}")
        asset = Path(expanded)
        if not asset.is_absolute():
            asset = objects_path.parent / asset
        if not asset.is_file():
            raise FileNotFoundError(f"Object {ordinal} asset does not exist: {asset}")
        if asset.stem != obj.get("fileIdentifier"):
            raise ValueError(f"Object {ordinal} asset filename does not match fileIdentifier")

    if slot_order != sorted(slot_order):
        raise ValueError("Object assignments must be ordered by placement_slot_index")

    print(f"Checked {expected_object_count} assets and saved placements for {segment}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    validate_example_inputs(args.objects, args.placement, args.segment, args.count)


if __name__ == "__main__":
    main()
