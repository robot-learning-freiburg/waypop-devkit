"""Build an evaluation registry from the distributed release manifests."""

import json
from collections import Counter, defaultdict
from pathlib import Path


def release_registry(data_root: Path) -> dict:
    with (data_root / "release.json").open(encoding="utf-8") as handle:
        release = json.load(handle)
    with (data_root / "sequences.jsonl").open(encoding="utf-8") as handle:
        sequences = [json.loads(line) for line in handle]
    if release.get("dataset") != "WayPOP":
        raise ValueError("release.json does not describe a WayPOP dataset")
    counts = Counter(row["split"] for row in sequences)
    if dict(counts) != release["split_variants"] or set(counts) != {"val", "v1", "xl"}:
        raise ValueError("Sequence membership disagrees with release.json")
    identities = [row["sequence_id"] for row in sequences]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate sequence_id in sequences.jsonl")
    sources = defaultdict(list)
    for row in sequences:
        if row["asset_source"] not in {"v1", "xl"} or row["clip_id"] not in {
            "clip_00",
            "clip_01",
        }:
            raise ValueError(f"Invalid domain or clip for {row['sequence_id']}")
        if row["split"] != "val" and row["asset_source"] != row["split"]:
            raise ValueError(f"Incorrect test domain for {row['sequence_id']}")
        sources[row["source_sequence"]].append(row)
    combinations = {
        (clip, domain) for clip in ("clip_00", "clip_01") for domain in ("v1", "xl")
    }
    for source, rows in sources.items():
        if (
            len(rows) != 4
            or {(r["clip_id"], r["asset_source"]) for r in rows} != combinations
        ):
            raise ValueError(f"Incomplete clip/domain membership for {source}")
        by_domain = {
            domain: {r["split"] for r in rows if r["asset_source"] == domain}
            for domain in ("v1", "xl")
        }
        if (by_domain["v1"], by_domain["xl"]) not in (
            ({"val"}, {"val"}),
            ({"val"}, {"xl"}),
            ({"v1"}, {"xl"}),
        ):
            raise ValueError(f"Invalid validation/test domain assignment for {source}")
    return {
        "version": 2,
        "dataset_version": release["version"],
        "collections": [
            {
                "id": split,
                "root": split,
                "source_domain": (
                    "mixed"
                    if len({r["asset_source"] for r in sequences if r["split"] == split}) > 1
                    else next(r["asset_source"] for r in sequences if r["split"] == split)
                ),
                "split": "val" if split == "val" else "test",
                "segments": sorted(
                    r["sequence_id"] for r in sequences if r["split"] == split
                ),
            }
            for split in ("val", "v1", "xl")
        ],
    }
