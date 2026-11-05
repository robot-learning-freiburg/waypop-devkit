"""Regression checks for the sparse-slot validation example."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from placement_checks import (  # noqa: E402
    assigned_point,
    check_sampled_position,
    checked_saved_object_ids,
)
from check_example_inputs import validate_example_inputs  # noqa: E402
from check_retry_state import CAMERA_NAMES, FRAME_TOKENS, validate_saved_render_state  # noqa: E402

from PIL import Image


class SparsePlacementTests(unittest.TestCase):
    def setUp(self):
        self.slots = [0, 1, 2, 4, 5, 6, 7, 8]
        self.rows = [
            {
                "fileIdentifier": f"asset-{slot}",
                "placement_slot_index": slot,
                "placement_candidate_id": 100 + slot,
                "placement_point_xyz": [float(slot), 2.0, 3.0],
            }
            for slot in self.slots
        ]

    def test_sparse_slots_keep_each_objects_saved_point(self):
        for ordinal, slot in enumerate(self.slots, 1):
            point = assigned_point(self.rows, ordinal, required=True)
            self.assertEqual(point, (float(slot), 2.0, 3.0))
            sampled_matrix = [
                [1, 0, 0, point[0]],
                [0, 1, 0, point[1]],
                [0, 0, 1, point[2]],
                [0, 0, 0, 1],
            ]
            check_sampled_position(sampled_matrix, point, ordinal)

        # The old dense lookup took slot 3 for OOD 4 and shifted the rest.
        with self.assertRaisesRegex(RuntimeError, "differs from its saved assignment"):
            check_sampled_position([3.0, 2.0, 3.0], assigned_point(self.rows, 4), 4)

    def test_saved_retry_ids_use_assignment_order_and_asset_identity(self):
        details = {
            str(ood_id): {"file_identifier": row["fileIdentifier"]}
            for ood_id, row in reversed(list(enumerate(self.rows, 1)))
        }
        self.assertEqual(list(checked_saved_object_ids(details, self.rows)), list(range(1, 9)))
        details.pop("4")
        with self.assertRaisesRegex(ValueError, "Saved object IDs differ"):
            checked_saved_object_ids(details, self.rows)
        details["4"] = {"file_identifier": "wrong-asset"}
        with self.assertRaisesRegex(ValueError, "Saved asset differs"):
            checked_saved_object_ids(details, self.rows)

    def test_preflight_checks_slot_candidate_point_relation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for row in self.rows:
                asset = root / f"{row['fileIdentifier']}.glb"
                asset.write_bytes(b"glTF")
                row["path"] = str(asset)
            candidates = [
                {"candidate_id": 100 + slot, "x": float(slot), "y": 2.0, "z": 3.0}
                for slot in range(10)
            ]
            placement = {
                "segment": "segment-example",
                "primary_candidate_ids": [100 + slot for slot in range(10)],
                "candidate_count": len(candidates),
                "candidate_table": candidates,
            }
            objects_path = root / "objects.json"
            placement_path = root / "placement.json"
            objects_path.write_text(json.dumps(self.rows))
            placement_path.write_text(json.dumps(placement))
            validate_example_inputs(objects_path, placement_path, "segment-example", 8)

            self.rows[3]["placement_candidate_id"] = 103
            objects_path.write_text(json.dumps(self.rows))
            with self.assertRaisesRegex(ValueError, "does not match placement slot"):
                validate_example_inputs(objects_path, placement_path, "segment-example", 8)

            self.rows[3]["placement_candidate_id"] = 104
            self.rows[3]["path"] = str(root / "asset-5.glb")
            objects_path.write_text(json.dumps(self.rows))
            with self.assertRaisesRegex(ValueError, "filename does not match fileIdentifier"):
                validate_example_inputs(objects_path, placement_path, "segment-example", 8)

            self.rows[3]["path"] = str(root / "asset-4.glb")
            self.rows[2], self.rows[3] = self.rows[3], self.rows[2]
            objects_path.write_text(json.dumps(self.rows))
            with self.assertRaisesRegex(ValueError, "ordered by placement_slot_index"):
                validate_example_inputs(objects_path, placement_path, "segment-example", 8)

    def test_retry_rejects_missing_layers_and_wrong_positions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = dict(self.rows[0])
            objects_path = root / "objects.json"
            objects_path.write_text(json.dumps([row]))
            segment_root = root / "outputs" / "segment-example"
            segment_root.mkdir(parents=True)
            details_path = segment_root / "ood_details.json"
            position = [[1, 0, 0, 0], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
            details_path.write_text(json.dumps({"1": {
                "file_identifier": "asset-0", "sampled_position": position
            }}))

            with self.assertRaisesRegex(FileNotFoundError, "Missing saved Blender layer"):
                validate_saved_render_state(root, objects_path, "segment-example")

            for camera in CAMERA_NAMES:
                layer_root = segment_root / "Renders" / "ood_1" / "camera" / camera
                layer_root.mkdir(parents=True)
                for token in FRAME_TOKENS:
                    for prefix in ("", "no_shadow_"):
                        Image.new("RGBA", (2, 2)).save(layer_root / f"{prefix}frame_00{token}.png")
            validate_saved_render_state(root, objects_path, "segment-example")

            position[0][3] = 3
            details_path.write_text(json.dumps({"1": {
                "file_identifier": "asset-0", "sampled_position": position
            }}))
            with self.assertRaisesRegex(RuntimeError, "differs from its saved assignment"):
                validate_saved_render_state(root, objects_path, "segment-example")


if __name__ == "__main__":
    unittest.main()
