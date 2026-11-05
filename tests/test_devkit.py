"""Small release-format integration checks for the public participant tools."""

from __future__ import annotations

import importlib.util
import json
import stat
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from waypop_devkit.data import CAMERAS, WaypopRelease
from waypop_devkit.labels import ids_to_rgb, rgb_to_ids, stable_instance_color
from waypop_devkit.submissions import SubmissionError, package_submission, validate_submission
from waypop_devkit.visualize import Viewer, _Handler, _ViewerServer

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def build_fixture(root: Path) -> tuple[Path, Path, Path]:
    data, predictions, val_predictions = root / "waypop", root / "predictions", root / "val_predictions"
    data.mkdir()
    categories = [
        {"id": 2, "name": "car", "color": [0, 0, 142], "isthing": 1},
        {"id": 20, "name": "road", "color": [128, 64, 128], "isthing": 0},
        {"id": 25, "name": "sky", "color": [70, 130, 180], "isthing": 0},
        {"id": 29, "name": "unknown", "color": [0, 255, 255], "isthing": 1},
    ]
    (data / "categories.json").write_text(json.dumps({"categories": categories}))
    (data / "release.json").write_text(
        json.dumps({"dataset": "WayPOP", "version": "test", "split_variants": {"val": 2, "v1": 2, "xl": 4}})
    )
    sequences = []
    sample_rows: dict[str, list[dict]] = {split: [] for split in ("val", "v1", "xl")}
    prediction_rows: dict[str, list[dict]] = {split: [] for split in ("val", "v1", "xl")}
    for source in ("validation-source", "test-source"):
        for clip in ("clip_00", "clip_01"):
            for domain in ("v1", "xl"):
                split = "val" if source == "validation-source" and domain == "v1" else domain
                sequence = f"{source}__{clip}__{domain}"
                sequences.append({"sequence_id": sequence, "source_sequence": source, "clip_id": clip, "asset_source": domain, "split": split})
                divisor = 1200
                for camera in CAMERAS:
                    stem = "1000000_000001"
                    relative = Path(sequence) / camera / f"{stem}.png"
                    sample = {
                        "sample_id": str(relative.with_suffix("")),
                        "sequence_id": sequence,
                        "source_sequence": source,
                        "clip_id": clip,
                        "asset_source": domain,
                        "split": split,
                        "camera_name": camera,
                        "frame_index": 1,
                        "sequence_position": 0,
                        "timestamp_micros": 1000000,
                        "width": 4,
                        "height": 4,
                        "image_path": str(Path("images") / relative),
                        "pointcloud_path": f"pointcloud/{sequence}/{stem}.npz",
                        "vehicle_pose_local_4x4": np.eye(4).tolist(),
                        "image_pose_local_4x4": np.eye(4).tolist(),
                    }
                    image = data / split / sample["image_path"]
                    image.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image)
                    labels = np.zeros((4, 4), dtype=np.uint32)
                    labels[0] = 29 * divisor + 1
                    labels[1] = 2 * divisor + 2
                    labels[2] = 20 * divisor
                    labels[3] = 25 * divisor
                    if split == "val":
                        sample.update({
                            "panoptic_path": str(Path("panoptic") / relative),
                            "wstq_meta_path": str((Path("wstq_meta") / relative).with_suffix(".npz")),
                            "panoptic_label_divisor": divisor,
                        })
                        label_path = data / split / sample["panoptic_path"]
                        label_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(ids_to_rgb(labels)).save(label_path)
                        aux = data / split / sample["wstq_meta_path"]
                        aux.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(aux, coverage=np.ones((4, 4), dtype=np.uint8), is_tracked_mask=np.ones((4, 4), dtype=np.uint8))
                    sample_rows[split].append(sample)
                    pred_ids = np.zeros((4, 4), dtype=np.uint32)
                    pred_ids[0], pred_ids[1], pred_ids[2], pred_ids[3] = 1, 2, 3, 4
                    prediction_root = val_predictions if split == "val" else predictions
                    pred_path = prediction_root / split / "predictions" / relative
                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(ids_to_rgb(pred_ids)).save(pred_path)
                    prediction_rows[split].append({
                        "file_name": str(relative),
                        "segments_info": [
                            {"id": 1, "category_id": 29},
                            {"id": 2, "category_id": 2},
                            {"id": 3, "category_id": 20},
                            {"id": 4, "category_id": 25},
                        ],
                    })
    write_jsonl(data / "sequences.jsonl", sequences)
    for split, rows in sample_rows.items():
        write_jsonl(data / split / "meta/samples.jsonl", rows)
        prediction_root = val_predictions if split == "val" else predictions
        (prediction_root / split / "predictions.json").write_text(json.dumps({"annotations": prediction_rows[split]}))
    return data, predictions, val_predictions


class PublicDevkitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data, self.predictions, self.val_predictions = build_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_reader_and_rgb_encoding(self) -> None:
        release = WaypopRelease(self.data)
        self.assertEqual([len(release.samples(split)) for split in ("val", "v1", "xl")], [10, 10, 20])
        self.assertEqual(len(release.frames("val", release.sequence_ids("val")[0])), 1)
        ids = np.array([[0, 1, 65536, 2**24 - 1]], dtype=np.uint32)
        np.testing.assert_array_equal(rgb_to_ids(ids_to_rgb(ids)), ids)
        self.assertEqual(stable_instance_color("a", 1), stable_instance_color("a", 1))
        self.assertNotEqual(stable_instance_color("a", 1), stable_instance_color("a", 2))

    def test_validates_directory_and_zip_and_rejects_missing_sample(self) -> None:
        report = validate_submission(self.predictions, self.data, "mv-ospt")
        self.assertEqual(report.required_samples, 30)
        self.assertEqual(validate_submission(self.val_predictions, self.data, "fv-ospt", split="val").required_samples, 2)
        front_report = validate_submission(self.predictions, self.data, "fv-ospt")
        self.assertEqual((front_report.required_samples, front_report.annotations), (6, 30))
        output = self.root / "submission.zip"
        package_submission(self.predictions, output, self.data, "mv-ospt")
        self.assertEqual(validate_submission(output, self.data, "mv-ospt").annotations, 30)

        manifest = self.predictions / "xl/predictions.json"
        payload = json.loads(manifest.read_text())
        payload["annotations"].pop()
        manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(SubmissionError, "Missing 1 required xl predictions"):
            validate_submission(self.predictions, self.data, "mv-ospt")

    def test_rejects_unlisted_pixels_and_unsafe_zip(self) -> None:
        manifest = self.predictions / "v1/predictions.json"
        payload = json.loads(manifest.read_text())
        payload["annotations"][0]["segments_info"].pop(0)
        manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(SubmissionError, "missing from segments_info"):
            validate_submission(self.predictions, self.data, "vi-osps")

        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("../escape.txt", "bad")
        with self.assertRaises(ValueError):
            validate_submission(archive, self.data, "mv-ospt")

        special_archive = self.root / "special.zip"
        special = zipfile.ZipInfo("v1/special")
        special.create_system = 3
        special.external_attr = (stat.S_IFIFO | 0o644) << 16
        with zipfile.ZipFile(special_archive, "w") as output:
            output.writestr(special, "bad")
        with self.assertRaisesRegex(SubmissionError, "Non-regular ZIP entry"):
            validate_submission(special_archive, self.data, "mv-ospt")

    def test_rejects_wrong_dimensions_and_unexpected_files(self) -> None:
        path = next((self.predictions / "v1/predictions").rglob("*.png"))
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(path)
        with self.assertRaisesRegex(SubmissionError, "Expected an RGB PNG of 4x4"):
            validate_submission(self.predictions, self.data, "mv-ospt")
        Image.fromarray(ids_to_rgb(np.ones((4, 4), dtype=np.uint32))).save(path)
        (self.predictions / "unexpected.txt").write_text("extra")
        with self.assertRaisesRegex(SubmissionError, "Unexpected file"):
            validate_submission(self.predictions, self.data, "mv-ospt")

    def test_viewer_uses_public_gt_only_on_val(self) -> None:
        val = Viewer(self.data, "val", self.val_predictions)
        frame = val.frame(val.sequences[0], 0)
        self.assertTrue(frame["has_gt"])
        self.assertTrue(frame["has_predictions"])
        self.assertEqual(len(frame["cameras"]), 5)
        self.assertTrue(val.overlay(0, "gt").startswith(b"\x89PNG"))
        self.assertTrue(val.overlay(0, "pred").startswith(b"\x89PNG"))
        test = Viewer(self.data, "v1", self.predictions)
        self.assertFalse(test.frame(test.sequences[0], 0)["has_gt"])
        with self.assertRaisesRegex(ValueError, "withheld"):
            test.overlay(0, "gt")

    def test_browser_serves_five_camera_frame_and_overlay(self) -> None:
        viewer = Viewer(self.data, "val", self.val_predictions)
        server = _ViewerServer(("127.0.0.1", 0), _Handler)
        server.viewer = viewer
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urllib.request.urlopen(base + "/api/sequences") as response:
                sequences = json.load(response)
            sequence_id = sequences["sequences"][0]["id"]
            with urllib.request.urlopen(base + f"/api/frame?sequence={sequence_id}&index=0") as response:
                frame = json.load(response)
            self.assertEqual(set(frame["cameras"]), set(CAMERAS))
            with urllib.request.urlopen(base + frame["cameras"]["FRONT"]["gt"]) as response:
                self.assertTrue(response.read().startswith(b"\x89PNG"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_packaged_config_matches_public_config(self) -> None:
        for name in ("classes.yaml", "format/coco.yaml"):
            self.assertEqual(
                (ROOT / "config" / name).read_bytes(),
                (ROOT / "src/waypop_devkit/config" / name).read_bytes(),
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("torchmetrics"),
        "Install the eval extra to test scoring",
    )
    def test_three_public_validation_tasks(self) -> None:
        from waypop_devkit.evaluate import evaluate_validation

        for task, metric, count in (("mv-ospt", "wstq", 10), ("fv-ospt", "stq", 2), ("vi-osps", "pq", 10)):
            with self.subTest(task=task):
                report = evaluate_validation(self.data, self.val_predictions, task)
                self.assertEqual((report["metric"], report["num_samples"]), (metric, count))
                self.assertAlmostEqual(report["scopes"]["all"]["unknown"]["STQ" if task != "vi-osps" else "PQ"], 1.0)


if __name__ == "__main__":
    unittest.main()
