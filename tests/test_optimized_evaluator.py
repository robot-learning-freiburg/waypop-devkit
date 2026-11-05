"""Compare accelerated public validation scoring with the reference path."""

from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from test_devkit import build_fixture

EVAL_DEPS = bool(
    importlib.util.find_spec("torch") and importlib.util.find_spec("torchmetrics")
)


@unittest.skipUnless(EVAL_DEPS, "Install the eval extra to test scoring")
class OptimizedEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.data_root, _, self.prediction_root = build_fixture(
            Path(self.temporary.name)
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_nested_close(self, expected, actual) -> None:
        if isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assert_nested_close(expected[key], actual[key])
        elif isinstance(expected, float):
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual))
            else:
                self.assertTrue(
                    math.isclose(expected, actual, rel_tol=5e-12, abs_tol=1e-10),
                    (expected, actual),
                )
        else:
            self.assertEqual(expected, actual)

    def test_fast_decoder_matches_reference_maps(self) -> None:
        from waypop_eval.benchmark import (
            BenchmarkPredictionResolver,
            BenchmarkRegistry,
            build_scope_samples,
            load_collection_data,
        )
        from waypop_eval.optimized.fast_loading import prediction, target

        config = Path(__file__).resolve().parents[1] / "src/waypop_devkit/config"
        specs = BenchmarkRegistry.from_json(self.data_root).select(
            collection_ids=("val",)
        )
        meta, collections = load_collection_data(specs, config / "classes.yaml")
        predictions = BenchmarkPredictionResolver(
            specs, self.prediction_root, meta, "coco", config / "format"
        )
        for sample in build_scope_samples(collections, collection_ids=("val",)):
            np.testing.assert_array_equal(
                target(sample), sample.load_panoptic().numpy()
            )
            np.testing.assert_array_equal(
                prediction(predictions[sample]),
                predictions[sample].load_panoptic().numpy(),
            )

    def test_cpu_pq_matches_reference_and_tracking_falls_back(self) -> None:
        from waypop_devkit.evaluate import evaluate_validation

        reference = evaluate_validation(
            self.data_root, self.prediction_root, "vi-osps", engine="reference"
        )
        optimized = evaluate_validation(
            self.data_root, self.prediction_root, "vi-osps", engine="optimized"
        )
        self.assertEqual(
            (reference["engine"], optimized["engine"]), ("reference", "optimized")
        )
        self.assert_nested_close(reference["scopes"], optimized["scopes"])
        for task in ("fv-ospt", "mv-ospt"):
            self.assertEqual(
                evaluate_validation(self.data_root, self.prediction_root, task)[
                    "engine"
                ],
                "reference",
            )
            with self.assertRaisesRegex(ValueError, "No optimized"):
                evaluate_validation(
                    self.data_root, self.prediction_root, task, engine="optimized"
                )

    def test_distinct_domains_keep_separate_scope_scores(self) -> None:
        from waypop_eval.benchmark import (
            BenchmarkPredictionResolver,
            BenchmarkRegistry,
            build_scope_samples,
            evaluate_task_scopes,
            load_collection_data,
        )

        config = Path(__file__).resolve().parents[1] / "src/waypop_devkit/config"
        specs = BenchmarkRegistry.from_json(self.data_root).select(
            collection_ids=("val",)
        )
        meta, collections = load_collection_data(specs, config / "classes.yaml")
        resolver = BenchmarkPredictionResolver(
            specs, self.prediction_root, meta, "coco", config / "format"
        )
        front = [
            sample
            for sample in build_scope_samples(collections, collection_ids=("val",))
            if sample.camera == "FRONT"
        ]
        first, second = front
        predictions = {
            first.prediction_key: resolver[first],
            second.prediction_key: resolver[second],
        }
        blank_path = Path(self.temporary.name) / "blank_prediction.png"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(blank_path)
        predictions[second.prediction_key] = replace(
            predictions[second.prediction_key],
            panoptic_path=blank_path,
            meta={"segments_info": {}},
        )

        class PredictionMap:
            def __getitem__(self, sample):
                return predictions[sample.prediction_key]

        samples = [
            replace(first, set_id="v1", source_domain="v1"),
            replace(second, set_id="xl", source_domain="xl"),
        ]
        reports = {
            engine: evaluate_task_scopes(
                samples, meta, PredictionMap(), "pq", "cpu", "segment", engine=engine
            )
            for engine in ("reference", "optimized")
        }
        self.assert_nested_close(reports["reference"], reports["optimized"])
        self.assertNotEqual(
            reports["optimized"]["source_domain"]["v1"]["unknown"]["PQ"],
            reports["optimized"]["source_domain"]["xl"]["unknown"]["PQ"],
        )

    def test_cuda_tasks_match_reference(self) -> None:
        import torch
        from waypop_devkit.evaluate import evaluate_validation

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        for task in ("vi-osps", "fv-ospt", "mv-ospt"):
            with self.subTest(task=task):
                reference = evaluate_validation(
                    self.data_root,
                    self.prediction_root,
                    task,
                    device="cuda",
                    engine="reference",
                )
                optimized = evaluate_validation(
                    self.data_root,
                    self.prediction_root,
                    task,
                    device="cuda",
                    engine="optimized",
                )
                self.assert_nested_close(reference["scopes"], optimized["scopes"])


if __name__ == "__main__":
    unittest.main()
