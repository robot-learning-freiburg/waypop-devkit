# WayPOP Devkit

## Overview

- [Introduction](#introduction)
- [Citation](#citation)
- [Setup](#setup)
- [Download the data](#download-the-data)
- [Folder structure](#folder-structure)
- [Read a sample](#read-a-sample)
- [Ground truth format](#ground-truth-format)
- [Visualization](#visualization)
- [Evaluation](#evaluation)
- [Prediction format](#prediction-format)
- [Submission checks and packaging](#submission-checks-and-packaging)
- [Classes](#classes)
- [Evaluation metrics](#evaluation-metrics)
- [Reconstruction example](#reconstruction-example)
- [License](#license)

## Introduction

<p align="center">
  <img src="assets/teaser.gif" width="960" alt="Five synchronized camera views with colored panoptic overlays">
</p>

<p align="center">
  <a href="https://waypop.cs.uni-freiburg.de/">Project Page</a> ·
  <a href="https://arxiv.org/abs/2611.05801">Paper</a> ·
  <a href="https://waypop.cs.uni-freiburg.de/gallery/">Gallery</a>
</p>

[WayPOP](https://huggingface.co/datasets/mohangrim/WayPOP) is a panoramic open-set panoptic segmentation and tracking benchmark built from Waymo PVPS driving sequences with inserted 3D objects from Objaverse 1.0 and Objaverse-XL. Methods assign each pixel a semantic label and a segment ID, covering 22 known classes and a single unknown class for objects outside that taxonomy. The tracking tasks additionally require persistent object identities over time, and the multi-view task requires consistent identities across camera views.

This devkit reads the release manifests, visualizes the five camera streams, evaluates predictions on public validation labels, and checks test submissions before upload.  All three tasks use the same prediction encoding and report metrics separately for known classes and unknown objects:

| Task | Input views | Metric | Identity scope | Codabench |
| --- | --- | --- | --- | --- |
| Multi-View Open-Set Panoptic Tracking (`mv-ospt`) | All five cameras | wSTQ | Across time and cameras within a sequence | [Benchmark](https://www.codabench.org/competitions/18460/) |
| Frontal-View Open-Set Panoptic Tracking (`fv-ospt`) | `FRONT` | STQ | Across time within a sequence | [Benchmark](https://www.codabench.org/competitions/18458/) |
| View-Independent Open-Set Panoptic Segmentation (`vi-osps`) | Individual images from all five cameras | PQ | Each image independently | [Benchmark](https://www.codabench.org/competitions/18459/) |

The six validation sequence variants have public panoptic labels. Test images belong to WayPOP-v1 (`v1`) or WayPOP-XL (`xl`); their labels are held by the benchmark evaluator. Test results for V1 and XL are reported separately.

## Citation

If you use WayPOP or this devkit, please cite the benchmark paper:

**[WayPOP: A Panoramic Open-Set Panoptic Tracking Benchmark](https://arxiv.org/abs/2611.05801)** (NeurIPS, 2026)

```bibtex
@inproceedings{waypop2026,
  title={WayPOP: A Panoramic Open-Set Panoptic Tracking Benchmark},
  author={Mohan, Rohit and Luz, Maximilian and Chowdhury, Swarnava and Lu, Yao and Drews, Florian and N{\"u}rnberg, Thomas and Miron, Yakov and Tombari, Federico and Gasperini, Stefano and Valada, Abhinav},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use Python 3.12 from the devkit directory:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[eval]"
```

The `eval` extra installs the packages needed for wSTQ, STQ, and PQ scoring. If you only need the reader, submission checker, and browser viewer, install `uv pip install -e .` instead. Local scoring defaults to CPU; pass `--device cuda` to use a compatible GPU.

## Download the data

After your access to the [WayPOP dataset](https://huggingface.co/datasets/mohangrim/WayPOP) is approved, download and reconstruct the release:

```bash
uv pip install huggingface_hub
hf auth login
hf download mohangrim/WayPOP --repo-type dataset --local-dir waypop_download
python waypop_download/reconstruct_release.py waypop_download waypop
python waypop/tools/verify_download.py waypop
```

All commands below use the extracted `waypop/` directory as `--data-root`.

## Folder structure

```text
waypop/
  release.json
  categories.json
  sequences.jsonl
  val/
    meta/samples.jsonl
    images/SEQUENCE_ID/CAMERA/FRAME.png
    panoptic/SEQUENCE_ID/CAMERA/FRAME.png
    wstq_meta/SEQUENCE_ID/CAMERA/FRAME.npz
    pointcloud/SEQUENCE_ID/FRAME.npz
  v1/
    meta/samples.jsonl
    images/SEQUENCE_ID/CAMERA/FRAME.png
    pointcloud/SEQUENCE_ID/FRAME.npz
  xl/
    meta/samples.jsonl
    images/SEQUENCE_ID/CAMERA/FRAME.png
    pointcloud/SEQUENCE_ID/FRAME.npz
```

Each `meta/samples.jsonl` row describes one camera image. Its `sequence_id`, `timestamp_micros`, `camera_name`, `width`, and `height` identify the observation and its expected prediction size. The row supplies vehicle and image poses, together with paths to the image, vehicle-frame XYZ point cloud and, on validation, the panoptic PNG and wSTQ metadata. Paths are relative to the split directory.
Read these paths from the manifest; `v1` and `xl` are directory IDs for the test splits named WayPOP-v1 and WayPOP-XL in the paper.

Check the extracted release with:

```bash
waypop inspect --data-root waypop --split val
```

## Read a sample

```python
from waypop_devkit.data import WaypopRelease
from waypop_devkit.labels import decode_ground_truth, load_ids

release = WaypopRelease("waypop")
sample = release.samples("val")[0]
image_path = sample.path("image_path")
panoptic = load_ids(sample.path("panoptic_path"))
semantic, instance = decode_ground_truth(
    panoptic, sample.row["panoptic_label_divisor"]
)
print(sample.sample_id, image_path, semantic.shape, instance.shape)
```

`examples/read_validation_sample.py` also loads the aligned XYZ point cloud. Test samples can be read in the same way, except that their panoptic and wSTQ files are private.

## Ground truth format

Each validation `panoptic/*.png` is an RGB PNG. Its channels encode one unsigned 24-bit panoptic ID per pixel:

```python
panoptic_id = R + 256 * G + 65536 * B
semantic_id = panoptic_id // panoptic_label_divisor
instance_id = panoptic_id % panoptic_label_divisor
```

Read `panoptic_label_divisor` from the corresponding validation sample row; it can differ between sequences. `panoptic_id == 0` means void. Semantic IDs are the raw category IDs in [Classes](#classes), not contiguous training IDs. Thing instances have positive instance IDs; stuff uses instance ID zero. The prediction encoding below is different: its PNG stores a segment ID, while `predictions.json` supplies the semantic category.

Each validation `wstq_meta/*.npz` contains two `height × width` arrays:

| Key | Type | Meaning |
| --- | --- | --- |
| `coverage` | `uint8` | Number of camera views covering the pixel, from 0 to 5. wSTQ weights covered pixels by `1 / coverage`. |
| `is_tracked_mask` | `uint8` | Whether the pixel contributes a ground-truth instance identity to wSTQ association. |

These arrays are reference annotations for multi-view scoring. Prediction files do not contain them.

## Visualization

Open the five-camera browser viewer on validation images and labels:

```bash
waypop visualize --data-root waypop --split val --open-browser
```

With `--pred-root predictions`, the viewer shows ground truth and predictions in synchronized rows. For test images, select `--split v1` or `--split xl` and provide predictions; the viewer shows prediction overlays without private labels. Instance colors remain stable within a sequence, and the overlay starts at 50% opacity. The viewer supports sequence selection, frame stepping, and playback. Use **FPS** to set the playback speed; if a frame takes longer to load, playback waits and keeps the previous frame visible until all camera views and overlays are ready.

The server binds to `127.0.0.1:8080` by default. If the data are on a remote machine, forward the server port over SSH. Use `--port` to choose a different server port.

## Evaluation

Use the same `predictions.json` and RGB PNG format on validation as for test. Score one task at a time:

```bash
waypop evaluate \
  --data-root waypop \
  --pred-root predictions \
  --task mv-ospt \
  --output-json results/mv-ospt-val.json
```

Replace the task with `fv-ospt` or `vi-osps` for the other evaluations. `fv-ospt` scores only `FRONT`; the other two score all five cameras. The result JSON contains `scopes.all.known`, `scopes.all.unknown`, and `scopes.all.per_class` for the selected metric. Metric values are fractions in `[0, 1]`; the hosted leaderboard displays percentages. Local evaluation uses public `val` labels; test scores are produced by the benchmark server.

The default `auto` engine uses compact PQ scoring on CPU and compact scoring for all three tasks on CUDA. CPU tracking uses the reference evaluator. Add `--device cuda` to use a GPU, or `--engine reference` to compare against the original scoring path. The result JSON records the engine used. Submission checks and the final metric summaries are the same for both engines.

## Prediction format

Create one `predictions.json` and one RGB prediction PNG per required image under each split:

```text
predictions/
  val/
    predictions.json
    predictions/SEQUENCE_ID/CAMERA/FRAME.png
  v1/
    predictions.json
    predictions/SEQUENCE_ID/CAMERA/FRAME.png
  xl/
    predictions.json
    predictions/SEQUENCE_ID/CAMERA/FRAME.png
```

For a test submission, include `v1/` and `xl/`; `val/` is only for local evaluation. Each PNG has the input image's dimensions and stores a 24-bit segment ID:

```python
segment_id = R + 256 * G + 65536 * B
```

ID zero is void. Each nonzero ID must appear in that image's `segments_info`. The JSON manifest gives the raw semantic `category_id` for each segment:

```json
{
  "annotations": [
    {
      "file_name": "SEQUENCE_ID/FRONT/FRAME.png",
      "segments_info": [{"id": 1, "category_id": 29}]
    }
  ]
}
```

`file_name` is relative to the split's `predictions/` directory. It corresponds to the sample's `image_path` after removing `images/`. For MV-OSPT, assign the same segment ID to an object across time and camera views within its sequence. FV-OSPT requires consistency across front-camera frames. VI-OSPS requires only per-image segment IDs. Use raw category ID `29` for unknown objects; `categories.json` defines all released categories.

## Submission checks and packaging

Check an extracted prediction directory or a ZIP before submitting:

```bash
waypop check-submission --data-root waypop --input predictions --task mv-ospt
waypop check-submission --data-root waypop --input submission.zip --task mv-ospt
```

Create a task-specific ZIP after checking it:

```bash
waypop package-submission \
  --data-root waypop \
  --pred-root predictions \
  --output submission-mv-ospt.zip \
  --task mv-ospt
```

The ZIP has `v1/` and `xl/` at its root. Upload it to the [Codabench competition for the selected task](#introduction). The checker verifies required samples and camera coverage, PNG dimensions and encoding, segment metadata, categories, tracking-ID category consistency, and safe file paths. It cannot assess accuracy without test labels. `fv-ospt` needs `FRONT` predictions; indexed predictions for the other cameras are accepted but ignored by its metric. `examples/make_void_predictions.py` creates an all-void bundle for testing the file format.

## Classes

Predictions and decoded validation labels use these raw category IDs. The evaluator maps them internally to contiguous class indices:

| ID | Category | Type |
| ---: | --- | --- |
| 2 | car | thing |
| 3 | truck | thing |
| 4 | bus | thing |
| 5 | other_large_vehicle | thing |
| 6 | bicycle | stuff |
| 7 | motorcycle | stuff |
| 8 | trailer | thing |
| 9 | pedestrian | thing |
| 10 | cyclist | thing |
| 11 | motorcyclist | stuff |
| 14 | construction_cone_pole | stuff |
| 15 | pole | stuff |
| 16 | pedestrian_object | stuff |
| 17 | sign | stuff |
| 18 | traffic_light | stuff |
| 19 | building | stuff |
| 20 | road | stuff |
| 21 | lane_marker | stuff |
| 22 | road_marker | stuff |
| 23 | sidewalk | stuff |
| 24 | vegetation | stuff |
| 25 | sky | stuff |
| 29 | unknown (`ood`) | thing |

Category `29` is the unknown class; the others are known. ID zero is void. Predictions may use only the listed nonzero category IDs. The `thing` and `stuff` designations follow the benchmark's scoring configuration.

## Evaluation metrics

**MV-OSPT — weighted Segmentation and Tracking Quality (wSTQ).** Semantic quality is mean intersection-over-union (mIoU), and association quality (AQ) measures whether predicted thing identities stay consistent across the whole sequence and its five camera views. The score is `wSTQ = sqrt(mIoU × AQ)`. Pixels visible in multiple views receive inverse-coverage weights, so the same surface is not counted five times. `is_tracked_mask` controls which ground-truth pixels contribute identities to AQ; semantic segmentation is still scored on valid labeled pixels. The report includes `STQ`, `mIoU`, and `AQ` for known and unknown classes.

**FV-OSPT — Segmentation and Tracking Quality (STQ).** This uses the `FRONT` stream and the same `STQ = sqrt(mIoU × AQ)` decomposition without cross-camera weighting. An object must retain its ID across front-camera frames in its sequence. The report includes known and unknown `STQ`, `mIoU`, and `AQ`.

**VI-OSPS — Panoptic Quality (PQ).** Each camera image is evaluated independently, so tracking consistency is not part of this metric. Predicted and ground-truth segments of the same class match when their intersection-over-union exceeds 0.5. For each class:

```text
PQ = sum IoU(matched pairs) / (TP + 0.5 * FP + 0.5 * FN)
SQ = sum IoU(matched pairs) / TP
RQ = TP / (TP + 0.5 * FP + 0.5 * FN)
PQ = SQ * RQ
```

The report includes known and unknown `PQ`, `SQ`, and `RQ`, plus per-class results.

## Reconstruction example

The optional [reconstruction example](reconstruction/README.md) demonstrates the geometry-guided object-insertion stage on three public-validation source frames. It uses a separate environment and separately acquired source inputs. It illustrates the original rendering stage; reproducing the final release images also requires later composition and quality-review steps.

## License

The WayPOP devkit source code is licensed under [GNU GPL version 3.0 only](LICENSE).
