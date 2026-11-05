# Reconstruction example

This optional component demonstrates the camera object-insertion stage used to
build WayPOP. It uses Waymo LiDAR geometry for placement and camera occlusion,
then renders objects into five synchronized camera views.

The prepared example uses three source frames (tokens `2,4,6`) from validation
sequence `14081240615915270380__clip_00__v1` and eight saved Objaverse v1.0
assignments. Tokens `2,4,6` correspond to public release `frame_index`
values `1,3,5`; their Waymo timestamps are checked when the source is prepared.

## Required inputs

1. The original Waymo validation TFRecord named
   `segment-14081240615915270380_4399_000_4419_000_with_camera_labels.tfrecord`.
2. The eight Objaverse v1.0 GLBs referenced by the example `objects.json`,
   under an asset root containing `hf-objaverse-v1/glbs/`.
3. The example metadata directory containing `objects.json` and
   `placement.json`.

The TFRecord and GLBs must be obtained from their respective [Waymo](https://waymo.com/open/)
and [Objaverse](https://objaverse.allenai.org/) distributions. Download the
[example metadata ZIP](https://drive.google.com/file/d/1r2uLqR1oQgfrNtXRqAvWtRjYS3tCNtC9/view?usp=sharing)
separately from this code. The archive's SHA-256 is
`47b530fe559fdfdde50d45e9e21c7439cb3bbccea4dd26993f49a976cff13423`.
Extract the archive and run `sha256sum -c SHA256SUMS` inside its
`waypop_reconstruction_example_val/` directory. Use its `val_140812_clip00/`
subdirectory as `--example-inputs`. No source images, LiDAR scans, GLBs, test
annotations, or generated renders are included in this code folder.

## Environment

Use Python 3.10 and Blender 3.6.13. The renderer has separate dependencies from
the core devkit. The pinned requirements match the tested environment; the Waymo
SDK wheel is installed without its dependency resolver because those compatible
packages are pinned here.

```bash
cd reconstruction
uv venv --python 3.10 .venv-reconstruction
source .venv-reconstruction/bin/activate
uv pip install -r requirements.txt
uv pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.7
```

Before running, confirm that `waymo_open_dataset.wdl_limited.camera.ops.py_camera_model_ops`
imports and that Blender can use the selected GPU for rendering. Set
`WAYPOP_RENDER_PYTHON` if the renderer should use a different Python executable.

## Download the example GLBs

The eight GLBs come from the [Objaverse 1.0 `glbs/` directory](https://huggingface.co/datasets/allenai/objaverse/tree/main/glbs).
Their exact UIDs are the `fileIdentifier` values in the downloaded `objects.json`.
Set the path to the extracted example, then install the official Objaverse
downloader in the activated environment:

```bash
export WAYPOP_EXAMPLE_INPUTS=/path/to/waypop_reconstruction_example_val/val_140812_clip00
uv pip install objaverse==0.1.7
python - <<'PY'
import json
import os
from pathlib import Path
import objaverse

objects = json.loads((Path(os.environ['WAYPOP_EXAMPLE_INPUTS']) / 'objects.json').read_text())
uids = [item['fileIdentifier'] for item in objects]
paths = objaverse.load_objects(uids=uids, download_processes=4)
if set(paths) != set(uids):
    raise RuntimeError(f'Missing Objaverse assets: {set(uids) - set(paths)}')
for uid in uids:
    print(paths[uid])
PY
```

Objaverse stores them under `~/.objaverse/hf-objaverse-v1/glbs/`, for example
`~/.objaverse/hf-objaverse-v1/glbs/000-027/b29b03e9703848a99a32ee9d9b4dd7ad.glb`.
Pass `--asset-root "$HOME/.objaverse"` to `run_example.sh` so the paths in
`objects.json` resolve to these downloaded files.

## Prepare and render

```bash
bash run_example.sh \
  --source-tfrecord /path/to/segment-14081240615915270380_4399_000_4419_000_with_camera_labels.tfrecord \
  --asset-root "$HOME/.objaverse" \
  --example-inputs "$WAYPOP_EXAMPLE_INPUTS" \
  --output-root /path/to/example_run

bash run_example.sh \
  --source-tfrecord /path/to/segment-14081240615915270380_4399_000_4419_000_with_camera_labels.tfrecord \
  --asset-root "$HOME/.objaverse" \
  --example-inputs "$WAYPOP_EXAMPLE_INPUTS" \
  --output-root /path/to/example_run \
  --blender /path/to/blender-3.6.13/blender \
  --mode render
```

The first command is the default `prepare` mode. It checks all eight GLBs and
the saved object-to-candidate assignments, verifies the requested Waymo
timestamps, and writes a three-frame mini TFRecord under `example_run/inputs/`.
It stops reading the source after token 6. Each invocation regenerates this
mini TFRecord from the supplied source, so a stale copy cannot be reused
silently.

The second command performs the render. Each object is locked to its own saved
placement point, including when the placement plan has gaps in its slot numbers.
The renderer rejects a sampled or retried position that differs from the saved
assignment. It writes intermediate layers under `example_run/outputs/`, merged
camera images and label products under `example_run/images/`, and Blender logs
under `example_run/blender_logs/`. Successful output has three frames from each
of the five cameras. The intermediate composites are JPEGs; these are not the
final public-release PNG images.

To resume an interrupted run, add `--retry`. Use `--skip-blender-render` only
with `--retry` and a complete saved render; the wrapper checks all eight saved
placements and all 240 expected Blender layers before reuse. An older run with
the wrong object placements is rejected. To start over, choose a new empty
`--output-root` and omit `--retry`.

The wrapper runs the independent output check before reporting success. To
check an existing run again, use:

```bash
python scripts/check_example_outputs.py \
  --output-root /path/to/example_run \
  --objects "$WAYPOP_EXAMPLE_INPUTS/objects.json" \
  --segment segment-14081240615915270380_4399_000_4419_000_with_camera_labels
```

It checks saved positions, render layers, 15 merged RGB/label sets, panoptic
encoding, and visibility of all eight object IDs. For the six validation clips,
`scripts/build_validation_manifest.py` maps public sample metadata to original
Waymo frame tokens; only the three-frame example has prepared assignments and
placements.

## License

The reconstruction code is covered by the devkit's [GPL-3.0-only license](../LICENSE).
