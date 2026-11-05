import bpy
from mathutils import Matrix, Vector
import mathutils
from bpy_extras.object_utils import world_to_camera_view
from inspect import signature
import numpy as np
try:
    import ephem
except ImportError:
    ephem = None
import time
import datetime
from datetime import datetime
from math import pi
import os
import json
import sys
import math
import threading
import multiprocessing
import tempfile
import subprocess
import shutil

from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if "numpy._core.multiarray" not in sys.modules:
    import numpy.core.multiarray as _np_multiarray
    sys.modules["numpy._core.multiarray"] = _np_multiarray

_BLENDER_CPU_THREAD_LOGGED = False


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _lerp(a, b, t):
    return a + (b - a) * t


def sun_params_linear(cloud_cover,
                      direct_norm,
                      min_energy=0.09,
                      max_energy=0.8,
                      min_angle=np.radians(0.53),
                      max_angle=np.radians(10)):
    """
    Compute sun energy and angle using a simple linear mapping.

    - cloud_cover: 0..100
    - direct_norm: 0..1 (direct illumination normalized). If outside range it'll be clamped.
    Returns: (energy, angle_in_radians)

    Defaults are conservative; change min_energy/max_energy or min_angle/max_angle
    at call sites to tune for your dataset. The user asked to leave ranges editable.
    """
    cloud_frac = _clamp(
        cloud_cover / 100.0 if cloud_cover is not None else 0.0, 0.0, 1.0)
    direct = _clamp(direct_norm if direct_norm is not None else 0.0, 0.0, 1.0)

    energy = _lerp(min_energy, max_energy, direct * (1.0 - cloud_frac))
    energy = _clamp(energy, min_energy, max_energy)

    angle = _lerp(min_angle, max_angle, cloud_frac)
    return energy, angle


def get_gpu_metrics(gpu_id):
    if gpu_id is None:
        gpu_id = 0

    try:

        command = [
            'nvidia-smi', '-i',
            str(gpu_id),
            '--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ]

        result = subprocess.run(command,
                                check=True,
                                capture_output=True,
                                text=True)

        utilization, mem_used, mem_total = result.stdout.strip().split(',')

        gpu_metrics = {
            f"GPU_{gpu_id}_Compute(%)": float(utilization),
            f"GPU_{gpu_id}_Memory_Used(MB)": float(mem_used),
            f"GPU_{gpu_id}_Memory_Total(MB)": float(mem_total)
        }
        return gpu_metrics
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:

        print(f"nvidia-smi command failed: {e}", file=sys.stderr)
        return {}


def resource_monitor_process(gpu_id, interval, log_file_path, stop_event):
    import time, json
    logs = []
    while not stop_event.is_set():
        metrics = get_gpu_metrics(gpu_id)
        if metrics:
            metrics['timestamp'] = time.time()
            logs.append(metrics)

            try:
                with open(log_file_path, 'w') as f:
                    json.dump(logs, f)
            except Exception as e:
                print(f"Resource monitor log write error: {e}",
                      file=sys.stderr)
        time.sleep(interval)


def _parse_visible_cuda_devices():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    parsed = []
    for token in raw.split(","):
        token = token.strip()
        if not token or token == "-1":
            continue
        try:
            parsed.append(int(token))
        except ValueError:

            return []
    return parsed


def _resolve_cycles_gpu_index(requested_gpu_id, num_gpu_devices):
    """
    Resolve Blender-local GPU index from a possibly global worker gpu_id.

    Each worker may set CUDA_VISIBLE_DEVICES=<global_id>, which reindexes the
    visible device to local index 0 inside Blender.
    """
    if num_gpu_devices <= 0:
        return None, "no GPU devices reported by Blender Cycles"

    if requested_gpu_id is None:
        return 0, "no gpu_id requested; selecting first visible GPU"

    try:
        requested = int(requested_gpu_id)
    except Exception:
        return 0, f"invalid gpu_id '{requested_gpu_id}'; selecting first visible GPU"

    if 0 <= requested < num_gpu_devices:
        return requested, "gpu_id already matches local visible-device index"

    visible_global_ids = _parse_visible_cuda_devices()
    if visible_global_ids:
        if requested in visible_global_ids:
            local_idx = visible_global_ids.index(requested)
            if 0 <= local_idx < num_gpu_devices:
                return (
                    local_idx,
                    f"mapped global gpu_id={requested} via CUDA_VISIBLE_DEVICES={visible_global_ids}",
                )
        if len(visible_global_ids) == 1 and num_gpu_devices >= 1:
            return (
                0,
                f"single visible GPU via CUDA_VISIBLE_DEVICES={visible_global_ids}; using local index 0",
            )

    if num_gpu_devices == 1:
        return 0, "single GPU detected by Blender; using local index 0"

    return None, f"gpu_id={requested} does not map to visible Blender GPU devices"


IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)


def _resolve_reference_dimension(dims_xyz, policy="auto"):
    dims = np.asarray(dims_xyz, dtype=np.float64).reshape(-1)
    if dims.size != 3:
        return 1.0, "fallback"
    dims = np.clip(np.abs(dims), 1e-6, None)
    max_dim = float(np.max(dims))
    median_dim = float(np.median(dims))
    geom_dim = float(np.cbrt(np.prod(dims)))
    policy = str(policy or "auto").strip().lower()
    if policy == "max_dim":
        return max_dim, "max_dim"
    if policy == "median_dim":
        return median_dim, "median_dim"
    if policy == "geom_mean":
        return geom_dim, "geom_mean"
    if policy == "height":
        return float(dims[2]), "height"

    auto_ref = 0.55 * max_dim + 0.30 * median_dim + 0.15 * geom_dim
    return float(max(auto_ref, 1e-6)), "auto"


def normalize_scene(
    ood_id,
    scale,
    render_path,
    size_multiplier=1.0,
    eval=False,
    target_ref_dim_m=None,
    scale_policy="auto",
    description="",
) -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    """
    if len(list(get_scene_root_objects())) > 1:

        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    raw_dims = np.abs(np.asarray(bbox_max) - np.asarray(bbox_min))
    reference_dim, resolved_policy = _resolve_reference_dimension(
        raw_dims, policy=scale_policy)

    if target_ref_dim_m is None:
        try:
            target_ref_dim = float(scale)
        except Exception:
            target_ref_dim = 1.0
    else:
        try:
            target_ref_dim = float(target_ref_dim_m)
        except Exception:
            target_ref_dim = float(scale)
    if not np.isfinite(target_ref_dim) or target_ref_dim <= 0:
        target_ref_dim = 1.0

    scale_factor = target_ref_dim / max(reference_dim, 1e-6)

    log_file_path = f'{render_path}/ood_{ood_id}.txt'

    with open(log_file_path, "w") as f:

        original_stdout = sys.stdout
        sys.stdout = f

        print(f"bbox_min: {bbox_min}, bbox_max: {bbox_max}")
        dimensions = bbox_max - bbox_min
        print(f"dimensions: {dimensions}")
        print(
            f"size_normalization: policy={resolved_policy}, "
            f"reference_dim={reference_dim:.6f}, target_ref_dim={target_ref_dim:.6f}, "
            f"scale_factor={scale_factor:.6f}, base_scale={float(scale):.6f}, "
            f"size_multiplier={float(size_multiplier):.6f}, eval={bool(eval)}, "
            f"description={str(description)[:120]}")

        for obj in get_scene_root_objects():
            obj.scale = obj.scale * scale_factor * size_multiplier

        bpy.context.view_layer.update()

        bbox_min_after, bbox_max_after = scene_bbox()
        dx = bbox_max_after.x - bbox_min_after.x
        dy = bbox_max_after.y - bbox_min_after.y
        dz = bbox_max_after.z - bbox_min_after.z
        volume = dx * dy * dz
        print(f"Volume of bbox after initial scaling: {volume}")

        while volume < 0.005:
            print(
                f"Retrying... Volume too small: {volume}. Scaling up by 1.5.")
            for obj in get_scene_root_objects():
                obj.scale *= 1.5
            bpy.context.view_layer.update()
            bbox_min_after, bbox_max_after = scene_bbox()
            dx = bbox_max_after.x - bbox_min_after.x
            dy = bbox_max_after.y - bbox_min_after.y
            dz = bbox_max_after.z - bbox_min_after.z
            volume = dx * dy * dz

        sys.stdout = original_stdout

    print(f"Logs written to {log_file_path}")

    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset

    camera = bpy.data.objects.get("Camera")
    if camera:
        camera.parent = None


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def scene_bbox(single_obj: Optional[bpy.types.Object] = None,
               ignore_matrix: bool = False) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf, ) * 3
    bbox_max = (-math.inf, ) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """

    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material, do_unlink=True)

    for texture in list(bpy.data.textures):
        if texture.users == 0:
            bpy.data.textures.remove(texture, do_unlink=True)

    for image in bpy.data.images:
        if image.name != 'Render Result':
            bpy.data.images.remove(image, do_unlink=True)

    for area in bpy.context.screen.areas:
        if area.type == 'IMAGE_EDITOR':
            area.spaces.active.image = None


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def set_camera_from_matrix(transformation_matrix,
                           projection,
                           image_width=None,
                           image_height=None,
                           spot=False):
    camera = bpy.data.objects.get('Camera')
    if camera is None or camera.type != 'CAMERA':
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.name = 'Camera'

    fx, fy, cx, cy = projection

    transformation_matrix = Matrix(transformation_matrix)
    rotation_matrix = transformation_matrix.to_3x3()
    translation_vector = transformation_matrix.to_translation()
    euler_angles = rotation_matrix.to_euler("XYZ")

    sensor_width_mm = 35.0
    if image_width is None or int(image_width) <= 0:
        image_width = max(1, int(round(float(cx) * 2.0)))
    if image_height is None or int(image_height) <= 0:
        image_height = max(1, int(round(float(cy) * 2.0)))
    width = float(max(1, int(image_width)))
    height = float(max(1, int(image_height)))
    focal_length_mm = (float(fx) * sensor_width_mm) / width

    sensor_height_mm = (focal_length_mm * height) / max(float(fy), 1e-6)

    camera.rotation_euler = euler_angles
    camera.location = translation_vector
    camera.data.lens = focal_length_mm
    camera.data.sensor_width = sensor_width_mm
    camera.data.sensor_height = sensor_height_mm
    camera.data.sensor_fit = 'HORIZONTAL'

    camera.data.shift_x = (0.5 - float(cx) / width)

    camera.data.shift_y = (float(cy) - 0.5 * height) / width

    if spot:
        spotlight = bpy.data.objects.get('Spot')
        if spotlight is None:

            bpy.ops.object.light_add(type='SPOT',
                                     radius=5,
                                     align='WORLD',
                                     location=translation_vector,
                                     scale=(1, 1, 1))
            spotlight = bpy.context.object

        spotlight.location = camera.location
        spotlight.rotation_euler = camera.rotation_euler
    bpy.context.view_layer.update()


def create_plane(pose, scale=1):
    transformation_matrix = mathutils.Matrix(pose)
    rotation_matrix = transformation_matrix.to_3x3()
    translation_vector = transformation_matrix.to_translation()
    euler_angles = rotation_matrix.to_euler("XYZ")
    euler_angles[0] += pi / 2
    euler_angles[2] -= pi / 2

    bpy.ops.mesh.primitive_plane_add(size=1,
                                     location=translation_vector,
                                     rotation=euler_angles)
    plane = bpy.context.active_object
    plane.scale = (scale, scale, 1)


def _rotation_from_normal(normal):
    """Return Euler rotation that aligns +Z with the provided normal."""
    n = Vector(normal)
    if n.length < 1e-8:
        n = Vector((0.0, 0.0, 1.0))
    else:
        n.normalize()
    quat = n.to_track_quat('Z', 'Y')
    return quat.to_euler()


def create_shadow_catcher(translation, normal=None, size=40.0):
    """Create a shadow catcher plane aligned to the local ground normal."""
    bpy.context.scene.render.film_transparent = True
    if normal is None:
        normal = (0.0, 0.0, 1.0)
    rotation = _rotation_from_normal(normal)
    bpy.ops.mesh.primitive_plane_add(
        size=float(max(1.0, size)),
        location=translation,
        rotation=rotation,
    )
    plane = bpy.context.active_object
    plane.name = "ShadowCatcher"
    plane.is_shadow_catcher = True


def create_cube(location, scale, rotation):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    cube = bpy.context.active_object
    cube.dimensions = scale
    cube.rotation_euler = rotation


def create_cube_from_matrix(transformation_matrix, dims):
    transformation_matrix = mathutils.Matrix(transformation_matrix)
    rotation_matrix = transformation_matrix.to_3x3()
    translation_vector = transformation_matrix.to_translation()
    bpy.ops.mesh.primitive_cube_add(size=1, location=translation_vector)
    cube = bpy.context.active_object
    cube.dimensions = dims
    euler_angles = rotation_matrix.to_euler("YXZ")
    cube.rotation_euler = euler_angles
    bpy.context.view_layer.update()


def clear_scene():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()


def render_scene(
    path,
    readout_time,
    resolution_x=None,
    resolution_y=None,
    cycles=False,
    cycles_samples_override=None,
    cycles_use_denoising=None,
    cloud_cover=None,
    gpu_id=None,
    direct_illumination=None,
    require_gpu=False,
):

    before, after = {}, {}
    resource_metrics = None
    if gpu_id is not None:
        before = get_gpu_metrics(gpu_id)
        before['timestamp'] = datetime.now().isoformat()

    camera = bpy.data.objects['Camera']
    bpy.context.scene.render.filepath = path
    bpy.context.scene.camera = camera
    if resolution_x is not None:
        bpy.context.scene.render.resolution_x = int(max(1, resolution_x))
    if resolution_y is not None:
        bpy.context.scene.render.resolution_y = int(max(1, resolution_y))

    bpy.context.scene.render.use_motion_blur = True
    bpy.context.scene.render.motion_blur_shutter = readout_time
    cpu_threads_env = os.environ.get("BLENDER_CPU_THREADS", "").strip()
    if cpu_threads_env:
        try:
            cpu_threads = max(1, int(cpu_threads_env))
            bpy.context.scene.render.threads_mode = 'FIXED'
            bpy.context.scene.render.threads = cpu_threads
            global _BLENDER_CPU_THREAD_LOGGED
            if not _BLENDER_CPU_THREAD_LOGGED:
                print(
                    f"Blender CPU thread cap active: render.threads={cpu_threads}"
                )
                _BLENDER_CPU_THREAD_LOGGED = True
        except Exception as e:
            print(
                f"Could not apply BLENDER_CPU_THREADS='{cpu_threads_env}': {e}",
                file=sys.stderr)

    if cycles:
        require_gpu = bool(require_gpu)

        bpy.context.scene.render.engine = 'CYCLES'
        if cycles_use_denoising is None:
            bpy.context.scene.cycles.use_denoising = True
        else:
            bpy.context.scene.cycles.use_denoising = bool(cycles_use_denoising)

        if os.environ.get('BLENDER_FORCE_CPU') == '1':
            if require_gpu:
                raise RuntimeError(
                    "BLENDER_FORCE_CPU=1 while require_gpu=True. "
                    "Refusing to render on CPU.")
            print(
                "BLENDER_FORCE_CPU is set. Forcing CPU rendering for Cycles.")
            bpy.context.scene.cycles.device = 'CPU'
        else:

            prefs = bpy.context.preferences.addons['cycles'].preferences

            override_compute = os.environ.get("BLENDER_CYCLES_COMPUTE_TYPE",
                                              "").strip().upper()
            selected_compute = None
            backend_candidates = []
            if override_compute:
                backend_candidates.append(override_compute)
            backend_candidates.extend(
                ('OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL'))
            tried = set()
            for preferred in backend_candidates:
                if preferred in tried:
                    continue
                tried.add(preferred)
                try:
                    prefs.compute_device_type = preferred
                    selected_compute = preferred
                    break
                except Exception:
                    continue

            try:
                prefs.get_devices()
            except Exception as e:
                print(f"Could not query Cycles devices: {e}", file=sys.stderr)

            devices = getattr(prefs, 'devices', None)
            if not devices:
                msg = 'No Cycles devices found.'
                if require_gpu:
                    raise RuntimeError(
                        f"{msg} require_gpu=True so CPU fallback is disabled.")
                print(f'{msg} Falling back to CPU.', file=sys.stderr)
                bpy.context.scene.cycles.device = 'CPU'
            else:

                for device in devices:
                    try:
                        device.use = False
                    except Exception:
                        pass

                gpu_devices = [
                    d for d in devices
                    if d.type in ('CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL')
                ]

                try:
                    print('Available Cycles devices:',
                          [(d.name, d.type, d.id) for d in devices])
                    print('Detected GPU devices:',
                          [(d.name, d.type, d.id) for d in gpu_devices])
                except Exception:
                    pass

                target_device = None
                resolved_idx, resolve_reason = _resolve_cycles_gpu_index(
                    gpu_id, len(gpu_devices))
                if resolved_idx is not None and gpu_devices:
                    target_device = gpu_devices[resolved_idx]
                    print(
                        "Cycles GPU selection: "
                        f"requested_gpu_id={gpu_id}, selected_local_index={resolved_idx}, reason={resolve_reason}"
                    )
                else:
                    print(
                        "Cycles GPU selection failed: "
                        f"requested_gpu_id={gpu_id}, reason={resolve_reason}",
                        file=sys.stderr,
                    )

                if target_device:
                    try:
                        target_device.use = True

                        for d in gpu_devices:
                            if d.name == target_device.name:
                                d.use = True
                        bpy.context.scene.cycles.device = 'GPU'
                        print(
                            f"Using GPU for Cycles: {target_device.name} (type={target_device.type})"
                        )

                        try:
                            if selected_compute == 'OPTIX' or target_device.type == 'OPTIX':
                                bpy.context.scene.cycles.denoiser = 'OPTIX'
                                print('Cycles denoiser set to OPTIX (GPU).')
                            else:
                                bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
                                print(
                                    'Cycles denoiser set to OpenImageDenoise (CPU).'
                                )
                        except Exception as e:
                            print(f'Could not set denoiser preference: {e}',
                                  file=sys.stderr)
                    except Exception as e:
                        if require_gpu:
                            raise RuntimeError(
                                f"Failed to enable selected GPU device for Cycles: {e}"
                            )
                        print(f"Failed to enable GPU device: {e}",
                              file=sys.stderr)
                        bpy.context.scene.cycles.device = 'CPU'
                else:
                    msg = "No suitable GPU found or specified for Cycles."
                    if require_gpu:
                        raise RuntimeError(
                            f"{msg} require_gpu=True so CPU fallback is disabled."
                        )
                    print(f"Warning: {msg} Falling back to CPU.",
                          file=sys.stderr)
                    bpy.context.scene.cycles.device = 'CPU'

        if cycles_samples_override is not None:
            try:
                target_samples = int(cycles_samples_override)
            except Exception:
                target_samples = 128
        else:
            samples_env = os.environ.get("BLENDER_CYCLES_SAMPLES", "").strip()
            try:
                target_samples = int(samples_env) if samples_env else 128
            except ValueError:
                target_samples = 128
        if target_samples < 1:
            target_samples = 1
        bpy.context.scene.cycles.samples = target_samples
        print(f"Cycles samples set to {target_samples}")

    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    bpy.context.scene.render.film_transparent = True

    try:
        cy = bpy.context.scene.cycles
        print(
            f"Final Cycles settings: device={getattr(cy, 'device', 'UNKNOWN')}, denoiser={getattr(cy, 'denoiser', 'UNKNOWN')}, samples={getattr(cy, 'samples', 'UNKNOWN')}"
        )
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            devices = getattr(prefs, 'devices', [])
            enabled = [(d.name, d.type, d.id) for d in devices
                       if getattr(d, 'use', False)]
            print('Enabled Cycles devices:', enabled)
        except Exception:
            pass
    except Exception as e:
        print(f'Could not log final Cycles settings: {e}', file=sys.stderr)

    if cycles and require_gpu and str(
            getattr(bpy.context.scene.cycles, 'device', '')).upper() != 'GPU':
        raise RuntimeError(
            "Cycles GPU requirement violated: scene.cycles.device is not 'GPU'. "
            "Aborting render instead of CPU fallback.")

    try:
        bpy.ops.render.render(write_still=True)
    except Exception as e:

        print(f"Initial render failed (likely GPU/EGL/CUDA issue): {e}",
              file=sys.stderr)

        error_text = str(e).lower()
        allow_gpu_oom_cpu_fallback = (os.environ.get(
            'BLENDER_ALLOW_GPU_OOM_CPU_FALLBACK', '0') == '1'
                                      and 'out of gpu memory' in error_text)

        if cycles and (not require_gpu or allow_gpu_oom_cpu_fallback):
            try:
                if allow_gpu_oom_cpu_fallback:
                    print(
                        "Attempting CPU fallback render after GPU OOM (reduced samples)...",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "Attempting CPU fallback render (reduced samples)...",
                        file=sys.stderr)

                try:
                    bpy.context.scene.cycles.device = 'CPU'
                except Exception:

                    pass
                try:
                    prefs = bpy.context.preferences.addons[
                        'cycles'].preferences
                    devices = getattr(prefs, 'devices', [])
                    for device in devices:
                        try:
                            device.use = (getattr(device, 'type',
                                                  '').upper() == 'CPU')
                        except Exception:
                            pass
                except Exception:
                    pass

                try:
                    fallback_samples_env = os.environ.get(
                        'BLENDER_FALLBACK_CYCLES_SAMPLES', '').strip()
                    try:
                        fallback_samples = int(
                            fallback_samples_env
                        ) if fallback_samples_env else 24
                    except ValueError:
                        fallback_samples = 24
                    fallback_samples = max(1, fallback_samples)
                    current_samples = int(
                        getattr(bpy.context.scene.cycles, 'samples',
                                fallback_samples))
                    bpy.context.scene.cycles.samples = min(
                        current_samples, fallback_samples)
                    bpy.context.scene.cycles.use_denoising = False
                    try:
                        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
                    except Exception:
                        pass
                except Exception:
                    pass

                try:
                    bpy.context.scene.render.engine = 'CYCLES'
                except Exception:
                    pass

                bpy.ops.render.render(write_still=True)
                print("CPU fallback render succeeded.", file=sys.stderr)
            except Exception as e2:

                print(f"CPU fallback render also failed: {e2}",
                      file=sys.stderr)
                raise
        else:

            raise

    if gpu_id is not None:

        after = get_gpu_metrics(gpu_id)
        after['timestamp'] = datetime.now().isoformat()
        resource_metrics = {"before": before, "after": after}

    return {"resource_metrics": resource_metrics}


def calculate_sun_euler(timestamp_utc, city='San Francisco'):
    if ephem is None:

        return 0.0, pi / 4

    timestamp_utc /= 1000000
    time_dt = datetime.utcfromtimestamp(timestamp_utc).strftime(
        '%Y/%m/%d %H:%M:%S')
    sanfran = ephem.Observer()
    sanfran.lat = '37.7749'
    sanfran.lon = '-122.4194'
    sanfran.date = time_dt
    sun = ephem.Sun()
    sun.compute(sanfran)

    azimuth_rad = sun.az
    altitude_rad = sun.alt

    rotation_z = pi - azimuth_rad
    rotation_x = pi / 2 - altitude_rad

    return rotation_z, rotation_x


def _reset_world_nodes(world):
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new(type='ShaderNodeOutputWorld')
    out.location = (200, 0)
    bg = nt.nodes.new(type='ShaderNodeBackground')
    bg.location = (0, 0)
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    return bg


def _clear_scene_lights():
    for obj in list(bpy.context.scene.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)


def _setup_world_lighting_for_frame(
    time_stamp,
    weather_api=None,
    city='San Francisco',
):
    """Recompute world + sun lights using per-frame timestamp/weather."""
    weather_api = weather_api or {}
    cloud_cover = weather_api.get('cloud_cover', 0) if weather_api else 0
    direct_illumination = weather_api.get('direct_illumination',
                                          None) if weather_api else None
    if direct_illumination is not None and direct_illumination > 1.5:

        direct_illumination = direct_illumination / 700.0

    try:
        cloud_cover = float(cloud_cover)
    except Exception:
        cloud_cover = 0.0
    cloud_cover = float(np.clip(cloud_cover, 0.0, 100.0))

    _clear_scene_lights()

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    bg_node = _reset_world_nodes(world)

    sun_euler_zx = calculate_sun_euler(time_stamp, city)
    sun_euler_xyz = (sun_euler_zx[1], 0.0, sun_euler_zx[0])

    if cloud_cover < 60.0:
        bpy.ops.object.light_add(
            type='SUN',
            radius=1,
            align='WORLD',
            location=(0.0, 0.0, 10000.0),
            rotation=sun_euler_xyz,
        )
        sunlight = bpy.context.object.data
        energy, angle = sun_params_linear(
            cloud_cover,
            direct_illumination,
            min_energy=0.2,
            max_energy=0.8,
            min_angle=np.radians(0.8),
            max_angle=np.radians(5.0),
        )
        sunlight.energy = energy
        sunlight.angle = angle
        bg_node.inputs['Color'].default_value = (0.55, 0.65, 0.95, 1.0)
        bg_node.inputs['Strength'].default_value = 0.2
    else:

        hdri_path = str(os.environ.get("WAYPOP_HDRI_PATH", "")).strip()
        if hdri_path and os.path.exists(hdri_path):
            env = world.node_tree.nodes.new(type='ShaderNodeTexEnvironment')
            env.location = (-250, 0)
            env.image = bpy.data.images.load(hdri_path, check_existing=True)
            world.node_tree.links.new(env.outputs['Color'],
                                      bg_node.inputs['Color'])
            bg_node.inputs['Strength'].default_value = 1.0
        else:
            bg_node.inputs['Color'].default_value = (0.7, 0.75, 0.85, 1.0)
            bg_node.inputs['Strength'].default_value = 2.0

        if direct_illumination is not None and direct_illumination > 0.0:
            bpy.ops.object.light_add(
                type='SUN',
                radius=1,
                align='WORLD',
                location=(0.0, 0.0, 10000.0),
                rotation=sun_euler_xyz,
            )
            sunlight = bpy.context.object.data
            energy, angle = sun_params_linear(
                cloud_cover,
                direct_illumination,
                min_energy=0.01,
                max_energy=0.2,
                min_angle=np.radians(10.0),
                max_angle=np.radians(15.0),
            )
            sunlight.energy = energy
            sunlight.angle = angle


def get_collection_lowest_point(collection):
    """Finds the lowest Z-coordinate of all objects in the collection."""
    min_z = float('inf')

    for obj in collection.all_objects:
        if obj.type != 'MESH':
            continue

        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_z = min(min_z, world_corner.z)

    return min_z if min_z != float('inf') else None


def get_scene_min_projection_on_axis(axis):
    """Find min projection of all mesh bbox corners onto a world-space axis."""
    try:
        axis_vec = Vector(axis)
        if axis_vec.length < 1e-8:
            axis_vec = Vector((0.0, 0.0, 1.0))
        else:
            axis_vec.normalize()
    except Exception:
        axis_vec = Vector((0.0, 0.0, 1.0))

    min_proj = float("inf")
    found = False
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            proj = float(axis_vec.dot(world_corner))
            if proj < min_proj:
                min_proj = proj
            found = True
    if not found:
        return None
    return float(min_proj)


def get_scene_projection_quantile_on_axis(axis,
                                          quantile=0.02,
                                          max_samples_per_object=50000):
    """Find robust low projection quantile of mesh vertices onto a world-space axis."""
    try:
        axis_vec = Vector(axis)
        if axis_vec.length < 1e-8:
            axis_vec = Vector((0.0, 0.0, 1.0))
        else:
            axis_vec.normalize()
    except Exception:
        axis_vec = Vector((0.0, 0.0, 1.0))

    q = float(np.clip(quantile, 0.0, 1.0))
    projections = []

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        verts = obj.data.vertices
        if verts is None or len(verts) == 0:
            continue
        step = max(
            1,
            int(math.ceil(len(verts) / float(max(1, max_samples_per_object)))))
        for idx in range(0, len(verts), step):
            world_v = obj.matrix_world @ verts[idx].co
            projections.append(float(axis_vec.dot(world_v)))

    if not projections:
        return None
    return float(np.quantile(np.asarray(projections, dtype=np.float64), q))


def get_scene_support_centroid_on_axis(
    axis,
    quantile=0.02,
    band_ratio=0.03,
    min_band_m=0.01,
    max_samples_per_object=50000,
):
    """Estimate the centroid of the object's support footprint near the ground-contact side.

    The centroid is computed from vertices whose projection onto `axis` lies within
    a narrow band above a low quantile of the scene's projection distribution.
    This is more stable than anchoring to the full bbox center for asymmetric assets.
    """
    try:
        axis_vec = Vector(axis)
        if axis_vec.length < 1e-8:
            axis_vec = Vector((0.0, 0.0, 1.0))
        else:
            axis_vec.normalize()
    except Exception:
        axis_vec = Vector((0.0, 0.0, 1.0))

    q = float(np.clip(quantile, 0.0, 1.0))
    band_ratio = float(max(0.0, band_ratio))
    min_band_m = float(max(1e-4, min_band_m))

    points = []
    projections = []

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        verts = obj.data.vertices
        if verts is None or len(verts) == 0:
            continue
        step = max(
            1,
            int(math.ceil(len(verts) / float(max(1, max_samples_per_object)))))
        for idx in range(0, len(verts), step):
            world_v = obj.matrix_world @ verts[idx].co
            points.append(
                (float(world_v.x), float(world_v.y), float(world_v.z)))
            projections.append(float(axis_vec.dot(world_v)))

    if not points:
        return None, None

    points_np = np.asarray(points, dtype=np.float64)
    projections_np = np.asarray(projections, dtype=np.float64)
    low_proj = float(np.quantile(projections_np, q))
    proj_span = float(np.max(projections_np) - np.min(projections_np))
    band_m = max(min_band_m, proj_span * band_ratio)
    mask = projections_np <= (low_proj + band_m)
    if not np.any(mask):
        mask = projections_np <= low_proj
    support_points = points_np[mask]
    support_projections = projections_np[mask]
    if support_points.size == 0:
        return None, None

    centroid = support_points.mean(axis=0)
    info = {
        "quantile_proj": low_proj,
        "band_m": float(band_m),
        "support_samples": int(support_points.shape[0]),
        "total_samples": int(points_np.shape[0]),
        "support_min_proj": float(np.min(support_projections)),
        "support_q01_proj": float(np.quantile(support_projections, 0.01)),
        "support_q05_proj": float(np.quantile(support_projections, 0.05)),
        "support_q10_proj": float(np.quantile(support_projections, 0.10)),
        "support_q25_proj": float(np.quantile(support_projections, 0.25)),
        "support_median_proj": float(np.quantile(support_projections, 0.50)),
    }
    return Vector(
        (float(centroid[0]), float(centroid[1]), float(centroid[2]))), info


def _downsample_indices(count, max_count):
    count = int(max(0, count))
    max_count = int(max(1, max_count))
    if count <= max_count:
        return np.arange(count, dtype=np.int32)
    step = int(math.ceil(float(count) / float(max_count)))
    return np.arange(0, count, step, dtype=np.int32)[:max_count]


def _color_from_component_index(index):
    idx = int(index)
    return (
        int((53 * idx + 97) % 255),
        int((97 * idx + 173) % 255),
        int((193 * idx + 29) % 255),
    )


def _write_ascii_point_ply(path, points, colors=None):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return False

    cols = None
    if colors is not None:
        cols = np.asarray(colors, dtype=np.uint8)
        if cols.shape != (pts.shape[0], 3):
            cols = None

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if cols is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if cols is None:
            for point in pts:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        else:
            for point, color in zip(pts, cols):
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                        f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
    return True


def _collect_support_debug_summary(
    axis_vec,
    support_band_limit_proj,
    contact_zone_limit_proj,
    max_export_points=25000,
):
    axis = Vector(axis_vec)
    axis.normalize()

    object_summaries = []
    component_summaries = []
    support_band_points = []
    support_band_colors = []
    contact_zone_points = []
    contact_zone_colors = []
    component_points = []
    component_colors = []
    global_component_index = 0

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        verts = obj.data.vertices
        edges = obj.data.edges
        if verts is None or len(verts) == 0:
            continue

        world = np.empty((len(verts), 3), dtype=np.float64)
        projections = np.empty((len(verts), ), dtype=np.float64)
        for idx, vert in enumerate(verts):
            world_v = obj.matrix_world @ vert.co
            world[idx, 0] = float(world_v.x)
            world[idx, 1] = float(world_v.y)
            world[idx, 2] = float(world_v.z)
            projections[idx] = float(axis.dot(world_v))

        obj_summary = {
            "name":
            obj.name,
            "vertex_count":
            int(len(verts)),
            "edge_count":
            int(len(edges)),
            "projection_min":
            float(np.min(projections)),
            "projection_q01":
            float(np.quantile(projections, 0.01)),
            "projection_q05":
            float(np.quantile(projections, 0.05)),
            "projection_q25":
            float(np.quantile(projections, 0.25)),
            "projection_median":
            float(np.quantile(projections, 0.50)),
            "projection_max":
            float(np.max(projections)),
            "bbox_min":
            world.min(axis=0).tolist(),
            "bbox_max":
            world.max(axis=0).tolist(),
            "support_band_vertex_count":
            int(np.count_nonzero(projections <= support_band_limit_proj)),
            "contact_zone_vertex_count":
            int(np.count_nonzero(projections <= contact_zone_limit_proj)),
            "bottom_components": [],
        }

        support_idx = np.nonzero(projections <= support_band_limit_proj)[0]
        contact_idx = np.nonzero(projections <= contact_zone_limit_proj)[0]

        if support_idx.size > 0:
            sampled = support_idx[_downsample_indices(
                len(support_idx), max(1, max_export_points // 4))]
            support_band_points.append(world[sampled])
            support_band_colors.append(
                np.tile(np.asarray([[80, 200, 255]], dtype=np.uint8),
                        (len(sampled), 1)))
        if contact_idx.size > 0:
            sampled = contact_idx[_downsample_indices(
                len(contact_idx), max(1, max_export_points // 4))]
            contact_zone_points.append(world[sampled])
            contact_zone_colors.append(
                np.tile(np.asarray([[255, 220, 0]], dtype=np.uint8),
                        (len(sampled), 1)))

        if support_idx.size == 0:
            object_summaries.append(obj_summary)
            continue

        support_local_map = {
            int(orig_idx): int(local_idx)
            for local_idx, orig_idx in enumerate(support_idx.tolist())
        }
        adjacency = [[] for _ in range(len(support_idx))]
        for edge in edges:
            v0 = int(edge.vertices[0])
            v1 = int(edge.vertices[1])
            i0 = support_local_map.get(v0)
            i1 = support_local_map.get(v1)
            if i0 is None or i1 is None:
                continue
            adjacency[i0].append(i1)
            adjacency[i1].append(i0)

        visited = np.zeros((len(support_idx), ), dtype=bool)
        object_components = []
        for start in range(len(support_idx)):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            component_local = []
            while stack:
                cur = stack.pop()
                component_local.append(cur)
                for nxt in adjacency[cur]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        stack.append(nxt)

            component_orig = support_idx[np.asarray(component_local,
                                                    dtype=np.int32)]
            component_world = world[component_orig]
            component_proj = projections[component_orig]
            component_id = int(global_component_index)
            global_component_index += 1
            color = _color_from_component_index(component_id)
            sampled = _downsample_indices(len(component_orig),
                                          max(1, max_export_points // 6))
            component_points.append(component_world[sampled])
            component_colors.append(
                np.tile(np.asarray([color], dtype=np.uint8),
                        (len(sampled), 1)))

            comp_summary = {
                "component_id":
                component_id,
                "object_name":
                obj.name,
                "vertex_count":
                int(len(component_orig)),
                "projection_min":
                float(np.min(component_proj)),
                "projection_q05":
                float(np.quantile(component_proj, 0.05)),
                "projection_q25":
                float(np.quantile(component_proj, 0.25)),
                "projection_median":
                float(np.quantile(component_proj, 0.50)),
                "bbox_min":
                component_world.min(axis=0).tolist(),
                "bbox_max":
                component_world.max(axis=0).tolist(),
                "centroid":
                component_world.mean(axis=0).tolist(),
                "xy_span_m": [
                    float(component_world[:, 0].max() -
                          component_world[:, 0].min()),
                    float(component_world[:, 1].max() -
                          component_world[:, 1].min()),
                ],
                "z_span_m":
                float(component_world[:, 2].max() -
                      component_world[:, 2].min()),
                "color_rgb": [int(color[0]),
                              int(color[1]),
                              int(color[2])],
            }
            object_components.append(comp_summary)
            component_summaries.append(comp_summary)

        object_components.sort(
            key=lambda item: (item["projection_min"], -item["vertex_count"]))
        for rank, component in enumerate(object_components):
            component["rank_by_min_projection"] = int(rank)
        obj_summary["bottom_components"] = object_components
        object_summaries.append(obj_summary)

    component_summaries.sort(
        key=lambda item: (item["projection_min"], -item["vertex_count"]))
    for rank, component in enumerate(component_summaries):
        component["global_rank_by_min_projection"] = int(rank)

    merged_support_points = (np.concatenate(support_band_points, axis=0)
                             if support_band_points else np.zeros(
                                 (0, 3), dtype=np.float32))
    merged_support_colors = (np.concatenate(support_band_colors, axis=0)
                             if support_band_colors else np.zeros(
                                 (0, 3), dtype=np.uint8))
    merged_contact_points = (np.concatenate(contact_zone_points, axis=0)
                             if contact_zone_points else np.zeros(
                                 (0, 3), dtype=np.float32))
    merged_contact_colors = (np.concatenate(contact_zone_colors, axis=0)
                             if contact_zone_colors else np.zeros(
                                 (0, 3), dtype=np.uint8))
    merged_component_points = (np.concatenate(component_points, axis=0)
                               if component_points else np.zeros(
                                   (0, 3), dtype=np.float32))
    merged_component_colors = (np.concatenate(component_colors, axis=0)
                               if component_colors else np.zeros(
                                   (0, 3), dtype=np.uint8))

    if merged_support_points.shape[0] > max_export_points:
        keep = _downsample_indices(merged_support_points.shape[0],
                                   max_export_points)
        merged_support_points = merged_support_points[keep]
        merged_support_colors = merged_support_colors[keep]
    if merged_contact_points.shape[0] > max_export_points:
        keep = _downsample_indices(merged_contact_points.shape[0],
                                   max_export_points)
        merged_contact_points = merged_contact_points[keep]
        merged_contact_colors = merged_contact_colors[keep]
    if merged_component_points.shape[0] > max_export_points:
        keep = _downsample_indices(merged_component_points.shape[0],
                                   max_export_points)
        merged_component_points = merged_component_points[keep]
        merged_component_colors = merged_component_colors[keep]

    return {
        "objects": object_summaries,
        "components": component_summaries,
        "support_band_points": merged_support_points,
        "support_band_colors": merged_support_colors,
        "contact_zone_points": merged_contact_points,
        "contact_zone_colors": merged_contact_colors,
        "component_points": merged_component_points,
        "component_colors": merged_component_colors,
    }


def _choose_grounding_component(
    axis_vec,
    support_info,
    robust_proj=None,
    min_proj=None,
    outlier_gap_threshold_m=0.005,
    lowest_count_ratio_max=0.25,
    lowest_area_ratio_max=0.35,
):
    if support_info is None:
        return None
    try:
        support_band_limit_proj = float(support_info["quantile_proj"]) + float(
            support_info["band_m"])
    except Exception:
        return None

    contact_zone_limit_proj = float(
        support_info.get("support_q05_proj", support_band_limit_proj)) + 0.002
    summary = _collect_support_debug_summary(
        axis_vec=axis_vec,
        support_band_limit_proj=support_band_limit_proj,
        contact_zone_limit_proj=contact_zone_limit_proj,
        max_export_points=1,
    )
    components = list(summary.get("components", []))
    if len(components) <= 1:
        return {
            "decision": "global_support",
            "chosen_component": None,
            "lowest_component": components[0] if components else None,
            "dominant_component": components[0] if components else None,
            "component_count": int(len(components)),
            "override_applied": False,
        }

    def _footprint_area(component):
        try:
            x_span, y_span = component.get("xy_span_m", [0.0, 0.0])
            x_span = float(max(0.0, x_span))
            y_span = float(max(0.0, y_span))
            area = x_span * y_span
            if area > 0.0:
                return area
            return max(x_span, y_span) * 0.01
        except Exception:
            return 0.0

    def _support_mass(component):
        return float(component.get("vertex_count", 0)) * max(
            _footprint_area(component), 1e-4)

    lowest_component = min(
        components,
        key=lambda item:
        (float(item.get("projection_min", float("inf"))), -int(
            item.get("vertex_count", 0))),
    )
    dominant_component = max(
        components,
        key=lambda item:
        (_support_mass(item), int(item.get("vertex_count", 0))),
    )

    lowest_area = _footprint_area(lowest_component)
    dominant_area = _footprint_area(dominant_component)
    lowest_q05 = float(
        lowest_component.get("projection_q05",
                             lowest_component.get("projection_min", 0.0)))
    dominant_q05 = float(
        dominant_component.get("projection_q05",
                               dominant_component.get("projection_min", 0.0)))
    q05_gap = float(dominant_q05 - lowest_q05)
    lowest_count = int(lowest_component.get("vertex_count", 0))
    dominant_count = int(max(1, dominant_component.get("vertex_count", 1)))
    count_ratio = float(lowest_count) / float(dominant_count)
    area_ratio = float(lowest_area) / float(max(dominant_area, 1e-6))

    override = (lowest_component.get("component_id") !=
                dominant_component.get("component_id")
                and q05_gap > float(outlier_gap_threshold_m)
                and count_ratio < float(lowest_count_ratio_max)
                and area_ratio < float(lowest_area_ratio_max))

    return {
        "decision":
        "dominant_component_override" if override else "global_support",
        "chosen_component": dominant_component if override else None,
        "lowest_component": lowest_component,
        "dominant_component": dominant_component,
        "component_count": int(len(components)),
        "override_applied": bool(override),
        "metrics": {
            "q05_gap_m": float(q05_gap),
            "lowest_count_ratio": float(count_ratio),
            "lowest_area_ratio": float(area_ratio),
            "lowest_area_m2": float(lowest_area),
            "dominant_area_m2": float(dominant_area),
            "outlier_gap_threshold_m": float(outlier_gap_threshold_m),
            "lowest_count_ratio_max": float(lowest_count_ratio_max),
            "lowest_area_ratio_max": float(lowest_area_ratio_max),
        },
    }


def _export_support_debug_artifacts(
    debug_dir,
    *,
    object_path,
    ood_id,
    root_object_name,
    scene_bbox_min,
    scene_bbox_max,
    contact_axis,
    contact_point,
    support_info,
    grounding_component_decision,
    grounding_component_metrics,
    chosen_component,
    lowest_component,
    dominant_component,
    source_name,
    robust_proj,
    min_proj,
    support_contact_proj,
    support_q25_proj,
    support_guard_proj,
    target_proj,
    delta,
    z_offset,
    outlier_gap,
    visual_sink_m,
    ground_contact_epsilon,
):
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)

    support_band_limit_proj = None
    if support_info is not None:
        try:
            support_band_limit_proj = float(
                support_info["quantile_proj"]) + float(support_info["band_m"])
        except Exception:
            support_band_limit_proj = None
    if support_band_limit_proj is None:
        if robust_proj is not None:
            support_band_limit_proj = float(robust_proj) + 0.01
        elif min_proj is not None:
            support_band_limit_proj = float(min_proj) + 0.01
        else:
            support_band_limit_proj = float(target_proj) + 0.01

    if support_contact_proj is not None:
        contact_zone_limit_proj = float(support_contact_proj) + 0.002
    elif min_proj is not None:
        contact_zone_limit_proj = float(min_proj) + 0.002
    else:
        contact_zone_limit_proj = float(target_proj) + 0.002

    summary = _collect_support_debug_summary(
        axis_vec=contact_axis,
        support_band_limit_proj=float(support_band_limit_proj),
        contact_zone_limit_proj=float(contact_zone_limit_proj),
    )

    profile = {
        "ood_id":
        int(ood_id) if ood_id is not None else None,
        "object_path":
        object_path,
        "root_object_name":
        root_object_name,
        "scene_bbox_min":
        list(scene_bbox_min) if scene_bbox_min is not None else None,
        "scene_bbox_max":
        list(scene_bbox_max) if scene_bbox_max is not None else None,
        "contact_axis": [
            float(contact_axis[0]),
            float(contact_axis[1]),
            float(contact_axis[2])
        ],
        "contact_point": [
            float(contact_point[0]),
            float(contact_point[1]),
            float(contact_point[2])
        ],
        "ground_snap": {
            "source":
            source_name,
            "robust_proj_q02":
            float(robust_proj) if robust_proj is not None else None,
            "global_min_proj":
            float(min_proj) if min_proj is not None else None,
            "support_contact_proj_q05":
            float(support_contact_proj)
            if support_contact_proj is not None else None,
            "support_q25_proj":
            float(support_q25_proj) if support_q25_proj is not None else None,
            "support_guard_proj_q01":
            float(support_guard_proj)
            if support_guard_proj is not None else None,
            "support_band_limit_proj":
            float(support_band_limit_proj),
            "contact_zone_limit_proj":
            float(contact_zone_limit_proj),
            "target_proj":
            float(target_proj),
            "delta":
            float(delta),
            "z_offset":
            float(z_offset),
            "outlier_gap":
            float(outlier_gap) if outlier_gap is not None else None,
            "visual_sink_m":
            float(visual_sink_m),
            "ground_contact_epsilon":
            float(ground_contact_epsilon),
        },
        "support_profile":
        support_info,
        "grounding_component_decision":
        grounding_component_decision,
        "grounding_component_metrics":
        grounding_component_metrics,
        "chosen_grounding_component":
        chosen_component,
        "lowest_component":
        lowest_component,
        "dominant_component":
        dominant_component,
        "mesh_objects":
        summary["objects"],
        "bottom_components":
        summary["components"],
    }

    with open(os.path.join(debug_dir, "support_profile.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    _write_ascii_point_ply(
        os.path.join(debug_dir, "support_band_points_world.ply"),
        summary["support_band_points"],
        summary["support_band_colors"],
    )
    _write_ascii_point_ply(
        os.path.join(debug_dir, "contact_zone_points_world.ply"),
        summary["contact_zone_points"],
        summary["contact_zone_colors"],
    )
    _write_ascii_point_ply(
        os.path.join(debug_dir, "support_components_world.ply"),
        summary["component_points"],
        summary["component_colors"],
    )


def _scene_bbox_to_lists():
    bbox_min, bbox_max = scene_bbox()
    return [float(v) for v in bbox_min], [float(v) for v in bbox_max]


def _matrix_to_nested_list(matrix):
    return [[float(v) for v in row] for row in matrix]


def _vector_to_list(vec):
    return [float(v) for v in vec]


def _project_world_points_raw(points_world):
    camera = bpy.context.scene.camera
    if camera is None or points_world is None or len(points_world) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    projected = []
    scene = bpy.context.scene
    for point in points_world:
        try:
            coord = world_to_camera_view(
                scene, camera,
                Vector((float(point[0]), float(point[1]), float(point[2]))))
        except Exception:
            continue
        if not np.isfinite(float(coord.x)) or not np.isfinite(float(
                coord.y)) or not np.isfinite(float(coord.z)):
            continue
        projected.append((float(coord.x), float(coord.y), float(coord.z)))
    if not projected:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(projected, dtype=np.float32)


def _project_world_points_to_image(points_world, image_width, image_height):
    raw = _project_world_points_raw(points_world)
    if raw.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    width = max(1, int(image_width))
    height = max(1, int(image_height))
    projected = []
    for coord_x, coord_y, _coord_z in raw:
        if float(coord_x) < 0.0 or float(coord_x) > 1.0 or float(
                coord_y) < 0.0 or float(coord_y) > 1.0:
            continue
        px = float(coord_x) * float(width - 1)
        py = (1.0 - float(coord_y)) * float(height - 1)
        projected.append((px, py))
    if not projected:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(projected, dtype=np.float32)


def _load_png_alpha_bbox(render_path):
    if not render_path or not os.path.isfile(render_path):
        return None
    img = None
    try:
        img = bpy.data.images.load(render_path, check_existing=False)
        width = int(img.size[0])
        height = int(img.size[1])
        if width <= 0 or height <= 0:
            return None
        pixels = np.asarray(img.pixels[:], dtype=np.float32)
        if pixels.size != width * height * 4:
            return None
        rgba = pixels.reshape((height, width, 4))
        alpha = np.flipud(rgba[:, :, 3])
        ys, xs = np.nonzero(alpha > 0.02)
        if ys.size == 0:
            return {
                "width": width,
                "height": height,
                "bbox": None,
                "bottom_row": None,
                "top_row": None,
                "left_col": None,
                "right_col": None,
                "alpha_count": 0,
            }
        return {
            "width":
            width,
            "height":
            height,
            "bbox": [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ],
            "bottom_row":
            int(ys.max()),
            "top_row":
            int(ys.min()),
            "left_col":
            int(xs.min()),
            "right_col":
            int(xs.max()),
            "alpha_count":
            int(ys.size),
        }
    finally:
        try:
            if img is not None:
                bpy.data.images.remove(img)
        except Exception:
            pass


def _export_camera_space_contact_debug(
    debug_dir,
    *,
    render_path,
    camera_name,
    frame_token,
    image_width,
    image_height,
):
    if not debug_dir:
        return
    profile_path = os.path.join(debug_dir, "support_profile.json")
    if not os.path.isfile(profile_path):
        return

    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    contact_axis = profile.get("contact_axis")
    ground_snap = profile.get("ground_snap", {})
    if not contact_axis or len(contact_axis) != 3:
        return
    support_info_current = None
    try:
        _, support_info_current = get_scene_support_centroid_on_axis(
            Vector((float(contact_axis[0]), float(contact_axis[1]),
                    float(contact_axis[2]))),
            quantile=0.02,
            band_ratio=0.03,
        )
    except Exception:
        support_info_current = None
    if support_info_current is not None:
        support_band_limit_proj = float(
            support_info_current["quantile_proj"]) + float(
                support_info_current["band_m"])
        contact_zone_limit_proj = float(
            support_info_current.get("support_q05_proj",
                                     support_band_limit_proj)) + 0.002
    else:
        support_band_limit_proj = ground_snap.get("support_band_limit_proj")
        contact_zone_limit_proj = ground_snap.get("contact_zone_limit_proj")
    if support_band_limit_proj is None or contact_zone_limit_proj is None:
        return

    summary = _collect_support_debug_summary(
        axis_vec=contact_axis,
        support_band_limit_proj=float(support_band_limit_proj),
        contact_zone_limit_proj=float(contact_zone_limit_proj),
    )
    support_raw = _project_world_points_raw(summary.get("support_band_points"))
    contact_raw = _project_world_points_raw(summary.get("contact_zone_points"))
    component_raw = _project_world_points_raw(summary.get("component_points"))

    support_proj = _project_world_points_to_image(
        summary.get("support_band_points"),
        image_width,
        image_height,
    )
    contact_proj = _project_world_points_to_image(
        summary.get("contact_zone_points"),
        image_width,
        image_height,
    )
    component_proj = _project_world_points_to_image(
        summary.get("component_points"),
        image_width,
        image_height,
    )
    alpha_bbox = _load_png_alpha_bbox(render_path)

    def _row_stats(arr):
        if arr is None or arr.size == 0:
            return None
        rows = arr[:, 1]
        return {
            "min_row": float(np.min(rows)),
            "q05_row": float(np.quantile(rows, 0.05)),
            "median_row": float(np.quantile(rows, 0.50)),
            "q95_row": float(np.quantile(rows, 0.95)),
            "max_row": float(np.max(rows)),
            "count": int(arr.shape[0]),
        }

    def _raw_stats(arr):
        if arr is None or arr.size == 0:
            return None
        xs = arr[:, 0]
        ys = arr[:, 1]
        zs = arr[:, 2]
        return {
            "min_x":
            float(np.min(xs)),
            "max_x":
            float(np.max(xs)),
            "min_y":
            float(np.min(ys)),
            "max_y":
            float(np.max(ys)),
            "min_z":
            float(np.min(zs)),
            "max_z":
            float(np.max(zs)),
            "count":
            int(arr.shape[0]),
            "in_bounds_count":
            int(
                np.count_nonzero((xs >= 0.0) & (xs <= 1.0) & (ys >= 0.0)
                                 & (ys <= 1.0))),
        }

    support_rows = _row_stats(support_proj)
    contact_rows = _row_stats(contact_proj)
    component_rows = _row_stats(component_proj)
    support_raw_stats = _raw_stats(support_raw)
    contact_raw_stats = _raw_stats(contact_raw)
    component_raw_stats = _raw_stats(component_raw)
    silhouette_bottom = alpha_bbox.get("bottom_row") if alpha_bbox else None

    contact_point_world = np.asarray([profile.get("contact_point", [])],
                                     dtype=np.float32)
    contact_point_raw = (
        _project_world_points_raw(contact_point_world)[0].tolist()
        if contact_point_world.size == 3 else None)

    metrics = {
        "camera_name": str(camera_name),
        "frame_token": str(frame_token),
        "render_path": render_path,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "alpha_bbox": alpha_bbox,
        "support_rows": support_rows,
        "contact_rows": contact_rows,
        "component_rows": component_rows,
        "support_raw_stats": support_raw_stats,
        "contact_raw_stats": contact_raw_stats,
        "component_raw_stats": component_raw_stats,
        "contact_point_raw": contact_point_raw,
        "silhouette_gap_px": {
            "support_max_minus_bottom":
            (float(support_rows["max_row"] -
                   silhouette_bottom) if support_rows is not None
             and silhouette_bottom is not None else None),
            "support_q95_minus_bottom":
            (float(support_rows["q95_row"] -
                   silhouette_bottom) if support_rows is not None
             and silhouette_bottom is not None else None),
            "contact_max_minus_bottom":
            (float(contact_rows["max_row"] -
                   silhouette_bottom) if contact_rows is not None
             and silhouette_bottom is not None else None),
            "contact_q95_minus_bottom":
            (float(contact_rows["q95_row"] -
                   silhouette_bottom) if contact_rows is not None
             and silhouette_bottom is not None else None),
        },
    }

    camera_dir = os.path.join(debug_dir, "camera", str(camera_name))
    os.makedirs(camera_dir, exist_ok=True)
    metrics_path = os.path.join(
        camera_dir, f"frame_{frame_token}_camera_space_contact.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.save(
        os.path.join(camera_dir,
                     f"frame_{frame_token}_projected_support_points.npy"),
        support_proj)
    np.save(
        os.path.join(camera_dir,
                     f"frame_{frame_token}_projected_contact_points.npy"),
        contact_proj)
    np.save(
        os.path.join(camera_dir,
                     f"frame_{frame_token}_projected_component_points.npy"),
        component_proj)


def clear_scene_and_import_object(
    object_path,
    pose,
    time_stamp,
    scale=1,
    ood_id=None,
    sun_en=1,
    cloud_cover=None,
    file_name=None,
    file_dir=None,
    size_multiplier=1.0,
    target_ref_dim_m=None,
    scale_policy="auto",
    description="",
    direct_illumination=None,
    indirect_illumination=None,
    eval=False,
    ground_contact_epsilon=0.01,
    shadow_plane_point=None,
    shadow_plane_normal=None,
    anchor_support_xy=False,
    support_anchor_quantile=0.02,
    support_anchor_band_ratio=0.03,
    support_debug_dir=None,
):
    """Delete all existing meshes and lights. Add sunlight. Load the object into the scene.
    Chose light source rotation by time. """
    city = "San Francisco"
    input_pose_matrix = Matrix(pose)
    transformation_matrix = Matrix(pose)

    inverse_camera_matrix = Matrix([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0],
                                    [0, 0, 0, 1]])
    transformation_matrix @= inverse_camera_matrix
    rotation_matrix = transformation_matrix.to_3x3()
    translation_vector = transformation_matrix.to_translation()
    euler_angles = rotation_matrix.to_euler("XYZ")
    rotation_matrix = Matrix.Rotation(
        euler_angles[2], 4, 'Z') @ Matrix.Rotation(
            euler_angles[1], 4, 'Y') @ Matrix.Rotation(euler_angles[0], 4, 'X')

    transform_trace = None
    if support_debug_dir:
        transform_trace = {
            "ood_id": int(ood_id) if ood_id is not None else None,
            "object_path": object_path,
            "input_pose_matrix": _matrix_to_nested_list(input_pose_matrix),
            "transformation_matrix":
            _matrix_to_nested_list(transformation_matrix),
            "translation_vector": _vector_to_list(translation_vector),
            "euler_angles_xyz": [float(v) for v in euler_angles],
            "stages": [],
        }

    def _record_transform_stage(stage_name, extra=None):
        if transform_trace is None:
            return
        bbox_min, bbox_max = _scene_bbox_to_lists()
        root_matrix = None
        if root_object is not None:
            try:
                root_matrix = _matrix_to_nested_list(root_object.matrix_world)
            except Exception:
                root_matrix = None
        entry = {
            "stage": str(stage_name),
            "scene_bbox_min": bbox_min,
            "scene_bbox_max": bbox_max,
            "root_object_name":
            root_object.name if root_object is not None else None,
            "root_matrix_world": root_matrix,
        }
        if extra:
            entry.update(extra)
        transform_trace["stages"].append(entry)

    reset_scene()

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.delete()
    old_objects = set(bpy.context.scene.objects)

    if object_path and os.path.isfile(object_path):

        load_object(object_path)
    else:
        create_cube((0, 0, 0), (1, 1, 1), (0, 0, 0))

    new_objects = set(bpy.context.scene.objects) - old_objects
    imported_objects = list(new_objects)
    old_objects.update(new_objects)
    root_object = next((obj for obj in imported_objects if not obj.parent),
                       None)

    _record_transform_stage(
        "post_import_pre_zero",
        {
            "imported_object_names":
            sorted([obj.name for obj in imported_objects]),
        },
    )

    for obj in bpy.context.scene.objects:

        if obj.location != Vector([0, 0, 0]):
            obj.location = Vector([0, 0, 0])
    bpy.context.view_layer.update()
    _record_transform_stage("post_zero_locations")

    normalize_scene(
        ood_id,
        scale,
        file_dir,
        size_multiplier=size_multiplier,
        eval=eval,
        target_ref_dim_m=target_ref_dim_m,
        scale_policy=scale_policy,
        description=description,
    )
    bpy.context.view_layer.update()
    _record_transform_stage(
        "post_normalize",
        {
            "normalize_scale":
            float(scale),
            "size_multiplier":
            float(size_multiplier),
            "target_ref_dim_m":
            float(target_ref_dim_m) if target_ref_dim_m is not None else None,
            "scale_policy":
            str(scale_policy),
        },
    )

    if isinstance(shadow_plane_normal,
                  (list, tuple)) and len(shadow_plane_normal) == 3:
        contact_axis = Vector(
            (float(shadow_plane_normal[0]), float(shadow_plane_normal[1]),
             float(shadow_plane_normal[2])))
    else:
        contact_axis = Vector((0.0, 0.0, 1.0))
    if contact_axis.length < 1e-8:
        contact_axis = Vector((0.0, 0.0, 1.0))
    else:
        contact_axis.normalize()

    if isinstance(shadow_plane_point,
                  (list, tuple)) and len(shadow_plane_point) == 3:
        contact_point = Vector(
            (float(shadow_plane_point[0]), float(shadow_plane_point[1]),
             float(shadow_plane_point[2])))
    else:
        contact_point = Vector((
            float(translation_vector[0]),
            float(translation_vector[1]),
            float(translation_vector[2]),
        ))

    z_offset1 = scene_bbox()[0][2]
    offset_translation = [0, 0, -z_offset1]

    matrix_world = None
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += Vector(offset_translation)
    bpy.context.view_layer.update()
    _record_transform_stage(
        "post_offset_translation",
        {
            "z_offset1": float(z_offset1),
            "offset_translation": [float(v) for v in offset_translation],
            "contact_axis": _vector_to_list(contact_axis),
            "contact_point": _vector_to_list(contact_point),
        },
    )
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += Vector(translation_vector)
    bpy.context.view_layer.update()
    _record_transform_stage("post_contact_translation")

    support_centroid = None
    support_info = None
    try:
        support_centroid, support_info = get_scene_support_centroid_on_axis(
            contact_axis,
            quantile=support_anchor_quantile,
            band_ratio=support_anchor_band_ratio,
        )
    except Exception as exc:
        print(f"support_profile: failed={exc}", file=sys.stdout)

    if anchor_support_xy and support_centroid is not None:
        try:
            delta_vec = contact_point - support_centroid
            lateral_delta = delta_vec - contact_axis * float(
                delta_vec.dot(contact_axis))
            if lateral_delta.length > 1e-8:
                for obj in get_scene_root_objects():
                    obj.matrix_world.translation += lateral_delta
                bpy.context.view_layer.update()
            print(
                "support_anchor:",
                f"enabled=True",
                f"lateral_delta={tuple(round(v, 6) for v in lateral_delta)}",
                f"support_samples={support_info.get('support_samples') if support_info else 'n/a'}",
                f"band_m={round(float(support_info.get('band_m', 0.0)), 6) if support_info else 'n/a'}",
                file=sys.stdout,
            )
        except Exception as exc:
            print(f"support_anchor: enabled=True failed={exc}",
                  file=sys.stdout)
    _record_transform_stage(
        "post_support_anchor_xy",
        {
            "anchor_support_xy":
            bool(anchor_support_xy),
            "support_centroid":
            _vector_to_list(support_centroid)
            if support_centroid is not None else None,
            "support_info_summary": {
                "support_samples":
                int(support_info.get("support_samples", 0))
                if support_info else None,
                "support_q05_proj":
                float(support_info.get("support_q05_proj")) if support_info
                and support_info.get("support_q05_proj") is not None else None,
                "support_q25_proj":
                float(support_info.get("support_q25_proj")) if support_info
                and support_info.get("support_q25_proj") is not None else None,
                "support_q01_proj":
                float(support_info.get("support_q01_proj")) if support_info
                and support_info.get("support_q01_proj") is not None else None,
                "quantile_proj":
                float(support_info.get("quantile_proj")) if support_info
                and support_info.get("quantile_proj") is not None else None,
                "band_m":
                float(support_info.get("band_m")) if support_info
                and support_info.get("band_m") is not None else None,
            },
        },
    )

    z_offset2 = 0.0
    try:
        robust_proj = get_scene_projection_quantile_on_axis(
            contact_axis,
            quantile=0.02,
            max_samples_per_object=50000,
        )
        min_proj = get_scene_min_projection_on_axis(contact_axis)
        support_samples = int(support_info.get("support_samples",
                                               0)) if support_info else 0
        support_contact_proj = (
            float(support_info.get("support_q05_proj")) if support_info
            and support_info.get("support_q05_proj") is not None else None)
        support_q25_proj = (
            float(support_info.get("support_q25_proj")) if support_info
            and support_info.get("support_q25_proj") is not None else None)
        support_guard_proj = (
            float(support_info.get("support_q01_proj")) if support_info
            and support_info.get("support_q01_proj") is not None else None)
        grounding_component = _choose_grounding_component(
            contact_axis,
            support_info,
            robust_proj=robust_proj,
            min_proj=min_proj,
        )
        grounding_decision = (grounding_component.get(
            "decision", "global_support") if grounding_component is not None
                              else "global_support")
        grounding_metrics = (grounding_component.get("metrics", {})
                             if grounding_component is not None else {})
        chosen_component = (grounding_component.get("chosen_component")
                            if grounding_component is not None else None)
        lowest_component = (grounding_component.get("lowest_component")
                            if grounding_component is not None else None)
        dominant_component = (grounding_component.get("dominant_component")
                              if grounding_component is not None else None)
        if chosen_component is not None:
            support_contact_proj = float(
                chosen_component.get("projection_q05",
                                     chosen_component.get("projection_min")))
            support_q25_proj = float(
                chosen_component.get("projection_q25",
                                     chosen_component.get("projection_q05")))
            support_guard_proj = float(
                chosen_component.get("projection_min",
                                     chosen_component.get("projection_q05")))
        if chosen_component is not None:
            support_samples = int(
                chosen_component.get("vertex_count", support_samples))
        use_support_contact = support_contact_proj is not None and support_samples >= 8
        visual_sink_m = 0.0
        if use_support_contact and support_q25_proj is not None:
            support_bottom_curve = max(
                0.0, float(support_q25_proj - support_contact_proj))
            if support_bottom_curve > 0.003:
                visual_sink_m = min(0.012, 0.5 * support_bottom_curve)
        source_proj = (support_contact_proj if use_support_contact else
                       (robust_proj if robust_proj is not None else min_proj))
        if source_proj is not None:

            target_proj = float(contact_axis.dot(contact_point))
            delta = float(target_proj - float(source_proj))
            if chosen_component is not None:
                source_name = "dominant_component_q05"
            else:
                source_name = 'support_q05' if use_support_contact else (
                    'q02' if robust_proj is not None else 'bbox_min')
            offset_vec = contact_axis * delta
            for obj in get_scene_root_objects():
                obj.matrix_world.translation += offset_vec

            post_min_proj = get_scene_min_projection_on_axis(contact_axis)
            post_support_centroid, post_support_info = get_scene_support_centroid_on_axis(
                contact_axis,
                quantile=support_anchor_quantile,
                band_ratio=support_anchor_band_ratio,
            )
            del post_support_centroid
            post_guard_proj = (float(post_support_info.get("support_q01_proj"))
                               if use_support_contact and post_support_info
                               and post_support_info.get("support_q01_proj")
                               is not None else post_min_proj)
            min_allowed = target_proj - max(float(ground_contact_epsilon),
                                            0.004)
            if post_guard_proj is not None and post_guard_proj < min_allowed:
                correction = float(min_allowed - float(post_guard_proj))
                correction_vec = contact_axis * correction
                for obj in get_scene_root_objects():
                    obj.matrix_world.translation += correction_vec
                offset_vec += correction_vec

            if visual_sink_m > 0.0:
                sink_vec = contact_axis * (-float(visual_sink_m))
                for obj in get_scene_root_objects():
                    obj.matrix_world.translation += sink_vec
                offset_vec += sink_vec

            z_offset2 = float(offset_vec.z)
            outlier_gap = (float(support_contact_proj -
                                 min_proj) if use_support_contact
                           and min_proj is not None else None)
            _record_transform_stage(
                "post_ground_snap",
                {
                    "ground_snap": {
                        "source":
                        str(source_name),
                        "target_proj":
                        float(target_proj),
                        "robust_proj_q02":
                        float(robust_proj)
                        if robust_proj is not None else None,
                        "global_min_proj":
                        float(min_proj) if min_proj is not None else None,
                        "support_contact_proj_q05":
                        float(support_contact_proj)
                        if support_contact_proj is not None else None,
                        "support_q25_proj":
                        float(support_q25_proj)
                        if support_q25_proj is not None else None,
                        "support_guard_proj_q01":
                        float(support_guard_proj)
                        if support_guard_proj is not None else None,
                        "delta":
                        float(delta),
                        "z_offset1":
                        float(z_offset1),
                        "z_offset2":
                        float(z_offset2),
                        "outlier_gap":
                        float(outlier_gap)
                        if outlier_gap is not None else None,
                        "visual_sink_m":
                        float(visual_sink_m),
                        "grounding_decision":
                        str(grounding_decision),
                    }
                },
            )
            print(
                "ground_snap:",
                f"axis={tuple(round(v, 6) for v in contact_axis)}",
                f"source={source_name}",
                f"delta={round(delta, 6)}",
                f"z_delta={round(z_offset2, 6)}",
                f"support_samples={support_samples if support_info else 'n/a'}",
                f"support_q05={round(float(support_contact_proj), 6) if support_contact_proj is not None else 'n/a'}",
                f"support_q25={round(float(support_q25_proj), 6) if support_q25_proj is not None else 'n/a'}",
                f"visual_sink_m={round(float(visual_sink_m), 6)}",
                f"global_min={round(float(min_proj), 6) if min_proj is not None else 'n/a'}",
                f"outlier_gap={round(float(outlier_gap), 6) if outlier_gap is not None else 'n/a'}",
                f"grounding_decision={grounding_decision}",
                file=sys.stdout,
            )
            try:
                _export_support_debug_artifacts(
                    support_debug_dir,
                    object_path=object_path,
                    ood_id=ood_id,
                    root_object_name=root_object.name
                    if root_object is not None else None,
                    scene_bbox_min=scene_bbox()[0],
                    scene_bbox_max=scene_bbox()[1],
                    contact_axis=contact_axis,
                    contact_point=contact_point,
                    support_info=support_info,
                    grounding_component_decision=grounding_decision,
                    grounding_component_metrics=grounding_metrics,
                    chosen_component=chosen_component,
                    lowest_component=lowest_component,
                    dominant_component=dominant_component,
                    source_name=source_name,
                    robust_proj=robust_proj,
                    min_proj=min_proj,
                    support_contact_proj=support_contact_proj,
                    support_q25_proj=support_q25_proj,
                    support_guard_proj=support_guard_proj,
                    target_proj=target_proj,
                    delta=delta,
                    z_offset=z_offset1 + z_offset2,
                    outlier_gap=outlier_gap,
                    visual_sink_m=visual_sink_m,
                    ground_contact_epsilon=ground_contact_epsilon,
                )
                if support_debug_dir:
                    print(f"support_debug: wrote={support_debug_dir}",
                          file=sys.stdout)
            except Exception as exc:
                print(f"support_debug: failed={exc}", file=sys.stdout)
        else:
            if root_object is not None:
                root_object.matrix_world = transformation_matrix
    except Exception:
        if root_object is not None:
            root_object.matrix_world = transformation_matrix

    bpy.context.view_layer.update()

    for obj in bpy.context.scene.objects:
        if obj.name == root_object.name:
            matrix_world = obj.matrix_world
        else:
            continue

    z_offset = z_offset1 + z_offset2

    bbox_min, bbox_max = scene_bbox()
    _record_transform_stage(
        "final",
        {
            "z_offset_total": float(z_offset),
            "bbox_min_final": [float(v) for v in bbox_min],
            "bbox_max_final": [float(v) for v in bbox_max],
        },
    )

    if transform_trace is not None:
        try:
            os.makedirs(support_debug_dir, exist_ok=True)
            with open(os.path.join(support_debug_dir, "transform_trace.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump(transform_trace, f, indent=2)
        except Exception as exc:
            print(f"support_debug_transform_trace: failed={exc}",
                  file=sys.stdout)

    bpy.ops.object.select_all(action="DESELECT")

    _setup_world_lighting_for_frame(
        time_stamp=time_stamp,
        weather_api={
            "cloud_cover": cloud_cover,
            "direct_illumination": direct_illumination,
            "indirect_illumination": indirect_illumination,
        },
        city=city,
    )

    matrix_world = Matrix([[round(x, 6) for x in row] for row in matrix_world])
    print('final matrix_world:', matrix_world, file=sys.stdout)
    return z_offset, euler_angles, translation_vector, scale, matrix_world, bbox_min, bbox_max


def sample_ood_object_and_render(data):

    pose = data['pose']
    ood_id = data['ood_id']
    path = data['path']
    scale = data['scale']
    base_scale = float(data.get('base_scale', scale))
    target_ref_dim_m = data.get('target_ref_dim_m', None)
    scale_policy = str(data.get('scale_policy', 'auto')
                       or 'auto').strip().lower()
    description = data.get('description', '')
    size_bucket = data.get('size_bucket', '')
    frame_id = data['frame_id']
    lvis = data['lvis']
    gpu_id = data['gpu_id']
    frame_map = data['frame_map']
    frames_with_seg = data['frames_with_seg']
    camera_data = data['camera_data']
    file_name = data['file_name']
    ood_details = data['ood_details']
    existing_entry = None
    if isinstance(ood_details, dict):
        if ood_id in ood_details:
            existing_entry = ood_details[ood_id]
        elif str(ood_id) in ood_details:
            existing_entry = ood_details[str(ood_id)]
            try:
                del ood_details[str(ood_id)]
            except Exception:
                pass
        if existing_entry is None:
            existing_entry = {}
        ood_details[ood_id] = existing_entry
    sensor = data['sensor']
    adjustment = data.get('adjustment', None)
    eval = data.get('eval', False)
    base_dir = data.get('base_dir', 'Transfer')
    size_multiplier = data.get('size_multiplier', 1.0)
    blender_require_gpu = bool(data.get('blender_require_gpu', False))
    no_shadow_cycles_samples = int(data.get('no_shadow_cycles_samples', 24))
    if no_shadow_cycles_samples < 1:
        no_shadow_cycles_samples = 1
    no_shadow_cycles_use_denoising = bool(
        data.get('no_shadow_cycles_use_denoising', False))
    shadow_plane_point = data.get('shadow_plane_point')
    shadow_plane_normal = data.get('shadow_plane_normal')
    shadow_catcher_size = float(data.get('shadow_catcher_size', 40.0))
    ground_contact_epsilon = float(data.get('ground_contact_epsilon', 0.01))
    enable_support_debug_exports = bool(
        data.get('enable_support_debug_exports', False))

    camera_pose = None
    all_resource_metrics = []
    if not eval:
        render_path = f"{base_dir}/{file_name}/Renders/"
    else:
        render_path = f"{base_dir}/{file_name}/Evaluation/Renders/"
    for sensor_type, len_frames in frames_with_seg.items():
        if len_frames == 0:
            continue

        if sensor_type == 'lidar' and sensor != 'lidar':
            print(f"Skipping lidar image rendering for OOD {ood_id}; "
                  f"sensor mode is '{sensor}'.")
            continue
        render_path_type = f"{render_path}ood_{ood_id}/{sensor_type}"
        if not os.path.exists(render_path_type):
            os.makedirs(render_path_type, exist_ok=True)
        else:

            try:
                for entry in os.listdir(render_path_type):
                    entry_path = os.path.join(render_path_type, entry)

                    if os.path.islink(entry_path) or os.path.isfile(
                            entry_path):
                        try:
                            os.unlink(entry_path)
                        except Exception as e:
                            print(f"Could not remove file {entry_path}: {e}",
                                  file=sys.stderr)

                    elif os.path.isdir(entry_path):
                        try:
                            shutil.rmtree(entry_path)
                        except Exception as e:
                            print(
                                f"Could not remove directory {entry_path}: {e}",
                                file=sys.stderr)
            except Exception as e:
                print(f"Error clearing directory {render_path_type}: {e}",
                      file=sys.stderr)

        if len_frames > 0:

            time_stamp = camera_data[sensor_type][0]["time_stamp"]
            light_energy = camera_data[sensor_type][0]["light"]
            weather = camera_data[sensor_type][0].get("weather_api", {})
            cloud_cover = weather.get('cloud_cover', 0) if weather else 0
            direct_illumination = weather.get('direct_illumination',
                                              None) if weather else None
            indirect_illumination = weather.get('indirect_illumination',
                                                None) if weather else None

            support_debug_dir = None
            if enable_support_debug_exports and ood_id is not None:
                if not eval:
                    support_debug_root = os.path.join(base_dir, file_name,
                                                      "Support_Debug")
                else:
                    support_debug_root = os.path.join(base_dir, file_name,
                                                      "Evaluation",
                                                      "Support_Debug")
                support_debug_dir = os.path.join(support_debug_root,
                                                 f"ood_{int(ood_id)}")

            print('before', pose)
            z_off, eu_an, transl, scale, matrix_world, bbox_min, bbox_max = clear_scene_and_import_object(
                path,
                pose,
                time_stamp,
                scale,
                ood_id,
                sun_en=light_energy,
                cloud_cover=cloud_cover,
                file_name=file_name,
                file_dir=render_path,
                size_multiplier=size_multiplier,
                target_ref_dim_m=target_ref_dim_m,
                scale_policy=scale_policy,
                description=description,
                direct_illumination=direct_illumination,
                indirect_illumination=indirect_illumination,
                eval=eval,
                ground_contact_epsilon=ground_contact_epsilon,
                shadow_plane_point=shadow_plane_point,
                shadow_plane_normal=shadow_plane_normal,
                support_debug_dir=support_debug_dir,
            )
            print('after', matrix_world)
            if 'bbox_3d' not in ood_details.get(ood_id, {}):
                ood_details.setdefault(ood_id, {})['bbox_3d'] = {}
            ood_details[ood_id]['bbox_3d']['min'] = np.array(bbox_min).tolist()
            ood_details[ood_id]['bbox_3d']['max'] = np.array(bbox_max).tolist()

            ood_details[ood_id]['pose'] = Matrix(matrix_world)
            ood_details[ood_id]['render_pose_override'] = [
                list(row) for row in matrix_world
            ]
            if shadow_plane_point is not None:
                ood_details[ood_id]['shadow_plane_point_override'] = [
                    float(shadow_plane_point[0]),
                    float(shadow_plane_point[1]),
                    float(shadow_plane_point[2]),
                ]
            if shadow_plane_normal is not None:
                ood_details[ood_id]['shadow_plane_normal_override'] = [
                    float(shadow_plane_normal[0]),
                    float(shadow_plane_normal[1]),
                    float(shadow_plane_normal[2]),
                ]
            ood_details[ood_id]['ground_contact_epsilon_override'] = float(
                ground_contact_epsilon)
            ood_details[ood_id]['shadow_catcher_size_override'] = float(
                shadow_catcher_size)

            ood_details[ood_id]['path'] = path
            ood_details[ood_id]['lvis'] = lvis
            ood_details[ood_id]['scale'] = float(base_scale)
            ood_details[ood_id]['effective_scale'] = float(scale)
            if target_ref_dim_m is not None:
                try:
                    ood_details[ood_id]['target_ref_dim_m'] = float(
                        target_ref_dim_m)
                except Exception:
                    pass
            ood_details[ood_id]['scale_policy'] = scale_policy
            if description:
                ood_details[ood_id]['description'] = str(description)
            if size_bucket:
                ood_details[ood_id]['size_bucket'] = str(size_bucket)

            def per_frame_camera_entries(frame_idx):
                frame_data = camera_data[sensor_type][frame_idx]
                if (sensor_type == 'camera' and isinstance(frame_data, dict)
                        and 'all_cameras' in frame_data):
                    return frame_data['all_cameras']
                return {'front': frame_data}

            def resolve_image_size(camera_entry):
                width = int(camera_entry.get("width", 0) or 0)
                height = int(camera_entry.get("height", 0) or 0)
                if width > 0 and height > 0:
                    return width, height
                projection = camera_entry.get("projection")
                if projection is not None and len(projection) >= 4:
                    cx = float(projection[2])
                    cy = float(projection[3])
                    return max(1, int(round(2.0 * cx))), max(
                        1, int(round(2.0 * cy)))
                return 1, 1

            for i in range(len_frames):
                frame_camera_entries = per_frame_camera_entries(i)
                first_entry = next(iter(frame_camera_entries.values())
                                   ) if frame_camera_entries else {}
                frame_time_stamp = first_entry.get(
                    "time_stamp",
                    camera_data[sensor_type][i].get("time_stamp", time_stamp))
                frame_weather = first_entry.get(
                    "weather_api",
                    camera_data[sensor_type][i].get("weather_api", {}))
                frame_cloud_cover = frame_weather.get(
                    'cloud_cover', 0) if frame_weather else 0
                frame_direct_illumination = frame_weather.get(
                    'direct_illumination', None) if frame_weather else None
                if frame_direct_illumination is not None and frame_direct_illumination > 1.5:
                    frame_direct_illumination = frame_direct_illumination / 700.0
                _setup_world_lighting_for_frame(frame_time_stamp,
                                                frame_weather)
                for camera_name, camera_entry in frame_camera_entries.items():
                    camera_pose = camera_entry["camera_pose"]
                    projection = camera_entry["projection"]
                    readout_time = camera_entry["readout_time"]
                    image_width, image_height = resolve_image_size(
                        camera_entry)
                    frame_render_dir = render_path_type
                    if sensor_type == 'camera':
                        frame_render_dir = os.path.join(
                            render_path_type, camera_name)
                        os.makedirs(frame_render_dir, exist_ok=True)
                    set_camera_from_matrix(
                        camera_pose,
                        projection,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    try:
                        render_scene(
                            f"{frame_render_dir}/no_shadow_frame_00{frame_map[sensor_type][i]}.png",
                            readout_time,
                            resolution_x=image_width,
                            resolution_y=image_height,
                            cycles=True,
                            cycles_samples_override=no_shadow_cycles_samples,
                            cycles_use_denoising=no_shadow_cycles_use_denoising,
                            cloud_cover=frame_cloud_cover,
                            gpu_id=gpu_id,
                            direct_illumination=frame_direct_illumination,
                            require_gpu=blender_require_gpu,
                        )
                        if support_debug_dir:
                            try:
                                _export_camera_space_contact_debug(
                                    support_debug_dir,
                                    render_path=
                                    f"{frame_render_dir}/no_shadow_frame_00{frame_map[sensor_type][i]}.png",
                                    camera_name=camera_name,
                                    frame_token=
                                    f"{frame_map[sensor_type][i]:04d}",
                                    image_width=image_width,
                                    image_height=image_height,
                                )
                            except Exception as exc:
                                print(
                                    f"camera_space_contact_debug: failed={exc}",
                                    file=sys.stdout)
                    except KeyboardInterrupt:
                        break

            spot = False
            if not light_energy < 0.3:
                if isinstance(shadow_plane_point,
                              (list, tuple)) and len(shadow_plane_point) == 3:
                    catcher_translation = tuple(
                        float(v) for v in shadow_plane_point)
                else:
                    catcher_translation = (float(transl[0]), float(transl[1]),
                                           float(transl[2]))
                create_shadow_catcher(
                    translation=catcher_translation,
                    normal=shadow_plane_normal,
                    size=shadow_catcher_size,
                )
            else:
                spot = True

            for i in range(len_frames):
                frame_camera_entries = per_frame_camera_entries(i)
                first_entry = next(iter(frame_camera_entries.values())
                                   ) if frame_camera_entries else {}
                frame_time_stamp = first_entry.get(
                    "time_stamp",
                    camera_data[sensor_type][i].get("time_stamp", time_stamp))
                frame_weather = first_entry.get(
                    "weather_api",
                    camera_data[sensor_type][i].get("weather_api", {}))
                _setup_world_lighting_for_frame(frame_time_stamp,
                                                frame_weather)
                for camera_name, camera_entry in frame_camera_entries.items():
                    camera_pose = camera_entry["camera_pose"]
                    projection = camera_entry["projection"]
                    readout_time = camera_entry["readout_time"]
                    image_width, image_height = resolve_image_size(
                        camera_entry)
                    weather = camera_entry.get(
                        "weather_api",
                        camera_data[sensor_type][i].get("weather_api"))
                    cloud_cover = weather.get('cloud_cover',
                                              0) if weather else 0
                    direct_illumination = weather.get(
                        'direct_illumination', None) if weather else None
                    max_illumination = 700
                    if direct_illumination is not None:
                        direct_illumination /= max_illumination

                    frame_render_dir = render_path_type
                    if sensor_type == 'camera':
                        frame_render_dir = os.path.join(
                            render_path_type, camera_name)
                        os.makedirs(frame_render_dir, exist_ok=True)
                    set_camera_from_matrix(
                        camera_pose,
                        projection,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    try:
                        result = render_scene(
                            f"{frame_render_dir}/frame_00{frame_map[sensor_type][i]}.png",
                            readout_time,
                            resolution_x=image_width,
                            resolution_y=image_height,
                            cycles=True,
                            cloud_cover=cloud_cover,
                            gpu_id=gpu_id,
                            direct_illumination=direct_illumination,
                            require_gpu=blender_require_gpu,
                        )
                        print(
                            f"Rendered frame {i} for OOD {ood_id} with sensor "
                            f"{sensor_type}/{camera_name}")
                        if 'resource_metrics' in result and result[
                                'resource_metrics']:
                            all_resource_metrics.append(
                                result['resource_metrics'])
                    except KeyboardInterrupt:
                        break

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.wm.quit_blender()

    return {
        "ood_details": ood_details,
        "render_path": render_path,
        "resource_metrics": all_resource_metrics,
    }


def render_ood_scale_sweep_frame(data):
    pose = data["pose"]
    ood_id = int(data.get("ood_id", 0) or 0)
    path = data["path"]
    lvis = data.get("lvis", "unknown")
    description = data.get("description", "")
    size_bucket = data.get("size_bucket", "")
    target_ref_dim_m = data.get("target_ref_dim_m", None)
    scale_policy = str(data.get("scale_policy", "auto")
                       or "auto").strip().lower()
    gpu_id = data.get("gpu_id", None)
    render_scale = float(data.get("scale", 1.0))
    base_size_multiplier = float(data.get("size_multiplier", 1.0))
    use_cycles = bool(data.get("use_cycles", False))
    cycles_samples = int(data.get("cycles_samples", 8))
    if cycles_samples < 1:
        cycles_samples = 1
    cycles_use_denoising = bool(data.get("cycles_use_denoising", False))
    blender_require_gpu = bool(data.get("blender_require_gpu", False))
    shadow_plane_point = data.get("shadow_plane_point")
    shadow_plane_normal = data.get("shadow_plane_normal")
    ground_contact_epsilon = float(data.get("ground_contact_epsilon", 0.01))
    anchor_support_xy = bool(data.get("anchor_support_xy", False))
    support_anchor_quantile = float(data.get("support_anchor_quantile", 0.02))
    support_anchor_band_ratio = float(
        data.get("support_anchor_band_ratio", 0.03))
    frame_token = str(data.get("frame_token", "0000"))
    camera_name = str(data.get("camera_name", "front"))
    camera_entry = data["camera_entry"]
    output_dir = str(data["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    candidate_size_multipliers = []
    raw_candidates = data.get("candidate_size_multipliers",
                              data.get("candidate_scales", []))
    for idx, value in enumerate(raw_candidates):
        try:
            parsed = float(value)
        except Exception:
            continue
        if not np.isfinite(parsed) or parsed <= 0:
            continue
        candidate_size_multipliers.append((idx, parsed))
    if not candidate_size_multipliers:
        raise ValueError(
            "candidate_size_multipliers must contain at least one positive float"
        )

    image_width = int(camera_entry.get("width", 0) or 0)
    image_height = int(camera_entry.get("height", 0) or 0)
    if image_width <= 0 or image_height <= 0:
        projection = camera_entry.get("projection")
        if projection is not None and len(projection) >= 4:
            image_width = max(1, int(round(2.0 * float(projection[2]))))
            image_height = max(1, int(round(2.0 * float(projection[3]))))
        else:
            image_width = 1
            image_height = 1

    time_stamp = camera_entry.get("time_stamp", 0)
    readout_time = camera_entry.get("readout_time", 0.0)
    camera_pose = camera_entry["camera_pose"]
    projection = camera_entry["projection"]
    weather = camera_entry.get("weather_api", {})
    cloud_cover = weather.get("cloud_cover", 0) if weather else 0
    direct_illumination = weather.get("direct_illumination",
                                      None) if weather else None
    if direct_illumination is not None and direct_illumination > 1.5:
        direct_illumination = direct_illumination / 700.0

    results = []
    render_root = os.path.join(output_dir, "renders")
    os.makedirs(render_root, exist_ok=True)

    for candidate_index, candidate_size_multiplier in candidate_size_multipliers:
        _setup_world_lighting_for_frame(time_stamp, weather)
        clear_scene_and_import_object(
            path,
            pose,
            time_stamp,
            scale=render_scale,
            ood_id=ood_id,
            sun_en=float(camera_entry.get("light", 0.8) or 0.8),
            cloud_cover=cloud_cover,
            file_name=f"scale_sweep_{ood_id}",
            file_dir=output_dir,
            size_multiplier=base_size_multiplier * candidate_size_multiplier,
            target_ref_dim_m=target_ref_dim_m,
            scale_policy=scale_policy,
            description=description,
            direct_illumination=direct_illumination,
            indirect_illumination=weather.get("indirect_illumination", None)
            if weather else None,
            eval=True,
            ground_contact_epsilon=ground_contact_epsilon,
            shadow_plane_point=shadow_plane_point,
            shadow_plane_normal=shadow_plane_normal,
            anchor_support_xy=anchor_support_xy,
            support_anchor_quantile=support_anchor_quantile,
            support_anchor_band_ratio=support_anchor_band_ratio,
        )
        set_camera_from_matrix(
            camera_pose,
            projection,
            image_width=image_width,
            image_height=image_height,
        )
        render_path = os.path.join(
            render_root,
            f"scale_{candidate_index:02d}_{candidate_size_multiplier:.6f}_frame_{frame_token}_{camera_name}.png",
        )
        result = render_scene(
            render_path,
            readout_time,
            resolution_x=image_width,
            resolution_y=image_height,
            cycles=use_cycles,
            cycles_samples_override=cycles_samples if use_cycles else None,
            cycles_use_denoising=cycles_use_denoising if use_cycles else None,
            cloud_cover=cloud_cover,
            gpu_id=gpu_id,
            direct_illumination=direct_illumination,
            require_gpu=blender_require_gpu,
        )
        results.append({
            "candidate_index": int(candidate_index),
            "scale": float(render_scale),
            "size_multiplier": float(candidate_size_multiplier),
            "camera_name": camera_name,
            "frame_token": frame_token,
            "render_path": render_path,
            "resource_metrics": result.get("resource_metrics"),
            "image_width": int(image_width),
            "image_height": int(image_height),
        })

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.wm.quit_blender()
    return {
        "ood_id": int(ood_id),
        "lvis": lvis,
        "description": description,
        "size_bucket": size_bucket,
        "target_ref_dim_m": target_ref_dim_m,
        "results": results,
    }


def visualize_ood_locations_and_get_data_for_pcd(data):
    def select_all_children(obj):
        obj.select_set(True)
        for child in obj.children:
            select_all_children(child)

    def preprocess_collection_for_ray_cast(obj):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True,
                                       rotation=True,
                                       scale=True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete_loose(use_verts=True,
                                  use_edges=True,
                                  use_faces=True)
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')

    file_name = data['file_name']
    frames_with_seg = data['frames_with_seg']
    ood_details = data['ood_details']
    file_dir = data.get('file_dir', 'Transfer')
    lidar_import_mode = str(
        data.get('lidar_import_mode', 'legacy') or 'legacy').strip().lower()
    if lidar_import_mode not in {'legacy', 'exact_render_path'}:
        lidar_import_mode = 'legacy'

    check_dir = f'{file_dir}/{file_name}/lidar_data'
    lidar_parameters = None
    lidar_range_pose = None
    azimuth = None
    inclination = None

    for file in os.listdir(check_dir):
        if file == 'lidar_parameters.npy':
            lidar_parameters = np.load(f'{check_dir}/lidar_parameters.npy',
                                       allow_pickle=True).item()
        if file == 'lidar_range_pose.npy':
            lidar_range_pose = np.load(f'{check_dir}/lidar_range_pose.npy',
                                       allow_pickle=True).item()
        if file == 'azimuth.npy':
            azimuth = np.load(f'{check_dir}/azimuth.npy',
                              allow_pickle=True).item()
        if file == 'inclination.npy':
            inclination = np.load(f'{check_dir}/inclination.npy',
                                  allow_pickle=True).item()

    missing_inputs = []
    if lidar_parameters is None:
        missing_inputs.append('lidar_parameters.npy')
    if lidar_range_pose is None:
        missing_inputs.append('lidar_range_pose.npy')
    if azimuth is None:
        missing_inputs.append('azimuth.npy')
    if inclination is None:
        missing_inputs.append('inclination.npy')
    if missing_inputs:
        raise RuntimeError("Missing LiDAR raycast prerequisites in "
                           f"'{check_dir}': {', '.join(missing_inputs)}")

    dataset_world_to_blender_world = Matrix(
        ((0, -1, 0, 0), (1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))

    beam_inclinations = 64
    azimuth_resolution = 2650

    results = {
        'hit_points': {},
        'hit_normals': {},
        'hit_targets': {},
        'lidar_positions': {},
        'vehicle_poses': {},
        'ray_directions': {},
        'mapping_dict': {},
        'intensity': {},
        'import_debug': {},
    }

    old_objects = set(bpy.context.scene.objects)

    def _extract_world_point(value, default=None):
        if value is None:
            return None if default is None else Vector(default)
        try:
            arr = np.asarray(value, dtype=np.float64)
        except Exception:
            return None if default is None else Vector(default)
        if arr.shape == (4, 4):
            return Vector((float(arr[0, 3]), float(arr[1, 3]), float(arr[2,
                                                                         3])))
        flat = arr.reshape(-1)
        if flat.size >= 3:
            return Vector((float(flat[0]), float(flat[1]), float(flat[2])))
        return None if default is None else Vector(default)

    def _clear_all_meshes_from_scene():
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def _precompute_frame_states():
        frame_states = {}
        for sensor_type, len_frames in frames_with_seg.items():
            if len_frames == 0:
                continue
            results['hit_points'][sensor_type] = {}
            results['hit_normals'][sensor_type] = {}
            results['hit_targets'][sensor_type] = {}
            results['lidar_positions'][sensor_type] = {}
            results['vehicle_poses'][sensor_type] = {}
            results['ray_directions'][sensor_type] = {}
            results['intensity'][sensor_type] = {}

            azimuth_rad = azimuth[sensor_type]
            inclination_rad = inclination[sensor_type]
            x = np.cos(azimuth_rad) * np.cos(inclination_rad)
            y = np.sin(azimuth_rad) * np.cos(inclination_rad)
            z = np.sin(inclination_rad)
            directions = np.stack([x, y, z], axis=-1)
            identity_matrix = np.eye(4)

            frame_states[sensor_type] = {}
            for frame_id in range(len_frames):
                if frame_id not in lidar_parameters[sensor_type]:
                    continue
                lidar_data = lidar_parameters[sensor_type][frame_id]
                lidar_pose_dataset_world = Matrix(
                    np.array(lidar_data['pose_global']))
                if lidar_import_mode == 'exact_render_path':
                    lidar_pose_blender_world = lidar_pose_dataset_world
                else:
                    lidar_pose_blender_world = dataset_world_to_blender_world @ lidar_pose_dataset_world
                lidar_position = lidar_pose_blender_world.translation
                results['lidar_positions'][sensor_type][frame_id] = list(
                    lidar_position)

                rotation_matrix = np.asarray(lidar_pose_blender_world.to_3x3())
                rotated_dirs = directions[frame_id].reshape(
                    -1, 3) @ rotation_matrix.T
                rotated_dirs /= np.linalg.norm(rotated_dirs,
                                               axis=1,
                                               keepdims=True)
                rotated_dirs = rotated_dirs.reshape(beam_inclinations,
                                                    azimuth_resolution, 3)
                results['ray_directions'][sensor_type][
                    frame_id] = rotated_dirs.tolist()

                frame_states[sensor_type][frame_id] = {
                    'lidar_pose_blender_world':
                    lidar_pose_blender_world,
                    'lidar_pose_np':
                    np.array(lidar_data['pose']),
                    'lidar_range_pose_frame':
                    lidar_range_pose[sensor_type][frame_id],
                    'rotated_dirs':
                    rotated_dirs,
                    'identity_matrix':
                    identity_matrix,
                }
        return frame_states

    def _raycast_current_scene_for_mapping(mapping_name):
        max_distance = 75.0
        depsgraph = bpy.context.evaluated_depsgraph_get()
        scene = bpy.context.scene
        material_cache = {}
        total_hits = 0
        for sensor_type, state_by_frame in frame_states.items():
            for frame_id, state in state_by_frame.items():
                lidar_pose_np = state['lidar_pose_np']
                lidar_range_pose_frame = state['lidar_range_pose_frame']
                rotated_dirs = state['rotated_dirs']
                identity_matrix = state['identity_matrix']

                hit_points_list = results['hit_points'][
                    sensor_type].setdefault(frame_id, [])
                hit_normals_list = results['hit_normals'][
                    sensor_type].setdefault(frame_id, [])
                hit_targets_list = results['hit_targets'][
                    sensor_type].setdefault(frame_id, [])
                intensity_list = results['intensity'][sensor_type].setdefault(
                    frame_id, [])

                for inc in range(beam_inclinations):
                    for azimuth_step in range(azimuth_resolution):
                        vehicle_top_pose = lidar_range_pose_frame[
                            inc, azimuth_step, :]
                        if np.allclose(vehicle_top_pose, identity_matrix):
                            continue

                        lidar_top_pose = vehicle_top_pose @ lidar_pose_np
                        x, y, z = lidar_top_pose[0, 3], lidar_top_pose[
                            1, 3], lidar_top_pose[2, 3]
                        if lidar_import_mode == 'exact_render_path':
                            ray_origin = Vector((x, y, z))
                        else:
                            ray_origin = Vector((-y, x, z))
                        ray_direction = Vector(rotated_dirs[inc, azimuth_step])

                        hit, location, normal, _, target, _ = scene.ray_cast(
                            depsgraph=depsgraph,
                            origin=ray_origin,
                            direction=ray_direction,
                            distance=max_distance,
                        )
                        if not hit:
                            continue
                        distance = (location - ray_origin).length
                        if distance >= max_distance:
                            continue

                        hit_points_list.append(
                            [location.x, location.y, location.z])
                        hit_normals_list.append([normal.x, normal.y, normal.z])
                        target_name_for_mapping = mapping_name if mapping_name is not None else target.name
                        hit_targets_list.append(target_name_for_mapping)

                        target_name = target.name
                        if target_name in material_cache:
                            metallic, roughness = material_cache[target_name]
                        else:
                            metallic, roughness = 0.0, 1.0
                            if target.material_slots:
                                for slot in target.material_slots:
                                    mat = slot.material
                                    if mat and mat.use_nodes:
                                        for node in mat.node_tree.nodes:
                                            if node.type == 'BSDF_PRINCIPLED':
                                                metallic = node.inputs[
                                                    "Metallic"].default_value
                                                roughness = node.inputs[
                                                    "Roughness"].default_value
                                                break
                                        if metallic > 0:
                                            break
                            material_cache[target_name] = (metallic, roughness)

                        intensity = 0.0
                        if metallic > 0 or roughness < 1.0:
                            angle = max(0.0, -ray_direction.dot(normal))
                            roughness_factor = 1.0 - roughness
                            intensity = max(
                                0.2, metallic * roughness_factor * angle)
                        intensity_list.append(intensity)
                        total_hits += 1
        return total_hits

    frame_states = _precompute_frame_states()

    print('before entering', ood_details)
    if lidar_import_mode == 'exact_render_path':
        total_hits = 0
        for ood_id, details in ood_details.items():
            if not isinstance(details, dict) or not details.get('path'):
                continue
            pose_value = details.get('render_pose_override',
                                     details.get('pose'))
            try:
                pose_arr = np.asarray(pose_value, dtype=np.float64)
            except Exception:
                continue
            if pose_arr.ndim == 1 and pose_arr.size == 16:
                pose_arr = pose_arr.reshape(4, 4)
            if pose_arr.shape != (4, 4) or not np.isfinite(pose_arr).all():
                continue

            _clear_all_meshes_from_scene()
            shadow_plane_point = details.get('shadow_plane_point_override')
            if shadow_plane_point is None:
                shadow_plane_point_vec = _extract_world_point(
                    details.get('sampled_position', details.get('position')),
                    default=(pose_arr[0, 3], pose_arr[1, 3], pose_arr[2, 3]),
                )
                shadow_plane_point = [
                    float(shadow_plane_point_vec.x),
                    float(shadow_plane_point_vec.y),
                    float(shadow_plane_point_vec.z),
                ]
            shadow_plane_normal = details.get('shadow_plane_normal_override')
            if shadow_plane_normal is None:
                shadow_plane_normal = [0.0, 0.0, 1.0]

            z_off, _eu_an, _transl, _scale, exact_matrix_world, bbox_min, bbox_max = clear_scene_and_import_object(
                object_path=details['path'],
                pose=pose_arr.tolist(),
                time_stamp=0,
                scale=float(
                    details.get('effective_scale', details.get('scale', 1.0))),
                ood_id=int(ood_id),
                file_name=file_name,
                file_dir=file_dir,
                size_multiplier=1.0,
                target_ref_dim_m=details.get('target_ref_dim_m', None),
                scale_policy=details.get('scale_policy', 'auto'),
                description=details.get('description', ''),
                eval=False,
                ground_contact_epsilon=float(
                    details.get('ground_contact_epsilon_override', 0.002)),
                shadow_plane_point=shadow_plane_point,
                shadow_plane_normal=shadow_plane_normal,
            )
            bpy.context.view_layer.update()
            mesh_objects = [
                obj for obj in bpy.context.scene.objects
                if obj.type == 'MESH' and obj.visible_get()
            ]
            if not mesh_objects:
                continue

            bpy.ops.object.select_all(action='DESELECT')
            for obj in mesh_objects:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = mesh_objects[0]
            if len(mesh_objects) > 1:
                bpy.ops.object.join()

            joined_mesh = bpy.context.view_layer.objects.active
            joined_mesh.name = f"OOD_{int(ood_id)}_joined"
            if joined_mesh.data is not None:
                joined_mesh.data.name = f"OOD_{int(ood_id)}_joined_mesh"
            preprocess_collection_for_ray_cast(joined_mesh)
            results['mapping_dict'][ood_id] = [joined_mesh.name]
            results['import_debug'][ood_id] = {
                'mode': 'exact_render_path',
                'matrix_world': [list(row) for row in exact_matrix_world],
                'bbox_min': [float(v) for v in bbox_min],
                'bbox_max': [float(v) for v in bbox_max],
                'z_offset': float(z_off),
            }
            total_hits += _raycast_current_scene_for_mapping(joined_mesh.name)

        lidar_total_frames = len(frame_states.get('lidar', {}))
        fail_count = 0
        for frame_id in frame_states.get('lidar', {}):
            if not results['hit_points']['lidar'].get(frame_id):
                fail_count += 1
        if lidar_total_frames > 0 and fail_count == lidar_total_frames and total_hits == 0:
            print("Error: Point Cloud cannot be generated for any frame.",
                  file=sys.stderr)
            sys.exit(1)
    else:
        for ood_id, details in ood_details.items():
            if not isinstance(details, dict):
                continue
            if 'pose' not in details:
                continue
            pose_value = details.get('pose')
            try:
                pose_arr = np.asarray(pose_value, dtype=np.float64)
            except Exception:
                continue
            if pose_arr.ndim == 1 and pose_arr.size == 16:
                pose_arr = pose_arr.reshape(4, 4)
            if pose_arr.shape != (4, 4) or not np.isfinite(pose_arr).all():
                continue
            linear = pose_arr[:3, :3]
            if float(np.linalg.norm(linear)) < 1e-8:
                repaired = np.eye(4, dtype=np.float64)
                repaired[:3, 3] = pose_arr[:3, 3]
                pose_arr = repaired
            if not details.get('path'):
                continue

            load_object(details['path'])

            new_objects = set(bpy.context.scene.objects) - old_objects
            imported_objects = list(new_objects)
            old_objects.update(new_objects)
            root_obj = next(
                (obj for obj in imported_objects if not obj.parent), None)

            if not root_obj:
                continue

            ood_pose_dataset_world = Matrix(pose_arr.tolist())
            ood_pose_blender_world = dataset_world_to_blender_world @ ood_pose_dataset_world
            root_obj.matrix_world = ood_pose_blender_world
            bpy.context.view_layer.update()

            bpy.ops.object.select_all(action='DESELECT')
            select_all_children(root_obj)
            mesh_objects = [
                obj for obj in bpy.context.selected_objects
                if obj.type == 'MESH' and obj.visible_get()
            ]
            if not mesh_objects:
                continue

            bpy.ops.object.select_all(action='DESELECT')
            for obj in mesh_objects:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = mesh_objects[0]
            if len(mesh_objects) > 1:
                bpy.ops.object.join()

            joined_mesh = bpy.context.view_layer.objects.active
            joined_mesh.name = f"OOD_{int(ood_id)}_joined"
            if joined_mesh.data is not None:
                joined_mesh.data.name = f"OOD_{int(ood_id)}_joined_mesh"
            preprocess_collection_for_ray_cast(joined_mesh)
            results['mapping_dict'][ood_id] = [joined_mesh.name]

        total_hits = _raycast_current_scene_for_mapping(None)
        lidar_total_frames = len(frame_states.get('lidar', {}))
        fail_count = 0
        for frame_id in frame_states.get('lidar', {}):
            if not results['hit_points']['lidar'].get(frame_id):
                fail_count += 1
        if lidar_total_frames > 0 and fail_count == lidar_total_frames and total_hits == 0:
            print("Error: Point Cloud cannot be generated for any frame.",
                  file=sys.stderr)
            sys.exit(1)

    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    return {
        'raycast_results': results,
    }


def main():
    """Main function to be called from the command line."""
    try:

        try:

            idx = sys.argv.index("--")
            custom_args = sys.argv[idx + 1:]

            method_name = custom_args[0]
            json_data_arg = custom_args[1]
            json_output_path = custom_args[2]
        except (ValueError, IndexError) as e:
            print(f"Error: Could not parse arguments. {e}", file=sys.stderr)
            sys.exit(1)

        method_map = {
            "sample_ood_object_and_render": sample_ood_object_and_render,
            "visualize_ood_locations_and_get_data_for_pcd":
            visualize_ood_locations_and_get_data_for_pcd,
            "render_ood_scale_sweep_frame": render_ood_scale_sweep_frame,
        }

        method = method_map.get(method_name, None)

        if method and callable(method):

            if json_data_arg.startswith("@"):
                payload_path = json_data_arg[1:]
                with open(payload_path, "r") as f:
                    blender_data = json.load(f)
            else:
                blender_data = json.loads(json_data_arg)
            result_data = method(blender_data)

            with open(json_output_path, "w") as f:
                if "ood_details" in result_data:
                    for ood_id, details in result_data["ood_details"].items():
                        if "pose" in details and isinstance(
                                details["pose"], Matrix):

                            details["pose"] = [
                                list(row) for row in details["pose"]
                            ]
                json.dump(result_data, f, default=str)

        else:
            print(f"Error: Method '{method_name}' not found or not callable.",
                  file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        import traceback
        print(f"An unexpected error occurred in Blender script: {e}",
              file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
