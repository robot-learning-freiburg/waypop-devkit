import os

os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_enable_xla_devices=false')

import re
import sys
import time
import json
import gzip
import copy
import pickle
import shutil
import threading
import functools
import subprocess
import multiprocessing
import hashlib
import tempfile
import argparse
import colorsys
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from inspect import signature
import fcntl

import tensorflow as tf

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree
import open3d as o3d
try:
    import wandb
except ImportError:
    wandb = None
import psutil
import cv2
import skimage as sk
from skimage.filters import sobel

try:
    import requests
except ImportError:
    requests = None

if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import camera_segmentation_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.v2.perception.utils import lidar_utils
from waymo_open_dataset.v2.perception import lidar as _v2_lidar
try:
    from waymo_to_semanticKITTI.lib.waymo2semantickitti import Waymo2SemanticKITTI
except ImportError:
    Waymo2SemanticKITTI = None
try:
    from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
except Exception as _pycam_err:
    py_camera_model_ops = None
    print(
        f"Warning: failed to import Waymo camera ops: {_pycam_err}. "
        "Camera projection via py_camera_model_ops will be unavailable.",
        file=sys.stderr,
    )

from utils.image_utils import histogram_match_with_reference, histogram_match_with_soft_blend, smooth_alpha, SEGMENTATION_COLOR_MAP, blend_image_with_shadow
from utils.depth_fill_utils import fill_in_multiscale as depth_fill_in_multiscale
from placement_checks import assigned_point, check_sampled_position, checked_saved_object_ids

from pathlib import Path


def _parse_range_image_and_camera_projection_compat(frame):
    range_images = {}
    camera_projections = {}
    seg_labels = {}
    range_image_top_pose = open_dataset.MatrixFloat()
    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_compressed) > 0:
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.range_image_compressed, 'ZLIB')
            ri = open_dataset.MatrixFloat()
            ri.ParseFromString(range_image_str_tensor.numpy())
            range_images[laser.name] = [ri]

            if laser.name == open_dataset.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                range_image_top_pose = open_dataset.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    range_image_top_pose_str_tensor.numpy())

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.camera_projection_compressed, 'ZLIB')
            cp = open_dataset.MatrixInt32()
            cp.ParseFromString(camera_projection_str_tensor.numpy())
            camera_projections[laser.name] = [cp]

            if len(laser.ri_return1.segmentation_label_compressed) > 0:
                seg_label_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.segmentation_label_compressed, 'ZLIB')
                seg_label = open_dataset.MatrixInt32()
                seg_label.ParseFromString(seg_label_str_tensor.numpy())
                seg_labels[laser.name] = [seg_label]
        if len(laser.ri_return2.range_image_compressed) > 0:
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.range_image_compressed, 'ZLIB')
            ri = open_dataset.MatrixFloat()
            ri.ParseFromString(range_image_str_tensor.numpy())
            range_images[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, 'ZLIB')
            cp = open_dataset.MatrixInt32()
            cp.ParseFromString(camera_projection_str_tensor.numpy())
            camera_projections[laser.name].append(cp)

            if len(laser.ri_return2.segmentation_label_compressed) > 0:
                seg_label_str_tensor = tf.io.decode_compressed(
                    laser.ri_return2.segmentation_label_compressed, 'ZLIB')
                seg_label = open_dataset.MatrixInt32()
                seg_label.ParseFromString(seg_label_str_tensor.numpy())
                seg_labels[laser.name].append(seg_label)
    return range_images, camera_projections, seg_labels, range_image_top_pose


frame_utils.parse_range_image_and_camera_projection = _parse_range_image_and_camera_projection_compat

torch = None
transforms = None
_TORCH_IMPORT_ERROR = None


def ensure_torch_imported(required=False, context=""):
    """Lazy-import torch/torchvision so non-PCT request-only runs avoid fragile libtorch startup."""
    global torch, transforms, _TORCH_IMPORT_ERROR
    if torch is not None and transforms is not None:
        return torch, transforms
    if _TORCH_IMPORT_ERROR is not None:
        if required:
            raise RuntimeError(
                f"Torch dependencies unavailable{f' during {context}' if context else ''}: {_TORCH_IMPORT_ERROR}"
            ) from _TORCH_IMPORT_ERROR
        return None, None
    try:
        import torch as _torch
        from torchvision import transforms as _transforms
        torch = _torch
        transforms = _transforms
        return torch, transforms
    except Exception as exc:
        _TORCH_IMPORT_ERROR = exc
        if required:
            raise RuntimeError(
                f"Failed to import torch/torchvision{f' during {context}' if context else ''}: {exc}"
            ) from exc
        return None, None


def str2bool(value):
    """Parse CLI boolean flags robustly."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, got '{value}'.")


def resolve_path(path_value):
    """Expand '~' and return absolute filesystem path."""
    return os.path.abspath(os.path.expanduser(path_value))


def parse_csv_numeric(value, cast=float):
    """Parse comma-separated numeric config values into a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        out = []
        for v in value:
            try:
                out.append(cast(v))
            except Exception:
                continue
        return out
    text = str(value).strip()
    if text == "":
        return []
    out = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(cast(token))
        except Exception:
            continue
    return out


def parse_csv_strings(value):
    """Parse comma-separated strings into a normalized lowercase list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        out = []
        for v in value:
            token = str(v).strip().lower()
            if token:
                out.append(token)
        return out
    text = str(value).strip()
    if text == "":
        return []
    return [tok.strip().lower() for tok in text.split(",") if tok.strip()]


def parse_csv_ints(value):
    """Parse comma-separated integers into a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        out = []
        for v in value:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    text = str(value).strip()
    if text == "":
        return []
    out = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except Exception:
            continue
    return out


def parse_camera_quota_spec(spec, valid_names):
    """Parse camera quota string like 'front:20,side_left:10,all:8'."""
    quotas = {}
    default_quota = None
    text = str(spec or "").strip()
    if text == "":
        return quotas, default_quota
    valid_set = {str(name).strip().lower() for name in valid_names}
    for token in text.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        key, value = token.split(":", 1)
        key = key.strip().lower()
        try:
            quota_value = int(value.strip())
        except Exception:
            continue
        if quota_value < 1:
            continue
        if key in {"all", "default", "*"}:
            default_quota = quota_value
        elif key in valid_set:
            quotas[key] = quota_value
    return quotas, default_quota


def parse_camera_name_list(spec, valid_names):
    """Parse comma-separated camera names and keep only known entries."""
    text = str(spec or "").strip().lower()
    if text == "":
        return []
    valid_set = {str(name).strip().lower() for name in valid_names}
    out = []
    for token in text.split(","):
        name = token.strip().lower()
        if not name or name not in valid_set:
            continue
        if name not in out:
            out.append(name)
    return out


def compose_export_camera_panoptic_label(semantic_label, instance_label,
                                         divisor):
    semantic_arr = np.asarray(semantic_label, dtype=np.int32)
    instance_arr = np.asarray(instance_label, dtype=np.int32)
    if semantic_arr.ndim == 3 and semantic_arr.shape[-1] == 1:
        semantic_arr = semantic_arr.squeeze(-1)
    if instance_arr.ndim == 3 and instance_arr.shape[-1] == 1:
        instance_arr = instance_arr.squeeze(-1)
    return semantic_arr * int(divisor) + instance_arr


def encode_export_panoptic_ids_to_rgb(panoptic_ids):
    ids_u32 = np.asarray(panoptic_ids, dtype=np.uint32)
    if ids_u32.ndim == 3 and ids_u32.shape[-1] == 1:
        ids_u32 = ids_u32.squeeze(-1)
    return np.stack(
        [
            (ids_u32 & 255).astype(np.uint8),
            ((ids_u32 >> 8) & 255).astype(np.uint8),
            ((ids_u32 >> 16) & 255).astype(np.uint8),
        ],
        axis=-1,
    )


INSERTION_PRESET_CONFIG = {
    "small": {
        "object_range": (10, 20),
        "candidate_budget": 1600,
        "max_objects_per_frame": 6,
        "distance_bucket_weights": "0.55,0.35,0.10",
    },
    "medium": {
        "object_range": (30, 50),
        "candidate_budget": 2600,
        "max_objects_per_frame": 10,
        "distance_bucket_weights": "0.45,0.35,0.20",
    },
    "stress": {
        "object_range": (50, 100),
        "candidate_budget": 4200,
        "max_objects_per_frame": 14,
        "distance_bucket_weights": "0.35,0.35,0.30",
    },
}

WAYMO_CAMERA_ORDER = [
    open_dataset.CameraName.SIDE_LEFT,
    open_dataset.CameraName.FRONT_LEFT,
    open_dataset.CameraName.FRONT,
    open_dataset.CameraName.FRONT_RIGHT,
    open_dataset.CameraName.SIDE_RIGHT,
]

WAYMO_CAMERA_ID_TO_NAME = {
    open_dataset.CameraName.FRONT: "front",
    open_dataset.CameraName.FRONT_LEFT: "front_left",
    open_dataset.CameraName.FRONT_RIGHT: "front_right",
    open_dataset.CameraName.SIDE_LEFT: "side_left",
    open_dataset.CameraName.SIDE_RIGHT: "side_right",
}

WAYMO_CAMERA_NAME_TO_ID = {
    name: int(cid)
    for cid, name in WAYMO_CAMERA_ID_TO_NAME.items()
}


def camera_id_to_name(camera_id):
    """Return a stable lowercase camera name for a Waymo camera enum id."""
    try:
        camera_id_int = int(camera_id)
    except Exception:
        return str(camera_id).lower()
    if camera_id_int in WAYMO_CAMERA_ID_TO_NAME:
        return WAYMO_CAMERA_ID_TO_NAME[camera_id_int]
    try:
        return open_dataset.CameraName.Name.Name(camera_id_int).lower()
    except Exception:
        return str(camera_id_int)


class ResourceLogger:
    _nvml_initialized = False
    GPU_COUNT = 0
    _gpu_id = -1
    wandb_enabled = False

    @classmethod
    def set_gpu_id(cls, gpu_id):
        cls._gpu_id = int(gpu_id)
        print(f"ResourceLogger: Monitoring GPU ID set to {cls._gpu_id}")

    @classmethod
    def _initialize_nvml(cls):
        if cls._nvml_initialized:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            cls.GPU_COUNT = pynvml.nvmlDeviceGetCount()
            print(f"NVML initialized with {cls.GPU_COUNT} GPUs")
            cls._nvml_initialized = True
        except Exception as e:
            print(f"NVML initialization skipped: {e}")
            cls.GPU_COUNT = 0

    @staticmethod
    def get_gpu_utilization():
        gpus_info = {}
        try:
            import pynvml
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpus_info[i] = {
                    'compute': util.gpu,
                    'memory': round(mem.used / (1024**3), 2),
                    'memory_total': round(mem.total / (1024**3), 2)
                }
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return gpus_info

    @staticmethod
    def _get_tf_gpu_metrics():
        metrics = {}
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if ResourceLogger._gpu_id < len(gpus):
                mem_info = tf.config.experimental.get_memory_info(
                    f'GPU:{ResourceLogger._gpu_id}')
                metrics[f"GPU_{ResourceLogger._gpu_id}_TF_Used(GB)"] = round(
                    mem_info['current'] / (1024**3), 2)
                metrics[f"GPU_{ResourceLogger._gpu_id}_TF_Peak(GB)"] = round(
                    mem_info['peak'] / (1024**3), 2)
        except Exception as e:
            print(f"TF GPU {ResourceLogger._gpu_id} monitoring failed: {e}")
        return metrics

    @staticmethod
    def _get_nvml_gpu_metrics():
        metrics = {}
        ResourceLogger._initialize_nvml()
        if ResourceLogger.GPU_COUNT > 0 and ResourceLogger._gpu_id < ResourceLogger.GPU_COUNT:
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(
                    ResourceLogger._gpu_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics[
                    f"GPU_{ResourceLogger._gpu_id}_NVML_Compute(%)"] = util.gpu
                metrics[
                    f"GPU_{ResourceLogger._gpu_id}_NVML_Memory_Used(GB)"] = round(
                        mem.used / (1024**3), 2)
            except Exception as e:
                print(
                    f"NVML GPU {ResourceLogger._gpu_id} monitoring failed: {e}"
                )
        return metrics

    @staticmethod
    def log_resources_to_wandb(prefix=""):
        logs = {}
        try:
            vm = psutil.virtual_memory()
            logs.update({
                f"{prefix}RAM_Used(GB)":
                round(vm.used / (1024**3), 2),
                f"{prefix}RAM_Available(GB)":
                round(vm.available / (1024**3), 2)
            })
        except Exception as e:
            print(f"CPU monitoring failed: {e}")

        try:
            logs.update(ResourceLogger._get_tf_gpu_metrics())
            logs.update(ResourceLogger._get_nvml_gpu_metrics())
        except Exception as e:
            print(f"GPU monitoring failed: {e}")

        if logs and ResourceLogger.wandb_enabled:
            try:
                wandb.log(logs)
            except Exception as e:
                print(f"WandB logging failed: {e}")

    @staticmethod
    def log_resources_during_execution(prefix=None, interval=5):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                _prefix = prefix if prefix is not None else func.__name__
                stop_event = threading.Event()

                def monitor():
                    while not stop_event.is_set():
                        ResourceLogger.log_resources_to_wandb(f"{_prefix}/")
                        time.sleep(interval)

                monitor_thread = threading.Thread(target=monitor, daemon=True)
                monitor_thread.start()

                try:
                    result = func(*args, **kwargs)
                finally:
                    stop_event.set()
                    monitor_thread.join(timeout=interval * 2)

                return result

            return wrapper

        return decorator


class Sampler():
    blender_path = os.environ.get("BLENDER_PATH") or shutil.which("blender") or ""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "utils", "render_utils.py")
    blender_rendered_frame_pattern = re.compile(
        r"Rendered frame\s+(\d+)\s+for OOD\s+(\d+)\s+with sensor\s+(.+)$")
    blender_sample_pattern = re.compile(r"Sample\s+(\d+)\s*/\s*(\d+)")
    blender_saved_pattern = re.compile(r"^Saved:\s+'(.+)'")
    color_map = {
        0: [0, 0, 0],
        1: [0, 0, 142],
        2: [0, 0, 70],
        3: [0, 60, 100],
        4: [61, 133, 198],
        5: [180, 0, 0],
        6: [255, 0, 0],
        7: [220, 20, 60],
        8: [246, 178, 107],
        9: [250, 170, 30],
        10: [153, 153, 153],
        11: [230, 145, 56],
        12: [119, 11, 32],
        13: [0, 0, 230],
        14: [70, 70, 70],
        15: [107, 142, 35],
        16: [234, 153, 153],
        17: [217, 210, 233],
        18: [128, 64, 128],
        19: [234, 209, 220],
        20: [102, 102, 102],
        21: [107, 142, 35],
        22: [244, 35, 232],
    }

    max_label = max(color_map.keys())
    color_array = np.zeros((max_label + 1, 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_array[label] = color

    color_dict = {
        1: [152, 255, 152],
        2: [255, 255, 153],
        3: [255, 110, 96],
        4: [134, 206, 235],
        5: [211, 211, 211],
        6: [200, 191, 231],
        7: [183, 209, 185],
        8: [255, 248, 220],
        9: [70, 80, 100],
        10: [255, 255, 255]
    }

    camera_surface_semantic_names = {
        20: "road",
        21: "lane_marker",
        22: "curb",
        23: "sidewalk",
        24: "walkable",
        26: "other_ground",
    }

    def __init__(self,
                 file_name,
                 base_dir='Transfer',
                 image_path=None,
                 runtime_config=None):
        runtime_config = runtime_config or {}

        self.runtime_config = runtime_config
        self._progress_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.blender_path = runtime_config.get('blender_path',
                                               Sampler.blender_path)
        self.blender_logs_dir = runtime_config.get(
            'blender_logs_dir', os.path.join(os.getcwd(),
                                             "blender_debug_logs"))
        self.blender_debug_all = bool(
            runtime_config.get('blender_debug_all', False))
        self.blender_require_gpu = bool(
            runtime_config.get('blender_require_gpu', False))
        self.blender_timeout_s = int(
            runtime_config.get('blender_timeout_s', 1800))
        if self.blender_timeout_s < 300:
            self.blender_timeout_s = 300
        self.blender_heartbeat_s = int(
            runtime_config.get('blender_heartbeat_s', 10))
        if self.blender_heartbeat_s < 5:
            self.blender_heartbeat_s = 5
        self.blender_timeout_per_cycle_render_s = float(
            runtime_config.get('blender_timeout_per_cycle_render_s', 4.0))
        if self.blender_timeout_per_cycle_render_s < 0.5:
            self.blender_timeout_per_cycle_render_s = 0.5
        self.blender_timeout_safety_factor = float(
            runtime_config.get('blender_timeout_safety_factor', 1.35))
        if self.blender_timeout_safety_factor < 1.0:
            self.blender_timeout_safety_factor = 1.0
        self.blender_timeout_overhead_s = int(
            runtime_config.get('blender_timeout_overhead_s', 300))
        if self.blender_timeout_overhead_s < 60:
            self.blender_timeout_overhead_s = 60
        self.blender_timeout_max_s = int(
            runtime_config.get('blender_timeout_max_s', 21600))
        if self.blender_timeout_max_s < self.blender_timeout_s:
            self.blender_timeout_max_s = self.blender_timeout_s
        self.blender_allow_partial_timeout = bool(
            runtime_config.get('blender_allow_partial_timeout', True))
        self.blender_partial_timeout_min_completion = float(
            runtime_config.get('blender_partial_timeout_min_completion', 0.85))
        self.blender_partial_timeout_min_completion = float(
            np.clip(self.blender_partial_timeout_min_completion, 0.0, 1.0))
        self.blender_worker_policy = str(
            runtime_config.get('blender_worker_policy',
                               'safe')).strip().lower()
        if self.blender_worker_policy not in {'safe', 'serial'}:
            print(
                f"\033[33mWarning: unsupported blender_worker_policy='{self.blender_worker_policy}', "
                "falling back to 'safe'.\033[0m")
            self.blender_worker_policy = 'safe'
        self.blender_max_workers = int(
            runtime_config.get('blender_max_workers', 0))
        if self.blender_max_workers < 0:
            self.blender_max_workers = 0
        self.blender_gpu_ids = self._resolve_blender_gpu_ids(
            runtime_config.get('blender_gpu_ids', ''))
        self.blender_worker_gpu_plan = self._build_blender_worker_gpu_plan()
        self.blender_effective_workers = int(
            max(1, len(self.blender_worker_gpu_plan)))
        print(
            "\033[36mBlender worker plan: "
            f"policy={self.blender_worker_policy}, "
            f"max_workers={'auto' if self.blender_max_workers <= 0 else self.blender_max_workers}, "
            f"workers={self.blender_effective_workers}, "
            f"gpu_plan={self.blender_worker_gpu_plan}\033[0m")
        self.blender_cpu_threads = int(
            runtime_config.get('blender_cpu_threads', 0))
        if self.blender_cpu_threads < 0:
            self.blender_cpu_threads = 0
        self.cpu_thread_cap = int(
            runtime_config.get('_effective_cpu_thread_cap', 0))
        if self.cpu_thread_cap < 0:
            self.cpu_thread_cap = 0
        self.objects_train_json = runtime_config.get(
            'objects_train_json',
            os.path.join(os.getcwd(), "objects_dicts_train.json"))
        self.objects_valid_json = runtime_config.get(
            'objects_valid_json',
            os.path.join(os.getcwd(), "objects_dicts_valid.json"))
        self.objects_eval_json = runtime_config.get(
            'objects_eval_json', os.path.join(os.getcwd(), "objects.json"))
        self.enable_sam = bool(runtime_config.get('enable_sam', False))
        self.sam_python_executable = runtime_config.get(
            'sam_python_executable')
        self.sam2_dir = runtime_config.get('sam2_dir')
        self.enable_pct = bool(runtime_config.get('enable_pct', False))
        self.pct_python_executable = runtime_config.get(
            'pct_python_executable', sys.executable)
        self.pct_gpu_device = runtime_config.get('pct_gpu_device', 'cuda:0')
        self.pct_require_gpu = bool(runtime_config.get('pct_require_gpu',
                                                       True))
        self.pct_model = str(runtime_config.get('pct_model',
                                                'vit')).strip().lower()
        self.pct_temporal_consistency = str(
            runtime_config.get('pct_temporal_consistency',
                               'per_camera')).strip().lower()
        if self.pct_temporal_consistency not in {
                'off', 'per_camera', 'global'
        }:
            print(
                f"\033[33mWarning: Unsupported pct_temporal_consistency "
                f"'{self.pct_temporal_consistency}'. Falling back to 'per_camera'.\033[0m"
            )
            self.pct_temporal_consistency = 'per_camera'
        self.pct_temporal_ema = float(
            runtime_config.get('pct_temporal_ema', 0.6))
        if not np.isfinite(self.pct_temporal_ema):
            self.pct_temporal_ema = 0.6
        self.pct_temporal_ema = float(np.clip(self.pct_temporal_ema, 0.0, 1.0))
        self.pct_temporal_min_pixels = int(
            runtime_config.get('pct_temporal_min_pixels', 200))
        if self.pct_temporal_min_pixels < 1:
            self.pct_temporal_min_pixels = 1
        pct_gain_clip = runtime_config.get('pct_temporal_gain_clip',
                                           [0.7, 1.3])
        if isinstance(pct_gain_clip,
                      (list, tuple)) and len(pct_gain_clip) >= 2:
            gmin = float(pct_gain_clip[0])
            gmax = float(pct_gain_clip[1])
        else:
            gmin = float(runtime_config.get('pct_temporal_gain_clip_min', 0.7))
            gmax = float(runtime_config.get('pct_temporal_gain_clip_max', 1.3))
        if not np.isfinite(gmin):
            gmin = 0.7
        if not np.isfinite(gmax):
            gmax = 1.3
        if gmin > gmax:
            gmin, gmax = gmax, gmin
        self.pct_temporal_gain_clip_min = float(max(0.05, gmin))
        self.pct_temporal_gain_clip_max = float(
            max(self.pct_temporal_gain_clip_min, gmax))
        self.pct_temporal_feather_sigma = float(
            runtime_config.get('pct_temporal_feather_sigma', 1.5))
        if not np.isfinite(self.pct_temporal_feather_sigma
                           ) or self.pct_temporal_feather_sigma < 0.0:
            self.pct_temporal_feather_sigma = 1.5
        self.pct_temporal_save_debug = bool(
            runtime_config.get('pct_temporal_save_debug', False))
        self.pct_stage2_global_match = bool(
            runtime_config.get('pct_stage2_global_match', True))
        self.pct_stage2_strength = float(
            runtime_config.get('pct_stage2_strength', 0.35))
        if not np.isfinite(self.pct_stage2_strength):
            self.pct_stage2_strength = 0.35
        self.pct_stage2_strength = float(
            np.clip(self.pct_stage2_strength, 0.0, 1.0))
        self.pct_stage2_min_pixels = int(
            runtime_config.get('pct_stage2_min_pixels', 200))
        if self.pct_stage2_min_pixels < 1:
            self.pct_stage2_min_pixels = 1
        pct_stage2_gain_clip = runtime_config.get('pct_stage2_gain_clip',
                                                  [0.75, 1.25])
        if isinstance(pct_stage2_gain_clip,
                      (list, tuple)) and len(pct_stage2_gain_clip) >= 2:
            s2_gmin = float(pct_stage2_gain_clip[0])
            s2_gmax = float(pct_stage2_gain_clip[1])
        else:
            s2_gmin = float(
                runtime_config.get('pct_stage2_gain_clip_min', 0.75))
            s2_gmax = float(
                runtime_config.get('pct_stage2_gain_clip_max', 1.25))
        if not np.isfinite(s2_gmin):
            s2_gmin = 0.75
        if not np.isfinite(s2_gmax):
            s2_gmax = 1.25
        if s2_gmin > s2_gmax:
            s2_gmin, s2_gmax = s2_gmax, s2_gmin
        self.pct_stage2_gain_clip_min = float(max(0.05, s2_gmin))
        self.pct_stage2_gain_clip_max = float(
            max(self.pct_stage2_gain_clip_min, s2_gmax))
        self.pct_stage2_save_debug = bool(
            runtime_config.get('pct_stage2_save_debug', False))
        self.enable_weather_api = bool(
            runtime_config.get('enable_weather_api', False))
        self.weather_api_timeout_s = float(
            runtime_config.get('weather_api_timeout_s', 1.5))
        if self.weather_api_timeout_s <= 0:
            self.weather_api_timeout_s = 1.5
        if self.pct_model not in {'vit', 'cnn', 'both'}:
            print(
                f"\033[33mWarning: Unsupported pct_model '{self.pct_model}'. Falling back to 'vit'.\033[0m"
            )
            self.pct_model = 'vit'
        self.ground_semantic_ids = np.array([17, 18, 19, 20, 21, 22],
                                            dtype=np.int32)
        self.ground_semantic_sparse_ratio = float(
            runtime_config.get('ground_semantic_sparse_ratio', 0.15))
        if not np.isfinite(self.ground_semantic_sparse_ratio):
            self.ground_semantic_sparse_ratio = 0.15
        self.ground_semantic_sparse_ratio = float(
            np.clip(self.ground_semantic_sparse_ratio, 0.0, 1.0))
        camera_ground_ids = runtime_config.get('camera_ground_semantic_ids',
                                               [20, 21, 22, 23, 24, 26])
        self.camera_ground_semantic_ids = set(
            int(v) for v in camera_ground_ids)
        self.enable_camera_ground_anchor_filter = bool(
            runtime_config.get('enable_camera_ground_anchor_filter', True))
        self.camera_ground_anchor_only_fallback = bool(
            runtime_config.get('camera_ground_anchor_only_fallback', True))
        self.camera_ground_min_hits = int(
            runtime_config.get('camera_ground_min_hits', 2))
        self.camera_ground_min_cameras = int(
            runtime_config.get('camera_ground_min_cameras', 1))
        self.camera_ground_min_ratio = float(
            runtime_config.get('camera_ground_min_ratio', 0.75))
        self.camera_non_ground_max_ratio = float(
            runtime_config.get('camera_non_ground_max_ratio', 0.25))
        self.camera_ground_label_patch_radius = int(
            runtime_config.get('camera_ground_label_patch_radius', 2))
        self.ground_plane_radius = float(
            runtime_config.get('ground_plane_radius', 1.2))
        self.ground_plane_min_neighbors = int(
            runtime_config.get('ground_plane_min_neighbors', 30))
        self.ground_plane_min_normal_z = float(
            runtime_config.get('ground_plane_min_normal_z', 0.9))
        self.local_floor_radius = float(
            runtime_config.get('local_floor_radius', 1.2))
        self.local_floor_percentile = float(
            runtime_config.get('local_floor_percentile', 10.0))
        self.local_floor_max_offset = float(
            runtime_config.get('local_floor_max_offset', 0.4))
        self.local_floor_min_neighbors = int(
            runtime_config.get('local_floor_min_neighbors', 20))
        self.min_fov_hits = int(runtime_config.get('min_fov_hits', 2))
        self.enable_adjacent_camera_overlap_filter = bool(
            runtime_config.get('enable_adjacent_camera_overlap_filter', True))
        self.adjacent_camera_overlap_min_hits = int(
            runtime_config.get('adjacent_camera_overlap_min_hits', 1))
        if self.adjacent_camera_overlap_min_hits < 1:
            self.adjacent_camera_overlap_min_hits = 1
        self.required_visible_camera_spec = str(
            runtime_config.get('required_visible_cameras', ''))
        self.required_visible_camera_min_hits = int(
            runtime_config.get('required_visible_camera_min_hits', 1))
        if self.required_visible_camera_min_hits < 1:
            self.required_visible_camera_min_hits = 1
        self.required_render_min_pixels = int(
            runtime_config.get('required_render_min_pixels', 64))
        if self.required_render_min_pixels < 1:
            self.required_render_min_pixels = 1
        self.required_render_max_retries = int(
            runtime_config.get('required_render_max_retries', 4))
        if self.required_render_max_retries < 1:
            self.required_render_max_retries = 1
        self.no_shadow_cycles_samples = int(
            runtime_config.get('no_shadow_cycles_samples', 24))
        if self.no_shadow_cycles_samples < 1:
            self.no_shadow_cycles_samples = 1
        self.no_shadow_cycles_use_denoising = bool(
            runtime_config.get('no_shadow_cycles_use_denoising', False))
        self.enable_shadow_debug_exports = bool(
            runtime_config.get('enable_shadow_debug_exports', False))
        self.shadow_debug_contact_dilate_px = int(
            runtime_config.get('shadow_debug_contact_dilate_px', 6))
        if self.shadow_debug_contact_dilate_px < 0:
            self.shadow_debug_contact_dilate_px = 0
        self.enable_shadow_quality_gate = bool(
            runtime_config.get('enable_shadow_quality_gate', True))
        self.shadow_quality_dilate_px = int(
            runtime_config.get('shadow_quality_dilate_px', 6))
        if self.shadow_quality_dilate_px < 0:
            self.shadow_quality_dilate_px = 0
        self.shadow_quality_min_ratio = float(
            runtime_config.get('shadow_quality_min_ratio', 0.03))
        self.shadow_quality_max_ratio = float(
            runtime_config.get('shadow_quality_max_ratio', 2.0))
        self.shadow_quality_min_contact_ratio = float(
            runtime_config.get('shadow_quality_min_contact_ratio', 0.002))
        self.shadow_quality_min_pass_frames = int(
            runtime_config.get('shadow_quality_min_pass_frames', 1))
        if self.shadow_quality_min_pass_frames < 1:
            self.shadow_quality_min_pass_frames = 1
        self.shadow_quality_min_pass_fraction = float(
            runtime_config.get('shadow_quality_min_pass_fraction', 0.34))
        if self.shadow_quality_min_pass_fraction < 0.0:
            self.shadow_quality_min_pass_fraction = 0.0
        if self.shadow_quality_min_pass_fraction > 1.0:
            self.shadow_quality_min_pass_fraction = 1.0
        self.shadow_quality_min_direct_illumination = float(
            runtime_config.get('shadow_quality_min_direct_illumination', 0.08))
        if self.shadow_quality_min_direct_illumination < 0.0:
            self.shadow_quality_min_direct_illumination = 0.0
        if self.shadow_quality_min_direct_illumination > 1.0:
            self.shadow_quality_min_direct_illumination = 1.0
        self.shadow_catcher_size = float(
            runtime_config.get('shadow_catcher_size', 40.0))
        if self.shadow_catcher_size < 5.0:
            self.shadow_catcher_size = 5.0
        self.ground_contact_epsilon = float(
            runtime_config.get('ground_contact_epsilon', 0.002))
        if self.ground_contact_epsilon < 0.0:
            self.ground_contact_epsilon = 0.0
        self.side_camera_bonus = float(
            runtime_config.get('side_camera_bonus', 0.5))
        self.occupied_xy_radius = float(
            runtime_config.get('occupied_xy_radius', 1.1))
        self.occupied_min_height = float(
            runtime_config.get('occupied_min_height', 0.25))
        self.occupied_max_height = float(
            runtime_config.get('occupied_max_height', 3.0))
        self.occupied_min_points = int(
            runtime_config.get('occupied_min_points', 25))
        self.occupancy_temporal_window_frames = int(
            runtime_config.get('occupancy_temporal_window_frames', 6))
        if self.occupancy_temporal_window_frames < 0:
            self.occupancy_temporal_window_frames = 0
        self.enable_fov_candidate_pools = bool(
            runtime_config.get('enable_fov_candidate_pools', True))
        self.placement_policy = str(
            runtime_config.get('placement_policy',
                               'deterministic')).strip().lower()
        if self.placement_policy not in {'deterministic', 'gpt_oneshot'}:
            print(
                f"\033[33mWarning: unsupported placement_policy='{self.placement_policy}'. "
                "Falling back to 'deterministic'.\033[0m")
            self.placement_policy = 'deterministic'
        self.placement_gpt_enabled = bool(
            runtime_config.get('placement_gpt_enabled', False))
        self.placement_gpt_model = str(
            runtime_config.get('placement_gpt_model', 'gpt-5.2')).strip()
        self.placement_gpt_api_key_env = str(
            runtime_config.get('placement_gpt_api_key_env',
                               'OPENAI_API_KEY')).strip()
        self.placement_gpt_timeout_s = float(
            runtime_config.get('placement_gpt_timeout_s', 60.0))
        if not np.isfinite(self.placement_gpt_timeout_s
                           ) or self.placement_gpt_timeout_s <= 1.0:
            self.placement_gpt_timeout_s = 60.0
        self.placement_gpt_top_k_candidates = int(
            runtime_config.get('placement_gpt_top_k_candidates', 600))
        if self.placement_gpt_top_k_candidates < 100:
            self.placement_gpt_top_k_candidates = 100
        self.placement_gpt_backup_multiplier = int(
            runtime_config.get('placement_gpt_backup_multiplier', 3))
        if self.placement_gpt_backup_multiplier < 1:
            self.placement_gpt_backup_multiplier = 1
        self.placement_gpt_response_json = runtime_config.get(
            'placement_gpt_response_json', '')
        if isinstance(self.placement_gpt_response_json,
                      str) and self.placement_gpt_response_json.strip():
            self.placement_gpt_response_json = os.path.abspath(
                os.path.expanduser(self.placement_gpt_response_json.strip()))
        self.placement_gpt_lock_primary = bool(
            runtime_config.get('placement_gpt_lock_primary', True))
        self.require_exact_placement = bool(
            runtime_config.get('require_exact_placement', False))
        self.placement_shift_max_radius_m = float(
            runtime_config.get('placement_shift_max_radius_m', 1.6))
        if not np.isfinite(self.placement_shift_max_radius_m
                           ) or self.placement_shift_max_radius_m < 0.0:
            self.placement_shift_max_radius_m = 1.6
        self.placement_shift_step_m = float(
            runtime_config.get('placement_shift_step_m', 0.4))
        if not np.isfinite(self.placement_shift_step_m
                           ) or self.placement_shift_step_m <= 0.0:
            self.placement_shift_step_m = 0.4
        self.placement_bev_export = bool(
            runtime_config.get('placement_bev_export', True))
        self.placement_bev_cell_size_m = float(
            runtime_config.get('placement_bev_cell_size_m', 0.3))
        if not np.isfinite(self.placement_bev_cell_size_m
                           ) or self.placement_bev_cell_size_m <= 0.0:
            self.placement_bev_cell_size_m = 0.3
        self.placement_bev_margin_m = float(
            runtime_config.get('placement_bev_margin_m', 3.0))
        if not np.isfinite(self.placement_bev_margin_m
                           ) or self.placement_bev_margin_m < 0.0:
            self.placement_bev_margin_m = 3.0
        self.placement_bev_max_side_px = int(
            runtime_config.get('placement_bev_max_side_px', 1800))
        if self.placement_bev_max_side_px < 256:
            self.placement_bev_max_side_px = 256
        self.placement_guard_mode = str(
            runtime_config.get('placement_guard_mode',
                               'reuse_external_candidates')).strip().lower()
        if self.placement_guard_mode not in {
                'strict', 'reuse_external_candidates'
        }:
            print(
                f"\033[33mWarning: unsupported placement_guard_mode='{self.placement_guard_mode}'. "
                "Falling back to 'reuse_external_candidates'.\033[0m")
            self.placement_guard_mode = 'reuse_external_candidates'
        self.stuff_occlusion_min_hits = int(
            runtime_config.get('stuff_occlusion_min_hits', 8))
        self.occlusion_projection_dilation = int(
            runtime_config.get('occlusion_projection_dilation', 9))
        self.stuff_occlusion_fallback_min_hits = int(
            runtime_config.get('stuff_occlusion_fallback_min_hits', 500))
        self.thing_occlusion_fallback_min_hits = int(
            runtime_config.get('thing_occlusion_fallback_min_hits', 24))
        if self.thing_occlusion_fallback_min_hits < 1:
            self.thing_occlusion_fallback_min_hits = 1
        self.camera_occlusion_depth_fill = str(
            runtime_config.get('camera_occlusion_depth_fill',
                               'none')).strip().lower()
        if self.camera_occlusion_depth_fill not in {'none', 'multiscale'}:
            self.camera_occlusion_depth_fill = 'none'
        self.camera_occlusion_depth_fill_bbox_pad_px = int(
            runtime_config.get('camera_occlusion_depth_fill_bbox_pad_px', 48))
        if self.camera_occlusion_depth_fill_bbox_pad_px < 0:
            self.camera_occlusion_depth_fill_bbox_pad_px = 0
        self.camera_occlusion_depth_fill_radius_px = int(
            runtime_config.get('camera_occlusion_depth_fill_radius_px', 96))
        if self.camera_occlusion_depth_fill_radius_px < 1:
            self.camera_occlusion_depth_fill_radius_px = 1
        self.camera_occlusion_depth_fill_max_depth = float(
            runtime_config.get('camera_occlusion_depth_fill_max_depth', 120.0))
        if (not np.isfinite(self.camera_occlusion_depth_fill_max_depth)
                or self.camera_occlusion_depth_fill_max_depth <= 1.0):
            self.camera_occlusion_depth_fill_max_depth = 120.0
        self.camera_occlusion_temporal_bridge = bool(
            runtime_config.get('camera_occlusion_temporal_bridge', True))
        self.camera_occlusion_temporal_min_source_px = int(
            runtime_config.get('camera_occlusion_temporal_min_source_px', 128))
        if self.camera_occlusion_temporal_min_source_px < 1:
            self.camera_occlusion_temporal_min_source_px = 1
        self.camera_occlusion_temporal_drop_ratio = float(
            runtime_config.get('camera_occlusion_temporal_drop_ratio', 0.15))
        if (not np.isfinite(self.camera_occlusion_temporal_drop_ratio)
                or self.camera_occlusion_temporal_drop_ratio < 0.0
                or self.camera_occlusion_temporal_drop_ratio > 1.0):
            self.camera_occlusion_temporal_drop_ratio = 0.15
        self.camera_occlusion_temporal_support_ratio = float(
            runtime_config.get('camera_occlusion_temporal_support_ratio',
                               0.25))
        if (not np.isfinite(self.camera_occlusion_temporal_support_ratio)
                or self.camera_occlusion_temporal_support_ratio < 0.0
                or self.camera_occlusion_temporal_support_ratio > 1.0):
            self.camera_occlusion_temporal_support_ratio = 0.25
        self.camera_occlusion_temporal_max_gap = int(
            runtime_config.get('camera_occlusion_temporal_max_gap', 2))
        if self.camera_occlusion_temporal_max_gap < 0:
            self.camera_occlusion_temporal_max_gap = 0
        self.enable_stuff_occlusion_expansion = bool(
            runtime_config.get('enable_stuff_occlusion_expansion', False))
        self.enable_stuff_occlusion_fallback = bool(
            runtime_config.get('enable_stuff_occlusion_fallback', False))
        self.enable_thing_occlusion_fallback = bool(
            runtime_config.get('enable_thing_occlusion_fallback', True))
        self.max_occlusion_fraction_of_ood = float(
            runtime_config.get('max_occlusion_fraction_of_ood', 0.85))
        self.occlusion_min_common_pixels = int(
            runtime_config.get('occlusion_min_common_pixels', 5))
        self.ood_semantic_class = int(
            runtime_config.get('ood_semantic_class', 29))
        self.camera_semantic_max_class = int(
            runtime_config.get('camera_semantic_max_class', 28))
        self.export_lidar_label_metadata = bool(
            runtime_config.get('export_lidar_label_metadata', False))
        self.compute_rays_aux = bool(
            runtime_config.get('compute_rays_aux', False))
        self.enable_projection_debug = bool(
            runtime_config.get('enable_projection_debug', False))
        self.enable_intensity_debug = bool(
            runtime_config.get('enable_intensity_debug', False))
        self.raycast_lidar_only = bool(
            runtime_config.get('raycast_lidar_only', True))
        self.enable_lidar_insertion = bool(
            runtime_config.get('enable_lidar_insertion', True))
        self.enable_raw_image_export = bool(
            runtime_config.get('enable_raw_image_export', True))
        self.lidar_insertion_frame_source = str(
            runtime_config.get('lidar_insertion_frame_source',
                               'camera')).strip().lower()
        if self.lidar_insertion_frame_source not in {
                'camera', 'lidar', 'auto'
        }:
            self.lidar_insertion_frame_source = 'camera'
        self.enable_lidar_panoptic_label_recovery = bool(
            runtime_config.get('enable_lidar_panoptic_label_recovery', False))
        self.reuse_saved_pose_for_render = bool(
            runtime_config.get('reuse_saved_pose_for_render', False))
        self.load_saved_ood_details = bool(
            runtime_config.get('load_saved_ood_details', False))
        self.lidar_import_mode = str(
            runtime_config.get('lidar_import_mode',
                               'exact_render_path')).strip().lower()
        if self.lidar_import_mode not in {'legacy', 'exact_render_path'}:
            self.lidar_import_mode = 'exact_render_path'
        self.save_range_image_viz = bool(
            runtime_config.get('save_range_image_viz', False))
        self.save_point_cloud_comparison = bool(
            runtime_config.get('save_point_cloud_comparison', False))
        self.cache_parsed_range_images = bool(
            runtime_config.get('cache_parsed_range_images', True))
        self.enable_precompute_cache = bool(
            runtime_config.get('enable_precompute_cache', True))
        self.force_recompute_precompute_cache = bool(
            runtime_config.get('force_recompute_precompute_cache', False))
        self.enable_run_progress = bool(
            runtime_config.get('enable_run_progress', True))
        self.precompute_cache_tag = str(
            runtime_config.get('precompute_cache_tag', 'v1'))
        self.retry = bool(runtime_config.get('retry', False))
        self.skip_blender_render = bool(
            runtime_config.get('skip_blender_render', False))
        self.stop_after_sampling = bool(
            runtime_config.get('stop_after_sampling', False))
        self.request_only = bool(runtime_config.get('request_only', False))
        self.lidar_only_resume = bool(
            runtime_config.get('lidar_only_resume', False))
        self.num_frames = int(runtime_config.get('num_frames', 10))
        if self.num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        self.frame_id_start = int(runtime_config.get('frame_id_start', 1))
        frame_id_end_cfg = runtime_config.get('frame_id_end', None)
        self.frame_id_end = None if frame_id_end_cfg is None else int(
            frame_id_end_cfg)
        self.frame_token_list = [
            int(v)
            for v in parse_csv_ints(runtime_config.get('frame_token_list', ''))
            if int(v) >= 1
        ]
        if self.frame_id_start < 1:
            raise ValueError("frame_id_start must be >= 1")
        if self.frame_id_end is not None and self.frame_id_end < 1:
            raise ValueError("frame_id_end must be >= 1")
        if self.frame_id_end is not None and self.frame_id_end < self.frame_id_start:
            raise ValueError("frame_id_end must be >= frame_id_start")
        if self.frame_token_list and len(
                self.frame_token_list) != self.num_frames:
            raise ValueError(
                "frame_token_list length must match num_frames when provided")
        self.enable_extended_accumulation_window = bool(
            runtime_config.get('enable_extended_accumulation_window', True))
        self.accumulation_window_offset_frames = int(
            runtime_config.get('accumulation_window_offset_frames', 10))
        if self.accumulation_window_offset_frames < 0:
            self.accumulation_window_offset_frames = 0
        self.accumulation_voxel_size = float(
            runtime_config.get('accumulation_voxel_size', 0.12))
        if not np.isfinite(self.accumulation_voxel_size
                           ) or self.accumulation_voxel_size < 0.0:
            self.accumulation_voxel_size = 0.0
        self.placement_pool_visualization_voxel = float(
            runtime_config.get('placement_pool_visualization_voxel', 0.25))
        if ((not np.isfinite(self.placement_pool_visualization_voxel))
                or self.placement_pool_visualization_voxel < 0.0):
            self.placement_pool_visualization_voxel = 0.25
        self.sampled_position_marker_radius = float(
            runtime_config.get('sampled_position_marker_radius', 0.35))
        if not np.isfinite(self.sampled_position_marker_radius
                           ) or self.sampled_position_marker_radius < 0.0:
            self.sampled_position_marker_radius = 0.35
        self.enable_ego_swept_exclusion = bool(
            runtime_config.get('enable_ego_swept_exclusion', True))
        self.ego_swept_exclusion_radius_xy = float(
            runtime_config.get('ego_swept_exclusion_radius_xy', 2.8))
        self.ego_swept_exclusion_z_margin = float(
            runtime_config.get('ego_swept_exclusion_z_margin', 2.0))
        if self.ego_swept_exclusion_radius_xy < 0:
            self.ego_swept_exclusion_radius_xy = 0.0
        if self.ego_swept_exclusion_z_margin < 0:
            self.ego_swept_exclusion_z_margin = 0.0
        self.object_count_policy = str(
            runtime_config.get('object_count_policy',
                               'distance')).strip().lower()
        if self.object_count_policy not in {'distance', 'range', 'fixed'}:
            print(
                f"\033[33mWarning: unsupported object_count_policy='{self.object_count_policy}'. "
                "Falling back to 'distance'.\033[0m")
            self.object_count_policy = 'distance'
        self.min_ood_objects = runtime_config.get('min_ood_objects')
        self.max_ood_objects = runtime_config.get('max_ood_objects')
        self.fixed_ood_objects = runtime_config.get('fixed_ood_objects')
        if self.min_ood_objects is not None:
            self.min_ood_objects = int(self.min_ood_objects)
        if self.max_ood_objects is not None:
            self.max_ood_objects = int(self.max_ood_objects)
        if self.fixed_ood_objects is not None:
            self.fixed_ood_objects = int(self.fixed_ood_objects)
        if self.fixed_ood_objects is not None and self.fixed_ood_objects < 1:
            raise ValueError("fixed_ood_objects must be >= 1")
        if self.min_ood_objects is not None and self.min_ood_objects < 1:
            raise ValueError("min_ood_objects must be >= 1")
        if self.max_ood_objects is not None and self.max_ood_objects < 1:
            raise ValueError("max_ood_objects must be >= 1")
        if (self.min_ood_objects is not None
                and self.max_ood_objects is not None
                and self.min_ood_objects > self.max_ood_objects):
            raise ValueError("min_ood_objects cannot exceed max_ood_objects")
        self.insertion_preset = str(
            runtime_config.get('insertion_preset', 'none')).strip().lower()
        if self.insertion_preset not in {'none', 'small', 'medium', 'stress'}:
            print(
                f"\033[33mWarning: unsupported insertion_preset='{self.insertion_preset}'. "
                "Falling back to 'none'.\033[0m")
            self.insertion_preset = 'none'
        self.strict_spread_controls = bool(
            runtime_config.get('strict_spread_controls', False))
        self.enable_object_size_priors = bool(
            runtime_config.get('enable_object_size_priors', True))
        self.enable_description_reject_filter = bool(
            runtime_config.get('enable_description_reject_filter', True))
        reject_kw_default = (
            "person,human,man,woman,boy,girl,character,avatar,mannequin,statue,"
            "shirt,sweater,jacket,coat,pants,trousers,dress,skirt,sock,shoe,sneaker,"
            "house,building,apartment,room,interior,landscape,mountain,terrain,island,"
            "road,sidewalk,bridge,wall,ceiling,floor")
        self.description_reject_keywords = parse_csv_strings(
            runtime_config.get('description_reject_keywords',
                               reject_kw_default))
        self.object_target_ref_dim_default_m = float(
            runtime_config.get('object_target_ref_dim_default_m', 1.0))
        self.object_target_ref_dim_min_m = float(
            runtime_config.get('object_target_ref_dim_min_m', 0.08))
        self.object_target_ref_dim_max_m = float(
            runtime_config.get('object_target_ref_dim_max_m', 4.5))
        if self.object_target_ref_dim_default_m <= 0:
            self.object_target_ref_dim_default_m = 1.0
        if self.object_target_ref_dim_min_m <= 0:
            self.object_target_ref_dim_min_m = 0.08
        if self.object_target_ref_dim_max_m < self.object_target_ref_dim_min_m:
            self.object_target_ref_dim_max_m = max(
                self.object_target_ref_dim_min_m,
                self.object_target_ref_dim_default_m)
        self.enable_distance_size_modulation = bool(
            runtime_config.get('enable_distance_size_modulation', True))
        self.distance_size_reference_m = float(
            runtime_config.get('distance_size_reference_m', 12.0))
        self.distance_size_exponent = float(
            runtime_config.get('distance_size_exponent', 0.35))
        self.distance_size_min_multiplier = float(
            runtime_config.get('distance_size_min_multiplier', 0.75))
        self.distance_size_max_multiplier = float(
            runtime_config.get('distance_size_max_multiplier', 1.6))
        self.object_scale_min_clip = float(
            runtime_config.get('object_scale_min_clip', 0.2))
        self.object_scale_max_clip = float(
            runtime_config.get('object_scale_max_clip', 6.0))
        if self.distance_size_reference_m <= 0:
            self.distance_size_reference_m = 12.0
        if self.distance_size_exponent < 0:
            self.distance_size_exponent = 0.0
        if self.distance_size_min_multiplier <= 0:
            self.distance_size_min_multiplier = 0.75
        if self.distance_size_max_multiplier < self.distance_size_min_multiplier:
            self.distance_size_max_multiplier = self.distance_size_min_multiplier
        if self.object_scale_min_clip <= 0:
            self.object_scale_min_clip = 0.2
        if self.object_scale_max_clip < self.object_scale_min_clip:
            self.object_scale_max_clip = self.object_scale_min_clip

        self.placement_candidate_budget = int(
            runtime_config.get('placement_candidate_budget', 2000))
        if self.placement_candidate_budget < 100:
            self.placement_candidate_budget = 100
        self.placement_fast_collision_radius = float(
            runtime_config.get('placement_fast_collision_radius', 0.9))
        self.placement_fast_collision_height_tol = float(
            runtime_config.get('placement_fast_collision_height_tol', 1.5))
        self.placement_max_distance = float(
            runtime_config.get('placement_max_distance', 15.0))
        self.placement_max_distance_eval = float(
            runtime_config.get('placement_max_distance_eval', 30.0))
        self.enable_occlusion_aware_sampling = bool(
            runtime_config.get('enable_occlusion_aware_sampling', True))
        self.min_clear_hits = int(runtime_config.get('min_clear_hits', 2))
        self.min_clear_cameras = int(runtime_config.get(
            'min_clear_cameras', 1))
        self.max_blocked_ratio = float(
            runtime_config.get('max_blocked_ratio', 0.85))
        self.occlusion_depth_margin_m = float(
            runtime_config.get('occlusion_depth_margin_m', 0.05))
        self.occlusion_pixel_radius = int(
            runtime_config.get('occlusion_pixel_radius', 10))
        self.enable_temporal_spread_sampling = bool(
            runtime_config.get('enable_temporal_spread_sampling', True))
        self.temporal_spread_max_delta_frames = int(
            runtime_config.get('temporal_spread_max_delta_frames', -1))
        if self.min_clear_hits < 0:
            self.min_clear_hits = 0
        if self.min_clear_cameras < 0:
            self.min_clear_cameras = 0
        if not np.isfinite(self.max_blocked_ratio):
            self.max_blocked_ratio = 1.0
        self.max_blocked_ratio = float(
            np.clip(self.max_blocked_ratio, 0.0, 1.0))
        if self.occlusion_depth_margin_m < 0:
            self.occlusion_depth_margin_m = 0.0
        if self.occlusion_pixel_radius < 1:
            self.occlusion_pixel_radius = 1
        if self.camera_ground_min_hits < 0:
            self.camera_ground_min_hits = 0
        if self.camera_ground_min_cameras < 0:
            self.camera_ground_min_cameras = 0
        if self.local_floor_radius <= 0:
            self.local_floor_radius = 1.2
        if not np.isfinite(self.local_floor_percentile):
            self.local_floor_percentile = 10.0
        self.local_floor_percentile = float(
            np.clip(self.local_floor_percentile, 0.0, 50.0))
        if self.local_floor_max_offset < 0:
            self.local_floor_max_offset = 0.4
        if self.local_floor_min_neighbors < 1:
            self.local_floor_min_neighbors = 1
        if not np.isfinite(self.camera_ground_min_ratio):
            self.camera_ground_min_ratio = 0.75
        self.camera_ground_min_ratio = float(
            np.clip(self.camera_ground_min_ratio, 0.0, 1.0))
        if not np.isfinite(self.camera_non_ground_max_ratio):
            self.camera_non_ground_max_ratio = 0.25
        self.camera_non_ground_max_ratio = float(
            np.clip(self.camera_non_ground_max_ratio, 0.0, 1.0))
        if self.camera_ground_label_patch_radius < 0:
            self.camera_ground_label_patch_radius = 0
        if not np.isfinite(self.max_occlusion_fraction_of_ood):
            self.max_occlusion_fraction_of_ood = 0.85
        self.max_occlusion_fraction_of_ood = float(
            np.clip(self.max_occlusion_fraction_of_ood, 0.0, 1.0))
        if self.occlusion_min_common_pixels < 1:
            self.occlusion_min_common_pixels = 1
        if self.temporal_spread_max_delta_frames == 0:
            self.temporal_spread_max_delta_frames = -1
        if self.temporal_spread_max_delta_frames < -1:
            self.temporal_spread_max_delta_frames = -1
        self._placement_occlusion_cache = {}
        self.max_objects_per_frame = runtime_config.get(
            'max_objects_per_frame')
        if self.max_objects_per_frame is not None:
            self.max_objects_per_frame = int(self.max_objects_per_frame)
            if self.max_objects_per_frame < 1:
                self.max_objects_per_frame = None

        self.enable_distance_buckets = bool(
            runtime_config.get('enable_distance_buckets', False))
        self.distance_bucket_edges = parse_csv_numeric(runtime_config.get(
            'distance_bucket_edges', '0,8,15,30'),
                                                       cast=float)
        self.distance_bucket_edges = sorted(
            {float(v)
             for v in self.distance_bucket_edges if np.isfinite(v)})
        if len(self.distance_bucket_edges) < 2:
            self.distance_bucket_edges = [0.0, 8.0, 15.0, 30.0]
        self.distance_bucket_weights = parse_csv_numeric(runtime_config.get(
            'distance_bucket_weights', ''),
                                                         cast=float)
        num_buckets = max(1, len(self.distance_bucket_edges) - 1)
        if len(self.distance_bucket_weights) != num_buckets:
            self.distance_bucket_weights = []
        else:
            total_weight = float(
                sum(max(0.0, float(v)) for v in self.distance_bucket_weights))
            if total_weight <= 0:
                self.distance_bucket_weights = []
            else:
                self.distance_bucket_weights = [
                    max(0.0, float(v)) / total_weight
                    for v in self.distance_bucket_weights
                ]
        if self.distance_bucket_weights:
            self.enable_distance_buckets = True
        if self.strict_spread_controls:
            self._apply_strict_spread_global_overrides()
        self.camera_fov_quota_spec = runtime_config.get('camera_fov_quota', '')
        if self.ood_semantic_class <= self.camera_semantic_max_class:
            print(
                f"\033[33mWarning: ood_semantic_class={self.ood_semantic_class} overlaps "
                "with native Waymo camera semantic ids (<= 28). Labels/colors may be ambiguous.\033[0m"
            )
        if self.ood_semantic_class not in SEGMENTATION_COLOR_MAP:

            SEGMENTATION_COLOR_MAP[self.ood_semantic_class] = [0, 255, 255]
            print(
                f"\033[33mWarning: Added fallback color for ood_semantic_class "
                f"{self.ood_semantic_class} to SEGMENTATION_COLOR_MAP.\033[0m")

        self.master = {}

        self.terrain_function = None
        self.file_name = file_name
        self.base_dir = base_dir
        self.image_path = image_path
        self.precompute_cache_dir = runtime_config.get(
            'precompute_cache_dir',
            os.path.join(self.base_dir, self.file_name, "precompute_cache"))
        self.progress_path = os.path.join(self.base_dir, self.file_name,
                                          "run_progress.json")
        self._dataset_cache_signature = {}

        self.boxes_for_scene = []
        self.points_for_terrain_estimation = []
        self.frames_with_seg = {}
        self.frame_map = {}
        self.window_frames = []
        self.window_frame_map = []
        self.all_frames = []
        self.semantic_labels = []
        self.instance_labels = []
        self.panoptic_labels = []
        self.panoptic_label_divisor = 1000
        self.camera_order = list(WAYMO_CAMERA_ORDER)
        self.camera_names = [
            camera_id_to_name(cam_id) for cam_id in self.camera_order
        ]
        self.camera_name_to_id = dict(WAYMO_CAMERA_NAME_TO_ID)
        (
            self.camera_fov_quota,
            self.camera_fov_default_quota,
        ) = parse_camera_quota_spec(self.camera_fov_quota_spec,
                                    self.camera_names)
        self.required_visible_cameras = parse_camera_name_list(
            self.required_visible_camera_spec,
            self.camera_names,
        )
        if self.required_visible_cameras:
            print(
                f"\033[36mPlacement camera requirement enabled: {self.required_visible_cameras} "
                f"(min_hits={self.required_visible_camera_min_hits}).\033[0m")
        if self.required_visible_camera_spec.strip(
        ) and not self.required_visible_cameras:
            print(
                f"\033[33mWarning: required_visible_cameras='{self.required_visible_camera_spec}' "
                "did not match known camera names; requirement disabled.\033[0m"
            )
        self.camera_fov_counts = {name: 0 for name in self.camera_names}
        self.panoptic_labels_by_camera = {
            name: []
            for name in self.camera_names
        }
        self.semantic_labels_by_camera = {
            name: []
            for name in self.camera_names
        }
        self.instance_labels_by_camera = {
            name: []
            for name in self.camera_names
        }
        self.ood_frame_id = []
        self.merged_images = {}
        self.lidar_sematic_labels = []
        self.ood_location = {}
        self.lidar_parameters = {}
        self.lidar_range_pose = {}
        self.azimuth = {}
        self.inclination = {}
        self.ri_indices = {}
        self.ri_ranges = {}
        self.ri_ranges_ood = {}
        self.ood_labels_to_indices = {}
        self.ood_labels_to_ranges = {}
        self.points_vehicle_frame = {}
        self.colors_vehicle_frame = {}
        self.projections_vehicle_frame = {}
        self.range_images_ood_indices = {}
        self.filtered_points = {}
        self.removed_points = {}
        self.removed_projections = {}
        self.vehicle_pose = {}
        self.merged_range_image = {}
        self.point_cloud_dict = {}
        self.point_cloud_rec_dict = {}
        self.frame_visibility_load = defaultdict(int)
        self.accepted_object_positions_world = []
        self.ood_target_distance_bucket = {}
        self.ood_resolved_distance_bucket = {}
        self.distance_bucket_target_counts = {}
        self.distance_bucket_counts = defaultdict(int)
        self.oneshot_plan_cache = None
        self.range_image_ood_exclusive = {}
        self.point_cloud = {}
        self.point_cloud_without_axes = {}
        self.point_cloud_vehicle = {}
        self.camera_accumulation_points_vehicle = {}
        self.lidar_insertion_sensor_type = None
        self.camera_accumulation_pose_world = {}

        self.road_projections_lidar = {}
        self.point_cloud_dataset_lidar = {}
        self.cp_reconstructed = {}
        self.pc_reconstructed = {}
        self.pc_reconstructed_indices = {}
        self.reconstructed_ood_points = {}
        self.cp_reconstructed_indices = {}
        self.pc_reconstructed_constant_pose = {}
        self.lidar_labels = {}
        self.points_lidar = {}
        self.intensity = {}
        self.ood_intensity = {}
        self.ood_details = {}
        self.shadow_color = {}
        self.framewise_ood_details = {}
        self.ood_locations = []
        self.ood_locations_dict = {}
        self.bev_map = []
        self.bev_ood = []
        self.intrinsic_parameters = []
        self.raw_images = {}
        self.frame_data = []
        self.global_box_transf = []
        self.global_box_dims = []
        self.camera_data = {}
        self.ood_range_image = {}
        self.ood_range_points = {}
        self.range_images_ood = {}
        self.range_images_reconstructed = {}
        self.reconstructed_panoptic_labels = {}
        self.reconstructed_pc = {}
        self.cp_actual = {}
        self.range_images = {}
        self.occlusion_masks = {}
        self.ood_occlusion_pixels = {}
        self.occlusion_pixels_from_projection = {}
        self._parsed_range_image_cache = {}
        self._top_lidar_first_return_cache = {}

        self.max_distance = 20
        self.max_objects = 4
        self.lvis = {}
        self.models = []
        self.sampled_model_metadata = []
        self.train = True

        self.max_sample_iterations = 100
        self.max_sample_distance = 20
        self.max_sample_angle = np.pi / 8
        self.max_object_scale = 2
        self.min_object_scale = .5

        self.color_dict = {
            int(k): list(v)
            for k, v in Sampler.color_dict.items()
        }
        prefill_target = 10
        for v in (self.max_ood_objects, self.fixed_ood_objects):
            if isinstance(v, int):
                prefill_target = max(prefill_target, v)
        self._ensure_ood_color_capacity(prefill_target)

    def _ensure_ood_color_capacity(self, max_ood_id):
        """Ensure self.color_dict contains unique colors for ids 1..max_ood_id."""
        target = int(max_ood_id)
        if target <= 0:
            return
        existing = {
            tuple(int(c) for c in color)
            for color in self.color_dict.values()
        }
        next_id = max(self.color_dict.keys(), default=0) + 1
        for ood_id in range(next_id, target + 1):
            hue = (ood_id * 0.6180339887498949) % 1.0
            saturation = 0.60
            value = 0.95
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            color = [int(round(channel * 255.0)) for channel in rgb]

            attempts = 0
            while tuple(color) in existing and attempts < 10:
                hue = (hue + 0.07) % 1.0
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                color = [int(round(channel * 255.0)) for channel in rgb]
                attempts += 1
            self.color_dict[ood_id] = color
            existing.add(tuple(color))

    def _size_bucket_from_target_dim(self, target_ref_dim_m):
        value = float(target_ref_dim_m)
        if value < 0.35:
            return "small"
        if value < 1.2:
            return "medium"
        if value < 2.5:
            return "large"
        return "very_large"

    def _description_reject_reason(self, description, lvis=None):
        """Reject obviously unsuitable assets for road insertion benchmarking."""
        if not self.enable_description_reject_filter:
            return None
        text = str(description or "").strip().lower()
        lvis_text = str(lvis or "").strip().lower()
        if not text and not lvis_text:
            return None
        haystack = f"{lvis_text} {text}".strip()
        for token in self.description_reject_keywords:
            if token and token in haystack:
                return f"keyword:{token}"
        return None

    def _infer_size_prior_from_text(self, description, lvis=None):
        """Infer a rough real-world target dimension prior from text metadata."""
        text = str(description or "").strip().lower()
        lvis_text = str(lvis or "").strip().lower()
        tokens = f"{lvis_text} {text}".strip()

        target = float(self.object_target_ref_dim_default_m)
        scale_policy = "auto"

        if any(
                k in tokens for k in
            ["bottle", "can", "cup", "mug", "glove", "phone", "toy", "helmet"
             ]):
            target = 0.25
            scale_policy = "median_dim"
        elif any(k in tokens for k in ["box", "package", "case", "crate"]):
            target = 0.45
            scale_policy = "median_dim"
        elif any(k in tokens for k in ["chair", "stool", "bench"]):
            target = 1.0
            scale_policy = "auto"
        elif any(k in tokens
                 for k in ["table", "desk", "cabinet", "payphone"]):
            target = 1.4
            scale_policy = "auto"
        elif any(k in tokens for k in ["sofa", "couch", "armchair"]):
            target = 1.8
            scale_policy = "max_dim"
        elif any(
                k in tokens
                for k in ["turret", "spacecraft", "ship", "machine", "robot"]):
            target = 1.3
            scale_policy = "max_dim"

        target = float(
            np.clip(
                target,
                self.object_target_ref_dim_min_m,
                self.object_target_ref_dim_max_m,
            ))
        size_bucket = self._size_bucket_from_target_dim(target)
        return {
            "target_ref_dim_m": target,
            "scale_policy": scale_policy,
            "size_bucket": size_bucket,
        }

    def _should_preserve_manual_reviewed_scale_at_render(self, model):
        """Preserve reviewed XL scale when text priors would incorrectly shrink it."""
        if not isinstance(model, dict):
            return False
        if bool(model.get("preserve_manual_scale_at_render", False)):
            return True
        if model.get("manual_scale_review_source") in (None, ""):
            return False
        if model.get("original_scale") is None:
            return False

        if model.get("original_target_ref_dim_m") is not None:
            return False
        if model.get("original_target_max_dim_m") is not None:
            return False
        return True

    def _resolve_model_size_prior(self, model):
        """Resolve per-object size prior from JSON fields or text fallback."""
        description = str(model.get("description", "") or "").strip()
        lvis = model.get("lvis", model.get("LVIS", None))
        target_ref_dim = None
        target_ref_dim_source = "default"
        target_ref_dim_raw = None

        for key in ("target_ref_dim_m", "target_max_dim_m"):
            raw_value = model.get(key, None)
            if raw_value is None:
                continue
            try:
                parsed_value = float(raw_value)
            except Exception:
                continue
            if not np.isfinite(parsed_value) or parsed_value <= 0:
                continue
            target_ref_dim = parsed_value
            target_ref_dim_raw = parsed_value
            target_ref_dim_source = key
            break

        recommended_blender_scaling = model.get("recommended_blender_scaling",
                                                None)
        if target_ref_dim is not None and recommended_blender_scaling is not None:
            try:
                scaling = float(recommended_blender_scaling)
            except Exception:
                scaling = None
            if scaling is not None and np.isfinite(scaling) and scaling > 0:
                target_ref_dim = target_ref_dim * scaling
                target_ref_dim_source = f"{target_ref_dim_source}*recommended_blender_scaling"

        scale_policy = str(model.get("scale_policy", "auto")
                           or "auto").strip().lower()
        if scale_policy not in {
                "auto", "max_dim", "median_dim", "geom_mean", "height"
        }:
            scale_policy = "auto"
        size_bucket = str(model.get("size_bucket", "") or "").strip().lower()

        if self._should_preserve_manual_reviewed_scale_at_render(model):
            if not size_bucket and self.enable_object_size_priors:
                inferred = self._infer_size_prior_from_text(description,
                                                            lvis=lvis)
                size_bucket = inferred["size_bucket"]
                if scale_policy == "auto":
                    scale_policy = inferred["scale_policy"]
            if not size_bucket:
                size_bucket = self._size_bucket_from_target_dim(
                    float(self.object_target_ref_dim_default_m))
            return {
                "description": description,
                "target_ref_dim_m": None,
                "target_ref_dim_source": "manual_review_scale_only",
                "target_ref_dim_raw": None,
                "scale_policy": scale_policy,
                "size_bucket": size_bucket,
            }

        if target_ref_dim is None and self.enable_object_size_priors:
            inferred = self._infer_size_prior_from_text(description, lvis=lvis)
            target_ref_dim = inferred["target_ref_dim_m"]
            target_ref_dim_source = "text_prior"
            if not size_bucket:
                size_bucket = inferred["size_bucket"]
            if scale_policy == "auto":
                scale_policy = inferred["scale_policy"]
        elif target_ref_dim is None:
            target_ref_dim = float(self.object_target_ref_dim_default_m)
            target_ref_dim_source = "default"

        target_ref_dim = float(
            np.clip(
                float(target_ref_dim),
                self.object_target_ref_dim_min_m,
                self.object_target_ref_dim_max_m,
            ))
        if not size_bucket:
            size_bucket = self._size_bucket_from_target_dim(target_ref_dim)

        return {
            "description": description,
            "target_ref_dim_m": target_ref_dim,
            "target_ref_dim_source": target_ref_dim_source,
            "target_ref_dim_raw": target_ref_dim_raw,
            "scale_policy": scale_policy,
            "size_bucket": size_bucket,
        }

    def _estimate_distance_to_vehicle_pose(self,
                                           position,
                                           preferred_sensor='camera'):
        """Estimate minimum distance from placement to ego trajectory for scale modulation."""
        if position is None:
            return None
        try:
            point_world = np.array(
                [
                    float(position[0][3]),
                    float(position[1][3]),
                    float(position[2][3])
                ],
                dtype=np.float64,
            )
        except Exception:
            return None

        pose_sequence = []
        for sensor_name in (preferred_sensor, 'lidar', 'camera'):
            seq = self.vehicle_pose.get(sensor_name, [])
            if len(seq) > 0:
                pose_sequence = seq
                break
        if len(pose_sequence) == 0:
            return None

        distances = []
        for pose in pose_sequence:
            try:
                ego = np.asarray(pose, dtype=np.float64)[:3, 3]
                distances.append(float(np.linalg.norm(point_world - ego)))
            except Exception:
                continue
        if len(distances) == 0:
            return None
        return float(min(distances))

    def _compute_distance_scale_multiplier(self, distance_m):
        if (not self.enable_distance_size_modulation) or distance_m is None:
            return 1.0
        if not np.isfinite(distance_m) or distance_m <= 0:
            return 1.0
        ratio = float(distance_m) / float(self.distance_size_reference_m)
        ratio = max(ratio, 1e-3)
        raw = ratio**float(self.distance_size_exponent)
        return float(
            np.clip(
                raw,
                self.distance_size_min_multiplier,
                self.distance_size_max_multiplier,
            ))

    def _resolve_num_objects_for_sequence(self, distance_m=None):
        """Resolve number of OOD objects for current sequence based on runtime policy."""
        if self.fixed_ood_objects is not None:
            return int(self.fixed_ood_objects)

        if self.min_ood_objects is not None or self.max_ood_objects is not None:
            min_ood = int(self.min_ood_objects
                          ) if self.min_ood_objects is not None else 1
            max_ood = int(self.max_ood_objects
                          ) if self.max_ood_objects is not None else min_ood
            if min_ood > max_ood:
                raise ValueError(
                    "min_ood_objects cannot exceed max_ood_objects")
            if min_ood == max_ood:
                return min_ood
            return int(self.seed.integers(min_ood, max_ood + 1))

        if self.object_count_policy == 'fixed':
            print(
                "\033[33mWarning: object_count_policy='fixed' without fixed_ood_objects. Falling back to distance policy.\033[0m"
            )

        if distance_m is None:
            return int(self.seed.integers(1, 4))
        if distance_m < 80:
            return int(self.seed.integers(1, 4))
        if distance_m < 200:
            return int(self.seed.integers(4, 7))
        return int(self.seed.integers(7, 10))

    def _apply_strict_spread_global_overrides(self):
        """Enable stricter defaults to reduce clustered placements."""
        self.enable_temporal_spread_sampling = True
        self.side_camera_bonus = max(float(self.side_camera_bonus), 0.75)
        self.min_clear_hits = max(int(self.min_clear_hits), 3)
        self.min_clear_cameras = max(int(self.min_clear_cameras), 2)
        self.enable_distance_buckets = True

        dist_cap = float(self.placement_max_distance)
        if not np.isfinite(dist_cap) or dist_cap <= 0:
            dist_cap = float(self.placement_max_distance_eval)
        if not np.isfinite(dist_cap) or dist_cap <= 0:
            dist_cap = float(self.distance_bucket_edges[-1]
                             ) if self.distance_bucket_edges else 30.0
        dist_cap = max(6.0, float(dist_cap))

        capped_edges = []
        for edge in self.distance_bucket_edges:
            try:
                value = float(edge)
            except Exception:
                continue
            if not np.isfinite(value):
                continue
            if 0.0 <= value < dist_cap:
                capped_edges.append(value)
        capped_edges.append(0.0)
        capped_edges.append(dist_cap)
        capped_edges = sorted(set(capped_edges))
        if len(capped_edges) < 2:
            capped_edges = [0.0, dist_cap]
        self.distance_bucket_edges = capped_edges

        num_buckets = max(1, len(self.distance_bucket_edges) - 1)
        if len(self.distance_bucket_weights) != num_buckets:
            self.distance_bucket_weights = [1.0 / float(num_buckets)
                                            ] * num_buckets
        else:
            total_weight = float(
                sum(max(0.0, float(v)) for v in self.distance_bucket_weights))
            if total_weight <= 0:
                self.distance_bucket_weights = [1.0 / float(num_buckets)
                                                ] * num_buckets
            else:
                self.distance_bucket_weights = [
                    max(0.0, float(v)) / total_weight
                    for v in self.distance_bucket_weights
                ]

    def _apply_strict_spread_sequence_overrides(self):
        """Apply per-sequence strict controls once num_objects is resolved."""
        if not hasattr(self, "num_objects") or int(self.num_objects) < 1:
            return
        adaptive_spacing = float(
            np.clip(30.0 / float(max(1, int(self.num_objects))), 1.5, 3.0))
        self.placement_fast_collision_radius = max(
            float(self.placement_fast_collision_radius),
            adaptive_spacing,
        )
        if len(
                self.camera_names
        ) > 0 and self.camera_fov_default_quota is None and not self.camera_fov_quota:
            per_camera_quota = int(
                max(
                    2,
                    np.ceil(
                        float(self.num_objects) /
                        float(len(self.camera_names))) + 1))
            self.camera_fov_default_quota = per_camera_quota
            self.camera_fov_quota = {
                name: per_camera_quota
                for name in self.camera_names
            }
        print(
            f"\033[36mSTRICT_SPREAD: enabled "
            f"(collision_radius={self.placement_fast_collision_radius:.2f}m, "
            f"bucket_edges={self.distance_bucket_edges}, "
            f"camera_quota_default={self.camera_fov_default_quota})\033[0m")

    def _initialize_distance_bucket_targets(self, num_objects):
        self.distance_bucket_target_counts = {}
        self.distance_bucket_counts = defaultdict(int)
        if not self.enable_distance_buckets:
            return
        num_buckets = max(1, len(self.distance_bucket_edges) - 1)
        if num_buckets <= 0:
            return
        if len(self.distance_bucket_weights) == num_buckets:
            weights = np.array(self.distance_bucket_weights, dtype=np.float64)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0:
                weights = np.ones(num_buckets,
                                  dtype=np.float64) / float(num_buckets)
            else:
                weights = weights / weight_sum
        else:
            weights = np.ones(num_buckets,
                              dtype=np.float64) / float(num_buckets)
        total = max(1, int(num_objects))
        raw = weights * float(total)
        counts = np.floor(raw).astype(np.int32)
        remainder = int(total - np.sum(counts))
        if remainder > 0:
            frac = raw - counts.astype(np.float64)
            order = np.argsort(-frac)
            for k in range(remainder):
                counts[int(order[k % len(order)])] += 1
        self.distance_bucket_target_counts = {
            int(i): int(v)
            for i, v in enumerate(counts.tolist())
        }

    def _initialize_sequence_placement_constraints(self):
        self.frame_visibility_load = defaultdict(int)
        self.accepted_object_positions_world = []
        self.ood_target_distance_bucket = {}
        self.ood_resolved_distance_bucket = {}
        self.camera_fov_counts = {name: 0 for name in self.camera_names}
        self.oneshot_plan_cache = None
        if self.strict_spread_controls:
            self._apply_strict_spread_sequence_overrides()
        self._initialize_distance_bucket_targets(self.num_objects)

    def _bucket_bounds(self, bucket_idx):
        idx = int(bucket_idx)
        if idx < 0 or idx >= (len(self.distance_bucket_edges) - 1):
            return None
        return (
            float(self.distance_bucket_edges[idx]),
            float(self.distance_bucket_edges[idx + 1]),
        )

    def _distance_to_bucket_index(self, distance_m):
        d = float(distance_m)
        num_buckets = len(self.distance_bucket_edges) - 1
        for idx in range(num_buckets):
            lo = float(self.distance_bucket_edges[idx])
            hi = float(self.distance_bucket_edges[idx + 1])
            if idx == num_buckets - 1:
                if lo <= d <= hi:
                    return idx
            else:
                if lo <= d < hi:
                    return idx
        return None

    def _distance_in_bucket(self, distance_m, bucket_idx):
        bounds = self._bucket_bounds(bucket_idx)
        if bounds is None:
            return True
        lo, hi = bounds
        idx = int(bucket_idx)
        last_idx = len(self.distance_bucket_edges) - 2
        if idx == last_idx:
            return lo <= float(distance_m) <= hi
        return lo <= float(distance_m) < hi

    def _choose_distance_bucket_for_object(self, ood_id):
        if not self.enable_distance_buckets or not self.distance_bucket_target_counts:
            return None
        remaining = {}
        for idx, target_count in self.distance_bucket_target_counts.items():
            used = int(self.distance_bucket_counts.get(idx, 0))
            remaining[idx] = int(target_count - used)
        positive = [(idx, rem) for idx, rem in remaining.items() if rem > 0]
        if not positive:
            return None
        rem_values = np.array([rem for _, rem in positive], dtype=np.float64)
        probs = rem_values / float(np.sum(rem_values))
        pick = int(self.seed.choice(len(positive), p=probs))
        bucket_idx = int(positive[pick][0])
        self.ood_target_distance_bucket[int(ood_id)] = bucket_idx
        return bucket_idx

    def _extract_world_translation(self, pose):
        if pose is None:
            return None
        try:
            pose_arr = np.asarray(pose, dtype=np.float64)
            if pose_arr.shape == (4, 4):
                return np.array(
                    [pose_arr[0, 3], pose_arr[1, 3], pose_arr[2, 3]],
                    dtype=np.float64)
            if pose_arr.shape == (3, ):
                return pose_arr.astype(np.float64)
        except Exception:
            return None
        return None

    @staticmethod
    def _voxel_downsample_points_np(points, voxel_size):
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return pts.reshape(-1, 3) if pts.ndim == 2 else np.empty(
                (0, 3), dtype=np.float32)
        if pts.shape[1] > 3:
            pts = pts[:, :3]
        if voxel_size is None or float(voxel_size) <= 0:
            return pts
        vs = float(voxel_size)
        quantized = np.floor(pts / vs).astype(np.int32)
        _, keep_idx = np.unique(quantized, axis=0, return_index=True)
        keep_idx = np.sort(keep_idx)
        return pts[keep_idx]

    def _camera_accumulation_source_frames(self):
        selected_frames = list(self.frames_with_seg.get('camera', []))
        selected_ids = [int(v) for v in self.frame_map.get('camera', [])]
        if len(selected_frames) == 0 or len(selected_ids) == 0:
            return []
        base_records = list(zip(selected_ids, selected_frames))
        if ((not self.enable_extended_accumulation_window)
                or len(self.window_frames) == 0
                or len(self.window_frame_map) == 0):
            return base_records

        lo = min(selected_ids) - int(self.accumulation_window_offset_frames)
        hi = max(selected_ids) + int(self.accumulation_window_offset_frames)
        lo = max(int(self.frame_id_start), int(lo))
        if self.frame_id_end is not None:
            hi = min(int(self.frame_id_end), int(hi))

        records = []
        for frame_id, frame in zip(self.window_frame_map, self.window_frames):
            fid = int(frame_id)
            if lo <= fid <= hi:
                records.append((fid, frame))
        if len(records) == 0:
            return base_records
        return records

    def _prepare_camera_accumulation_points(self):
        """Precompute top-LiDAR points for placement accumulation over camera window."""
        self.camera_accumulation_points_vehicle = {}
        self.camera_accumulation_pose_world = {}

        source_records = self._camera_accumulation_source_frames()
        if len(source_records) == 0:
            return

        source_ids = tuple(int(fid) for fid, _ in source_records)
        cache_name = "top_lidar_camera_accumulation"
        cache_meta = {
            "dataset_path":
            self._dataset_cache_signature.get("dataset_path"),
            "dataset_size":
            self._dataset_cache_signature.get("dataset_size"),
            "dataset_mtime_ns":
            self._dataset_cache_signature.get("dataset_mtime_ns"),
            "source_frame_ids":
            source_ids,
            "voxel_size":
            float(self.accumulation_voxel_size),
            "extended_window":
            bool(self.enable_extended_accumulation_window),
            "window_offset_frames":
            int(self.accumulation_window_offset_frames),
        }

        cached_payload = self._load_precompute_cache(cache_name,
                                                     expected_meta=cache_meta)
        if isinstance(cached_payload, dict):
            frame_ids = [int(v) for v in cached_payload.get("frame_ids", [])]
            points_list = cached_payload.get("points_vehicle", [])
            poses_list = cached_payload.get("poses_world", [])
            for idx, frame_id in enumerate(frame_ids):
                if idx >= len(points_list) or idx >= len(poses_list):
                    continue
                points_np = np.asarray(points_list[idx], dtype=np.float32)
                pose_np = np.asarray(poses_list[idx], dtype=np.float64)
                if points_np.ndim != 2 or points_np.shape[0] == 0:
                    continue
                if pose_np.shape != (4, 4):
                    continue
                self.camera_accumulation_points_vehicle[int(
                    frame_id)] = points_np[:, :3]
                self.camera_accumulation_pose_world[int(frame_id)] = pose_np
            if len(self.camera_accumulation_points_vehicle) > 0:
                print(
                    f"[ACCUM] Reused camera accumulation points for "
                    f"{len(self.camera_accumulation_points_vehicle)} frames.",
                    flush=True,
                )
                return

        cached_frame_ids = []
        cached_points = []
        cached_poses = []
        for local_idx, (frame_global_id, frame) in enumerate(source_records):
            range_images, camera_projections, _seg_labels, range_image_top_pose = (
                self._get_cached_range_parse("camera_accumulation", local_idx,
                                             frame))
            points, _cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=0,
                keep_polar_features=False,
            )
            points_np = np.asarray(points[0], dtype=np.float32)
            points_np = self._voxel_downsample_points_np(
                points_np, self.accumulation_voxel_size)
            if points_np.ndim != 2 or points_np.shape[0] == 0:
                continue
            pose_np = np.array(frame.pose.transform,
                               dtype=np.float64).reshape(4, 4)
            fid = int(frame_global_id)
            self.camera_accumulation_points_vehicle[fid] = points_np[:, :3]
            self.camera_accumulation_pose_world[fid] = pose_np
            cached_frame_ids.append(fid)
            cached_points.append(points_np[:, :3])
            cached_poses.append(pose_np)

        self._save_precompute_cache(
            cache_name,
            {
                "frame_ids": cached_frame_ids,
                "points_vehicle": cached_points,
                "poses_world": cached_poses,
            },
            meta=cache_meta,
        )
        if len(cached_frame_ids) > 0:
            print(
                f"[ACCUM] Prepared camera accumulation points for {len(cached_frame_ids)} frames "
                f"(voxel={self.accumulation_voxel_size:.3f}m).",
                flush=True,
            )

    def _candidate_in_ego_swept_volume(self, candidate_world, pose_sensor):
        if not self.enable_ego_swept_exclusion:
            return False
        if self.ego_swept_exclusion_radius_xy <= 0:
            return False

        poses = []
        if pose_sensor == 'camera' and len(
                self.camera_accumulation_pose_world) > 0:
            for frame_id in sorted(self.camera_accumulation_pose_world.keys()):
                poses.append(self.camera_accumulation_pose_world[frame_id])
        else:
            poses = list(self.vehicle_pose.get(pose_sensor, []))

        if len(poses) == 0:
            return False

        point = np.asarray(candidate_world, dtype=np.float64).reshape(-1)[:3]
        radius_xy = float(self.ego_swept_exclusion_radius_xy)
        z_margin = float(self.ego_swept_exclusion_z_margin)

        for pose in poses:
            pose_arr = np.asarray(pose, dtype=np.float64)
            if pose_arr.shape != (4, 4):
                continue
            ego_center = pose_arr[:3, 3]
            if np.linalg.norm(point[:2] - ego_center[:2]) > radius_xy:
                continue
            if abs(float(point[2]) - float(ego_center[2])) > z_margin:
                continue
            return True
        return False

    def _min_distance_to_pose(self, point_world, pose_sensor):
        pose_list = []
        if pose_sensor == 'camera' and len(
                self.camera_accumulation_pose_world) > 0:
            pose_list = [
                self.camera_accumulation_pose_world[k]
                for k in sorted(self.camera_accumulation_pose_world.keys())
            ]
        elif pose_sensor in self.vehicle_pose:
            pose_list = list(self.vehicle_pose[pose_sensor])
        if len(pose_list) == 0:
            return np.inf
        min_distance = np.inf
        for vehicle_pose in pose_list:
            vehicle_position = np.asarray(vehicle_pose[:3, -1],
                                          dtype=np.float64)
            distance = np.linalg.norm(
                np.asarray(point_world, dtype=np.float64) - vehicle_position)
            min_distance = min(min_distance, float(distance))
        return float(min_distance)

    def _get_camera_quota_limit(self, camera_name):
        if camera_name is None:
            return self.camera_fov_default_quota
        if camera_name in self.camera_fov_quota:
            return self.camera_fov_quota[camera_name]
        return self.camera_fov_default_quota

    def _register_accepted_placement(self, ood_id, point_world,
                                     visibility_sensor):
        point = self._extract_world_translation(point_world)
        if point is None:
            return
        self.accepted_object_positions_world.append(point)
        (
            _score,
            _hits,
            visible_by_camera,
            visible_frames,
        ) = self._collect_world_point_visibility_stats(
            point, sensor_type=visibility_sensor)
        for frame_id in visible_frames:
            self.frame_visibility_load[int(frame_id)] += 1
        dominant_camera = None
        dominant_hits = 0
        for camera_name, hits in visible_by_camera.items():
            if int(hits) > dominant_hits:
                dominant_hits = int(hits)
                dominant_camera = camera_name
        if dominant_camera is not None and dominant_hits > 0:
            self.camera_fov_counts[
                dominant_camera] = self.camera_fov_counts.get(
                    dominant_camera, 0) + 1
        min_distance = self._min_distance_to_pose(point, visibility_sensor)
        bucket_idx = self.ood_target_distance_bucket.get(int(ood_id))
        if bucket_idx is None or not self._distance_in_bucket(
                min_distance, bucket_idx):
            bucket_idx = self._distance_to_bucket_index(min_distance)
        if bucket_idx is not None:
            self.distance_bucket_counts[int(bucket_idx)] += 1
            self.ood_resolved_distance_bucket[int(ood_id)] = int(bucket_idx)

    def _get_search_locations_dir(self, eval=False):
        if not eval:
            return f"{self.base_dir}/{self.file_name}/Placement"
        return f"{self.base_dir}/{self.file_name}/Evaluation/Placement"

    def _get_placement_debug_dir(self, eval=False):
        debug_dir = os.path.join(self._get_search_locations_dir(eval=eval),
                                 "replay_inputs")
        os.makedirs(debug_dir, exist_ok=True)
        return debug_dir

    def _get_image_segment_dir(self, eval=False):
        train_valid = "Train" if self.train else "Valid"
        base_image_path = self.image_path if self.image_path else f"Pipeline_2_{train_valid}"
        if not eval:
            return os.path.join(base_image_path, self.file_name)
        return os.path.join(base_image_path, self.file_name, "Evaluation")

    @staticmethod
    def _write_point_cloud_dual_formats(path_pcd, point_cloud):
        """Write debug point cloud to both .pcd and .ply paths."""
        pcd_path = os.path.abspath(path_pcd)
        root, ext = os.path.splitext(pcd_path)
        if ext.lower() != ".pcd":
            pcd_path = f"{pcd_path}.pcd"
            root = pcd_path[:-4]
        ply_path = f"{root}.ply"
        o3d.io.write_point_cloud(pcd_path, point_cloud)
        o3d.io.write_point_cloud(ply_path, point_cloud)
        return pcd_path, ply_path

    @staticmethod
    def _expanded_marker_points(points_xyz, radius=0.35):
        """Create a small cross marker around each sampled point for visibility."""
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float64)
        r = float(max(0.0, radius))
        if r <= 0:
            return points[:, :3]
        offsets = np.array(
            [
                [0.0, 0.0, 0.0],
                [r, 0.0, 0.0],
                [-r, 0.0, 0.0],
                [0.0, r, 0.0],
                [0.0, -r, 0.0],
                [0.0, 0.0, r],
                [0.0, 0.0, -r],
            ],
            dtype=np.float64,
        )
        marker_points = points[:, None, :3] + offsets[None, :, :]
        return marker_points.reshape(-1, 3)

    def _fused_ground_mask(self, points_vehicle, labels_labeled=None):
        """Fuse semantic-ground with geometric fallback to handle sparse labels."""
        points = np.asarray(points_vehicle, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0:
            return np.zeros((0, ), dtype=bool), np.zeros((0, ), dtype=bool)

        z_vals = points[:, 2]
        z_cut = float(np.percentile(z_vals, 30.0)) + 0.2
        geometric_ground = z_vals <= z_cut
        if int(np.count_nonzero(geometric_ground)) < 100:
            z_cut = float(np.percentile(z_vals, 45.0)) + 0.1
            geometric_ground = z_vals <= z_cut

        if labels_labeled is None:
            non_ground = ~geometric_ground
            return geometric_ground.astype(bool), non_ground.astype(bool)

        labels = np.asarray(labels_labeled, dtype=np.int32).reshape(-1)
        if labels.shape[0] != points.shape[0]:
            non_ground = ~geometric_ground
            return geometric_ground.astype(bool), non_ground.astype(bool)

        semantic_ground = np.isin(labels, self.ground_semantic_ids)
        semantic_non_ground = (labels > 0) & (~semantic_ground)
        unlabeled = labels == 0

        fused_ground = semantic_ground | (unlabeled & geometric_ground)

        labeled_nonzero = labels > 0
        semantic_ground_ratio = float(
            np.count_nonzero(semantic_ground)) / float(
                max(1, int(np.count_nonzero(labeled_nonzero))))
        if semantic_ground_ratio < float(self.ground_semantic_sparse_ratio):
            fused_ground = fused_ground | (
                (~semantic_non_ground) & geometric_ground)

        non_ground = semantic_non_ground | (~fused_ground)
        return fused_ground.astype(bool), non_ground.astype(bool)

    @staticmethod
    def _expand_temporal_token_window(seed_tokens, sorted_tokens, window_size):
        if len(sorted_tokens) == 0:
            return []
        if window_size <= 0:
            return [
                int(t) for t in seed_tokens if int(t) in set(
                    int(v) for v in sorted_tokens)
            ]
        token_to_idx = {int(tok): idx for idx, tok in enumerate(sorted_tokens)}
        expanded_idx = set()
        for tok in seed_tokens:
            t = int(tok)
            if t not in token_to_idx:
                continue
            center = token_to_idx[t]
            lo = max(0, center - int(window_size))
            hi = min(len(sorted_tokens) - 1, center + int(window_size))
            for idx in range(lo, hi + 1):
                expanded_idx.add(idx)
        if len(expanded_idx) == 0:
            return []
        return [int(sorted_tokens[idx]) for idx in sorted(expanded_idx)]

    def _export_fov_candidate_pool_summary(
        self,
        visibility_sensor,
        random_points,
        candidate_visibility_cache,
        camera_frame_pool,
        eval=False,
    ):
        try:
            if not self.enable_fov_candidate_pools:
                return
            if visibility_sensor != 'camera':
                return
            debug_dir = self._get_placement_debug_dir(eval=eval)
            summary_path = os.path.join(debug_dir,
                                        "fov_candidate_pool_summary.json")
            by_camera = {}
            for camera_name in sorted(camera_frame_pool.keys()):
                frame_counts = camera_frame_pool[camera_name]
                by_camera[camera_name] = {
                    str(int(frame_token)): int(count)
                    for frame_token, count in sorted(frame_counts.items(),
                                                     key=lambda kv: int(kv[0]))
                }

            visible_candidate_count = int(
                sum(1 for item in candidate_visibility_cache
                    if int(item.get("visibility_hits", 0)) > 0))
            payload = {
                "visibility_sensor": str(visibility_sensor),
                "candidate_count": int(len(random_points)),
                "visible_candidate_count": int(visible_candidate_count),
                "camera_frame_candidate_counts": by_camera,
            }
            with open(summary_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(
                f"\033[33mSampling debug export skipped (fov candidate pool summary): {e}\033[0m",
                flush=True,
            )

    @staticmethod
    def _normalize_01(values):
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return arr
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if vmax <= vmin + 1e-9:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - vmin) / (vmax - vmin)

    def _semantic_label_name_for_surface(self, semantic_class):
        try:
            sid = int(semantic_class)
        except Exception:
            return "unknown"
        if sid == 0:
            return "unknown"
        return self.camera_surface_semantic_names.get(sid, f"class_{sid}")

    def _surface_type_from_semantic_label(self, semantic_label):
        label = str(semantic_label or "").strip().lower()
        if label in {"road", "lane_marker"}:
            return "road"
        if label in {"sidewalk", "curb", "walkable"}:
            return "sidewalk"
        if label in {"other_ground", "bridge", "floor"}:
            return "other_plane"
        if label in {"", "unknown"}:
            return "unknown"
        return "non_ground"

    def _empty_ground_label_stats(self):
        ground_by_camera = {name: 0 for name in self.camera_names}
        non_ground_by_camera = {name: 0 for name in self.camera_names}
        return {
            "label_hits": 0,
            "ground_hits": 0,
            "non_ground_hits": 0,
            "unknown_hits": 0,
            "ground_ratio": 0.0,
            "non_ground_ratio": 0.0,
            "ground_camera_count": 0,
            "ground_by_camera": ground_by_camera,
            "non_ground_by_camera": non_ground_by_camera,
            "semantic_class_counts": {},
            "semantic_label_counts": {},
            "surface_type_counts": {},
            "dominant_semantic_class": None,
            "dominant_semantic_label": "unknown",
            "surface_type": "unknown",
        }

    def _select_diverse_candidate_indices(self,
                                          points_xyz,
                                          scores,
                                          k,
                                          diversity_weight=0.35):
        """Greedy score+distance selection for deterministic diverse planning."""
        points = np.asarray(points_xyz, dtype=np.float64)
        score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        n = int(points.shape[0]) if points.ndim == 2 else 0
        if n <= 0 or k <= 0:
            return []
        if score_arr.shape[0] != n:
            score_arr = np.zeros((n, ), dtype=np.float64)
        if n <= int(k):
            return list(range(n))

        score_norm = self._normalize_01(score_arr)
        selected = []
        remaining = np.ones((n, ), dtype=bool)
        first_idx = int(np.argmax(score_norm))
        selected.append(first_idx)
        remaining[first_idx] = False
        min_dist = np.linalg.norm(points - points[first_idx], axis=1)
        min_dist[first_idx] = 0.0

        for _ in range(1, int(k)):
            rem_idx = np.where(remaining)[0]
            if rem_idx.size == 0:
                break
            dist_norm = self._normalize_01(min_dist[rem_idx])
            objective = score_norm[rem_idx] + float(
                diversity_weight) * dist_norm
            best_rel = int(np.argmax(objective))
            best_idx = int(rem_idx[best_rel])
            selected.append(best_idx)
            remaining[best_idx] = False
            d = np.linalg.norm(points - points[best_idx], axis=1)
            min_dist = np.minimum(min_dist, d)
        return selected

    def _rank_backup_candidate_indices(self, points_xyz, scores,
                                       selected_indices, top_k):
        points = np.asarray(points_xyz, dtype=np.float64)
        score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        n = int(points.shape[0]) if points.ndim == 2 else 0
        if n <= 0 or top_k <= 0:
            return []
        selected_mask = np.zeros((n, ), dtype=bool)
        for idx in selected_indices:
            if 0 <= int(idx) < n:
                selected_mask[int(idx)] = True
        rem_idx = np.where(~selected_mask)[0]
        if rem_idx.size == 0:
            return []

        score_norm = self._normalize_01(score_arr)
        if np.count_nonzero(selected_mask) > 0:
            selected_pts = points[np.where(selected_mask)[0]]
            dists = np.linalg.norm(points[rem_idx, None, :] -
                                   selected_pts[None, :, :],
                                   axis=2)
            min_d = np.min(dists, axis=1)
        else:
            min_d = np.zeros((rem_idx.size, ), dtype=np.float64)
        dist_norm = self._normalize_01(min_d)
        rank_score = 0.65 * score_norm[rem_idx] + 0.35 * dist_norm
        order = np.argsort(-rank_score)
        ranked = [int(rem_idx[i]) for i in order[:int(top_k)]]
        return ranked

    def _generate_shifted_candidate_points(self, base_point):
        """Generate local XY offsets around one base candidate for correction."""
        base = np.asarray(base_point, dtype=np.float64).reshape(-1)
        if base.size < 3:
            return np.empty((0, 3), dtype=np.float64)
        base = base[:3]
        offsets = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
        max_radius = float(self.placement_shift_max_radius_m)
        step = float(self.placement_shift_step_m)
        if max_radius <= 0.0 or step <= 0.0:
            return np.asarray([base], dtype=np.float64)
        ring = step
        while ring <= max_radius + 1e-6:
            ring_steps = max(
                8, int(np.ceil((2.0 * np.pi * ring) / max(step, 1e-3))))
            for i in range(ring_steps):
                theta = (2.0 * np.pi * float(i)) / float(ring_steps)
                offsets.append(
                    np.array([ring * np.cos(theta), ring * np.sin(theta), 0.0],
                             dtype=np.float64))
            ring += step
        out = np.asarray([base + off for off in offsets], dtype=np.float64)
        return out

    def _resolve_requested_placement_plan_name(self):
        """Infer the placement plan expected by the current per-segment object pool."""
        raw_names = []
        for item in getattr(self, "sampled_model_metadata", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("placement_plan", "") or "").strip()
            if name != "":
                raw_names.append(name)
        if len(raw_names) == 0:
            return None
        counts = Counter(raw_names)
        plan_name, plan_count = counts.most_common(1)[0]
        if len(counts) > 1:
            summary = ", ".join(f"{name}={int(count)}"
                                for name, count in sorted(counts.items()))
            print(
                "\033[33mPlacement plan metadata is mixed within the sampled object pool: "
                f"{summary}. Using dominant plan '{plan_name}' ({plan_count}/{len(raw_names)}).\033[0m"
            )
        return str(plan_name)

    def _get_policy_candidate_rng(self, visibility_sensor, render_mode, eval,
                                  dist):
        """Return RNG used for candidate subset sampling."""

        if self.placement_policy != 'gpt_oneshot':
            return self.seed

        dist_tag = "none" if dist is None else f"{float(dist):.4f}"
        frame_end_tag = "end" if self.frame_id_end is None else str(
            int(self.frame_id_end))
        seed_material = (
            f"{self.file_name}|gpt_oneshot_candidates_v1|"
            f"f{int(self.frame_id_start)}-{frame_end_tag}|"
            f"n{int(self.num_frames)}|"
            f"ext{int(bool(self.enable_extended_accumulation_window))}|"
            f"off{int(self.accumulation_window_offset_frames)}|"
            f"budget{int(self.placement_candidate_budget)}|"
            f"vis{str(visibility_sensor)}|mode{str(render_mode)}|"
            f"eval{int(bool(eval))}|dist{dist_tag}")
        seed = int(
            hashlib.sha256(seed_material.encode("utf-8")).hexdigest(),
            16) % (2**32)
        return np.random.default_rng(seed)

    @staticmethod
    def _compute_visibility_rank_arrays(candidate_visibility_cache, n):
        vis_scores = np.zeros((n, ), dtype=np.float64)
        vis_hits = np.zeros((n, ), dtype=np.int32)
        cam_counts = np.zeros((n, ), dtype=np.int32)
        for i in range(min(int(n), len(candidate_visibility_cache))):
            item = candidate_visibility_cache[i]
            vis_scores[i] = float(item.get("visibility_score", 0.0))
            vis_hits[i] = int(item.get("visibility_hits", 0))
            cam_counts[i] = int(
                sum(1 for v in item.get("visible_by_camera", {}).values()
                    if int(v) > 0))
        base_score = vis_scores + 0.25 * cam_counts.astype(np.float64)
        return vis_scores, vis_hits, cam_counts, base_score

    def _prefilter_request_only_camera_candidates(
        self,
        candidate_points,
        candidate_visibility_cache,
        accumulated_points,
        placement_pool_points,
        points_non_ground_by_frame,
        visibility_sensor,
        pose_sensor,
        has_lidar_semantic,
        eval=False,
    ):
        """Apply strong geometric/ground/occupancy guards before writing GPT request pools."""
        points = np.asarray(accumulated_points, dtype=np.float64)
        points_ground = np.asarray(placement_pool_points, dtype=np.float64)
        candidates = np.asarray(candidate_points, dtype=np.float64)
        if (candidates.ndim != 2 or candidates.shape[0] == 0
                or candidates.shape[1] < 3 or points.ndim != 2
                or points.shape[0] == 0 or points.shape[1] < 3
                or points_ground.ndim != 2 or points_ground.shape[0] == 0
                or points_ground.shape[1] < 3):
            return candidates, candidate_visibility_cache
        if visibility_sensor != "camera":
            return candidates[:, :3], candidate_visibility_cache

        points = points[:, :3]
        points_ground = points_ground[:, :3]
        candidates = candidates[:, :3]
        ground_tree = cKDTree(points_ground)
        all_xy_tree = cKDTree(points[:, :2])
        non_ground_points_by_token = {}
        non_ground_xy_tree_by_token = {}
        for frame_token, chunks in points_non_ground_by_frame.items():
            merged_chunks = np.asarray(chunks, dtype=np.float64)
            if merged_chunks.ndim == 3:
                try:
                    merged_chunks = np.vstack(list(merged_chunks))
                except Exception:
                    merged_chunks = np.asarray([],
                                               dtype=np.float64).reshape(0, 3)
            elif isinstance(chunks, list):
                if len(chunks) == 0:
                    merged_chunks = np.asarray([],
                                               dtype=np.float64).reshape(0, 3)
                else:
                    merged_chunks = np.vstack(chunks)
            if merged_chunks.ndim != 2 or merged_chunks.shape[
                    0] == 0 or merged_chunks.shape[1] < 3:
                continue
            token = int(frame_token)
            merged_chunks = merged_chunks[:, :3]
            non_ground_points_by_token[token] = merged_chunks
            non_ground_xy_tree_by_token[token] = cKDTree(merged_chunks[:, :2])
        sorted_non_ground_tokens = sorted(non_ground_points_by_token.keys())

        filtered_points = []
        filtered_visibility_cache = []
        reject_stats = defaultdict(int)
        dist_limit = self.placement_max_distance_eval if eval else self.placement_max_distance

        for cand_idx, point in enumerate(candidates):
            min_distance_to_pose = self._min_distance_to_pose(
                point, pose_sensor)
            if np.isfinite(
                    dist_limit
            ) and dist_limit > 0 and min_distance_to_pose >= dist_limit:
                reject_stats["distance_cap"] += 1
                continue

            neighborhood_idx = ground_tree.query_ball_point(
                point, self.ground_plane_radius)
            if len(neighborhood_idx) < self.ground_plane_min_neighbors:
                reject_stats["insufficient_neighbors"] += 1
                continue
            neighborhood = points_ground[neighborhood_idx]
            centered = neighborhood - np.mean(
                neighborhood, axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
            if abs(float(normal[2])) < self.ground_plane_min_normal_z:
                reject_stats["slope_reject"] += 1
                continue
            if (np.max(neighborhood[:, 2]) -
                    np.min(neighborhood[:, 2])) > 0.35:
                reject_stats["height_variation_reject"] += 1
                continue

            candidate = np.array(point, copy=True)
            candidate[-1] = float(np.median(neighborhood[:, 2]))

            local_floor_idx = all_xy_tree.query_ball_point(
                candidate[:2], self.local_floor_radius)
            if len(local_floor_idx) >= self.local_floor_min_neighbors:
                local_floor_z = float(
                    np.percentile(
                        points[local_floor_idx, 2],
                        self.local_floor_percentile,
                    ))
                if (float(candidate[2]) - local_floor_z) > float(
                        self.local_floor_max_offset):
                    reject_stats["elevated_floor_reject"] += 1
                    continue

            if self._candidate_in_ego_swept_volume(candidate, pose_sensor):
                reject_stats["ego_swept_reject"] += 1
                continue

            visibility_entry = (candidate_visibility_cache[cand_idx] if
                                0 <= cand_idx < len(candidate_visibility_cache)
                                else {})
            visible_by_camera = dict(
                visibility_entry.get("visible_by_camera", {}))
            visible_frames = []
            for raw_frame in visibility_entry.get("visible_frames", []):
                try:
                    visible_frames.append(int(raw_frame))
                except Exception:
                    continue
            visible_frame_tokens = []
            for raw_token in visibility_entry.get("visible_frame_tokens", []):
                try:
                    visible_frame_tokens.append(int(raw_token))
                except Exception:
                    continue
            visibility_hits = int(visibility_entry.get("visibility_hits", 0))
            if visibility_hits < self.min_fov_hits:
                reject_stats["fov_hits_reject"] += 1
                continue

            if len(non_ground_xy_tree_by_token) > 0:
                seed_tokens = [int(t) for t in visible_frame_tokens]
                if len(seed_tokens) == 0 and len(sorted_non_ground_tokens) > 0:
                    seed_tokens = [
                        int(sorted_non_ground_tokens[
                            len(sorted_non_ground_tokens) // 2])
                    ]
                local_tokens = self._expand_temporal_token_window(
                    seed_tokens,
                    sorted_non_ground_tokens,
                    int(self.occupancy_temporal_window_frames),
                )
                if len(local_tokens) == 0:
                    local_tokens = list(sorted_non_ground_tokens)

                occupied_points_count = 0
                for token in local_tokens:
                    token = int(token)
                    tree = non_ground_xy_tree_by_token.get(token)
                    if tree is None:
                        continue
                    nearby_idx = tree.query_ball_point(candidate[:2],
                                                       self.occupied_xy_radius)
                    if not nearby_idx:
                        continue
                    nearby_points = non_ground_points_by_token[token][
                        nearby_idx]
                    vertical_mask = (
                        (nearby_points[:, 2] >
                         (candidate[2] + self.occupied_min_height))
                        & (nearby_points[:, 2] <
                           (candidate[2] + self.occupied_max_height)))
                    occupied_points_count += int(
                        np.count_nonzero(vertical_mask))
                    if occupied_points_count >= self.occupied_min_points:
                        break
                if occupied_points_count >= self.occupied_min_points:
                    reject_stats["occupied_reject"] += 1
                    continue

            if self.enable_camera_ground_anchor_filter:
                enforce_ground_anchor = True
                if self.camera_ground_anchor_only_fallback and has_lidar_semantic:
                    enforce_ground_anchor = False
                if enforce_ground_anchor:
                    camera_filter = [
                        cam_name
                        for cam_name, cam_hits in visible_by_camera.items()
                        if int(cam_hits) > 0
                    ]
                    if len(camera_filter) == 0:
                        camera_filter = None
                    ground_stats = self._collect_world_point_ground_label_stats(
                        candidate,
                        sensor_type="camera",
                        frame_ids=visible_frames
                        if len(visible_frames) > 0 else None,
                        camera_names=camera_filter,
                    )
                    if ground_stats["label_hits"] < self.camera_ground_min_hits:
                        reject_stats["ground_anchor_low_label_hits"] += 1
                        continue
                    if ground_stats[
                            "ground_hits"] < self.camera_ground_min_hits:
                        reject_stats["ground_anchor_low_hits"] += 1
                        continue
                    if ground_stats[
                            "ground_camera_count"] < self.camera_ground_min_cameras:
                        reject_stats["ground_anchor_low_cameras"] += 1
                        continue
                    if float(ground_stats["ground_ratio"]
                             ) < self.camera_ground_min_ratio:
                        reject_stats["ground_anchor_low_ratio"] += 1
                        continue
                    if float(ground_stats["non_ground_ratio"]
                             ) > self.camera_non_ground_max_ratio:
                        reject_stats[
                            "ground_anchor_high_non_ground_ratio"] += 1
                        continue

            filtered_points.append(candidate)
            if cand_idx < len(candidate_visibility_cache):
                filtered_visibility_cache.append(
                    copy.deepcopy(candidate_visibility_cache[cand_idx]))

        if len(filtered_points) == 0:
            print(
                "\033[33mStrong request-only candidate prefilter rejected every camera candidate; "
                "falling back to the original visible pool.\033[0m")
            return candidates, candidate_visibility_cache

        summary = ", ".join(f"{k}={int(v)}"
                            for k, v in sorted(reject_stats.items()))
        print("\033[36mStrong request-only candidate prefilter: "
              f"kept {len(filtered_points)}/{len(candidates)} candidates" +
              (f" ({summary})" if summary else "") + "\033[0m")
        return np.asarray(filtered_points,
                          dtype=np.float64), filtered_visibility_cache

    def _build_planner_candidate_payload(self, candidate_points,
                                         candidate_visibility_cache, top_k):
        points = np.asarray(candidate_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] == 0:
            return [], []
        n = int(points.shape[0])
        vis_scores, vis_hits, cam_counts, base_score = self._compute_visibility_rank_arrays(
            candidate_visibility_cache,
            n,
        )
        keep = self._select_diverse_candidate_indices(
            points_xyz=points[:, :3],
            scores=base_score,
            k=min(int(top_k), n),
            diversity_weight=0.60,
        )
        table = []
        for idx in keep:
            idx = int(idx)
            visibility_entry = (candidate_visibility_cache[idx]
                                if 0 <= idx < len(candidate_visibility_cache)
                                else {})
            visible_by_camera = dict(
                visibility_entry.get("visible_by_camera", {}))
            visible_frame_ids = []
            for raw_frame_id in visibility_entry.get("visible_frames", []):
                try:
                    visible_frame_ids.append(int(raw_frame_id))
                except Exception:
                    continue
            visible_frame_tokens = []
            for raw_token in visibility_entry.get("visible_frame_tokens", []):
                try:
                    visible_frame_tokens.append(int(raw_token))
                except Exception:
                    continue
            visible_cameras = [
                cam_name for cam_name, cam_hits in visible_by_camera.items()
                if int(cam_hits) > 0
            ]
            surface_stats = self._empty_ground_label_stats()
            if (int(vis_hits[idx]) > 0 and "camera" in self.camera_data
                    and len(self.camera_data.get("camera", [])) > 0):
                surface_stats = self._collect_world_point_ground_label_stats(
                    points[idx, :3],
                    sensor_type="camera",
                    frame_ids=visible_frame_ids
                    if len(visible_frame_ids) > 0 else None,
                    camera_names=visible_cameras
                    if len(visible_cameras) > 0 else None,
                )
            row = {
                "candidate_id":
                idx,
                "x":
                float(points[idx, 0]),
                "y":
                float(points[idx, 1]),
                "z":
                float(points[idx, 2]),
                "visibility_score":
                float(vis_scores[idx]),
                "visibility_hits":
                int(vis_hits[idx]),
                "visible_camera_count":
                int(cam_counts[idx]),
                "visible_frame_ids":
                visible_frame_ids,
                "visible_frame_tokens":
                visible_frame_tokens,
                "visible_cameras":
                visible_cameras,
                "plane_type":
                str(surface_stats.get("surface_type", "unknown")),
                "dominant_surface_label":
                str(surface_stats.get("dominant_semantic_label", "unknown")),
                "dominant_surface_semantic_id":
                surface_stats.get(
                    "dominant_semantic_class",
                    None,
                ),
                "surface_label_hits":
                int(surface_stats.get("label_hits", 0)),
                "surface_ground_hits":
                int(surface_stats.get("ground_hits", 0)),
                "surface_ground_ratio":
                float(surface_stats.get("ground_ratio", 0.0)),
                "surface_non_ground_ratio":
                float(surface_stats.get("non_ground_ratio", 0.0)),
                "surface_label_counts":
                dict(surface_stats.get("semantic_label_counts", {})),
                "plane_type_counts":
                dict(surface_stats.get("surface_type_counts", {})),
            }
            table.append(row)
        keep_values = np.asarray(keep).reshape(-1).tolist()
        return table, [int(v) for v in keep_values]

    def _parse_planner_candidate_ids(self, payload, field_name):
        if not isinstance(payload, dict):
            return []
        values = payload.get(field_name, [])
        if not isinstance(values, list):
            return []
        out = []
        for v in values:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out

    def _candidate_table_signature(self, candidate_table):
        """Stable signature for planner candidate rows."""
        rows = []
        if isinstance(candidate_table, list):
            for row in candidate_table:
                if not isinstance(row, dict):
                    continue
                try:
                    rows.append({
                        "candidate_id": int(row.get("candidate_id")),
                        "x": round(float(row.get("x")), 4),
                        "y": round(float(row.get("y")), 4),
                        "z": round(float(row.get("z")), 4),
                    })
                except Exception:
                    continue
        payload = {
            "rows": rows,
            "count": int(len(rows)),
        }
        blob = json.dumps(payload, separators=(",", ":"),
                          sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _guard_external_oneshot_response(self, response_path, request_payload):
        """
        Validate external one-shot response against the current candidate table.
        In strict mode, mismatches raise.
        In reuse_external_candidates mode, mismatches warn and caller can
        reuse the external candidate table directly.
        """
        result = {
            "reuse_external_candidates": False,
            "external_candidate_table": None,
        }
        try:
            request_table = request_payload.get("candidate_table", [])
            current_sig = self._candidate_table_signature(request_table)
        except Exception:
            current_sig = None

        response_dir = os.path.dirname(os.path.abspath(response_path))
        response_payload = None
        try:
            with open(response_path, "r") as f:
                response_payload = json.load(f)
        except Exception:
            response_payload = None

        embedded_table = None
        if isinstance(response_payload, dict):
            candidate_rows = response_payload.get("candidate_table", None)
            if isinstance(candidate_rows, list) and len(candidate_rows) > 0:
                embedded_table = candidate_rows
                result["external_candidate_table"] = embedded_table
        peer_request = os.path.join(response_dir, "gpt_oneshot_request.json")
        if embedded_table is not None:
            peer_sig = self._candidate_table_signature(embedded_table)
            if current_sig is not None and peer_sig != current_sig:
                if self.placement_guard_mode == "strict":
                    raise RuntimeError(
                        "Placement guard failed: embedded candidate table in placement response "
                        "does not match the current candidate table. "
                        f"current_sig={current_sig[:12]} embedded_sig={peer_sig[:12]} "
                        f"response={response_path}. "
                        "Use a matching placement artifact for this exact frame window."
                    )
                print(
                    "\033[33mPlacement guard mismatch: embedded candidate table "
                    f"differs from current run (current_sig={current_sig[:12]} "
                    f"embedded_sig={peer_sig[:12]}). Reusing embedded candidate table "
                    "from placement response for plan decoding.\033[0m")
                result["reuse_external_candidates"] = True

        elif os.path.isfile(peer_request):
            try:
                with open(peer_request, "r") as f:
                    peer_payload = json.load(f)
                result["external_candidate_table"] = peer_payload.get(
                    "candidate_table", None)
                peer_sig = self._candidate_table_signature(
                    peer_payload.get("candidate_table", []))
            except Exception as e:
                raise RuntimeError(
                    f"Placement guard failed: could not read peer request file "
                    f"'{peer_request}': {e}") from e

            if current_sig is not None and peer_sig != current_sig:
                if self.placement_guard_mode == "strict":
                    raise RuntimeError(
                        "Placement guard failed: external GPT response does not match the "
                        "current candidate table. "
                        f"current_sig={current_sig[:12]} peer_sig={peer_sig[:12]} "
                        f"response={response_path} peer_request={peer_request}. "
                        "Use a matching placement artifact for this exact frame window."
                    )
                print(
                    "\033[33mPlacement guard mismatch: external response candidate table "
                    f"differs from current run (current_sig={current_sig[:12]} "
                    f"peer_sig={peer_sig[:12]}). Reusing external candidate table "
                    "from request folder for plan decoding.\033[0m")
                result["reuse_external_candidates"] = True

        segment_root = os.path.dirname(os.path.dirname(response_dir))
        peer_manifest = os.path.join(segment_root, "run_manifest.json")
        if os.path.isfile(peer_manifest):
            try:
                with open(peer_manifest, "r") as f:
                    manifest = json.load(f)
                rc = manifest.get("runtime_config", {}) if isinstance(
                    manifest, dict) else {}
            except Exception as e:
                raise RuntimeError(
                    f"Placement guard failed: could not read peer manifest "
                    f"'{peer_manifest}': {e}") from e

            def _ival(dct, key):
                v = dct.get(key, None)
                if v is None:
                    return None
                try:
                    return int(v)
                except Exception:
                    return None

            mismatches = []
            check_pairs = [
                ("frame_id_start", _ival(rc, "frame_id_start"),
                 int(self.frame_id_start)),
                ("frame_id_end", _ival(rc, "frame_id_end"), None
                 if self.frame_id_end is None else int(self.frame_id_end)),
                ("num_frames", _ival(rc, "num_frames"), int(self.num_frames)),
                (
                    "enable_extended_accumulation_window",
                    bool(rc.get("enable_extended_accumulation_window", True)),
                    bool(self.enable_extended_accumulation_window),
                ),
                (
                    "accumulation_window_offset_frames",
                    _ival(rc, "accumulation_window_offset_frames"),
                    int(self.accumulation_window_offset_frames),
                ),
                (
                    "placement_candidate_budget",
                    _ival(rc, "placement_candidate_budget"),
                    int(self.placement_candidate_budget),
                ),
            ]
            for key, peer_v, cur_v in check_pairs:
                if peer_v is None:
                    continue
                if peer_v != cur_v:
                    mismatches.append((key, peer_v, cur_v))
            if mismatches:
                mismatch_txt = ", ".join(f"{k}:peer={pv},current={cv}"
                                         for k, pv, cv in mismatches)
                if self.placement_guard_mode == "strict":
                    raise RuntimeError(
                        "Placement guard failed: pass1/pass2 context mismatch for external "
                        f"GPT response ({mismatch_txt}). "
                        "Use a matching pass1 placement artifact for this pass2 run."
                    )
                print(
                    "\033[33mPlacement guard context mismatch: "
                    f"{mismatch_txt}. Reusing external candidate table from request folder.\033[0m"
                )
                result["reuse_external_candidates"] = True

        if not result["reuse_external_candidates"]:
            print(
                f"\033[36mPlacement guard passed for external response: {response_path}\033[0m"
            )
        return result

    def _try_openai_oneshot_plan(self, request_payload):
        """Optional OpenAI request for one-shot placement IDs. Returns dict or None."""
        api_key = os.environ.get(
            str(self.placement_gpt_api_key_env).strip(), "").strip()
        if api_key == "":
            return None
        prompt = (
            "Select candidate IDs for OOD placement.\n"
            "Return strict JSON with keys: primary_candidate_ids, backup_candidate_ids.\n"
            "Use unique candidate IDs from the provided table only.\n\n"
            f"REQUEST_JSON:\n{json.dumps(request_payload, separators=(',', ':'))}\n"
        )
        body = {
            "model": str(self.placement_gpt_model),
            "input": prompt,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=body,
                timeout=float(self.placement_gpt_timeout_s),
            )
            if resp.status_code < 200 or resp.status_code >= 300:
                print(
                    f"\033[33mGPT placement request failed: status={resp.status_code}\033[0m"
                )
                return None
            data = resp.json()
            text = ""
            if isinstance(data, dict):
                text = str(data.get("output_text", "") or "")
                if text == "":
                    output = data.get("output", [])
                    if isinstance(output, list):
                        for item in output:
                            if not isinstance(item, dict):
                                continue
                            for content in item.get("content", []):
                                if isinstance(content, dict):
                                    chunk = content.get("text", "")
                                    if chunk:
                                        text += str(chunk)
            text = text.strip()
            if text == "":
                return None
            try:
                return json.loads(text)
            except Exception:
                m = re.search(r"\{.*\}", text, flags=re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        return None
            return None
        except Exception as e:
            print(f"\033[33mGPT placement request error: {e}\033[0m")
            return None

    def _export_bev_grid_colored(
        self,
        accumulated_points,
        placement_pool_points,
        fov_candidate_points,
        occupied_points,
        ego_positions,
        eval=False,
    ):
        """Export colored BEV raster for placement inspection."""
        if not self.placement_bev_export:
            return
        try:
            acc = np.asarray(accumulated_points, dtype=np.float64)
            pool = np.asarray(placement_pool_points, dtype=np.float64)
            fov = np.asarray(fov_candidate_points, dtype=np.float64)
            occ = np.asarray(occupied_points, dtype=np.float64)
            ego = np.asarray(ego_positions, dtype=np.float64)
            if acc.ndim != 2 or acc.shape[0] == 0:
                return
            acc = acc[:, :3]
            if pool.ndim == 2 and pool.shape[1] >= 3:
                pool = pool[:, :3]
            else:
                pool = np.empty((0, 3), dtype=np.float64)
            if fov.ndim == 2 and fov.shape[1] >= 3:
                fov = fov[:, :3]
            else:
                fov = np.empty((0, 3), dtype=np.float64)
            if occ.ndim == 2 and occ.shape[1] >= 3:
                occ = occ[:, :3]
            else:
                occ = np.empty((0, 3), dtype=np.float64)
            if ego.ndim == 2 and ego.shape[1] >= 3:
                ego = ego[:, :3]
            else:
                ego = np.empty((0, 3), dtype=np.float64)

            margin = float(self.placement_bev_margin_m)
            x_min = float(np.min(acc[:, 0])) - margin
            x_max = float(np.max(acc[:, 0])) + margin
            y_min = float(np.min(acc[:, 1])) - margin
            y_max = float(np.max(acc[:, 1])) + margin
            if x_max <= x_min or y_max <= y_min:
                return

            cell = float(self.placement_bev_cell_size_m)
            width = int(np.ceil((x_max - x_min) / cell)) + 1
            height = int(np.ceil((y_max - y_min) / cell)) + 1
            max_side = int(self.placement_bev_max_side_px)
            if max(width, height) > max_side:
                scale = float(max(width, height)) / float(max_side)
                cell = cell * scale
                width = int(np.ceil((x_max - x_min) / cell)) + 1
                height = int(np.ceil((y_max - y_min) / cell)) + 1
            width = max(32, width)
            height = max(32, height)

            img = np.full((height, width, 3), 248, dtype=np.uint8)

            def _paint_points(points_xyz, color):
                pts = np.asarray(points_xyz, dtype=np.float64)
                if pts.ndim != 2 or pts.shape[0] == 0:
                    return
                u = ((pts[:, 0] - x_min) / cell).astype(np.int32)
                v = ((y_max - pts[:, 1]) / cell).astype(np.int32)
                mask = ((u >= 0) & (u < width) & (v >= 0) & (v < height))
                if not np.any(mask):
                    return
                img[v[mask],
                    u[mask], :] = np.asarray(color,
                                             dtype=np.uint8).reshape(1, 1, 3)

            _paint_points(acc, (140, 140, 140))
            _paint_points(pool, (0, 210, 210))
            _paint_points(fov, (30, 90, 255))
            _paint_points(occ, (240, 80, 80))

            if ego.shape[0] > 0:
                radius_px = max(
                    1,
                    int(
                        np.round(
                            float(self.ego_swept_exclusion_radius_xy) / cell)))
                for p in ego:
                    u = int(np.round((p[0] - x_min) / cell))
                    v = int(np.round((y_max - p[1]) / cell))
                    if 0 <= u < width and 0 <= v < height:
                        cv2.circle(img, (u, v),
                                   radius_px, (255, 186, 60),
                                   thickness=-1)
                        cv2.circle(img, (u, v), 1, (255, 255, 0), thickness=-1)

            debug_dir = self._get_placement_debug_dir(eval=eval)
            out_img = os.path.join(debug_dir, "bev_grid_colored.png")
            out_meta = os.path.join(debug_dir, "bev_grid_metadata.json")
            Image.fromarray(img).save(out_img)
            with open(out_meta, "w") as f:
                json.dump(
                    {
                        "cell_size_m": float(cell),
                        "bounds": {
                            "x_min": float(x_min),
                            "x_max": float(x_max),
                            "y_min": float(y_min),
                            "y_max": float(y_max),
                        },
                        "shape": {
                            "height": int(height),
                            "width": int(width),
                        },
                        "legend_rgb": {
                            "accumulated": [140, 140, 140],
                            "placement_pool": [0, 210, 210],
                            "fov_conditioned_candidates": [30, 90, 255],
                            "occupied": [240, 80, 80],
                            "ego_forbidden": [255, 186, 60],
                        },
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"\033[33mBEV export skipped: {e}\033[0m")

    def _ensure_oneshot_plan_cache(
        self,
        accumulated_points,
        placement_pool_points,
        candidate_points,
        candidate_visibility_cache,
        points_non_ground_by_frame,
        visibility_sensor,
        pose_sensor,
        eval=False,
    ):
        if self.oneshot_plan_cache is not None:
            return self.oneshot_plan_cache
        points = np.asarray(candidate_points, dtype=np.float64)
        n = int(points.shape[0]) if points.ndim == 2 else 0
        if n <= 0:
            self.oneshot_plan_cache = {}
            return self.oneshot_plan_cache

        target_count = int(max(1, int(getattr(self, "num_objects", 1))))
        backup_count = int(target_count *
                           int(self.placement_gpt_backup_multiplier))

        vis_scores = np.zeros((n, ), dtype=np.float64)
        cam_counts = np.zeros((n, ), dtype=np.int32)
        for i in range(min(n, len(candidate_visibility_cache))):
            item = candidate_visibility_cache[i]
            vis_scores[i] = float(item.get("visibility_score", 0.0))
            cam_counts[i] = int(
                sum(1 for v in item.get("visible_by_camera", {}).values()
                    if int(v) > 0))
        planner_score = vis_scores + 0.25 * cam_counts.astype(np.float64)

        primary_det = self._select_diverse_candidate_indices(
            points_xyz=points[:, :3],
            scores=planner_score,
            k=min(target_count, n),
            diversity_weight=0.45,
        )
        backup_det = self._rank_backup_candidate_indices(
            points_xyz=points[:, :3],
            scores=planner_score,
            selected_indices=primary_det,
            top_k=min(max(backup_count, target_count),
                      max(0, n - len(primary_det))),
        )

        planner_rows, planner_idx_subset = self._build_planner_candidate_payload(
            candidate_points=points,
            candidate_visibility_cache=candidate_visibility_cache,
            top_k=self.placement_gpt_top_k_candidates,
        )
        request_payload = {
            "segment": str(self.file_name),
            "target_count": int(target_count),
            "backup_count": int(backup_count),
            "candidate_table": planner_rows,
        }
        debug_dir = self._get_placement_debug_dir(eval=eval)
        request_json_path = os.path.join(debug_dir, "gpt_oneshot_request.json")
        with open(request_json_path, "w") as f:
            json.dump(request_payload, f, indent=2)

        planner_payload = None
        guard_result = {
            "reuse_external_candidates": False,
            "external_candidate_table": None,
        }
        response_path = str(self.placement_gpt_response_json or "").strip()
        if response_path != "" and os.path.isfile(response_path):
            guard_result = self._guard_external_oneshot_response(
                response_path=response_path,
                request_payload=request_payload,
            )
            try:
                with open(response_path, "r") as f:
                    planner_payload = json.load(f)
            except Exception:
                planner_payload = None
        elif self.placement_gpt_enabled:
            planner_payload = self._try_openai_oneshot_plan(request_payload)

        primary_final = []
        backup_final = []
        valid_set = set(range(n))
        external_primary_points = []
        external_backup_points = []

        if isinstance(planner_payload, dict):
            response_json_path = os.path.join(debug_dir,
                                              "gpt_oneshot_response.json")
            try:
                with open(response_json_path, "w") as f:
                    json.dump(planner_payload, f, indent=2)
            except Exception:
                pass
            active_plan_name = self._resolve_requested_placement_plan_name()
            active_plan_payload = planner_payload
            if active_plan_name:
                placement_plans = planner_payload.get("placement_plans", [])
                if isinstance(placement_plans, list):
                    matched_plan = None
                    for plan_entry in placement_plans:
                        if not isinstance(plan_entry, dict):
                            continue
                        if str(plan_entry.get("plan_name", "")
                               or "").strip() == active_plan_name:
                            matched_plan = plan_entry
                            break
                    if matched_plan is not None:
                        active_plan_payload = matched_plan
                        print(
                            "\033[36mUsing placement plan "
                            f"'{active_plan_name}' from multi-plan response for current object pool.\033[0m"
                        )
                    else:
                        print(
                            "\033[33mRequested placement plan "
                            f"'{active_plan_name}' was not found in placement response; "
                            "falling back to top-level candidate IDs.\033[0m")

            raw_primary = self._parse_planner_candidate_ids(
                active_plan_payload,
                "primary_candidate_ids",
            )
            raw_backup = self._parse_planner_candidate_ids(
                active_plan_payload,
                "backup_candidate_ids",
            )
            raw_primary_custom = active_plan_payload.get(
                "primary_custom_world_xy",
                planner_payload.get("primary_custom_world_xy", []),
            )
            raw_backup_custom = active_plan_payload.get(
                "backup_custom_world_xy",
                planner_payload.get("backup_custom_world_xy", []),
            )

            def _parse_custom_world_xy(raw_items):
                out = []
                if not isinstance(raw_items, list):
                    return out
                for item in raw_items:
                    x = None
                    y = None
                    z = None
                    if isinstance(item, dict):
                        x = item.get("x", None)
                        y = item.get("y", None)
                        z = item.get("z", None)
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        x = item[0]
                        y = item[1]
                        z = item[2] if len(item) >= 3 else None
                    try:
                        xf = float(x)
                        yf = float(y)
                    except Exception:
                        continue
                    zf = None
                    try:
                        if z is not None:
                            zf = float(z)
                            if not np.isfinite(zf):
                                zf = None
                    except Exception:
                        zf = None
                    if not np.isfinite(xf) or not np.isfinite(yf):
                        continue
                    out.append([xf, yf, zf])
                return out

            def _lift_custom_xy_to_xyz(items_xy_or_xyz):
                lifted = []
                if not isinstance(items_xy_or_xyz,
                                  list) or len(items_xy_or_xyz) == 0:
                    return lifted
                if points.ndim != 2 or points.shape[0] == 0:
                    return lifted
                xy_points = points[:, :2]
                z_default = float(np.median(
                    points[:, 2])) if points.shape[1] >= 3 else 0.0
                for rec in items_xy_or_xyz:
                    if not isinstance(rec, (list, tuple)) or len(rec) < 3:
                        continue
                    x, y, z = rec[0], rec[1], rec[2]
                    z_use = None
                    if z is not None and np.isfinite(float(z)):
                        z_use = float(z)
                    else:
                        try:
                            deltas = xy_points - np.asarray(
                                [float(x), float(y)],
                                dtype=np.float64).reshape(1, 2)
                            dist2 = np.sum(deltas * deltas, axis=1)
                            nearest_idx = int(np.argmin(dist2))
                            z_use = float(points[
                                nearest_idx,
                                2]) if points.shape[1] >= 3 else z_default
                        except Exception:
                            z_use = z_default
                    lifted.append([float(x), float(y), float(z_use)])
                return lifted

            primary_custom_xyz = _lift_custom_xy_to_xyz(
                _parse_custom_world_xy(raw_primary_custom))
            backup_custom_xyz = _lift_custom_xy_to_xyz(
                _parse_custom_world_xy(raw_backup_custom))
            if len(primary_custom_xyz) > 0:
                seen_custom = set()
                for xyz in primary_custom_xyz:
                    key = tuple(round(float(v), 4) for v in xyz)
                    if key in seen_custom:
                        continue
                    external_primary_points.append([float(v) for v in xyz])
                    seen_custom.add(key)
                    if len(external_primary_points) >= target_count:
                        break
                for xyz in backup_custom_xyz:
                    if len(external_backup_points) >= backup_count:
                        break
                    key = tuple(round(float(v), 4) for v in xyz)
                    if key in seen_custom:
                        continue
                    external_backup_points.append([float(v) for v in xyz])
                    seen_custom.add(key)

            if bool(guard_result.get("reuse_external_candidates", False)):
                external_rows = guard_result.get("external_candidate_table",
                                                 None)
                if isinstance(external_rows, list) and len(external_rows) > 0:
                    by_candidate_id = {}
                    by_row_index = {}
                    for ridx, row in enumerate(external_rows):
                        if not isinstance(row, dict):
                            continue
                        try:
                            xyz = [
                                float(row.get("x")),
                                float(row.get("y")),
                                float(row.get("z")),
                            ]
                        except Exception:
                            continue
                        by_row_index[int(ridx)] = xyz
                        try:
                            cid = int(row.get("candidate_id"))
                            by_candidate_id[cid] = xyz
                        except Exception:
                            pass

                    def _ext_xyz(cid):
                        c = int(cid)
                        if c in by_candidate_id:
                            return by_candidate_id[c]
                        return by_row_index.get(c, None)

                    seen_ext = set()
                    for cid in raw_primary:
                        xyz = _ext_xyz(cid)
                        if xyz is None:
                            continue
                        key = tuple(round(float(v), 4) for v in xyz)
                        if key in seen_ext:
                            continue
                        external_primary_points.append(xyz)
                        seen_ext.add(key)
                        if len(external_primary_points) >= target_count:
                            break
                    for cid in raw_backup:
                        xyz = _ext_xyz(cid)
                        if xyz is None:
                            continue
                        key = tuple(round(float(v), 4) for v in xyz)
                        if key in seen_ext:
                            continue
                        external_backup_points.append(xyz)
                        seen_ext.add(key)
                        if len(external_backup_points) >= backup_count:
                            break

            subset_index_set = set(planner_idx_subset)
            subset_row_to_full = {
                row_idx: int(full_idx)
                for row_idx, full_idx in enumerate(planner_idx_subset)
            }

            def _map_candidate_id(cid):
                if int(cid) in valid_set:
                    return int(cid)
                if int(cid) in subset_row_to_full:
                    return int(subset_row_to_full[int(cid)])
                return None

            seen = set()
            for cid in raw_primary:
                mapped = _map_candidate_id(cid)
                if mapped is None or mapped in seen:
                    continue
                primary_final.append(mapped)
                seen.add(mapped)
                if len(primary_final) >= target_count:
                    break
            for cid in raw_backup:
                mapped = _map_candidate_id(cid)
                if mapped is None or mapped in seen:
                    continue
                backup_final.append(mapped)
                seen.add(mapped)
                if len(backup_final) >= backup_count:
                    break

        used = set(primary_final)
        for idx in primary_det:
            if len(primary_final) >= target_count:
                break
            if idx in used:
                continue
            primary_final.append(int(idx))
            used.add(int(idx))

        used_all = set(primary_final)
        for idx in backup_final:
            used_all.add(int(idx))
        for idx in backup_det:
            if len(backup_final) >= backup_count:
                break
            if int(idx) in used_all:
                continue
            backup_final.append(int(idx))
            used_all.add(int(idx))

        self.oneshot_plan_cache = {
            "primary_indices": [int(v) for v in primary_final],
            "backup_indices": [int(v) for v in backup_final],
        }
        if len(external_primary_points) > 0:
            self.oneshot_plan_cache[
                "external_primary_points"] = external_primary_points
        if len(external_backup_points) > 0:
            self.oneshot_plan_cache[
                "external_backup_points"] = external_backup_points
        try:
            with open(
                    os.path.join(debug_dir, "gpt_oneshot_plan_resolved.json"),
                    "w") as f:
                json.dump(self.oneshot_plan_cache, f, indent=2)
        except Exception:
            pass

        occupied_concat = []
        for chunks in points_non_ground_by_frame.values():
            if len(chunks) > 0:
                occupied_concat.append(np.vstack(chunks))
        occupied_points = np.vstack(
            occupied_concat) if len(occupied_concat) > 0 else np.empty(
                (0, 3), dtype=np.float64)
        ego_positions = []
        if pose_sensor == 'camera' and len(
                self.camera_accumulation_pose_world) > 0:
            for frame_id in sorted(self.camera_accumulation_pose_world.keys()):
                pose = np.asarray(
                    self.camera_accumulation_pose_world[frame_id],
                    dtype=np.float64)
                if pose.shape == (4, 4):
                    ego_positions.append(pose[:3, 3])
        else:
            for pose in self.vehicle_pose.get(pose_sensor, []):
                pose_np = np.asarray(pose, dtype=np.float64)
                if pose_np.shape == (4, 4):
                    ego_positions.append(pose_np[:3, 3])
        ego_positions = np.asarray(
            ego_positions,
            dtype=np.float64) if len(ego_positions) > 0 else np.empty(
                (0, 3), dtype=np.float64)

        self._export_bev_grid_colored(
            accumulated_points=accumulated_points,
            placement_pool_points=placement_pool_points,
            fov_candidate_points=points,
            occupied_points=occupied_points,
            ego_positions=ego_positions,
            eval=eval,
        )
        return self.oneshot_plan_cache

    def _exact_assignment_rows(self):
        """Load the canonical input assignment, independent of saved retry state."""
        if not hasattr(self, '_exact_placement_rows'):
            objects_path = (self.objects_train_json if self.train else
                            self.objects_valid_json)
            with open(objects_path, 'r', encoding='utf-8') as handle:
                rows = json.load(handle)
            if not isinstance(rows, list):
                raise ValueError(f'Invalid object assignments: {objects_path}')
            self._exact_placement_rows = sorted(
                rows, key=lambda row: int(row['placement_slot_index']))
        return self._exact_placement_rows

    def _saved_placement_point(self, ood_id):
        """Return the point assigned to this object, regardless of sparse slot IDs."""
        point = assigned_point(self.sampled_model_metadata, ood_id, required=False)
        if point is None and self.require_exact_placement:
            # Retry bypasses sample_random_models. Validate against the input
            # assignment, never the saved render metadata being checked.
            point = assigned_point(self._exact_assignment_rows(), ood_id, required=True)
        return None if point is None else np.asarray(point, dtype=np.float64)

    def _assert_saved_placement(self, ood_id, position):
        if not self.require_exact_placement:
            return
        expected = self._saved_placement_point(ood_id)
        check_sampled_position(position, expected, ood_id)

    def _build_policy_candidate_subset(
        self,
        ood_id,
        accumulated_points,
        placement_pool_points,
        candidate_points,
        candidate_visibility_cache,
        points_non_ground_by_frame,
        visibility_sensor,
        pose_sensor,
        eval=False,
    ):
        if self.placement_policy != 'gpt_oneshot':
            return None
        if visibility_sensor != 'camera':
            return None

        plan = self._ensure_oneshot_plan_cache(
            accumulated_points=accumulated_points,
            placement_pool_points=placement_pool_points,
            candidate_points=candidate_points,
            candidate_visibility_cache=candidate_visibility_cache,
            points_non_ground_by_frame=points_non_ground_by_frame,
            visibility_sensor=visibility_sensor,
            pose_sensor=pose_sensor,
            eval=eval,
        )
        if not isinstance(plan, dict):
            return None

        # The accepted object pool can omit placement slots. Dense OOD order
        # must not index the full plan's primary-candidate list in that case.
        if self.require_exact_placement:
            return self._saved_placement_point(ood_id).reshape(1, 3)

        points = np.asarray(candidate_points, dtype=np.float64)
        n = int(points.shape[0]) if points.ndim == 2 else 0
        if n <= 0:
            return None
        primary = [int(v) for v in plan.get("primary_indices", [])]
        backup = [int(v) for v in plan.get("backup_indices", [])]
        external_primary = plan.get("external_primary_points", None)
        external_backup = plan.get("external_backup_points", None)
        if isinstance(external_primary, list) and len(external_primary) > 0:
            order_points = []
            target_idx = int(ood_id) - 1
            if 0 <= target_idx < len(external_primary):
                try:
                    order_points.append(
                        np.asarray(external_primary[target_idx],
                                   dtype=np.float64)[:3])
                except Exception:
                    pass
            if self.placement_gpt_lock_primary:
                if len(order_points) > 0:
                    return np.asarray([order_points[0]], dtype=np.float64)
                return None
            if isinstance(external_backup, list):
                for item in external_backup:
                    if len(order_points) >= 16:
                        break
                    try:
                        order_points.append(
                            np.asarray(item, dtype=np.float64)[:3])
                    except Exception:
                        continue
            if len(order_points) > 0:
                return np.asarray(order_points, dtype=np.float64)
        order = []
        target_idx = int(ood_id) - 1
        if 0 <= target_idx < len(primary):
            order.append(primary[target_idx])
        if self.placement_gpt_lock_primary:
            if len(order) == 0:
                return None
            idx = int(order[0])
            if idx < 0 or idx >= n:
                return None
            return np.asarray([points[idx, :3]], dtype=np.float64)
        for idx in backup:
            if idx not in order:
                order.append(idx)
            if len(order) >= 16:
                break

        subset = []
        for idx in order:
            if idx < 0 or idx >= n:
                continue
            shifted = self._generate_shifted_candidate_points(points[idx, :3])
            for p in shifted:
                subset.append(p[:3])
        if len(subset) == 0:
            return None
        return np.asarray(subset, dtype=np.float64)

    def _export_fov_conditioned_candidates_cloud(self,
                                                 candidate_points_world,
                                                 eval=False):
        """Export FOV-conditioned candidate subset for placement debugging."""
        try:
            points = np.asarray(candidate_points_world, dtype=np.float64)
            if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
                return
            points = points[:, :3]
            debug_dir = self._get_placement_debug_dir(eval=eval)
            out_path = os.path.join(debug_dir,
                                    "fov_conditioned_candidates_only.pcd")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            pcd.paint_uniform_color([0.10, 0.45, 1.00])
            self._write_point_cloud_dual_formats(out_path, pcd)
        except Exception as e:
            print(
                f"\033[33mSampling debug export skipped (fov-conditioned candidates): {e}\033[0m",
                flush=True,
            )

    def _export_sampling_pool_cloud(self, pool_points_world, eval=False):
        """Write placement-pool points (candidate ground set) for visualization."""
        try:
            points = np.asarray(pool_points_world, dtype=np.float64)
            if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
                return
            points = points[:, :3]
            points = self._voxel_downsample_points_np(
                points,
                self.placement_pool_visualization_voxel,
            )
            if points.shape[0] == 0:
                return
            debug_dir = self._get_placement_debug_dir(eval=eval)
            pool_path = os.path.join(debug_dir,
                                     "placement_pool_points_only.pcd")
            pool_ply = os.path.splitext(pool_path)[0] + ".ply"
            if os.path.exists(pool_path):
                if not os.path.exists(pool_ply):
                    try:
                        pcd_existing = o3d.io.read_point_cloud(pool_path)
                        o3d.io.write_point_cloud(pool_ply, pcd_existing)
                    except Exception:
                        pass
                return
            pool_pcd = o3d.geometry.PointCloud()
            pool_pcd.points = o3d.utility.Vector3dVector(
                points.astype(np.float64))

            pool_pcd.paint_uniform_color([0.0, 0.78, 1.0])
            self._write_point_cloud_dual_formats(pool_path, pool_pcd)
        except Exception as e:
            print(
                f"\033[33mSampling debug export skipped (pool cloud): {e}\033[0m",
                flush=True,
            )

    def _export_sampling_accumulated_cloud(self,
                                           accumulated_points_world,
                                           eval=False):
        """Write a single accumulated cloud for placement spread visualization."""
        try:
            points = np.asarray(accumulated_points_world, dtype=np.float64)
            if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
                return
            points = points[:, :3]
            debug_dir = self._get_placement_debug_dir(eval=eval)
            accumulated_path = os.path.join(debug_dir,
                                            "accumulated_points_only.pcd")
            accumulated_ply_path = os.path.splitext(
                accumulated_path)[0] + ".ply"
            if os.path.exists(accumulated_path):
                if not os.path.exists(accumulated_ply_path):
                    try:
                        pcd_existing = o3d.io.read_point_cloud(
                            accumulated_path)
                        o3d.io.write_point_cloud(accumulated_ply_path,
                                                 pcd_existing)
                    except Exception:
                        pass
                return
            accumulated_pcd = o3d.geometry.PointCloud()
            accumulated_pcd.points = o3d.utility.Vector3dVector(points)
            accumulated_pcd.paint_uniform_color([0.55, 0.55, 0.55])
            self._write_point_cloud_dual_formats(accumulated_path,
                                                 accumulated_pcd)
        except Exception as e:
            print(
                f"\033[33mSampling debug export skipped (accumulated cloud): {e}\033[0m",
                flush=True,
            )

    def _export_sampling_selected_spread(self, eval=False, finalize=False):
        """Export currently selected placement spots and optional merged overlay."""
        try:
            if len(self.accepted_object_positions_world) == 0:
                return
            points = np.asarray(self.accepted_object_positions_world,
                                dtype=np.float64)
            if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
                return
            points = points[:, :3]

            debug_dir = self._get_placement_debug_dir(eval=eval)
            selected_pcd = o3d.geometry.PointCloud()
            selected_pcd.points = o3d.utility.Vector3dVector(points)

            selected_pcd.paint_uniform_color([1.0, 0.0, 1.0])

            selected_path = os.path.join(debug_dir,
                                         "sampled_positions_only.pcd")
            self._write_point_cloud_dual_formats(selected_path, selected_pcd)

            marker_points = self._expanded_marker_points(
                points,
                radius=self.sampled_position_marker_radius,
            )
            marker_pcd = o3d.geometry.PointCloud()
            marker_pcd.points = o3d.utility.Vector3dVector(
                marker_points.astype(np.float64))

            marker_pcd.paint_uniform_color([1.0, 1.0, 0.0])
            marker_path = os.path.join(debug_dir,
                                       "sampled_positions_markers_only.pcd")
            self._write_point_cloud_dual_formats(marker_path, marker_pcd)

            sampled_json = []
            for idx, point in enumerate(points, start=1):
                sampled_json.append({
                    "order": int(idx),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                })
            sampled_json_path = os.path.join(debug_dir,
                                             "sampled_positions_only.json")
            with open(sampled_json_path, "w") as f:
                json.dump(sampled_json, f, indent=2)

            if finalize:
                accumulated_path = os.path.join(debug_dir,
                                                "accumulated_points_only.pcd")
                pool_path = os.path.join(debug_dir,
                                         "placement_pool_points_only.pcd")
                merged = o3d.geometry.PointCloud()
                if os.path.exists(accumulated_path):
                    merged += o3d.io.read_point_cloud(accumulated_path)
                if os.path.exists(pool_path):
                    merged += o3d.io.read_point_cloud(pool_path)
                merged += selected_pcd
                merged += marker_pcd
                merged_path = os.path.join(
                    debug_dir,
                    "accumulated_with_sampled_positions.pcd",
                )
                self._write_point_cloud_dual_formats(merged_path, merged)
        except Exception as e:
            print(
                f"\033[33mSampling debug export skipped (selected spread): {e}\033[0m",
                flush=True,
            )

    def _get_cached_range_parse(self, sensor_type, frame_id, frame):
        if not self.cache_parsed_range_images:
            return frame_utils.parse_range_image_and_camera_projection(frame)
        sensor_cache = self._parsed_range_image_cache.setdefault(
            sensor_type, {})
        if frame_id not in sensor_cache:
            sensor_cache[
                frame_id] = frame_utils.parse_range_image_and_camera_projection(
                    frame)
        return sensor_cache[frame_id]

    def _get_cached_top_lidar_first_return(self, sensor_type, frame_id, frame):
        if not self.cache_parsed_range_images:
            range_images, _, _, _ = frame_utils.parse_range_image_and_camera_projection(
                frame)
            original_top_ri = range_images[open_dataset.LaserName.TOP][0]
            h, w = original_top_ri.shape.dims[0], original_top_ri.shape.dims[1]
            return np.array(original_top_ri.data,
                            dtype=np.float32).reshape(h, w, 4)

        sensor_cache = self._top_lidar_first_return_cache.setdefault(
            sensor_type, {})
        if frame_id not in sensor_cache:
            range_images, _, _, _ = self._get_cached_range_parse(
                sensor_type, frame_id, frame)
            original_top_ri = range_images[open_dataset.LaserName.TOP][0]
            h, w = original_top_ri.shape.dims[0], original_top_ri.shape.dims[1]
            sensor_cache[frame_id] = np.array(original_top_ri.data,
                                              dtype=np.float32).reshape(
                                                  h, w, 4)
        return sensor_cache[frame_id]

    def _clear_parsed_range_image_cache(self):
        self._parsed_range_image_cache.clear()
        self._top_lidar_first_return_cache.clear()

    def _ensure_precompute_cache_dir(self):
        if not self.enable_precompute_cache:
            return None
        os.makedirs(self.precompute_cache_dir, exist_ok=True)
        return self.precompute_cache_dir

    def _precompute_cache_path(self, cache_name):
        cache_dir = self._ensure_precompute_cache_dir()
        if cache_dir is None:
            return None
        safe_name = str(cache_name).strip().replace("/", "_")
        return os.path.join(cache_dir, f"{safe_name}.pkl.gz")

    @staticmethod
    def _dataset_cache_identity_matches(expected_meta, payload_meta):
        """Allow cache reuse when a TFRecord moved to a different filesystem root."""
        if not isinstance(expected_meta, dict) or not isinstance(
                payload_meta, dict):
            return False

        exp_path = str(expected_meta.get("dataset_path", "") or "")
        got_path = str(payload_meta.get("dataset_path", "") or "")
        exp_size = expected_meta.get("dataset_size", None)
        got_size = payload_meta.get("dataset_size", None)
        exp_mtime = expected_meta.get("dataset_mtime_ns", None)
        got_mtime = payload_meta.get("dataset_mtime_ns", None)

        def _known(value):
            return value is not None and value not in {"", -1}

        if exp_path == got_path and exp_path != "":
            size_ok = (not _known(exp_size)) or exp_size == got_size
            mtime_ok = (not _known(exp_mtime)) or exp_mtime == got_mtime
            return bool(size_ok and mtime_ok)

        exp_base = os.path.basename(exp_path)
        got_base = os.path.basename(got_path)
        if exp_base == "" or got_base == "" or exp_base != got_base:
            return False

        size_match = _known(exp_size) and _known(
            got_size) and exp_size == got_size
        mtime_match = _known(exp_mtime) and _known(
            got_mtime) and exp_mtime == got_mtime
        return bool(size_match or mtime_match)

    def _load_precompute_cache(self, cache_name, expected_meta=None):
        if not self.enable_precompute_cache or self.force_recompute_precompute_cache:
            return None
        cache_path = self._precompute_cache_path(cache_name)
        if cache_path is None or not os.path.exists(cache_path):
            return None
        try:
            with gzip.open(cache_path, "rb") as cache_file:
                wrapped = pickle.load(cache_file)
            if not isinstance(wrapped, dict):
                return None
            if str(wrapped.get("cache_tag", "")) != self.precompute_cache_tag:
                return None
            payload = wrapped.get("payload")
            payload_meta = wrapped.get("meta", {})
            if expected_meta:
                dataset_alias_ok = self._dataset_cache_identity_matches(
                    expected_meta,
                    payload_meta,
                )
                if (dataset_alias_ok and payload_meta.get("dataset_path") !=
                        expected_meta.get("dataset_path")):
                    print(
                        f"[CACHE] dataset alias accepted for {cache_name}: "
                        f"cached={payload_meta.get('dataset_path')} "
                        f"expected={expected_meta.get('dataset_path')}",
                        flush=True,
                    )
                for key, value in expected_meta.items():
                    if dataset_alias_ok and key in {
                            "dataset_path",
                            "dataset_size",
                            "dataset_mtime_ns",
                    }:
                        continue
                    if payload_meta.get(key) != value:
                        return None
            print(f"[CACHE] HIT {cache_name}: {cache_path}", flush=True)
            return payload
        except Exception as exc:
            print(f"[CACHE] MISS {cache_name}: failed to load ({exc})",
                  flush=True)
            return None

    def _save_precompute_cache(self, cache_name, payload, meta=None):
        if not self.enable_precompute_cache:
            return
        cache_path = self._precompute_cache_path(cache_name)
        if cache_path is None:
            return
        wrapped = {
            "cache_tag": self.precompute_cache_tag,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "meta": meta or {},
            "payload": payload,
        }
        tmp_path = f"{cache_path}.tmp"
        try:
            with gzip.open(tmp_path, "wb") as cache_file:
                pickle.dump(wrapped,
                            cache_file,
                            protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            print(f"[CACHE] SAVE {cache_name}: {cache_path}", flush=True)
        except Exception as exc:
            print(f"[CACHE] WARN {cache_name}: failed to save ({exc})",
                  flush=True)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _capture_dataset_signature(self, dataset_name):
        signature = {
            "dataset_path": os.path.abspath(dataset_name),
            "num_frames": int(self.num_frames),
        }
        try:
            signature["dataset_size"] = int(os.path.getsize(dataset_name))
            signature["dataset_mtime_ns"] = int(
                os.stat(dataset_name).st_mtime_ns)
        except Exception:
            signature["dataset_size"] = -1
            signature["dataset_mtime_ns"] = -1
        self._dataset_cache_signature = signature
        return signature

    def _update_run_progress(self, stage, status, **details):
        if not self.enable_run_progress:
            return
        progress_payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "segment": self.file_name,
            "stage": stage,
            "status": status,
            "details": details or {},
        }
        progress_path = self.progress_path
        progress_dir = os.path.dirname(progress_path)
        if progress_dir:
            os.makedirs(progress_dir, exist_ok=True)
        tmp_path = f"{progress_path}.tmp"
        with self._progress_lock:
            with open(tmp_path, "w") as progress_file:
                json.dump(progress_payload, progress_file, indent=2)
            os.replace(tmp_path, progress_path)

    def _decode_camera_panoptic_labels_from_selected_frames(self):
        self.panoptic_labels = []
        self.instance_labels = []
        self.semantic_labels = []
        self.panoptic_labels_by_camera = {
            name: []
            for name in self.camera_names
        }
        self.semantic_labels_by_camera = {
            name: []
            for name in self.camera_names
        }
        self.instance_labels_by_camera = {
            name: []
            for name in self.camera_names
        }
        self.has_semantic_label_6 = False

        if len(self.frames_with_seg.get('camera', [])) == 0:
            return

        segmentation_protos_ordered = []
        for frame in self.frames_with_seg['camera']:
            segmentation_proto_dict = {
                image.name: image.camera_segmentation_label
                for image in frame.images
            }
            segmentation_protos_ordered.append(
                [segmentation_proto_dict[name] for name in self.camera_order])

        segmentation_protos_flat = sum(segmentation_protos_ordered, [])
        sequence_ids = {label.sequence_id for label in segmentation_protos_flat}
        if len(sequence_ids) != 1:
            raise ValueError("Expected camera labels from one Waymo sequence")
        max_global_id = max(
            (mapping.global_instance_id
             for label in segmentation_protos_flat
             for mapping in label.instance_id_to_global_id_mapping),
            default=0,
        )
        # Waymo requires the divisor to be strictly greater than every instance ID.
        # Its multi-frame helper can otherwise choose a divisor equal to max_global_id.
        safe_divisor = max(
            max(label.panoptic_label_divisor for label in segmentation_protos_flat),
            max_global_id + 1,
        )
        panoptic_labels, _num_cameras_covered, _is_tracked_masks, panoptic_label_divisor = (
            camera_segmentation_utils.
            decode_multi_frame_panoptic_labels_from_segmentation_labels(
                segmentation_protos_flat, remap_to_global=True,
                new_panoptic_label_divisor=safe_divisor))
        self.panoptic_label_divisor = int(max(1, panoptic_label_divisor))

        num_camera_frames = len(self.camera_order)
        for i in range(0, len(segmentation_protos_flat), num_camera_frames):
            for cam_offset, cam_id in enumerate(self.camera_order):
                camera_name = camera_id_to_name(cam_id)
                panoptic_label = panoptic_labels[i + cam_offset]
                semantic_label, instance_label = (
                    camera_segmentation_utils.
                    decode_semantic_and_instance_labels_from_panoptic_label(
                        panoptic_label,
                        panoptic_label_divisor,
                    ))

                if 6 in semantic_label:
                    self.has_semantic_label_6 = True

                self.panoptic_labels_by_camera[camera_name].append(
                    panoptic_label)
                self.instance_labels_by_camera[camera_name].append(
                    instance_label)
                self.semantic_labels_by_camera[camera_name].append(
                    semantic_label)

                if cam_id == open_dataset.CameraName.FRONT:
                    self.panoptic_labels.append(panoptic_label)
                    self.instance_labels.append(instance_label)
                    self.semantic_labels.append(semantic_label)

    def get_weather_from_api(self,
                             timestamp_micros,
                             lat='37.7749',
                             lon='-122.4194'):
        """Fetches weather data from Open-Meteo for a specific time and location."""
        if requests is None:
            raise RuntimeError(
                "requests is required only when weather API support is enabled."
            )
        try:
            dt_object = datetime.fromtimestamp(timestamp_micros / 1e6,
                                               tz=timezone.utc)

            query_date = dt_object
            if query_date.date() >= datetime.now(timezone.utc).date():
                query_date = query_date.replace(year=query_date.year - 1)

            date_str = query_date.strftime('%Y-%m-%d')
            hour = dt_object.hour

            base_url = "https://archive-api.open-meteo.com/v1/archive"
            hourly_params = "cloud_cover,direct_radiation,diffuse_radiation"

            url = (
                f"{base_url}?latitude={lat}&longitude={lon}&start_date={date_str}"
                f"&end_date={date_str}&hourly={hourly_params}&timezone=UTC")

            response = requests.get(url, timeout=self.weather_api_timeout_s)
            response.raise_for_status()
            data = response.json()['hourly']

            return {
                "cloud_cover": data['cloud_cover'][hour],
                "direct_radiation": data['direct_radiation'][hour],
                "diffuse_radiation": data['diffuse_radiation'][hour]
            }

        except (requests.exceptions.RequestException, KeyError,
                IndexError) as e:
            print(
                f"Warning: Could not fetch weather data. Using defaults. Error: {e}",
                file=sys.stderr)
            return {
                "cloud_cover": -1,
                "direct_radiation": -1,
                "diffuse_radiation": -1
            }

    def log_failed_object(self, ood_id, error_message):
        """Logs details of a failed object to a JSON file in a process-safe manner."""
        failed_log_path = "objects_failed.json"
        lock_path = f"{failed_log_path}.lock"

        if isinstance(error_message, bytes):
            error_message = error_message.decode('utf-8', errors='replace')

        failed_object_details = {
            "file_name": self.file_name,
            "ood_id": ood_id,
            "lvis": self.ood_details.get(ood_id, {}).get('lvis', 'N/A'),
            "path": self.ood_details.get(ood_id, {}).get('path', 'N/A'),
            "error": error_message
        }

        lock = None
        try:

            lock = open(lock_path, 'w')

            fcntl.flock(lock, fcntl.LOCK_EX)

            failed_data = []
            if os.path.exists(
                    failed_log_path) and os.path.getsize(failed_log_path) > 0:
                with open(failed_log_path, 'r') as f:
                    try:
                        failed_data = json.load(f)
                        if not isinstance(failed_data, list):
                            failed_data = [failed_data]
                    except json.JSONDecodeError:

                        failed_data = []

            failed_data.append(failed_object_details)

            with open(failed_log_path, 'w') as f:
                json.dump(failed_data, f, indent=4)

            print(f"Logged failed object {ood_id} to {failed_log_path}")

        except Exception as e:
            print(f"Failed to log failed object: {e}", file=sys.stderr)
        finally:
            if lock:

                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()

                if os.path.exists(lock_path):
                    os.remove(lock_path)

    def save_ood_locations(self):
        """Store all sampled OOD locations into JSON file."""
        print(
            f"\033[32mSAMPLER: Saving OOD locations for dataset {self.file_name}\033[0m"
        )
        with open(
                f"{self.base_dir}/{self.file_name}/{self.file_name}_ood_locations.json",
                "w") as ood_file:
            json.dump(self.ood_locations, ood_file)

    def load_ood_locations(self, path):
        """Load OOD locations from JSON file."""
        with open(
                f"{self.base_dir}/{self.file_name}/{self.file_name}_ood_locations.json",
                "r") as ood_file:
            self.ood_locations = json.load(ood_file)

    @ResourceLogger.log_resources_during_execution()
    def find_frames_with_seg(self, files, sensor_type):

        FILE_NAME = self.file_name
        print("-" * 90, flush=True)
        print(f"\033[35mRETRIEVER: In Dataset {FILE_NAME}\033[0m", flush=True)
        if self.train:
            train_valid = "Train"
        else:
            train_valid = "Valid"

        dataset_name = os.path.join(files, f"{FILE_NAME}.tfrecord")
        dataset_signature = self._capture_dataset_signature(dataset_name)
        if not os.path.exists(f"{self.base_dir}/{self.file_name}"):
            os.makedirs(f"{self.base_dir}/{self.file_name}")
        """Stores all frames from scene that have panoptic segmentation labels
        Args: max_length: collect only this number of frames, useful for debugging
        Returns: List of frames"""

        self.frames_with_seg['lidar'] = []
        self.frame_map['lidar'] = []
        self.frames_with_seg['camera'] = []
        self.frame_map['camera'] = []
        self.window_frames = []
        self.window_frame_map = []
        self.all_frames = []

        camera_frames = []
        lidar_frames = []
        both_frames = []
        num_cam = 0
        num_lidar = 0
        num_both = 0
        attempts = 0
        if self.frame_token_list:
            frame_window_tag = "tokens-" + "_".join(
                str(int(v)) for v in self.frame_token_list)
        else:
            frame_window_tag = (
                f"f{int(self.frame_id_start)}-{int(self.frame_id_end)}"
                if self.frame_id_end is not None else
                f"f{int(self.frame_id_start)}-end")
        cache_name = f"frames_sensor-{sensor_type}_n{self.num_frames}_{frame_window_tag}"
        cache_meta = {
            "dataset_path":
            dataset_signature.get("dataset_path"),
            "dataset_size":
            dataset_signature.get("dataset_size"),
            "dataset_mtime_ns":
            dataset_signature.get("dataset_mtime_ns"),
            "sensor_type":
            sensor_type,
            "num_frames":
            int(self.num_frames),
            "frame_id_start":
            int(self.frame_id_start),
            "frame_id_end":
            None if self.frame_id_end is None else int(self.frame_id_end),
            "frame_token_list":
            list(self.frame_token_list),
            "frames_cache_version":
            2,
        }
        cached_payload = self._load_precompute_cache(cache_name,
                                                     expected_meta=cache_meta)

        if isinstance(cached_payload, dict):
            frame_map_cached = cached_payload.get("frame_map", {})
            self.frame_map['lidar'] = [
                int(v) for v in frame_map_cached.get('lidar', [])
            ]
            self.frame_map['camera'] = [
                int(v) for v in frame_map_cached.get('camera', [])
            ]

            for serialized in cached_payload.get("frames_lidar", []):
                frame = open_dataset.Frame()
                frame.ParseFromString(serialized)
                self.frames_with_seg['lidar'].append(frame)

            for serialized in cached_payload.get("frames_camera", []):
                frame = open_dataset.Frame()
                frame.ParseFromString(serialized)
                self.frames_with_seg['camera'].append(frame)

            all_serialized = cached_payload.get("all_frames", [])
            if all_serialized:
                for serialized in all_serialized:
                    frame = open_dataset.Frame()
                    frame.ParseFromString(serialized)
                    self.all_frames.append(frame)
            else:

                self.all_frames = list(self.frames_with_seg['lidar'])
            window_serialized = cached_payload.get("window_frames", [])
            if window_serialized:
                self.window_frame_map = [
                    int(v) for v in cached_payload.get("window_frame_map", [])
                ]
                for serialized in window_serialized:
                    frame = open_dataset.Frame()
                    frame.ParseFromString(serialized)
                    self.window_frames.append(frame)
            else:

                self.window_frames = list(self.all_frames)
                if len(self.window_frame_map) == 0:
                    candidate_map = list(self.frame_map.get("camera", []))
                    if len(candidate_map) != len(self.window_frames):
                        candidate_map = list(self.frame_map.get("lidar", []))
                    if len(candidate_map) != len(self.window_frames):
                        candidate_map = list(
                            range(
                                int(self.frame_id_start),
                                int(self.frame_id_start) +
                                len(self.window_frames),
                            ))
                    self.window_frame_map = candidate_map

            counts = cached_payload.get("counts", {})
            camera_frames = [int(v) for v in counts.get("camera_frames", [])]
            lidar_frames = [int(v) for v in counts.get("lidar_frames", [])]
            both_frames = [int(v) for v in counts.get("both_frames", [])]
            num_cam = int(counts.get("num_cam", len(camera_frames)))
            num_lidar = int(counts.get("num_lidar", len(lidar_frames)))
            num_both = int(counts.get("num_both", len(both_frames)))
            attempts = int(counts.get("attempts", 0))
        else:
            dataset = tf.data.TFRecordDataset(dataset_name,
                                              compression_type='')
            print(
                f"[FIND_FRAMES] Starting to iterate through dataset frames...",
                flush=True)
            explicit_frame_tokens = list(self.frame_token_list)
            for i, data in enumerate(dataset):
                attempts = i + 1
                if i % 10 == 0:
                    print(f"[FIND_FRAMES] Processing frame {i}...", flush=True)
                frame = open_dataset.Frame()
                frame.ParseFromString(data.numpy())
                if explicit_frame_tokens:
                    if i >= len(explicit_frame_tokens):
                        break
                    frame_id = int(explicit_frame_tokens[i])
                else:
                    frame_id = i + 1
                    if frame_id < self.frame_id_start:
                        continue
                    if self.frame_id_end is not None and frame_id > self.frame_id_end:
                        break
                self.window_frames.append(frame)
                self.window_frame_map.append(frame_id)
                has_semantic = any(
                    (hasattr(laser, "ri_return1") and
                     len(laser.ri_return1.segmentation_label_compressed) > 0)
                    or (hasattr(laser, "ri_return2") and len(
                        laser.ri_return2.segmentation_label_compressed) > 0)
                    for laser in frame.lasers)
                has_panoptic = any(
                    bool(img.camera_segmentation_label.panoptic_label)
                    for img in frame.images)

                if has_panoptic and not has_semantic:
                    num_cam += 1
                    camera_frames.append(frame_id)
                elif has_semantic and not has_panoptic:
                    num_lidar += 1
                    lidar_frames.append(frame_id)
                elif has_panoptic and has_semantic:
                    num_both += 1
                    both_frames.append(frame_id)

                if sensor_type == 'both':

                    if has_semantic and len(
                            self.frames_with_seg['lidar']) < self.num_frames:
                        self.frames_with_seg['lidar'].append(frame)
                        self.frame_map['lidar'].append(frame_id)
                    if has_panoptic and len(
                            self.frames_with_seg['camera']) < self.num_frames:
                        self.frames_with_seg['camera'].append(frame)
                        self.frame_map['camera'].append(frame_id)
                    if has_semantic or has_panoptic:
                        self.all_frames.append(frame)
                else:
                    if has_semantic and len(
                            self.frames_with_seg['lidar']) < self.num_frames:
                        self.frames_with_seg['lidar'].append(frame)
                        self.frame_map['lidar'].append(frame_id)
                        self.all_frames.append(frame)
                    if (has_panoptic and sensor_type in ['camera'] and len(
                            self.frames_with_seg['camera']) < self.num_frames):
                        self.frames_with_seg['camera'].append(frame)
                        self.frame_map['camera'].append(frame_id)

                if sensor_type == 'both':
                    if (len(self.frames_with_seg['lidar']) == self.num_frames
                            and len(self.frames_with_seg['camera'])
                            == self.num_frames):
                        break
                elif sensor_type == 'camera':
                    if (len(self.frames_with_seg['lidar']) == self.num_frames
                            and len(self.frames_with_seg['camera'])
                            == self.num_frames):
                        break
                elif len(self.frames_with_seg['lidar']) == self.num_frames:
                    break

            cache_payload = {
                "frame_map": {
                    "lidar": list(self.frame_map['lidar']),
                    "camera": list(self.frame_map['camera']),
                },
                "frames_lidar": [
                    frame.SerializeToString()
                    for frame in self.frames_with_seg['lidar']
                ],
                "frames_camera": [
                    frame.SerializeToString()
                    for frame in self.frames_with_seg['camera']
                ],
                "all_frames":
                [frame.SerializeToString() for frame in self.all_frames],
                "window_frames":
                [frame.SerializeToString() for frame in self.window_frames],
                "window_frame_map":
                list(self.window_frame_map),
                "counts": {
                    "camera_frames": list(camera_frames),
                    "lidar_frames": list(lidar_frames),
                    "both_frames": list(both_frames),
                    "num_cam": int(num_cam),
                    "num_lidar": int(num_lidar),
                    "num_both": int(num_both),
                    "attempts": int(attempts),
                },
            }
            self._save_precompute_cache(cache_name,
                                        cache_payload,
                                        meta=cache_meta)

        print(
            f"Total frames with lidar segmentation labels: {len(self.frames_with_seg['lidar'])}"
        )
        print(
            f"Total frames with camera segmentation labels: {len(self.frames_with_seg['camera'])}"
        )
        print(self.num_frames)

        log_file_dir = f'{self.base_dir}/{self.file_name}'
        log_file_path = f'{log_file_dir}/frame_log.txt'
        with open(log_file_path, "w") as f:
            f.write('-' * 60 + '\n')
            f.write(f'{self.file_name}\n')
            f.write(f'Parsed frames: {attempts}\n')
            f.write(f'Camera-only segmentation frames: {num_cam}\n')
            f.write(f'LiDAR-only segmentation frames: {num_lidar}\n')
            f.write(f'Both camera+LiDAR segmentation frames: {num_both}\n')
            f.write(f'camera_frames: {camera_frames}\n')
            f.write(f'lidar_frames: {lidar_frames}\n')
            f.write(f'both_frames: {both_frames}\n')
            f.write(f'window_frame_count: {len(self.window_frame_map)}\n')
            if len(self.window_frame_map) > 0:
                f.write(
                    f'window_frame_span: [{int(min(self.window_frame_map))}, '
                    f'{int(max(self.window_frame_map))}]\n')

        print(f"Logs written to {log_file_path}")
        self.master[self.file_name] = {
            'train': self.train,
            'num_frames': attempts,
            'num_camera_seg_frames': num_cam,
            'camera_seg_frames': camera_frames,
            'num_lidar_seg_frames': num_lidar,
            'lidar_seg_frames': lidar_frames,
            'num_both_seg_frames': num_both,
            'both_seg_frames': both_frames,
            'frame_seg_info': log_file_path,
        }

        if len(self.frames_with_seg['lidar']
               ) == self.num_frames and self.export_lidar_label_metadata:
            load_dir = files
            save_dir = f'{self.base_dir}/{self.file_name}/lidar_label_metadata'
            points_dir = os.path.join(save_dir, "velodyne")
            labels_dir = os.path.join(save_dir, "labels")
            projections_dir = os.path.join(save_dir, "projections")
            metadata_ready = (
                os.path.isdir(points_dir) and os.path.isdir(labels_dir)
                and os.path.isdir(projections_dir) and
                len([f for f in os.listdir(points_dir) if f.endswith(".bin")])
                >= self.num_frames and len([
                    f for f in os.listdir(labels_dir) if f.endswith(".label")
                ]) >= self.num_frames and len([
                    f
                    for f in os.listdir(projections_dir) if f.endswith(".bin")
                ]) >= self.num_frames)
            if metadata_ready:
                print(
                    "[FIND_FRAMES] Reusing existing lidar_label_metadata cache "
                    "(export already complete).",
                    flush=True,
                )
            else:
                if Waymo2SemanticKITTI is None:
                    raise RuntimeError(
                        "waymo_to_semanticKITTI is required only when lidar label metadata export is enabled."
                    )
                converter = Waymo2SemanticKITTI(load_dir, save_dir)
                all_files = [f'{self.file_name}.tfrecord']

                converter.convert_all(all_files, self.num_frames)
        elif len(self.frames_with_seg['lidar']) == self.num_frames:
            print(
                "[FIND_FRAMES] Skipping lidar label metadata export (export_lidar_label_metadata=False).",
                flush=True)

        self._decode_camera_panoptic_labels_from_selected_frames()

    def collect_global_bounding_boxes(self):
        """Method to collect all bounding boxes from all frames.
        Store camera data in dict self.camera_data.
        keys:
            camera_pose: list of 4x4 transformation matrix for camera position including extrinsic for each frame
            extrinsic: list of 4x4 transformation matrix for camera extrinsic for each frame
            focal_length: list of float numbers representing the focal length of the camera
        Store 3D box data in self.global_box_transf mainly for testing purposes
        keys:
            object_id (str): list of 4x4 transf matrices representing locatio and rotatio of objects for each frame
        Store 3D box dimensions in self.global_box_dims
        keys:
            object_id (str): list of 3d vectors [length, width, height] for each frame
        Store bev map in self.bev_map:
            list of tuples [c_x, c_y, length, width, heading] for collision detection
        Store camera locations and box locations in self.boxes_for_terrain_estimation:
            list of tuples [center_x, center_y, z_coordinate] representing bottom of vehicles and ego vehicle"""
        print("\033[36mSAMPLER: Collecting global bounding boxes.\033[0m")

        cache_name = "camera_data"
        cache_meta = {
            "dataset_path":
            self._dataset_cache_signature.get("dataset_path"),
            "dataset_size":
            self._dataset_cache_signature.get("dataset_size"),
            "dataset_mtime_ns":
            self._dataset_cache_signature.get("dataset_mtime_ns"),
            "num_frames":
            int(self.num_frames),
            "frame_map_lidar":
            tuple(int(v) for v in self.frame_map.get("lidar", [])),
            "frame_map_camera":
            tuple(int(v) for v in self.frame_map.get("camera", [])),
        }
        cached_payload = self._load_precompute_cache(cache_name,
                                                     expected_meta=cache_meta)
        if isinstance(cached_payload, dict):
            self.camera_data = copy.deepcopy(
                cached_payload.get("camera_data", {}))
            self.points_for_terrain_estimation = [
                [float(v) for v in row] for row in cached_payload.get(
                    "points_for_terrain_estimation", [])
            ]
            cam_dict_file_dir = f"{self.base_dir}/{self.file_name}/camera_data/"
            self.master.setdefault(self.file_name, {})
            self.master[self.file_name]["camera_data"] = cam_dict_file_dir
            print("[CACHE] Reused camera_data precompute.", flush=True)
            return

        for sensor_type, frames in self.frames_with_seg.items():
            if len(frames) == 0:
                continue
            if sensor_type not in self.camera_data.keys():
                self.camera_data[sensor_type] = []

            cam_dict_file_dir = f"{self.base_dir}/{self.file_name}/camera_data/"
            cam_dict_file_dir_type = f"{cam_dict_file_dir}{sensor_type}"
            if not os.path.isdir(cam_dict_file_dir_type):
                os.makedirs(cam_dict_file_dir_type, exist_ok=True)

            for frame_id, frame in enumerate(frames):

                transf_data = {}
                dim_data = {}
                camera_data = {}

                FILTER_AVAILABLE = any([
                    label.num_top_lidar_points_in_box > 0
                    for label in frame.laser_labels
                ])
                pose_matrix = np.array(frame.pose.transform).reshape(4, 4)
                self.points_for_terrain_estimation.append(
                    [pose_matrix[0][3], pose_matrix[1][3], pose_matrix[2][3]])
                time_stamp = frame.timestamp_micros

                time_of_day = frame.context.stats.time_of_day

                datetime_obj = datetime.fromtimestamp(frame.timestamp_micros /
                                                      1e6,
                                                      tz=timezone.utc)
                time_of_day_str = datetime_obj.strftime('%H:%M:%S')

                is_night_2_to_5 = 2 <= datetime_obj.hour < 5

                weather = frame.context.stats.weather

                light_cond = 0.3

                if weather == "Rain":
                    light_cond *= 0
                if time_of_day == "Night":
                    light_cond *= 0
                if time_of_day == "Dawn/Dusk":
                    light_cond *= 0.3
                if time_of_day == "Dawn/Dusk" and is_night_2_to_5:
                    light_cond *= 0

                if self.enable_weather_api:
                    weather_api = self.get_weather_from_api(time_stamp)
                else:
                    weather_api = {
                        "cloud_cover": -1,
                        "direct_radiation": -1,
                        "diffuse_radiation": -1,
                    }

                camera_entries = {}
                for img in frame.images:
                    calibration = next(
                        cc for cc in frame.context.camera_calibrations
                        if cc.name == img.name)
                    extrinsic = tf.reshape(
                        tf.constant(list(calibration.extrinsic.transform),
                                    dtype=tf.float32), [4, 4])
                    intrinsic = tf.constant(list(calibration.intrinsic),
                                            dtype=tf.float32)
                    cam_pose_matrix = np.array([[0, 0, -1, 0], [-1, 0, 0, 0],
                                                [0, 1, 0, 0], [0, 0, 0, 1]])
                    cam_pose_matrix = np.matmul(extrinsic, cam_pose_matrix)
                    cam_pose_matrix = np.matmul(pose_matrix, cam_pose_matrix)

                    camera_trigger_time = img.camera_trigger_time
                    camera_readout_done_time = img.camera_readout_done_time
                    camera_shutter = img.shutter
                    velocity = img.velocity
                    camera_name = camera_id_to_name(img.name)

                    camera_entries[camera_name] = {
                        "camera_id":
                        int(img.name),
                        "focal_length":
                        float(intrinsic[0]),
                        "projection": (
                            float(intrinsic[0]),
                            float(intrinsic[1]),
                            float(intrinsic[2]),
                            float(intrinsic[3]),
                        ),
                        "width":
                        int(calibration.width),
                        "height":
                        int(calibration.height),
                        "camera_pose":
                        np.array(cam_pose_matrix).tolist(),
                        "extrinsic":
                        np.array(extrinsic).tolist(),
                        "time_stamp":
                        time_stamp,
                        "readout_time":
                        camera_readout_done_time - camera_trigger_time,
                        "shutter":
                        camera_shutter,
                        "velocity": {
                            "v_x": velocity.v_x,
                            "v_y": velocity.v_y,
                            "v_z": velocity.v_z,
                            "w_x": velocity.w_x,
                            "w_y": velocity.w_y,
                            "w_z": velocity.w_z,
                        },
                        "time_of_day":
                        time_of_day,
                        "weather":
                        weather,
                        "weather_api":
                        weather_api,
                        "light":
                        light_cond,
                    }

                front_entry = camera_entries.get("front")
                if front_entry is None:

                    first_key = sorted(camera_entries.keys())[0]
                    front_entry = camera_entries[first_key]
                camera_data.update(copy.deepcopy(front_entry))
                camera_data["all_cameras"] = camera_entries

                self.camera_data[sensor_type].append(camera_data)

                cam_dict_file_name = f"{cam_dict_file_dir_type}/camera_data_frame_00{self.frame_map[sensor_type][frame_id]}.json"
                with open(cam_dict_file_name, "w") as json_file:
                    json.dump(camera_data, json_file, indent=4)

            self.master[self.file_name]["camera_data"] = cam_dict_file_dir

        cache_payload = {
            "camera_data":
            copy.deepcopy(self.camera_data),
            "points_for_terrain_estimation":
            copy.deepcopy(self.points_for_terrain_estimation),
        }
        self._save_precompute_cache(cache_name, cache_payload, meta=cache_meta)

    def fit_triangulation(self):
        """Method to fit self.terrain_function using triangular interpolation and all boxes for terrain
        estimation.
        self.terrain_function (location: [x: float, y: float]) -> z_coordinate: float
        self.points_for_terrain_estimation (list of tuples [x: float, y: float, z: float])"""
        self.terrain_function = fit_triangulation(
            self.points_for_terrain_estimation)

    def _get_camera_entries_for_frame(self, sensor_type, frame_id):
        """Return per-camera metadata for a frame (fallback to legacy front-only format)."""
        if sensor_type not in self.camera_data:
            return {}
        if frame_id >= len(self.camera_data[sensor_type]):
            return {}
        frame_camera_data = self.camera_data[sensor_type][frame_id]
        if isinstance(frame_camera_data,
                      dict) and "all_cameras" in frame_camera_data:
            return frame_camera_data["all_cameras"]
        return {"front": frame_camera_data}

    def _resolve_camera_image_size(self, camera_entry):
        """Resolve camera image size from metadata without assuming fixed dimensions."""
        if not isinstance(camera_entry, dict):
            return 0, 0

        width = int(camera_entry.get("width", 0) or 0)
        height = int(camera_entry.get("height", 0) or 0)
        if width > 0 and height > 0:
            return width, height

        projection = camera_entry.get("projection")
        if projection is not None and len(projection) >= 4:
            cx = float(projection[2])
            cy = float(projection[3])

            width = max(1, int(round(2.0 * cx)))
            height = max(1, int(round(2.0 * cy)))
            return width, height

        return 0, 0

    def _project_world_point_to_camera(self, point_world, sensor_type,
                                       frame_id, camera_entry):
        """Project a world point into one camera and return visibility/projection metadata."""
        if sensor_type not in self.vehicle_pose:
            return False, None, None, None, None
        if frame_id >= len(self.vehicle_pose[sensor_type]):
            return False, None, None, None, None
        vehicle_pose = self.vehicle_pose[sensor_type][frame_id]
        point_world_h = np.array(
            [point_world[0], point_world[1], point_world[2], 1.0],
            dtype=np.float64)
        point_vehicle = np.linalg.inv(vehicle_pose) @ point_world_h

        extrinsic = np.asarray(camera_entry["extrinsic"]).reshape(4, 4)
        point_camera = np.linalg.inv(extrinsic) @ point_vehicle

        z_cam = point_camera[0]
        if z_cam <= 1e-3:
            return False, None, None, None, None
        x_cam = -point_camera[1]
        y_cam = -point_camera[2]
        fx, fy, cx, cy = camera_entry["projection"]
        u = (fx * x_cam / z_cam) + cx
        v = (fy * y_cam / z_cam) + cy

        width, height = self._resolve_camera_image_size(camera_entry)
        if width <= 0 or height <= 0:
            return False, None, None, None, None
        in_bounds = (0 <= u < width) and (0 <= v < height)
        radial_distance = float(np.linalg.norm(point_vehicle[:3]))
        return in_bounds, float(u), float(v), float(z_cam), radial_distance

    def _is_world_point_visible_in_camera(self, point_world, sensor_type,
                                          frame_id, camera_entry):
        """Project a world point to camera pixels and check in-bounds visibility."""
        in_bounds, _u, _v, _z_cam, _radial_distance = self._project_world_point_to_camera(
            point_world,
            sensor_type,
            frame_id,
            camera_entry,
        )
        return bool(in_bounds)

    def _prepare_sampling_occlusion_cache(self):
        """Build sparse per-frame/per-camera LiDAR depth grids for placement occlusion checks."""
        if not self.enable_occlusion_aware_sampling:
            return

        source_sensor = 'camera'
        cell_px = 8

        if self._placement_occlusion_cache.get("ready", False):
            if (self._placement_occlusion_cache.get("sensor_type")
                    == source_sensor
                    and int(self._placement_occlusion_cache.get(
                        "cell_px", -1)) == int(cell_px)):
                return

        cache = {
            "ready": False,
            "sensor_type": source_sensor,
            "cell_px": int(cell_px),
            "grids": {},
        }

        if (source_sensor not in self.point_cloud_vehicle
                or source_sensor not in self.cp_actual
                or len(self.point_cloud_vehicle[source_sensor]) == 0
                or len(self.cp_actual[source_sensor]) == 0):
            self._placement_occlusion_cache = cache
            return

        for frame_id, pcd in self.point_cloud_vehicle[source_sensor].items():
            if frame_id not in self.cp_actual[source_sensor]:
                continue

            points = np.asarray(pcd.points)
            cp_points = np.asarray(self.cp_actual[source_sensor][frame_id])
            if points.ndim != 2 or points.shape[0] == 0:
                continue
            if cp_points.ndim != 2 or cp_points.shape[0] == 0:
                continue

            n = min(points.shape[0], cp_points.shape[0])
            if n <= 0:
                continue
            points = points[:n]
            cp_points = cp_points[:n]
            distances = np.linalg.norm(points, axis=1).astype(np.float32)

            frame_grids = {}
            frame_camera_entries = self._get_camera_entries_for_frame(
                source_sensor, frame_id)
            for camera_name, camera_id in self.camera_name_to_id.items():
                camera_entry = frame_camera_entries.get(camera_name, {})
                cam_w, cam_h = self._resolve_camera_image_size(camera_entry)
                if cam_w <= 0 or cam_h <= 0:
                    continue
                pix_parts = []
                dist_parts = []
                if cp_points.shape[1] >= 3:
                    mask_primary = cp_points[:, 0] == camera_id
                    if np.any(mask_primary):
                        pix_parts.append(cp_points[mask_primary][:, 1:3])
                        dist_parts.append(distances[mask_primary])
                if cp_points.shape[1] >= 6:
                    mask_secondary = cp_points[:, 3] == camera_id
                    if np.any(mask_secondary):
                        pix_parts.append(cp_points[mask_secondary][:, 4:6])
                        dist_parts.append(distances[mask_secondary])
                if not pix_parts:
                    continue

                pix = np.vstack(pix_parts)
                depth_vals = np.concatenate(dist_parts).astype(np.float32)
                pix = np.rint(pix).astype(np.int32)
                valid = ((pix[:, 0] >= 0) & (pix[:, 0] < cam_w) &
                         (pix[:, 1] >= 0) & (pix[:, 1] < cam_h))
                if not np.any(valid):
                    continue
                pix = pix[valid]
                depth_vals = depth_vals[valid]

                grid_h = int(np.ceil(float(cam_h) / float(cell_px)))
                grid_w = int(np.ceil(float(cam_w) / float(cell_px)))
                grid = np.full((grid_h, grid_w), np.inf, dtype=np.float32)
                uu = pix[:, 0] // cell_px
                vv = pix[:, 1] // cell_px
                np.minimum.at(grid, (vv, uu), depth_vals)
                frame_grids[camera_name] = grid

            if frame_grids:
                cache["grids"][int(frame_id)] = frame_grids

        cache["ready"] = True
        self._placement_occlusion_cache = cache

    def _collect_world_point_occlusion_stats(self,
                                             point_world,
                                             sensor_type='camera'):
        """Estimate foreground blocking stats for a candidate point from projected LiDAR depth."""
        visible_by_camera = {name: 0 for name in self.camera_names}
        clear_by_camera = {name: 0 for name in self.camera_names}
        if sensor_type not in self.camera_data:
            return {
                "total_visible_hits": 0,
                "clear_visible_hits": 0,
                "blocked_visible_hits": 0,
                "blocked_ratio": 1.0,
                "visible_by_camera": visible_by_camera,
                "clear_by_camera": clear_by_camera,
                "visible_frames": [],
                "clear_frames": [],
            }

        self._prepare_sampling_occlusion_cache()
        cache = self._placement_occlusion_cache
        grids_by_frame = cache.get("grids", {}) if isinstance(cache,
                                                              dict) else {}
        cell_px = int(cache.get("cell_px", 8)) if isinstance(cache,
                                                             dict) else 8
        radius_cells = max(
            0,
            int(np.ceil(float(self.occlusion_pixel_radius) / float(cell_px))))
        depth_margin = float(self.occlusion_depth_margin_m)

        total_visible_hits = 0
        clear_visible_hits = 0
        blocked_visible_hits = 0
        visible_frames = []
        clear_frames = []

        num_frames = min(len(self.camera_data.get(sensor_type, [])),
                         len(self.vehicle_pose.get(sensor_type, [])))
        for frame_id in range(num_frames):
            frame_hits = 0
            frame_clear_hits = 0
            frame_cache = grids_by_frame.get(int(frame_id), {})
            camera_entries = self._get_camera_entries_for_frame(
                sensor_type, frame_id)
            for camera_name, camera_entry in camera_entries.items():
                in_bounds, u, v, _z_cam, candidate_distance = self._project_world_point_to_camera(
                    point_world,
                    sensor_type,
                    frame_id,
                    camera_entry,
                )
                if not in_bounds:
                    continue

                total_visible_hits += 1
                frame_hits += 1
                visible_by_camera[camera_name] = visible_by_camera.get(
                    camera_name, 0) + 1

                blocked = False
                grid = frame_cache.get(camera_name)
                if grid is not None:
                    cu = int(u) // cell_px
                    cv = int(v) // cell_px
                    u0 = max(0, cu - radius_cells)
                    u1 = min(grid.shape[1] - 1, cu + radius_cells)
                    v0 = max(0, cv - radius_cells)
                    v1 = min(grid.shape[0] - 1, cv + radius_cells)
                    if u0 <= u1 and v0 <= v1:
                        min_depth = float(np.min(grid[v0:v1 + 1, u0:u1 + 1]))
                        if np.isfinite(min_depth) and min_depth < (
                                float(candidate_distance) - depth_margin):
                            blocked = True

                if blocked:
                    blocked_visible_hits += 1
                else:
                    clear_visible_hits += 1
                    frame_clear_hits += 1
                    clear_by_camera[camera_name] = clear_by_camera.get(
                        camera_name, 0) + 1

            if frame_hits > 0:
                visible_frames.append(frame_id)
            if frame_clear_hits > 0:
                clear_frames.append(frame_id)

        blocked_ratio = (float(blocked_visible_hits) /
                         float(total_visible_hits)
                         if total_visible_hits > 0 else 1.0)
        return {
            "total_visible_hits": int(total_visible_hits),
            "clear_visible_hits": int(clear_visible_hits),
            "blocked_visible_hits": int(blocked_visible_hits),
            "blocked_ratio": float(blocked_ratio),
            "visible_by_camera": visible_by_camera,
            "clear_by_camera": clear_by_camera,
            "visible_frames": visible_frames,
            "clear_frames": clear_frames,
        }

    def _collect_world_point_ground_label_stats(
        self,
        point_world,
        sensor_type='camera',
        frame_ids=None,
        camera_names=None,
    ):
        """Measure how often a projected world point lands on ground semantics in camera labels."""
        empty_stats = self._empty_ground_label_stats()
        ground_by_camera = dict(empty_stats["ground_by_camera"])
        non_ground_by_camera = dict(empty_stats["non_ground_by_camera"])
        semantic_class_counts = defaultdict(int)
        semantic_label_counts = defaultdict(int)
        surface_type_counts = defaultdict(int)
        if sensor_type not in self.camera_data:
            return empty_stats

        label_hits = 0
        ground_hits = 0
        non_ground_hits = 0
        unknown_hits = 0
        patch_radius = int(self.camera_ground_label_patch_radius)
        num_frames = min(
            len(self.camera_data.get(sensor_type, [])),
            len(self.vehicle_pose.get(sensor_type, [])),
        )
        if frame_ids is None:
            frame_iter = range(num_frames)
        else:
            frame_iter = [
                int(fid) for fid in frame_ids if 0 <= int(fid) < num_frames
            ]
        camera_name_set = set(
            camera_names) if camera_names is not None else None
        for frame_id in frame_iter:
            camera_entries = self._get_camera_entries_for_frame(
                sensor_type, frame_id)
            for camera_name, camera_entry in camera_entries.items():
                if camera_name_set is not None and camera_name not in camera_name_set:
                    continue
                in_bounds, u, v, _z_cam, _radial_distance = self._project_world_point_to_camera(
                    point_world,
                    sensor_type,
                    frame_id,
                    camera_entry,
                )
                if not in_bounds:
                    continue

                sem_seq = self.semantic_labels_by_camera.get(camera_name)
                if sem_seq is None or frame_id >= len(sem_seq):
                    continue

                sem_map = np.asarray(sem_seq[frame_id])
                if sem_map.ndim == 3 and sem_map.shape[-1] == 1:
                    sem_map = sem_map.squeeze(-1)
                if sem_map.ndim != 2 or sem_map.size == 0:
                    continue

                label_h, label_w = sem_map.shape
                cam_w, cam_h = self._resolve_camera_image_size(camera_entry)
                if cam_w <= 0 or cam_h <= 0:
                    continue

                if (label_h, label_w) != (cam_h, cam_w):
                    u_label = int(
                        np.clip(
                            np.rint(
                                float(u) * (float(label_w) / float(cam_w))),
                            0,
                            label_w - 1,
                        ))
                    v_label = int(
                        np.clip(
                            np.rint(
                                float(v) * (float(label_h) / float(cam_h))),
                            0,
                            label_h - 1,
                        ))
                else:
                    u_label = int(np.clip(np.rint(float(u)), 0, label_w - 1))
                    v_label = int(np.clip(np.rint(float(v)), 0, label_h - 1))

                u0 = max(0, u_label - patch_radius)
                u1 = min(label_w - 1, u_label + patch_radius)
                v0 = max(0, v_label - patch_radius)
                v1 = min(label_h - 1, v_label + patch_radius)
                patch = sem_map[v0:v1 + 1, u0:u1 + 1]
                if patch.size == 0:
                    continue

                patch_vals = patch.reshape(-1).astype(np.int32, copy=False)
                patch_vals = patch_vals[patch_vals >= 0]
                if patch_vals.size == 0:
                    continue
                label_hits += 1
                semantic_class = int(np.bincount(patch_vals).argmax())
                semantic_label = self._semantic_label_name_for_surface(
                    semantic_class)
                semantic_class_counts[int(semantic_class)] += 1
                semantic_label_counts[semantic_label] += 1
                surface_type_counts[self._surface_type_from_semantic_label(
                    semantic_label)] += 1
                if semantic_class in self.camera_ground_semantic_ids:
                    ground_hits += 1
                    ground_by_camera[camera_name] = ground_by_camera.get(
                        camera_name, 0) + 1
                elif semantic_class == 0:
                    unknown_hits += 1
                else:
                    non_ground_hits += 1
                    non_ground_by_camera[
                        camera_name] = non_ground_by_camera.get(
                            camera_name, 0) + 1

        denom = max(1, int(label_hits))
        ground_ratio = float(ground_hits) / float(denom)
        non_ground_ratio = float(non_ground_hits) / float(denom)
        ground_camera_count = sum(1 for v in ground_by_camera.values()
                                  if int(v) > 0)
        dominant_semantic_class = None
        dominant_semantic_label = "unknown"
        if semantic_class_counts:
            dominant_semantic_class = max(
                semantic_class_counts.items(),
                key=lambda item: (int(item[1]), -int(item[0])),
            )[0]
            dominant_semantic_label = self._semantic_label_name_for_surface(
                dominant_semantic_class)
        surface_type = self._surface_type_from_semantic_label(
            dominant_semantic_label)
        if surface_type_counts:
            surface_type = max(
                surface_type_counts.items(),
                key=lambda item: (int(item[1]), str(item[0])),
            )[0]
        return {
            "label_hits":
            int(label_hits),
            "ground_hits":
            int(ground_hits),
            "non_ground_hits":
            int(non_ground_hits),
            "unknown_hits":
            int(unknown_hits),
            "ground_ratio":
            float(ground_ratio),
            "non_ground_ratio":
            float(non_ground_ratio),
            "ground_camera_count":
            int(ground_camera_count),
            "ground_by_camera":
            ground_by_camera,
            "non_ground_by_camera":
            non_ground_by_camera,
            "semantic_class_counts": {
                str(int(k)): int(v)
                for k, v in sorted(semantic_class_counts.items())
            },
            "semantic_label_counts":
            {str(k): int(v)
             for k, v in sorted(semantic_label_counts.items())},
            "surface_type_counts":
            {str(k): int(v)
             for k, v in sorted(surface_type_counts.items())},
            "dominant_semantic_class": (None if dominant_semantic_class is None
                                        else int(dominant_semantic_class)),
            "dominant_semantic_label":
            str(dominant_semantic_label),
            "surface_type":
            str(surface_type),
        }

    def _frame_token_for_sensor_index(self, sensor_type, frame_index):
        """Map local frame index to TFRecord frame token when available."""
        try:
            idx = int(frame_index)
        except Exception:
            return frame_index
        frame_map_obj = self.frame_map
        if isinstance(frame_map_obj, dict):
            frame_map = frame_map_obj.get(sensor_type, [])
        elif isinstance(frame_map_obj, (list, tuple)):
            frame_map = frame_map_obj
        else:
            return idx
        if isinstance(frame_map, list) and 0 <= idx < len(frame_map):
            try:
                return int(frame_map[idx])
            except Exception:
                return frame_map[idx]
        return idx

    def _collect_world_point_visibility_detail(self,
                                               point_world,
                                               sensor_type='lidar'):
        """Collect detailed per-frame/per-camera visibility support for one world point."""
        visible_by_camera = {name: 0 for name in self.camera_names}
        if sensor_type not in self.camera_data:
            return 0.0, 0, visible_by_camera, [], {}

        total_visible_hits = 0
        visible_frames = []
        visible_camera_frames = {}
        num_frames = min(len(self.camera_data.get(sensor_type, [])),
                         len(self.vehicle_pose.get(sensor_type, [])))
        for frame_id in range(num_frames):
            frame_hits = 0
            camera_entries = self._get_camera_entries_for_frame(
                sensor_type, frame_id)
            for camera_name, camera_entry in camera_entries.items():
                if self._is_world_point_visible_in_camera(
                        point_world, sensor_type, frame_id, camera_entry):
                    visible_by_camera[camera_name] = visible_by_camera.get(
                        camera_name, 0) + 1
                    total_visible_hits += 1
                    frame_hits += 1
                    visible_camera_frames.setdefault(int(frame_id),
                                                     []).append(camera_name)
            if frame_hits > 0:
                visible_frames.append(frame_id)

        side_arc_hits = (visible_by_camera.get("front_left", 0) +
                         visible_by_camera.get("front_right", 0) +
                         visible_by_camera.get("side_left", 0) +
                         visible_by_camera.get("side_right", 0))
        score = float(total_visible_hits) + (self.side_camera_bonus *
                                             float(side_arc_hits))
        for frame_id, camera_names in list(visible_camera_frames.items()):
            visible_camera_frames[frame_id] = sorted(set(camera_names))
        return score, total_visible_hits, visible_by_camera, visible_frames, visible_camera_frames

    def _collect_world_point_visibility_stats(self,
                                              point_world,
                                              sensor_type='lidar'):
        """Collect visibility statistics for one world point across cameras/frames."""
        (
            score,
            total_visible_hits,
            visible_by_camera,
            visible_frames,
            _visible_camera_frames,
        ) = self._collect_world_point_visibility_detail(
            point_world, sensor_type=sensor_type)
        return score, total_visible_hits, visible_by_camera, visible_frames

    def _get_temporal_target_frame(self, ood_id, sensor_type):
        if not self.enable_temporal_spread_sampling:
            return None
        total_frames = len(self.vehicle_pose.get(sensor_type, []))
        if total_frames < 2:
            return None
        if not hasattr(self, "num_objects") or int(self.num_objects) < 1:
            return None
        numerator = max(0.0, float(ood_id) - 0.5)
        denominator = max(1.0, float(self.num_objects))
        ratio = min(1.0, max(0.0, numerator / denominator))
        return int(round(ratio * float(total_frames - 1)))

    def _resolve_temporal_spread_max_delta(self, total_frames):
        if self.strict_spread_controls:
            strict_auto = max(2, int(round(float(total_frames) * 0.12)))
            if int(self.temporal_spread_max_delta_frames) > 0:
                return max(
                    1,
                    min(int(self.temporal_spread_max_delta_frames),
                        strict_auto))
            return strict_auto
        if int(self.temporal_spread_max_delta_frames) > 0:
            return int(self.temporal_spread_max_delta_frames)

        return max(6, int(round(float(total_frames) * 0.25)))

    @staticmethod
    def _representative_frame_id(visible_frames):
        if not visible_frames:
            return None
        return int(np.median(np.asarray(visible_frames, dtype=np.float32)))

    def _score_world_point_visibility(self, point_world, sensor_type='lidar'):
        """Score candidate visibility across all camera FOVs over the selected frames."""
        score, total_visible_hits, visible_by_camera, _visible_frames = (
            self._collect_world_point_visibility_stats(
                point_world, sensor_type=sensor_type))
        return score, total_visible_hits, visible_by_camera

    def _adjacent_camera_overlap_mismatches(self, visible_by_camera):
        """Return adjacent camera-pair mismatches for left/right arcs."""
        min_hits = int(max(1, self.adjacent_camera_overlap_min_hits))

        def hits(name):
            return int(visible_by_camera.get(name, 0))

        fl = hits("front_left")
        sl = hits("side_left")
        fr = hits("front_right")
        sr = hits("side_right")

        mismatches = []
        if (fl >= min_hits) != (sl >= min_hits):
            mismatches.append("left_pair")
        if (fr >= min_hits) != (sr >= min_hits):
            mismatches.append("right_pair")
        return mismatches

    @staticmethod
    def _estimate_patch_normal(patch_points):
        """Estimate upward-facing normal from local ground patch points."""
        try:
            pts = np.asarray(patch_points, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 3:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)
            centered = pts - np.mean(pts, axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
            if normal[2] < 0:
                normal = -normal
            n = np.linalg.norm(normal)
            if n < 1e-8:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)
            return (normal / n).astype(np.float64)
        except Exception:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _build_shadow_plane_payload(self, ood_id, position):
        """Build shadow-catcher point/normal payload for Blender."""
        details = self.ood_details.get(int(ood_id), {}) or {}
        point = None
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        point_overridden = False
        normal_overridden = False
        override_point = details.get("shadow_plane_point_override")
        if override_point is not None:
            try:
                override_arr = np.asarray(override_point, dtype=np.float64)
                if override_arr.shape[0] >= 3 and np.isfinite(
                        override_arr[:3]).all():
                    point = override_arr[:3].astype(np.float64)
                    point_overridden = True
            except Exception:
                point = None
        override_normal = details.get("shadow_plane_normal_override")
        if override_normal is not None:
            try:
                override_arr = np.asarray(override_normal, dtype=np.float64)
                if override_arr.shape[0] >= 3 and np.isfinite(
                        override_arr[:3]).all():
                    normal = override_arr[:3].astype(np.float64)
                    n = np.linalg.norm(normal)
                    if n >= 1e-8:
                        normal = (normal / n).astype(np.float64)
                        normal_overridden = True
            except Exception:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        try:
            if not point_overridden:
                pos_arr = np.asarray(position, dtype=np.float64)
                if pos_arr.shape == (4, 4):
                    point = np.array(
                        [pos_arr[0, 3], pos_arr[1, 3], pos_arr[2, 3]],
                        dtype=np.float64)
                elif pos_arr.shape[0] >= 3:
                    point = pos_arr[:3].astype(np.float64)
        except Exception:
            if not point_overridden:
                point = None

        patch = self.ood_location.get(int(ood_id))
        if patch is not None and np.asarray(patch).size > 0:
            if not normal_overridden:
                normal = self._estimate_patch_normal(patch)
            if point is None and not point_overridden:
                try:
                    point = np.mean(np.asarray(patch, dtype=np.float64),
                                    axis=0)
                except Exception:
                    point = None

        if point is None:
            point = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        catcher_size = float(
            details.get("shadow_catcher_size_override",
                        self.shadow_catcher_size))
        try:
            dist = float(
                self._min_distance_to_pose(point, pose_sensor='camera'))
            if np.isfinite(dist):
                catcher_size = max(catcher_size,
                                   min(120.0, catcher_size + 0.6 * dist))
        except Exception:
            pass
        return {
            "shadow_plane_point":
            [float(point[0]),
             float(point[1]),
             float(point[2])],
            "shadow_plane_normal":
            [float(normal[0]),
             float(normal[1]),
             float(normal[2])],
            "shadow_catcher_size":
            float(catcher_size),
            "ground_contact_epsilon":
            float(self.ground_contact_epsilon),
        }

    def _render_root_dir(self, eval=False):
        if eval:
            return Path(
                self.base_dir) / self.file_name / "Evaluation" / "Renders"
        return Path(self.base_dir) / self.file_name / "Renders"

    def _infer_shadow_quality_cameras(self,
                                      ood_id,
                                      eval=False,
                                      min_hits=1,
                                      min_pixels=1):
        """Infer cameras with sufficient rendered object visibility for shadow checks."""
        if 'camera' not in self.frame_map:
            return []

        render_root = self._render_root_dir(eval=eval)
        ood_camera_dir = render_root / f"ood_{int(ood_id)}" / "camera"
        if not ood_camera_dir.exists():
            return []

        camera_frame_ids = self.frame_map.get('camera', [])
        camera_names = []
        try:
            camera_names = sorted(
                [p.name for p in ood_camera_dir.iterdir() if p.is_dir()])
        except Exception:
            camera_names = []
        if not camera_names:
            camera_names = list(self.camera_name_to_id.keys())

        inferred = []
        for camera_name in camera_names:
            hit_count = 0
            pixel_sum = 0
            for frame_token in camera_frame_ids:
                no_shadow_path = (render_root / f"ood_{int(ood_id)}" /
                                  "camera" / camera_name /
                                  f"no_shadow_frame_00{frame_token}.png")
                if not no_shadow_path.exists():
                    continue
                try:
                    no_shadow_rgba = np.asarray(Image.open(no_shadow_path))
                except Exception:
                    continue
                mask = self._extract_alpha_mask_with_fallback(
                    no_shadow_rgba,
                    hard_alpha_threshold=100,
                    min_pixels=min_pixels,
                )
                visible_pixels = int(np.count_nonzero(mask))
                if visible_pixels >= int(max(1, min_pixels)):
                    hit_count += 1
                    pixel_sum += visible_pixels
            if hit_count >= int(max(1, min_hits)):
                inferred.append({
                    "camera": camera_name,
                    "hits": int(hit_count),
                    "pixel_sum": int(pixel_sum),
                })

        inferred.sort(key=lambda x: (-x["hits"], -x["pixel_sum"], x["camera"]))
        return [entry["camera"] for entry in inferred]

    def _delete_ood_render_artifacts(self, ood_id, eval=False):
        """Remove existing render artifacts for one OOD id before a resample/rerender."""
        render_root = self._render_root_dir(eval=eval)
        ood_dir = render_root / f"ood_{int(ood_id)}"
        if ood_dir.exists():
            shutil.rmtree(ood_dir, ignore_errors=True)
        summary_txt = render_root / f"ood_{int(ood_id)}.txt"
        if summary_txt.exists():
            try:
                summary_txt.unlink()
            except OSError:
                pass

    def _validate_required_camera_render_visibility(self, ood_id, eval=False):
        """Validate that rendered no-shadow masks are visible in required cameras."""
        if not self.required_visible_cameras:
            return {"ok": True, "counts": {}, "frames": {}, "missing": []}
        if 'camera' not in self.frame_map:
            return {"ok": True, "counts": {}, "frames": {}, "missing": []}

        min_hits = int(max(1, self.required_visible_camera_min_hits))
        min_pixels = int(max(1, self.required_render_min_pixels))
        render_root = self._render_root_dir(eval=eval)
        camera_frame_ids = self.frame_map.get('camera', [])

        counts = {}
        frames = {}
        missing = []
        for camera_name in self.required_visible_cameras:
            hit_frames = []
            for frame_token in camera_frame_ids:
                image_path = (render_root / f"ood_{int(ood_id)}" / "camera" /
                              camera_name /
                              f"no_shadow_frame_00{frame_token}.png")
                if not image_path.exists():
                    continue
                try:
                    rgba = np.asarray(Image.open(image_path))
                except Exception:
                    continue
                if rgba.ndim != 3 or rgba.shape[2] < 4:
                    continue
                alpha = rgba[:, :, 3]
                visible_pixels = int(np.count_nonzero(alpha > 0))
                if visible_pixels >= min_pixels:
                    hit_frames.append(int(frame_token))

            counts[camera_name] = int(len(hit_frames))
            frames[camera_name] = hit_frames
            if len(hit_frames) < min_hits:
                missing.append(camera_name)

        return {
            "ok": len(missing) == 0,
            "counts": counts,
            "frames": frames,
            "missing": missing,
            "min_hits": int(min_hits),
            "min_pixels": int(min_pixels),
        }

    @staticmethod
    def _extract_alpha_mask_with_fallback(
        rgba,
        hard_alpha_threshold=100,
        min_pixels=64,
        fallback_rgba=None,
    ):
        """Extract an object alpha mask with robust fallback for semi-transparent/noisy alpha."""
        if rgba is None:
            return None
        if rgba.ndim != 3 or rgba.shape[2] < 4:
            return None

        alpha = np.asarray(rgba[:, :, 3])
        mask_hard = alpha > int(max(0, hard_alpha_threshold))
        if int(np.count_nonzero(mask_hard)) >= int(max(1, min_pixels)):
            return mask_hard

        mask_soft = alpha > 0
        if int(np.count_nonzero(mask_soft)) > 0:
            return mask_soft

        if fallback_rgba is not None and fallback_rgba.ndim == 3 and fallback_rgba.shape[
                2] >= 4:
            fb = np.asarray(fallback_rgba[:, :, 3])
            if fb.shape != alpha.shape:
                fb = cv2.resize(
                    fb.astype(np.uint8),
                    (alpha.shape[1], alpha.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            fb_mask = fb > int(max(0, hard_alpha_threshold))
            if int(np.count_nonzero(fb_mask)) > 0:
                return fb_mask
            fb_mask = fb > 0
            if int(np.count_nonzero(fb_mask)) > 0:
                return fb_mask

        return mask_hard

    @staticmethod
    def _shadow_metrics_from_rgba(full_rgba, no_shadow_rgba, dilate_px=6):
        """Compute shadow-only and contact metrics from full vs object-only alpha."""
        if full_rgba is None or no_shadow_rgba is None:
            return None
        if full_rgba.ndim != 3 or no_shadow_rgba.ndim != 3:
            return None
        if full_rgba.shape[2] < 4 or no_shadow_rgba.shape[2] < 4:
            return None

        alpha_obj = Sampler._extract_alpha_mask_with_fallback(
            no_shadow_rgba,
            hard_alpha_threshold=100,
            min_pixels=64,
            fallback_rgba=full_rgba,
        )
        alpha_full = np.asarray(full_rgba[:, :, 3]) > 0
        if alpha_full.shape != alpha_obj.shape:
            alpha_full = cv2.resize(
                alpha_full.astype(np.uint8),
                (alpha_obj.shape[1], alpha_obj.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0

        k = int(max(0, dilate_px))
        if k > 0:
            kernel = np.ones((k * 2 + 1, k * 2 + 1), dtype=np.uint8)
            alpha_obj_dil = cv2.dilate(
                alpha_obj.astype(np.uint8), kernel, iterations=1) > 0
        else:
            alpha_obj_dil = alpha_obj
        shadow_only = np.logical_and(alpha_full, np.logical_not(alpha_obj_dil))
        contact_band = np.logical_and(alpha_obj_dil, np.logical_not(alpha_obj))
        shadow_pixels = int(np.count_nonzero(shadow_only))
        object_pixels = int(np.count_nonzero(alpha_obj))
        contact_pixels = int(
            np.count_nonzero(np.logical_and(shadow_only, contact_band)))
        ratio = float(shadow_pixels / max(1, object_pixels))
        contact_ratio = float(contact_pixels / max(1, shadow_pixels))

        near_contact_ratio = 0.0
        median_shadow_distance_px = None
        if shadow_pixels > 0:
            inv_obj = np.logical_not(alpha_obj).astype(np.uint8)
            dist_to_obj = cv2.distanceTransform(inv_obj, cv2.DIST_L2, 5)
            shadow_distances = dist_to_obj[shadow_only]
            if shadow_distances.size > 0:
                median_shadow_distance_px = float(np.median(shadow_distances))
                near_threshold_px = max(1.5, float(k) * 0.75)
                near_contact_ratio = float(
                    np.count_nonzero(shadow_distances <= near_threshold_px) /
                    shadow_distances.size)

        return {
            "shadow_only": shadow_only,
            "object_mask": alpha_obj,
            "shadow_pixels": shadow_pixels,
            "object_pixels": object_pixels,
            "contact_pixels": contact_pixels,
            "shadow_ratio": ratio,
            "contact_ratio": contact_ratio,
            "near_contact_ratio": float(near_contact_ratio),
            "median_shadow_distance_px": median_shadow_distance_px,
        }

    def _validate_required_camera_shadow_quality(self, ood_id, eval=False):
        """Validate shadow quality in required cameras before accepting a render."""
        if not self.enable_shadow_quality_gate:
            return {
                "ok": True,
                "missing": [],
                "camera_stats": {},
                "fail_reasons": {}
            }
        if 'camera' not in self.frame_map:
            return {
                "ok": True,
                "missing": [],
                "camera_stats": {},
                "fail_reasons": {}
            }

        def _frame_direct_illumination(camera_name, frame_idx):
            try:
                frame_entries = self._get_camera_entries_for_frame(
                    'camera', int(frame_idx))
                camera_entry = frame_entries.get(camera_name)
                if not isinstance(camera_entry, dict):
                    return None
                weather = camera_entry.get('weather_api', {})
                if not isinstance(weather, dict):
                    return None
                raw_direct = weather.get('direct_illumination')
                if raw_direct is None:
                    return None
                direct = float(raw_direct)
                if direct > 1.5:
                    direct /= 700.0
                return float(np.clip(direct, 0.0, 1.0))
            except Exception:
                return None

        render_root = self._render_root_dir(eval=eval)
        camera_frame_ids = self.frame_map.get('camera', [])
        min_hits = int(max(1, self.required_visible_camera_min_hits))
        min_pass_frames = int(max(1, self.shadow_quality_min_pass_frames))
        min_required = int(max(min_hits, min_pass_frames))
        min_pixels = int(max(1, self.required_render_min_pixels))
        min_pass_fraction = float(
            np.clip(self.shadow_quality_min_pass_fraction, 0.0, 1.0))
        min_direct_illum = float(
            np.clip(self.shadow_quality_min_direct_illumination, 0.0, 1.0))
        required_cameras = list(self.required_visible_cameras)
        if not required_cameras:
            required_cameras = self._infer_shadow_quality_cameras(
                ood_id=ood_id,
                eval=eval,
                min_hits=min_hits,
                min_pixels=min_pixels,
            )
        if not required_cameras:
            return {
                "ok": True,
                "missing": [],
                "camera_stats": {},
                "fail_reasons": {},
                "required_cameras": [],
            }

        camera_stats = {}
        missing = []
        fail_reasons = {}

        for camera_name in required_cameras:
            visible_frames = []
            passing_frames = []
            strict_visible_frames = []
            strict_passing_frames = []
            low_light_frames = []
            frame_metrics = []
            strict_shadow_ratios = []
            strict_contact_ratios = []
            reason_counts = {}
            for frame_idx, frame_token in enumerate(camera_frame_ids):
                no_shadow_path = (render_root / f"ood_{int(ood_id)}" /
                                  "camera" / camera_name /
                                  f"no_shadow_frame_00{frame_token}.png")
                full_path = (render_root / f"ood_{int(ood_id)}" / "camera" /
                             camera_name / f"frame_00{frame_token}.png")
                if not no_shadow_path.exists() or not full_path.exists():
                    continue
                try:
                    no_shadow_rgba = np.asarray(Image.open(no_shadow_path))
                    full_rgba = np.asarray(Image.open(full_path))
                except Exception:
                    continue

                metrics = self._shadow_metrics_from_rgba(
                    full_rgba=full_rgba,
                    no_shadow_rgba=no_shadow_rgba,
                    dilate_px=self.shadow_quality_dilate_px,
                )
                if metrics is None:
                    continue
                if int(metrics["object_pixels"]) < min_pixels:
                    continue

                visible_frames.append(int(frame_token))
                shadow_ratio = float(metrics["shadow_ratio"])
                contact_ratio = float(metrics["contact_ratio"])
                direct_illum = _frame_direct_illumination(
                    camera_name, frame_idx)
                low_light = ((direct_illum is not None)
                             and (float(direct_illum) < min_direct_illum))
                if low_light:
                    low_light_frames.append(int(frame_token))

                passes = True
                reason = "pass"
                if low_light:

                    passes = True
                    reason = "low_light_relaxed"
                else:
                    strict_visible_frames.append(int(frame_token))
                    strict_shadow_ratios.append(shadow_ratio)
                    strict_contact_ratios.append(contact_ratio)
                    near_contact_ratio = float(
                        metrics.get("near_contact_ratio", 0.0))
                    if int(metrics["shadow_pixels"]) <= 0:
                        passes = False
                        reason = "no_shadow"
                    elif (contact_ratio < float(
                            self.shadow_quality_min_contact_ratio)
                          and near_contact_ratio < max(
                              0.06,
                              float(self.shadow_quality_min_contact_ratio) *
                              8.0)):
                        passes = False
                        reason = "disconnected_shadow"
                    elif shadow_ratio < float(self.shadow_quality_min_ratio):
                        passes = False
                        reason = "shadow_too_weak"
                    elif shadow_ratio > float(self.shadow_quality_max_ratio):
                        passes = False
                        reason = "shadow_too_strong"
                if passes:
                    passing_frames.append(int(frame_token))
                    if not low_light:
                        strict_passing_frames.append(int(frame_token))
                else:
                    reason_counts[reason] = int(reason_counts.get(reason,
                                                                  0)) + 1
                frame_metrics.append({
                    "frame":
                    int(frame_token),
                    "shadow_ratio":
                    shadow_ratio,
                    "contact_ratio":
                    contact_ratio,
                    "shadow_pixels":
                    int(metrics["shadow_pixels"]),
                    "object_pixels":
                    int(metrics["object_pixels"]),
                    "pass":
                    bool(passes),
                    "reason":
                    reason,
                    "near_contact_ratio":
                    float(metrics.get("near_contact_ratio", 0.0)),
                    "median_shadow_distance_px":
                    metrics.get("median_shadow_distance_px"),
                    "direct_illumination":
                    (None if direct_illum is None else float(direct_illum)),
                    "low_light_relaxed":
                    bool(low_light),
                })

            visible_count = int(len(visible_frames))
            passing_count = int(len(strict_passing_frames))
            strict_visible_count = int(len(strict_visible_frames))
            fraction_required = int(
                math.ceil(
                    float(strict_visible_count) * float(min_pass_fraction)))
            required_frames = int(max(min_required, fraction_required))
            if strict_visible_count > 0:
                required_frames = int(
                    min(required_frames, strict_visible_count))
            else:
                required_frames = 0

            median_shadow_ratio = (float(np.median(strict_shadow_ratios))
                                   if strict_shadow_ratios else 0.0)
            median_contact_ratio = (float(np.median(strict_contact_ratios))
                                    if strict_contact_ratios else 0.0)
            p75_contact_ratio = (float(np.percentile(
                strict_contact_ratios, 75)) if strict_contact_ratios else 0.0)
            relaxed_shadow_low = float(self.shadow_quality_min_ratio) * 0.5
            relaxed_shadow_high = float(self.shadow_quality_max_ratio) * 1.5
            percentile_contact_min = float(
                self.shadow_quality_min_contact_ratio) * 0.5

            camera_reason_list = []
            if visible_count < min_hits:
                camera_reason_list.append(
                    f"insufficient_visible_frames({visible_count}<{min_hits})")
            if strict_visible_count > 0 and passing_count < required_frames:
                camera_reason_list.append(
                    f"insufficient_pass_frames({passing_count}<{required_frames})"
                )
            if strict_visible_count > 0 and passing_count < 1:
                camera_reason_list.append("no_contact_shadow_frame")
            if strict_visible_count > 0 and not (relaxed_shadow_low <=
                                                 median_shadow_ratio <=
                                                 relaxed_shadow_high):
                camera_reason_list.append("shadow_ratio_median_out_of_range")
            if strict_visible_count > 0 and p75_contact_ratio < percentile_contact_min:
                camera_reason_list.append("contact_ratio_p75_too_low")

            camera_stats[camera_name] = {
                "visible_frames": visible_frames,
                "passing_frames": passing_frames,
                "strict_visible_frames": strict_visible_frames,
                "strict_passing_frames": strict_passing_frames,
                "low_light_frames": low_light_frames,
                "frame_metrics": frame_metrics,
                "required_min_frames": int(required_frames),
                "visible_count": visible_count,
                "passing_count": passing_count,
                "strict_visible_count": strict_visible_count,
                "median_shadow_ratio": median_shadow_ratio,
                "median_contact_ratio": median_contact_ratio,
                "p75_contact_ratio": p75_contact_ratio,
                "reason_histogram": reason_counts,
            }
            fail_reasons[camera_name] = camera_reason_list
            if len(camera_reason_list) > 0:
                missing.append(camera_name)

        return {
            "ok": len(missing) == 0,
            "missing": missing,
            "camera_stats": camera_stats,
            "fail_reasons": fail_reasons,
            "min_required_frames": int(min_required),
            "required_cameras": list(required_cameras),
        }

    def _resolve_no_shadow_render_path(self,
                                       ood_id,
                                       sensor_type,
                                       frame_id,
                                       eval=False,
                                       camera_name=None):
        """Resolve no-shadow render path for camera-aware and legacy flat layouts."""
        frame_token = self.frame_map[sensor_type][frame_id]
        render_root = self._render_root_dir(
            eval=eval) / f"ood_{int(ood_id)}" / str(sensor_type)
        filename = f"no_shadow_frame_00{frame_token}.png"
        candidate_paths = []
        if sensor_type == 'camera' and camera_name:
            candidate_paths.append(render_root / str(camera_name) / filename)
        candidate_paths.append(render_root / filename)
        for path in candidate_paths:
            if path.exists():
                return str(path)
        return str(candidate_paths[0])

    def _recover_camera_raw_images_from_disk(self):
        """Best-effort recovery of per-camera raw images for retry/debug flows."""
        camera_root = Path(
            self.base_dir) / self.file_name / "Raw_Images" / "camera"
        if not camera_root.is_dir():
            return None
        frame_tokens = self.frame_map.get('camera', [])
        if len(frame_tokens) == 0:
            return None

        recovered = {}
        for camera_dir in sorted(camera_root.iterdir()):
            if not camera_dir.is_dir():
                continue
            frame_images = []
            for frame_token in frame_tokens:
                image_path = camera_dir / f"frame_00{frame_token}.png"
                if not image_path.exists():
                    frame_images = []
                    break
                try:
                    frame_images.append(np.array(Image.open(image_path)))
                except Exception:
                    frame_images = []
                    break
            if frame_images:
                recovered[camera_dir.name] = frame_images
        return recovered or None

    def _compute_camera_occlusion_mask_from_lidar(self, ood_id, frame_id,
                                                  camera_name, ood_mask):
        """Estimate camera occlusion mask from projected LiDAR points for one camera/frame."""
        panoptic_divisor = int(
            max(1, getattr(self, "panoptic_label_divisor", 1000)))

        def _panoptic_semantic(pid):
            return int(pid) // panoptic_divisor

        def _panoptic_instance(pid):
            return int(pid) % panoptic_divisor

        if ('camera' not in self.point_cloud_vehicle
                or frame_id not in self.point_cloud_vehicle['camera']
                or 'camera' not in self.cp_actual
                or frame_id not in self.cp_actual['camera']):
            return np.zeros_like(ood_mask, dtype=np.uint8)

        if ood_id not in self.ood_details or 'pose' not in self.ood_details[
                ood_id]:
            return np.zeros_like(ood_mask, dtype=np.uint8)

        object_pose_world = np.asarray(self.ood_details[ood_id]['pose'])
        object_pose_vehicle = np.linalg.inv(
            self.vehicle_pose['camera'][frame_id]) @ object_pose_world
        instance_point = object_pose_vehicle[:3, 3]
        object_distance = np.linalg.norm(np.array(instance_point), ord=2)

        points = np.asarray(
            self.point_cloud_vehicle['camera'][frame_id].points)
        cp_points = np.asarray(self.cp_actual['camera'][frame_id])
        if points.shape[0] == 0 or cp_points.shape[0] == 0:
            return np.zeros_like(ood_mask, dtype=np.uint8)

        distances = np.linalg.norm(points, ord=2, axis=1)
        probable_occluders = distances < (object_distance - 0.05)
        cp_occlusion = cp_points[probable_occluders]
        if cp_occlusion.shape[0] == 0:
            return np.zeros_like(ood_mask, dtype=np.uint8)

        camera_id = self.camera_name_to_id.get(camera_name)
        if camera_id is None:
            return np.zeros_like(ood_mask, dtype=np.uint8)

        proj_pixels = []
        proj_depths = []
        occluder_depths = distances[probable_occluders].astype(np.float32)
        if cp_occlusion.shape[1] >= 3:
            mask_primary = cp_occlusion[:, 0] == camera_id
            if np.any(mask_primary):
                proj_pixels.append(cp_occlusion[mask_primary][:, 1:3])
                proj_depths.append(occluder_depths[mask_primary])
        if cp_occlusion.shape[1] >= 6:
            mask_secondary = cp_occlusion[:, 3] == camera_id
            if np.any(mask_secondary):
                proj_pixels.append(cp_occlusion[mask_secondary][:, 4:6])
                proj_depths.append(occluder_depths[mask_secondary])
        if not proj_pixels or not proj_depths:
            return np.zeros_like(ood_mask, dtype=np.uint8)

        proj = np.vstack(proj_pixels)
        proj_depth = np.concatenate(proj_depths).astype(np.float32)
        proj = np.rint(proj).astype(int)
        h, w = ood_mask.shape
        valid = ((proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) &
                 (proj[:, 1] < h) & np.isfinite(proj_depth))
        proj = proj[valid]
        proj_depth = proj_depth[valid]
        if proj.shape[0] == 0:
            return np.zeros_like(ood_mask, dtype=np.uint8)

        proj_mask = np.zeros((h, w), dtype=np.uint8)
        depth_front_threshold = float(object_distance) - 0.05
        use_depth_fill = (self.camera_occlusion_depth_fill == 'multiscale'
                          and depth_front_threshold > 0.0
                          and np.count_nonzero(ood_mask) > 0)
        depth_fill_applied = False
        if use_depth_fill:
            rows, cols = np.where(ood_mask)
            if rows.size > 0:
                pad = int(self.camera_occlusion_depth_fill_bbox_pad_px)
                y0 = max(0, int(rows.min()) - pad)
                y1 = min(h - 1, int(rows.max()) + pad)
                x0 = max(0, int(cols.min()) - pad)
                x1 = min(w - 1, int(cols.max()) + pad)
                in_crop = ((proj[:, 0] >= x0) & (proj[:, 0] <= x1) &
                           (proj[:, 1] >= y0) & (proj[:, 1] <= y1))
                if np.any(in_crop):
                    u_crop = proj[in_crop, 0] - x0
                    v_crop = proj[in_crop, 1] - y0
                    d_crop = proj_depth[in_crop]
                    crop_h = y1 - y0 + 1
                    crop_w = x1 - x0 + 1
                    sparse_crop = np.zeros((crop_h, crop_w), dtype=np.float32)
                    seed_mask_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
                    min_depth = np.full((crop_h, crop_w),
                                        np.inf,
                                        dtype=np.float32)
                    np.minimum.at(min_depth, (v_crop, u_crop), d_crop)
                    valid_depth = np.isfinite(min_depth)
                    sparse_crop[valid_depth] = min_depth[valid_depth]
                    seed_mask_crop[v_crop, u_crop] = 1
                    max_depth = max(
                        float(self.camera_occlusion_depth_fill_max_depth),
                        float(object_distance) + 5.0,
                    )
                    try:
                        dense_crop, _ = depth_fill_in_multiscale(
                            sparse_crop,
                            max_depth=max_depth,
                            depth_range=(1.0, 15.0, 30.0),
                            extrapolate=False,
                            blur_type='bilateral',
                            show_process=False,
                        )
                    except Exception:
                        dense_crop = None
                    if dense_crop is not None:
                        front_crop = ((dense_crop > 0.1)
                                      & np.isfinite(dense_crop)
                                      & (dense_crop < depth_front_threshold))
                        radius_px = int(
                            self.camera_occlusion_depth_fill_radius_px)
                        if radius_px > 0 and np.count_nonzero(
                                seed_mask_crop) > 0:
                            dist_to_seed = cv2.distanceTransform(
                                (1 - seed_mask_crop).astype(np.uint8),
                                cv2.DIST_L2,
                                5,
                            )
                            front_crop = front_crop & (dist_to_seed <=
                                                       float(radius_px))
                        proj_mask[y0:y1 + 1,
                                  x0:x1 + 1] = front_crop.astype(np.uint8)

                        proj_mask[proj[:, 1], proj[:, 0]] = 1
                        depth_fill_applied = True

        if not depth_fill_applied:
            proj_mask[proj[:, 1], proj[:, 0]] = 1
            if self.occlusion_projection_dilation > 1:
                k = int(self.occlusion_projection_dilation)
                kernel = np.ones((k, k), dtype=np.uint8)
                proj_mask = cv2.dilate(proj_mask, kernel, iterations=1)

        panoptic_label = None
        if (camera_name in self.panoptic_labels_by_camera and
                frame_id < len(self.panoptic_labels_by_camera[camera_name])):
            panoptic_label = np.asarray(
                self.panoptic_labels_by_camera[camera_name][frame_id])
            if panoptic_label.ndim == 3 and panoptic_label.shape[-1] == 1:
                panoptic_label = panoptic_label.squeeze(-1)

        common_rows, common_cols = np.where((proj_mask > 0) & ood_mask)
        common_pixels = set(zip(common_cols.tolist(), common_rows.tolist()))
        occlusion_mask = np.zeros((h, w), dtype=np.uint8)
        min_common_pixels = int(self.occlusion_min_common_pixels)
        expanded_with_thing_ids = False

        def _expand_overlap_ids_from_pixels(rows, cols, existing_mask=None):
            nonlocal expanded_with_thing_ids
            if panoptic_label is None or panoptic_label.ndim != 2:
                return np.zeros((h, w), dtype=np.uint8)
            if rows.size == 0:
                return np.zeros((h, w), dtype=np.uint8)

            label_h, label_w = panoptic_label.shape
            if (label_h, label_w) != (h, w):
                u_label = np.clip(
                    (cols.astype(np.float32) * (label_w / float(w))).astype(
                        np.int32),
                    0,
                    label_w - 1,
                )
                v_label = np.clip(
                    (rows.astype(np.float32) * (label_h / float(h))).astype(
                        np.int32),
                    0,
                    label_h - 1,
                )
            else:
                u_label, v_label = cols, rows

            overlap_ids, overlap_counts = np.unique(panoptic_label[v_label,
                                                                   u_label],
                                                    return_counts=True)
            overlap_mask = np.zeros((h, w), dtype=np.uint8)
            ground_semantic_set = self.camera_ground_semantic_ids
            for pid, cnt in zip(overlap_ids, overlap_counts):
                pid = int(pid)
                cnt = int(cnt)
                if pid == 0:
                    continue
                sem_class = _panoptic_semantic(pid)
                if sem_class == self.ood_semantic_class:
                    continue
                inst_id = _panoptic_instance(pid)
                if inst_id != 0:
                    if not self.enable_thing_occlusion_fallback:
                        continue
                    if cnt < self.thing_occlusion_fallback_min_hits:
                        continue
                    expanded_with_thing_ids = True
                else:
                    if not self.enable_stuff_occlusion_fallback:
                        continue
                    if sem_class in ground_semantic_set:
                        continue
                    if cnt < self.stuff_occlusion_fallback_min_hits:
                        continue
                pid_mask = (panoptic_label == pid).astype(np.uint8)
                if (label_h, label_w) != (h, w):
                    pid_mask = cv2.resize(pid_mask, (w, h),
                                          interpolation=cv2.INTER_NEAREST)
                overlap_mask[pid_mask > 0] = 1

            if np.count_nonzero(
                    overlap_mask) > 0 and existing_mask is not None:
                overlap_mask = np.maximum(overlap_mask,
                                          existing_mask.astype(np.uint8))
            return overlap_mask.astype(np.uint8)

        if len(common_pixels) < min_common_pixels:
            if not (self.enable_stuff_occlusion_fallback
                    or self.enable_thing_occlusion_fallback):
                return occlusion_mask.astype(np.uint8)
            if panoptic_label is None or panoptic_label.ndim != 2:
                return occlusion_mask.astype(np.uint8)
            ood_rows, ood_cols = np.where(ood_mask)
            if ood_rows.size == 0:
                return occlusion_mask.astype(np.uint8)
            occlusion_mask = _expand_overlap_ids_from_pixels(
                rows=ood_rows,
                cols=ood_cols,
            )
            if np.count_nonzero(occlusion_mask) > 0:
                occlusion_mask = cv2.dilate(occlusion_mask,
                                            np.ones((3, 3), np.uint8),
                                            iterations=1)
            return occlusion_mask.astype(np.uint8)

        if panoptic_label is not None and panoptic_label.ndim == 2:
            label_h, label_w = panoptic_label.shape
            common_array = np.array(list(common_pixels), dtype=np.int32)
            u = common_array[:, 0]
            v = common_array[:, 1]
            if (label_h, label_w) != (h, w):
                u_label = np.clip((u.astype(np.float32) *
                                   (label_w / float(w))).astype(np.int32), 0,
                                  label_w - 1)
                v_label = np.clip((v.astype(np.float32) *
                                   (label_h / float(h))).astype(np.int32), 0,
                                  label_h - 1)
            else:
                u_label, v_label = u, v

            panoptic_ids = panoptic_label[v_label, u_label]
            unique_ids, unique_counts = np.unique(panoptic_ids,
                                                  return_counts=True)
            id_hit_count = {
                int(pid): int(cnt)
                for pid, cnt in zip(unique_ids, unique_counts)
            }

            expand_ids = set()
            for pid in unique_ids:
                pid = int(pid)
                if pid == 0:
                    continue
                sem_class = _panoptic_semantic(pid)
                if sem_class == self.ood_semantic_class:
                    continue

                if _panoptic_instance(pid) != 0:
                    expand_ids.add(pid)
                    expanded_with_thing_ids = True
                    continue

                if not self.enable_stuff_occlusion_expansion:
                    continue
                if sem_class in self.camera_ground_semantic_ids:
                    continue
                if id_hit_count.get(pid, 0) >= self.stuff_occlusion_min_hits:
                    expand_ids.add(pid)

            for pid in expand_ids:
                pid_mask = (panoptic_label == pid).astype(np.uint8)
                if (label_h, label_w) != (h, w):
                    pid_mask = cv2.resize(pid_mask, (w, h),
                                          interpolation=cv2.INTER_NEAREST)
                occlusion_mask[pid_mask > 0] = 1

        if panoptic_label is not None and panoptic_label.ndim == 2 and len(
                common_pixels) > 0:
            label_h, label_w = panoptic_label.shape
            common_array = np.array(list(common_pixels), dtype=np.int32)
            u = common_array[:, 0]
            v = common_array[:, 1]
            if (label_h, label_w) != (h, w):
                u_label = np.clip((u.astype(np.float32) *
                                   (label_w / float(w))).astype(np.int32), 0,
                                  label_w - 1)
                v_label = np.clip((v.astype(np.float32) *
                                   (label_h / float(h))).astype(np.int32), 0,
                                  label_h - 1)
            else:
                u_label, v_label = u, v

            common_panoptic = panoptic_label[v_label, u_label]
            for idx, pid in enumerate(common_panoptic):
                pid = int(pid)
                if pid == 0:
                    continue
                sem_class = _panoptic_semantic(pid)
                if sem_class == self.ood_semantic_class:
                    continue
                inst_id = _panoptic_instance(pid)
                if inst_id != 0:
                    occlusion_mask[v[idx], u[idx]] = 1
                    continue
                if self.enable_stuff_occlusion_expansion and sem_class not in self.camera_ground_semantic_ids:
                    occlusion_mask[v[idx], u[idx]] = 1
        else:
            for u, v in common_pixels:
                occlusion_mask[v, u] = 1

        if ((self.enable_stuff_occlusion_fallback
             or self.enable_thing_occlusion_fallback)
                and panoptic_label is not None and panoptic_label.ndim == 2):
            residual_rows, residual_cols = np.where(ood_mask
                                                    & (occlusion_mask == 0))
            if residual_rows.size > 0:
                residual_mask = _expand_overlap_ids_from_pixels(
                    rows=residual_rows,
                    cols=residual_cols,
                    existing_mask=occlusion_mask,
                )
                occlusion_mask = np.maximum(occlusion_mask,
                                            residual_mask).astype(np.uint8)

        ood_area = int(np.count_nonzero(ood_mask))
        if ood_area > 0:
            occluded_area = int(np.count_nonzero(occlusion_mask))
            occluded_fraction = float(occluded_area) / float(ood_area)
            if (occluded_fraction > float(self.max_occlusion_fraction_of_ood)
                    and not expanded_with_thing_ids):

                return np.zeros((h, w), dtype=np.uint8)

        occlusion_mask = cv2.dilate(occlusion_mask,
                                    np.ones((3, 3), np.uint8),
                                    iterations=1)
        return occlusion_mask.astype(np.uint8)

    def sample_ood_location(self,
                            scale=1,
                            ood_id=None,
                            sensor_type='lidar',
                            render_mode='multi',
                            eval=False,
                            dist=None):
        print(
            "\033[36mSAMPLING: Sampling random OOD locations and checking.\033[0m"
        )
        """Method to sample possible locations for ood object and check for feasibility.
        Sample random frame of all frames available. Sample random angle and distance.
        Compute location including z_coordinate from terrain estimation. Check whether
        location colides with any existing bouding box. Check whether test plane rendered
        actually appears on final image that is labeled street or similar. Repeat and
        decrease parameters until successful.
        Returns: 4x4 transformation representing a randomly sampled pose including rotation."""

        if render_mode == 'multi':
            selected_point_dataset_world = self.check_render_points_for_lidar(
                ood_id, scale, sensor_type, render_mode=render_mode, eval=eval)
        else:
            selected_point_dataset_world = self.check_render_points_for_lidar(
                ood_id,
                scale,
                sensor_type,
                render_mode=render_mode,
                eval=eval,
                dist=dist)

        if np.array_equal(selected_point_dataset_world, np.zeros((3, ))):

            return None, None

        x, y, z = selected_point_dataset_world[
            0], selected_point_dataset_world[1], selected_point_dataset_world[
                2]

        random_pose = [[0, 0, 0, x], [0, 0, 0, y], [0, 0, 0, z], [0, 0, 0, 1]]

        print(f"\033[32mLocation found.\033[0m")

        return random_pose, 0

        return None, None

    def convert_lidar_points_to_world_points(self, points, frame_id):

        if points.ndim == 1:
            point_dataset_lidar = np.asarray(points).reshape(1, -1)
        else:
            point_dataset_lidar = np.asarray(points)

        N = point_dataset_lidar.shape[0]

        points_homogeneous = np.hstack((point_dataset_lidar, np.ones((N, 1))))

        transformation = np.asarray(
            self.lidar_parameters[frame_id]['pose_global'])

        points_dataset_world = (transformation @ points_homogeneous.T).T

        return points_dataset_world[:, :3]

    def _parse_gpu_id_csv(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, np.ndarray)):
            raw_tokens = value
        else:
            text = str(value).strip()
            if text == "":
                return []
            raw_tokens = [tok.strip() for tok in text.split(",")]
        out = []
        seen = set()
        for token in raw_tokens:
            if token is None or str(token).strip() == "":
                continue
            try:
                gpu_id = int(str(token).strip())
            except Exception:
                continue
            if gpu_id < 0 or gpu_id in seen:
                continue
            out.append(gpu_id)
            seen.add(gpu_id)
        return out

    def _resolve_blender_gpu_ids(self, configured_gpu_ids):
        configured = self._parse_gpu_id_csv(configured_gpu_ids)
        if configured:
            return configured

        visible = self._parse_gpu_id_csv(
            os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        if visible:
            return visible

        worker_gpu = getattr(ResourceLogger, "_gpu_id", -1)
        if isinstance(worker_gpu, int) and worker_gpu >= 0:
            return [int(worker_gpu)]
        return [0]

    def _build_blender_worker_gpu_plan(self):
        gpu_ids = list(self.blender_gpu_ids)
        if len(gpu_ids) == 0:
            gpu_ids = [0]

        if self.blender_worker_policy == "serial":
            return [int(gpu_ids[0])]

        max_workers = int(self.blender_max_workers)
        if max_workers <= 0:
            max_workers = len(gpu_ids)
        max_workers = int(max(1, min(max_workers, len(gpu_ids))))
        if len(gpu_ids) <= 1:
            return [int(gpu_ids[0])]
        return [int(g) for g in gpu_ids[:max_workers]]

    def _gpu_for_parallel_render_task(self, task_index):
        if not self.blender_worker_gpu_plan:
            return int(getattr(ResourceLogger, "_gpu_id", 0))
        return int(
            self.blender_worker_gpu_plan[int(task_index) %
                                         len(self.blender_worker_gpu_plan)])

    def _build_blender_payload_for_ood(
        self,
        *,
        ood_id,
        position,
        fallback_scale,
        render_sensor_mode,
        eval,
        size_multiplier,
        loaded_ood_details_present,
        gpu_id_override=None,
    ):
        serializable_frames_with_seg = {
            sensor_type: len(frames)
            for sensor_type, frames in self.frames_with_seg.items()
        }
        details = self.ood_details[int(ood_id)]

        render_pose = None
        pose_template = details.get("render_pose_override", None)
        if pose_template is None:
            pose_template = details.get("pose", None)
        if pose_template is not None:
            render_pose = pose_template
        if render_pose is None:
            render_pose = position
        payload_gpu_id = (int(gpu_id_override) if gpu_id_override is not None
                          else int(getattr(ResourceLogger, "_gpu_id", 0)))
        blender_data = {
            "pose":
            copy.deepcopy(render_pose),
            "ood_id":
            int(ood_id),
            "lvis":
            details.get("lvis", "unknown"),
            "path":
            details.get("path"),
            "scale":
            float(
                details.get("effective_scale",
                            details.get("scale", fallback_scale))),
            "base_scale":
            float(details.get("base_scale", fallback_scale)),
            "target_ref_dim_m":
            details.get(
                "target_ref_dim_m",
                details.get("target_max_dim_m", None),
            ),
            "scale_policy":
            details.get("scale_policy", "auto"),
            "size_bucket":
            details.get("size_bucket", ""),
            "description":
            details.get("description", ""),
            "frame_id":
            None,
            "gpu_id":
            payload_gpu_id,
            "frames_with_seg":
            serializable_frames_with_seg,
            "frame_map":
            self.frame_map,
            "camera_data":
            self.camera_data,
            "file_name":
            self.file_name,
            "ood_details":
            copy.deepcopy(self.ood_details),
            "sensor":
            render_sensor_mode,
            "adjustment":
            False if loaded_ood_details_present else True,
            "eval":
            eval,
            "base_dir":
            self.base_dir,
            "size_multiplier":
            size_multiplier,
            "blender_require_gpu":
            self.blender_require_gpu,
            "no_shadow_cycles_samples":
            self.no_shadow_cycles_samples,
            "no_shadow_cycles_use_denoising":
            self.no_shadow_cycles_use_denoising,
            "enable_support_debug_exports":
            self.enable_shadow_debug_exports,
            "lidar_import_mode":
            self.lidar_import_mode,
        }
        blender_data.update(
            self._build_shadow_plane_payload(
                ood_id=int(ood_id),
                position=position,
            ))
        return blender_data

    def _run_parallel_blender_renders(self, render_tasks):
        if not render_tasks:
            return {}

        effective_workers = int(
            max(1, min(self.blender_effective_workers, len(render_tasks))))
        if effective_workers <= 1:
            for task_idx, task in enumerate(render_tasks):
                gpu_id = self._gpu_for_parallel_render_task(task_idx)
                task["blender_data"]["gpu_id"] = int(gpu_id)
                self.run_blender_render_subprocess(
                    ood_id=int(task["ood_id"]),
                    blender_data=task["blender_data"],
                )
            return {}

        print(
            f"\033[36m[BLENDER-PARALLEL] Launching {len(render_tasks)} OOD renders "
            f"with workers={effective_workers}, gpu_plan={self.blender_worker_gpu_plan}\033[0m"
        )
        self._update_run_progress(
            stage="blender_parallel",
            status="running",
            total_tasks=int(len(render_tasks)),
            workers=int(effective_workers),
            gpu_plan=[int(g) for g in self.blender_worker_gpu_plan],
        )

        errors = {}

        def _task_runner(task_idx, task):
            ood_id = int(task["ood_id"])
            gpu_id = self._gpu_for_parallel_render_task(task_idx)
            blender_data = task["blender_data"]
            blender_data["gpu_id"] = int(gpu_id)
            start_t = time.time()
            self.run_blender_render_subprocess(ood_id=ood_id,
                                               blender_data=blender_data)
            return ood_id, gpu_id, float(time.time() - start_t)

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(_task_runner, idx, task): int(task["ood_id"])
                for idx, task in enumerate(render_tasks)
            }
            for future in as_completed(futures):
                ood_id = futures[future]
                try:
                    done_ood_id, gpu_id, elapsed_s = future.result()
                    print(
                        f"\033[36m[BLENDER-PARALLEL] OOD {done_ood_id} rendered on GPU {gpu_id} "
                        f"in {elapsed_s:.2f}s\033[0m")
                except Exception as exc:
                    errors[int(ood_id)] = str(exc)
                    print(
                        f"\033[31m[BLENDER-PARALLEL] OOD {ood_id} failed: {exc}\033[0m"
                    )

        if errors:
            self._update_run_progress(
                stage="blender_parallel",
                status="failed",
                failed_oods=errors,
            )
            failed_list = ", ".join(str(k) for k in sorted(errors.keys()))
            raise RuntimeError(
                f"Parallel Blender render failed for OOD ids: {failed_list}")

        self._update_run_progress(
            stage="blender_parallel",
            status="completed",
            total_tasks=int(len(render_tasks)),
            workers=int(effective_workers),
        )
        return errors

    def _post_render_validate_with_resample(
        self,
        *,
        ood_id,
        base_scale,
        render_mode,
        eval,
        dist,
        render_sensor_mode,
        size_multiplier,
        loaded_ood_details_present,
    ):
        position = self.ood_details[int(ood_id)].get('position', None)
        if position is None:
            return False

        visibility_attempt = 0
        max_visibility_attempts = int(max(1, self.required_render_max_retries))
        needs_render = False

        while position is not None:
            if needs_render:
                blender_data = self._build_blender_payload_for_ood(
                    ood_id=int(ood_id),
                    position=position,
                    fallback_scale=float(base_scale),
                    render_sensor_mode=render_sensor_mode,
                    eval=eval,
                    size_multiplier=size_multiplier,
                    loaded_ood_details_present=bool(
                        loaded_ood_details_present),
                )
                self.run_blender_render_subprocess(
                    ood_id=int(ood_id),
                    blender_data=blender_data,
                )
                needs_render = False

            if (render_sensor_mode in {"camera", "both"}
                    and self.required_visible_cameras):
                visibility_check = self._validate_required_camera_render_visibility(
                    ood_id=int(ood_id),
                    eval=eval,
                )
                if not visibility_check["ok"]:
                    visibility_attempt += 1
                    print(
                        f"\033[33mRender visibility miss for OOD {ood_id}: "
                        f"missing={visibility_check['missing']} "
                        f"counts={visibility_check['counts']} "
                        f"frames={visibility_check['frames']} "
                        f"(attempt {visibility_attempt}/{max_visibility_attempts}).\033[0m"
                    )
                    if visibility_attempt >= max_visibility_attempts:
                        print(
                            f"\033[31mFailed to satisfy required camera render visibility "
                            f"for OOD {ood_id} after {max_visibility_attempts} attempts.\033[0m"
                        )
                        self.ood_details[int(ood_id)]['position'] = None
                        self.ood_details[int(ood_id)].pop('pose', None)
                        return False

                    self._delete_ood_render_artifacts(ood_id=int(ood_id),
                                                      eval=eval)
                    position, _frame_id = self.sample_ood_location(
                        scale=float(base_scale),
                        ood_id=int(ood_id),
                        sensor_type='lidar',
                        render_mode=render_mode,
                        eval=eval,
                        dist=dist,
                    )
                    if position is None:
                        self.ood_details[int(ood_id)]['position'] = None
                        self.ood_details[int(ood_id)].pop('pose', None)
                        return False
                    self.ood_details[int(ood_id)]['position'] = position
                    self.ood_details[int(ood_id)]['pose'] = position
                    needs_render = True
                    continue

            if (render_sensor_mode in {"camera", "both"}
                    and self.enable_shadow_quality_gate):
                shadow_check = self._validate_required_camera_shadow_quality(
                    ood_id=int(ood_id),
                    eval=eval,
                )
                if not shadow_check["ok"]:
                    reason_map = shadow_check.get("fail_reasons", {})
                    compact_reasons = {
                        cam: reasons
                        for cam, reasons in reason_map.items() if reasons
                    }
                    visibility_attempt += 1
                    print(
                        f"\033[33mShadow quality miss for OOD {ood_id}: "
                        f"missing={shadow_check['missing']} "
                        f"reasons={compact_reasons} "
                        f"(attempt {visibility_attempt}/{max_visibility_attempts}).\033[0m"
                    )
                    if visibility_attempt >= max_visibility_attempts:
                        print(
                            f"\033[31mFailed to satisfy required camera shadow quality "
                            f"for OOD {ood_id} after {max_visibility_attempts} attempts.\033[0m"
                        )
                        self.ood_details[int(ood_id)]['position'] = None
                        self.ood_details[int(ood_id)].pop('pose', None)
                        return False

                    self._delete_ood_render_artifacts(ood_id=int(ood_id),
                                                      eval=eval)
                    position, _frame_id = self.sample_ood_location(
                        scale=float(base_scale),
                        ood_id=int(ood_id),
                        sensor_type='lidar',
                        render_mode=render_mode,
                        eval=eval,
                        dist=dist,
                    )
                    if position is None:
                        self.ood_details[int(ood_id)]['position'] = None
                        self.ood_details[int(ood_id)].pop('pose', None)
                        return False
                    self.ood_details[int(ood_id)]['position'] = position
                    self.ood_details[int(ood_id)]['pose'] = position
                    needs_render = True
                    continue

            return True

        return False

    def _estimate_blender_workload(self, blender_data):
        """
        Estimate number of render calls for timeout scaling and progress reporting.
        Returns:
            dict with:
              - cycle_renders: number of cycles pass renders
              - total_render_calls: no-shadow + cycles render calls
        """
        frames_with_seg = blender_data.get("frames_with_seg", {}) or {}
        camera_data = blender_data.get("camera_data", {}) or {}
        sensor_mode = str(blender_data.get("sensor", "both")).lower()

        cycle_renders = 0
        total_render_calls = 0

        for sensor_type, frame_count_raw in frames_with_seg.items():
            try:
                frame_count = int(frame_count_raw)
            except Exception:
                frame_count = 0
            if frame_count <= 0:
                continue

            if sensor_type == "lidar" and sensor_mode != "lidar":
                continue

            if sensor_type == "camera":
                camera_entries = camera_data.get(sensor_type, [])
                usable = min(frame_count, len(camera_entries))
                for frame_idx in range(usable):
                    frame_entry = camera_entries[frame_idx]
                    camera_count = 1
                    if (isinstance(frame_entry, dict) and isinstance(
                            frame_entry.get("all_cameras"), dict)
                            and len(frame_entry["all_cameras"]) > 0):
                        camera_count = len(frame_entry["all_cameras"])
                    cycle_renders += int(camera_count)
                    total_render_calls += int(camera_count) * 2

                if frame_count > usable:
                    missing = frame_count - usable
                    cycle_renders += int(missing)
                    total_render_calls += int(missing) * 2
            else:
                cycle_renders += int(frame_count)
                total_render_calls += int(frame_count) * 2

        return {
            "cycle_renders": int(cycle_renders),
            "total_render_calls": int(total_render_calls),
        }

    def _read_blender_stdout_progress(self, stdout_log, last_offset):
        """
        Incrementally parse blender stdout log.
        Returns:
            new_offset, new_cycles_completed, new_render_calls_completed,
            last_render_info, last_sample_info
        """
        try:
            if not os.path.exists(stdout_log):
                return last_offset, 0, 0, None, None
            file_size = os.path.getsize(stdout_log)
            if file_size <= 0:
                return 0, 0, 0, None, None
            if last_offset > file_size:
                last_offset = 0
            with open(stdout_log, "rb") as f:
                f.seek(last_offset)
                chunk = f.read()
                new_offset = f.tell()
        except Exception:
            return last_offset, 0, 0, None, None

        if not chunk:
            return new_offset, 0, 0, None, None

        text = chunk.decode("utf-8", errors="replace")
        cycles_completed = 0
        render_calls_completed = 0
        last_render = None
        last_sample = None

        for line in text.splitlines():
            if self.blender_saved_pattern.search(line):
                render_calls_completed += 1
            render_match = self.blender_rendered_frame_pattern.search(line)
            if render_match:
                cycles_completed += 1
                last_render = {
                    "frame_idx": int(render_match.group(1)),
                    "sensor": str(render_match.group(3)).strip(),
                }
            sample_match = self.blender_sample_pattern.search(line)
            if sample_match:
                last_sample = {
                    "sample": int(sample_match.group(1)),
                    "sample_total": int(sample_match.group(2)),
                }

        return new_offset, cycles_completed, render_calls_completed, last_render, last_sample

    def _count_existing_render_files_for_ood(self, ood_id, eval=False):
        """Count existing Blender render files for an OOD id."""
        render_root = (
            f"{self.base_dir}/{self.file_name}/Evaluation/Renders/ood_{ood_id}"
            if eval else
            f"{self.base_dir}/{self.file_name}/Renders/ood_{ood_id}")
        if not os.path.isdir(render_root):
            return 0
        count = 0
        for _root, _dirs, files in os.walk(render_root):
            for name in files:
                if name.startswith("frame_00") or name.startswith(
                        "no_shadow_frame_00"):
                    count += 1
        return int(count)

    def _has_complete_existing_renders_for_ood(self,
                                               ood_id,
                                               blender_data,
                                               eval=False):
        """Return whether an OOD already has the full expected render set on disk."""
        existing_render_files = self._count_existing_render_files_for_ood(
            ood_id=ood_id, eval=eval)
        workload = self._estimate_blender_workload(blender_data)
        expected_render_files = int(workload.get("total_render_calls", 0))
        is_complete = expected_render_files > 0 and existing_render_files >= expected_render_files
        return {
            "is_complete": bool(is_complete),
            "existing_render_files": int(existing_render_files),
            "expected_render_files": int(expected_render_files),
        }

    def run_blender_render_subprocess(self, ood_id, blender_data):
        """
        Execute Blender subprocess for OOD object rendering.

        Args:
            ood_id: The OOD object ID
            blender_data: Dictionary containing all data to pass to Blender
            sensor_type: Type of sensor ('lidar', 'camera', 'both')

        Returns:
            updated_data: Dictionary with updated OOD details from Blender

        Raises:
            subprocess.CalledProcessError: If Blender process fails
            subprocess.TimeoutExpired: If Blender process times out
        """
        if not os.path.isfile(self.blender_path):
            raise FileNotFoundError(
                f"Blender executable not found at '{self.blender_path}'. "
                "Set a valid path with --blender_path or BLENDER_PATH.")

        with tempfile.NamedTemporaryFile(mode='w+',
                                         delete=False,
                                         suffix='.json') as temp_payload_file:
            json.dump(blender_data, temp_payload_file)
            payload_json_path = temp_payload_file.name

        with tempfile.NamedTemporaryFile(mode='w+',
                                         delete=False) as temp_json_file:
            json_output_path = temp_json_file.name

        logs_dir = os.path.join(self.blender_logs_dir, self.file_name)
        os.makedirs(logs_dir, exist_ok=True)

        stdout_log = os.path.join(logs_dir, f"blender_ood_{ood_id}_stdout.log")
        stderr_log = os.path.join(logs_dir, f"blender_ood_{ood_id}_stderr.log")
        workload = {"cycle_renders": 0, "total_render_calls": 0}
        cycle_renders = 0
        estimated_timeout_s = int(self.blender_timeout_s)
        timeout_s = int(self.blender_timeout_s)
        completed_cycles = 0
        completed_render_calls = 0
        last_render_info = None
        last_sample_info = None

        try:

            command = [
                self.blender_path,
                "--background",
                "--python",
                self.script_path,
                "--",
                "sample_ood_object_and_render",
                f"@{payload_json_path}",
                json_output_path,
            ]
            if self.blender_debug_all:
                command.insert(2, "--debug-all")

            env = os.environ.copy()
            env.pop('DISPLAY', None)
            env.pop('XAUTHORITY', None)
            env.pop('WAYLAND_DISPLAY', None)
            requested_gpu_id = blender_data.get("gpu_id", None)
            if requested_gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(requested_gpu_id)
                print(
                    f"[BLENDER] OOD {ood_id} subprocess CUDA_VISIBLE_DEVICES={requested_gpu_id}"
                )
            blender_cpu_threads = int(self.blender_cpu_threads)
            if blender_cpu_threads <= 0:
                blender_cpu_threads = int(self.cpu_thread_cap)
            if blender_cpu_threads > 0:
                env["BLENDER_CPU_THREADS"] = str(blender_cpu_threads)
                for var_name in CPU_THREAD_ENV_VARS:
                    if not env.get(var_name, "").strip():
                        env[var_name] = str(blender_cpu_threads)
                print(f"[CPU] Blender subprocess thread cap for OOD {ood_id}: "
                      f"BLENDER_CPU_THREADS={blender_cpu_threads}")
            if not env.get("BLENDER_ALLOW_GPU_OOM_CPU_FALLBACK", "").strip():
                env["BLENDER_ALLOW_GPU_OOM_CPU_FALLBACK"] = "1"
            if not env.get("BLENDER_FALLBACK_CYCLES_SAMPLES", "").strip():
                env["BLENDER_FALLBACK_CYCLES_SAMPLES"] = "24"

            workload = self._estimate_blender_workload(blender_data)
            cycle_renders = int(workload.get("cycle_renders", 0))
            total_render_calls = int(workload.get("total_render_calls", 0))
            no_shadow_samples = int(
                max(1, blender_data.get("no_shadow_cycles_samples", 24)))
            full_samples_env = os.environ.get("BLENDER_CYCLES_SAMPLES", "128")
            try:
                full_samples = int(full_samples_env)
            except Exception:
                full_samples = 128
            full_samples = int(max(1, full_samples))
            no_shadow_cost = float(
                np.clip(no_shadow_samples / float(full_samples), 0.05, 1.0))
            effective_cycle_units = float(cycle_renders) * (1.0 +
                                                            no_shadow_cost)
            estimated_timeout_s = int(self.blender_timeout_overhead_s +
                                      effective_cycle_units *
                                      self.blender_timeout_per_cycle_render_s *
                                      self.blender_timeout_safety_factor)
            timeout_s = max(self.blender_timeout_s, estimated_timeout_s)
            timeout_s = min(timeout_s, self.blender_timeout_max_s)
            heartbeat_s = int(self.blender_heartbeat_s)
            timeout_extend_count = 0
            print(
                f"[BLENDER] Starting OOD {ood_id} render. "
                f"stdout={stdout_log} stderr={stderr_log} timeout={timeout_s}s "
                f"(cycle_renders={cycle_renders}, total_calls={total_render_calls}, "
                f"safety_factor={self.blender_timeout_safety_factor:.2f})")
            self._update_run_progress(
                stage="blender_render",
                status="running",
                ood_id=int(ood_id),
                timeout_s=int(timeout_s),
                estimated_timeout_s=int(estimated_timeout_s),
                expected_cycle_renders=int(cycle_renders),
                expected_total_render_calls=int(
                    workload.get("total_render_calls", 0)),
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            )
            with open(stdout_log, 'wb') as out_f, open(stderr_log,
                                                       'wb') as err_f:
                proc = subprocess.Popen(
                    command,
                    stdout=out_f,
                    stderr=err_f,
                    env=env,
                )
                start_t = time.time()
                last_heartbeat = start_t
                stdout_offset = 0

                def ingest_progress():
                    nonlocal stdout_offset, completed_cycles, completed_render_calls
                    nonlocal last_render_info, last_sample_info
                    (
                        stdout_offset,
                        new_cycles,
                        new_render_calls,
                        parsed_last_render,
                        parsed_last_sample,
                    ) = self._read_blender_stdout_progress(
                        stdout_log, stdout_offset)
                    if new_cycles > 0:
                        completed_cycles += int(new_cycles)
                    if new_render_calls > 0:
                        completed_render_calls += int(new_render_calls)
                    if parsed_last_render is not None:
                        last_render_info = parsed_last_render
                    if parsed_last_sample is not None:
                        last_sample_info = parsed_last_sample

                while True:
                    return_code = proc.poll()
                    if return_code is not None:
                        if return_code != 0:
                            raise subprocess.CalledProcessError(
                                return_code, command)
                        break

                    now_t = time.time()
                    elapsed = now_t - start_t
                    if elapsed > timeout_s:

                        ingest_progress()
                        completion_ratio = 0.0
                        if total_render_calls > 0:
                            completion_ratio = completed_render_calls / float(
                                total_render_calls)
                        elif cycle_renders > 0:
                            completion_ratio = completed_cycles / float(
                                cycle_renders)
                        completion_ratio = float(
                            np.clip(completion_ratio, 0.0, 1.0))

                        if (completion_ratio >= 0.75
                                and timeout_s < self.blender_timeout_max_s):
                            extra_s = 0
                            if total_render_calls > 0 and completed_render_calls > 0:
                                avg_call_s = elapsed / float(
                                    max(1, completed_render_calls))
                                remaining_calls = max(
                                    0, total_render_calls -
                                    completed_render_calls)
                                extra_s = int(
                                    max(240,
                                        remaining_calls * avg_call_s * 1.5))
                            elif cycle_renders > 0 and completed_cycles > 0:
                                avg_cycle_s = elapsed / float(
                                    max(1, completed_cycles))
                                remaining_cycles = max(
                                    0, cycle_renders - completed_cycles)
                                extra_s = int(
                                    max(240,
                                        remaining_cycles * avg_cycle_s * 1.5))
                            else:
                                extra_s = 300
                            new_timeout = min(
                                self.blender_timeout_max_s,
                                int(max(timeout_s + 60, elapsed + extra_s)),
                            )
                            if new_timeout > timeout_s:
                                timeout_extend_count += 1
                                timeout_s = int(new_timeout)
                                print(
                                    f"[BLENDER] OOD {ood_id} timeout auto-extended to {timeout_s}s "
                                    f"(progress={completion_ratio * 100:.1f}%, extends={timeout_extend_count})."
                                )
                                self._update_run_progress(
                                    stage="blender_render",
                                    status="running",
                                    ood_id=int(ood_id),
                                    elapsed_s=int(elapsed),
                                    timeout_s=int(timeout_s),
                                    timeout_auto_extended=True,
                                    timeout_extend_count=int(
                                        timeout_extend_count),
                                    completed_total_render_calls=int(
                                        completed_render_calls),
                                    expected_total_render_calls=int(
                                        total_render_calls),
                                    completed_cycle_renders=int(
                                        completed_cycles),
                                    expected_cycle_renders=int(cycle_renders),
                                    stdout_log=stdout_log,
                                    stderr_log=stderr_log,
                                )
                                last_heartbeat = now_t
                                time.sleep(1.0)
                                continue

                        proc.kill()
                        raise subprocess.TimeoutExpired(command,
                                                        timeout=timeout_s)

                    if (now_t - last_heartbeat) >= heartbeat_s:
                        ingest_progress()

                        progress_parts = []
                        if total_render_calls > 0:
                            pct_calls = (100.0 * min(completed_render_calls,
                                                     total_render_calls) /
                                         float(total_render_calls))
                            progress_parts.append(
                                f"calls={completed_render_calls}/{total_render_calls} ({pct_calls:.1f}%)"
                            )
                        elif completed_render_calls > 0:
                            progress_parts.append(
                                f"calls={completed_render_calls}")
                        if cycle_renders > 0:
                            pct = 100.0 * min(
                                completed_cycles,
                                cycle_renders) / float(cycle_renders)
                            progress_parts.append(
                                f"cycles={completed_cycles}/{cycle_renders} ({pct:.1f}%)"
                            )
                        elif completed_cycles > 0:
                            progress_parts.append(f"cycles={completed_cycles}")
                        if last_render_info is not None:
                            progress_parts.append(
                                f"last={last_render_info['sensor']} frame_idx={last_render_info['frame_idx']}"
                            )
                        if last_sample_info is not None:
                            progress_parts.append(
                                f"sample={last_sample_info['sample']}/{last_sample_info['sample_total']}"
                            )
                        progress_msg = ""
                        if progress_parts:
                            progress_msg = " " + "; ".join(progress_parts)
                        print(f"[BLENDER] OOD {ood_id} still rendering "
                              f"({int(elapsed)}s elapsed).{progress_msg}")
                        progress_details = {
                            "stage":
                            "blender_render",
                            "status":
                            "running",
                            "ood_id":
                            int(ood_id),
                            "elapsed_s":
                            int(elapsed),
                            "timeout_s":
                            int(timeout_s),
                            "estimated_timeout_s":
                            int(estimated_timeout_s),
                            "expected_cycle_renders":
                            int(cycle_renders),
                            "expected_total_render_calls":
                            int(workload.get("total_render_calls", 0)),
                            "completed_total_render_calls":
                            int(completed_render_calls),
                            "completed_cycle_renders":
                            int(completed_cycles),
                            "stdout_log":
                            stdout_log,
                            "stderr_log":
                            stderr_log,
                        }
                        if last_render_info is not None:
                            progress_details[
                                "last_render_sensor"] = last_render_info[
                                    "sensor"]
                            progress_details["last_render_frame_idx"] = int(
                                last_render_info["frame_idx"])
                        if last_sample_info is not None:
                            progress_details["last_sample"] = int(
                                last_sample_info["sample"])
                            progress_details["last_sample_total"] = int(
                                last_sample_info["sample_total"])
                        self._update_run_progress(**progress_details)
                        last_heartbeat = now_t
                    time.sleep(1.0)

            if os.path.getsize(json_output_path) == 0:
                raise RuntimeError(
                    f"Blender returned empty JSON output for OOD {ood_id}. "
                    f"See logs: {stdout_log}, {stderr_log}")
            with open(json_output_path, 'r') as f:
                updated_data = json.load(f)

            print(
                f'updated data {updated_data["ood_details"]} for ood {ood_id}')
            with self._state_lock:
                self.master[self.file_name]['num_ood_objects'] = int(
                    self.num_objects)
                self.master[self.file_name]['ood_renders'] = updated_data[
                    'render_path']

            resource_metrics_list = updated_data.pop('resource_metrics', [])
            if resource_metrics_list:
                self._log_resource_metrics(ood_id, resource_metrics_list)

            updated_ood_details = updated_data['ood_details']
            with self._state_lock:
                for ood_id_key, updated_details in updated_ood_details.items():
                    ood_id_int = int(ood_id_key)
                    if ood_id_int in self.ood_details:

                        if 'position' in self.ood_details[ood_id_int]:
                            updated_details['position'] = self.ood_details[
                                ood_id_int]['position']
                        if 'file_identifier' in self.ood_details[ood_id_int]:
                            updated_details[
                                'file_identifier'] = self.ood_details[
                                    ood_id_int]['file_identifier']
                    self.ood_details[ood_id_int] = updated_details
            print("Blender operation successful. State updated.")
            self._update_run_progress(
                stage="blender_render",
                status="completed",
                ood_id=int(ood_id),
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            )

            return updated_data

        except subprocess.CalledProcessError as e:
            print('CalledProcessError')
            print(f"Blender subprocess failed with exit code {e.returncode}:",
                  file=sys.stderr)
            if e.stdout:
                print(f"Stdout:\n{e.stdout.decode('utf-8', errors='replace')}",
                      file=sys.stderr)
            if e.stderr:
                print(f"Stderr:\n{e.stderr.decode('utf-8', errors='replace')}",
                      file=sys.stderr)
            error_info = e.stderr if e.stderr else b"Stderr not captured"
            self.log_failed_object(ood_id, error_info)
            self._update_run_progress(
                stage="blender_render",
                status="failed",
                ood_id=int(ood_id),
                error=f"exit_code={e.returncode}",
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            )
            raise e

        except subprocess.TimeoutExpired as e:
            print('TimeoutExpired')
            print(f"Blender subprocess timed out:", file=sys.stderr)
            if e.stdout:
                print(f"Stdout:\n{e.stdout.decode('utf-8', errors='replace')}",
                      file=sys.stderr)
            if e.stderr:
                print(f"Stderr:\n{e.stderr.decode('utf-8', errors='replace')}",
                      file=sys.stderr)
            total_render_calls = int(workload.get("total_render_calls", 0))
            completion_ratio = 0.0
            if total_render_calls > 0:
                completion_ratio = completed_render_calls / float(
                    total_render_calls)
            elif cycle_renders > 0:
                completion_ratio = completed_cycles / float(cycle_renders)
            completion_ratio = float(np.clip(completion_ratio, 0.0, 1.0))

            render_root = (
                f"{self.base_dir}/{self.file_name}/Evaluation/Renders" if bool(
                    blender_data.get("eval", False)) else
                f"{self.base_dir}/{self.file_name}/Renders")
            ood_render_root = os.path.join(render_root, f"ood_{ood_id}")
            partial_render_files = 0
            if os.path.isdir(ood_render_root):
                for root, _dirs, files in os.walk(ood_render_root):
                    for name in files:
                        if name.startswith("frame_00") or name.startswith(
                                "no_shadow_frame_00"):
                            partial_render_files += 1

            can_salvage = (self.blender_allow_partial_timeout
                           and partial_render_files > 0 and completion_ratio >=
                           self.blender_partial_timeout_min_completion)
            if can_salvage:
                print(
                    f"[BLENDER] OOD {ood_id} timeout salvaged with partial renders "
                    f"(completion={completion_ratio * 100:.1f}%, files={partial_render_files})."
                )
                self.master[self.file_name]['num_ood_objects'] = int(
                    self.num_objects)
                self.master[self.file_name]['ood_renders'] = f"{render_root}/"
                self._update_run_progress(
                    stage="blender_render",
                    status="partial_timeout_salvaged",
                    ood_id=int(ood_id),
                    error=f"timeout={e.timeout}",
                    completed_total_render_calls=int(completed_render_calls),
                    expected_total_render_calls=int(total_render_calls),
                    completed_cycle_renders=int(completed_cycles),
                    expected_cycle_renders=int(cycle_renders),
                    partial_render_files=int(partial_render_files),
                    completion_ratio=float(completion_ratio),
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                )
                return {
                    "ood_details": self.ood_details,
                    "render_path": f"{render_root}/",
                    "resource_metrics": [],
                }

            error_info = e.stderr if e.stderr else b"Stderr not captured on timeout"
            self.log_failed_object(ood_id, error_info)
            self._update_run_progress(
                stage="blender_render",
                status="failed",
                ood_id=int(ood_id),
                error=f"timeout={e.timeout}",
                completed_total_render_calls=int(completed_render_calls),
                expected_total_render_calls=int(total_render_calls),
                completed_cycle_renders=int(completed_cycles),
                expected_cycle_renders=int(cycle_renders),
                partial_render_files=int(partial_render_files),
                completion_ratio=float(completion_ratio),
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            )
            raise e

        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)
            self.log_failed_object(ood_id, str(e))
            self._update_run_progress(
                stage="blender_render",
                status="failed",
                ood_id=int(ood_id),
                error=str(e),
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            )
            raise e

        finally:

            if os.path.exists(json_output_path):
                os.remove(json_output_path)
            if os.path.exists(payload_json_path):
                os.remove(payload_json_path)

    def _log_resource_metrics(self, ood_id, resource_metrics_list):
        """Helper method to log resource metrics to wandb."""
        table_data = []
        columns = ["Frame", "Compute Delta (%)", "Memory Delta (MB)"]

        for frame_idx, frame_metrics in enumerate(resource_metrics_list):
            if not frame_metrics:
                continue

            before_metrics = frame_metrics.get('before', {})
            after_metrics = frame_metrics.get('after', {})

            if before_metrics and after_metrics:
                compute_delta = after_metrics.get(
                    f'GPU_{ResourceLogger._gpu_id}_Compute(%)',
                    0) - before_metrics.get(
                        f'GPU_{ResourceLogger._gpu_id}_Compute(%)', 0)
                mem_delta = after_metrics.get(
                    f'GPU_{ResourceLogger._gpu_id}_Memory_Used(MB)',
                    0) - before_metrics.get(
                        f'GPU_{ResourceLogger._gpu_id}_Memory_Used(MB)', 0)
                table_data.append([frame_idx + 1, compute_delta, mem_delta])

        if table_data and ResourceLogger.wandb_enabled:
            wandb.log({
                f"ood_{ood_id}/resource_metrics_per_frame":
                wandb.Table(columns=columns, data=table_data)
            })

    def _update_sam_mask_log(self, file_name, new_entries):
        """
        Update SAM mask log JSON file with nested dictionary structure.

        Structure:
        {
            "file_name_1": [
                {"ood_id": ..., "frame_id": ..., "best_class": ..., "best_confidence": ..., "best_mask_path": ...},
                ...
            ],
            "file_name_2": [...],
            ...
        }

        Args:
            file_name (str): The primary key (file name) for grouping entries
            new_entries (list): List of entry dictionaries to add/update

        Returns:
            None
        """
        log_path = "./sam_best_mask_log.json"

        if not new_entries:
            return

        try:

            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        print(
                            f"Warning: Log file format is not a dictionary. Reinitializing."
                        )
                        data = {}
            else:
                data = {}

            data[file_name] = new_entries

            with open(log_path, "w") as f:
                json.dump(data, f, indent=4)

            print(
                f"SAM mask log updated for '{file_name}' with {len(new_entries)} entries."
            )

        except json.JSONDecodeError as e:
            print(f"Error: Corrupted JSON in {log_path}. {e}")
            print(f"Reinitializing log file with current entries.")
            data = {file_name: new_entries}
            with open(log_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error updating SAM mask log: {e}")

    def check_collision(self, point, scale=2):
        """Method to check whether point collides with any existing bounding box including previously found
        OOD locations i n BEV.
        Arguments:
            point (x, y): tuple of float representing center x, center y
            scale (float): global scale of the object to be placed
        Returns:
            bool: True if no collision with any box or any other OOD location is detected. False else."""
        scale = min(self.max_object_scale, scale * 100)
        scale = max(self.min_object_scale, scale)

        corner_1 = [point[0] - scale / 2, point[1] - scale / 2]
        corner_2 = [point[0] - scale / 2, point[1] + scale / 2]
        corner_3 = [point[0] + scale / 2, point[1] - scale / 2]
        corner_4 = [point[0] + scale / 2, point[1] + scale / 2]
        corners = np.array([corner_1, corner_2, corner_3, corner_4])
        box = [point[0], point[1], scale, scale]
        for bev_box in self.bev_map:
            if check_collision(box, bev_box):
                return False
        for ood_bev in self.bev_ood:
            if check_collision(box, ood_bev):
                return False
        return True
        print("\033[101mPrint after return\033[0m]")
        any_collision = box_utils.is_within_box_2d(corners,
                                                   np.array(self.bev_map))
        if tf.reduce_any(any_collision):
            return False
        if len(self.ood_locations) > 0:
            ood_points = np.array(
                [point + [1, 1, 0] for point in self.bev_ood])
            ood_collision = box_utils.is_within_box_2d(corners, ood_points)
            if tf.reduce_any(ood_collision):
                return False
        return True

    def check_is_within_map(self, location):
        """Method to check whether sampled location is still within boundaries of map.
        Arguments:
            location (x: float, y: float): sample point
        Returns:
            bool: True if point is within map. False else."""
        x_coordinates = [
            point[0] for point in self.points_for_terrain_estimation
        ]
        y_coordinates = [
            point[1] for point in self.points_for_terrain_estimation
        ]
        x_max = max(x_coordinates)
        x_min = min(x_coordinates)
        y_max = max(y_coordinates)
        y_min = min(y_coordinates)
        x_location = location[0]
        y_location = location[1]
        if x_min < x_location < x_max and y_min < y_location < y_max:
            return True
        return False

    def check_render_on_road(self, pose, frame_id, scale=1):
        """Creates a blender scene and place camera at location. Place plane at location of point.
        Render the scene. Check whether plane is within road label
        Args: point (x, y, z), frame_id (int)
        Returns (bool): True if final render is on road label. False else"""
        focal_length = self.camera_data[frame_id]["focal_length"]
        camera_pose = self.camera_data[frame_id]["camera_pose"]

        test_render_scene = "sample_render_plane.png"
        clear_scene_and_import_object(None, pose, 1)
        semantic_label = self.semantic_labels[frame_id]
        clear_scene()
        create_plane(pose, scale)
        set_camera_from_matrix(camera_pose, focal_length)
        render_scene(test_render_scene)
        test_image = Image.open(test_render_scene)
        test_array = np.array(test_image)
        non_transparent = test_array[:, :, 3] != 0
        if any(semantic_label[non_transparent]):

            if np.all(
                    np.isin(semantic_label[non_transparent],
                            [20, 21, 22, 26])):
                print("\033[42mObject can be rendered on road.\033[0m")
                return True
        print("\033[43mObject cannot be rendered on road.\033[0m")
        return False

    def save_raw_images(self):
        print("\033[32mSAMPLING: Saving raw images.\033[0m")
        """Method to save all the raw imaged of the scene into a folder."""
        write_raw_images = bool(getattr(self, "enable_raw_image_export", True))

        def load_model(ckpt_path, device):
            """Load SDDNet model from checkpoint"""
            model = SDDNet(backbone='efficientnet-b3',
                           proj_planes=16,
                           pred_planes=32,
                           use_pretrained=True,
                           fix_backbone=False,
                           has_se=False,
                           dropout_2d=0,
                           normalize=True,
                           mu_init=0.4,
                           reweight_mode='manual')

            if isinstance(device, torch.device):
                device_str = str(device)
            else:
                device_str = device

            checkpoint = torch.load(ckpt_path, map_location=device_str)
            model.load_state_dict(checkpoint['model'])

            model.to(device)
            model.eval()
            return model

        def preprocess_image(image_path, target_size=512):
            """Preprocess image for SDDNet input"""

            image = Image.open(image_path).convert('RGB')
            original_size = image.size

            transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
            ])

            image_tensor = transform(image).unsqueeze(0)
            return image_tensor, original_size, image

        def filter_road_shadows(selected_regions,
                                shadow_colors,
                                sensor_type,
                                frame_id,
                                radius=10):
            projection_array = self.road_projections_lidar[frame_id]
            print(projection_array)
            filtered_shadow_colors = []
            for i, (x1, y1, x2, y2) in enumerate(selected_regions):
                if np.array_equal(shadow_colors[i], np.empty(3)):
                    continue

                for j in range(20):
                    random_pixel = (self.seed.integers(x1, x2),
                                    self.seed.integers(y1, y2))
                    print(f'Checking pixel {random_pixel} in patch {i+1}')
                    diffs = projection_array - np.array(random_pixel)
                    dists = np.linalg.norm(diffs, axis=1)
                    print(dists)
                    if np.any(dists <= radius):

                        filtered_shadow_colors.append(shadow_colors[i])
                        print('Found')
                        break
                if j == 19:
                    print(f'No valid pixel found in patch {i+1}')

            return filtered_shadow_colors

        def detect_shadow_two_pass(image_path, ckpt_path, save_dir,
                                   sensor_type, frame_id):
            """Two-pass shadow detection with proper overlay visualization"""
            ensure_torch_imported(required=True,
                                  context="detect_shadow_two_pass")
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.basename(image_path).split('.')[0]

            patch_save_dir = os.path.join(save_dir, 'patch_results')
            os.makedirs(patch_save_dir, exist_ok=True)

            model = load_model(ckpt_path, device)
            original_image = Image.open(image_path).convert('RGB')
            original_size = original_image.size
            original_np = np.array(original_image)

            print("Running first pass on full image...")
            start_time = time.time()

            input_tensor, _, _ = preprocess_image(image_path)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                output = model(input_tensor)
            first_pass_time = time.time() - start_time
            print(f"First pass completed in {first_pass_time:.2f}s")

            first_pass_mask = torch.sigmoid(output['logit']).cpu()
            first_pass_mask = transforms.Resize(original_size[::-1])(
                first_pass_mask[0]).squeeze().numpy()

            plt.imsave(f'{save_dir}/{base_name}_first_pass.png',
                       first_pass_mask,
                       cmap='jet',
                       vmin=0,
                       vmax=1)

            print("Identifying high-confidence regions for refinement...")
            high_conf_mask = (first_pass_mask > 0.75).astype(np.uint8)

            labeled = label(high_conf_mask)
            regions = regionprops(labeled)

            min_area = 500
            selected_regions = []
            for region in regions:
                if region.area >= min_area:
                    min_row, min_col, max_row, max_col = region.bbox
                    height = max_row - min_row
                    width = max_col - min_col

                    pad_h = int(height * 0.1)
                    pad_w = int(width * 0.1)

                    y1 = max(0, min_row - pad_h)
                    y2 = min(original_size[1], max_row + pad_h)
                    x1 = max(0, min_col - pad_w)
                    x2 = min(original_size[0], max_col + pad_w)

                    selected_regions.append((x1, y1, x2, y2))

            if len(selected_regions) > 10:
                areas = [(r[2] - r[0]) * (r[3] - r[1])
                         for r in selected_regions]
                selected_regions = [
                    r for _, r in sorted(zip(areas, selected_regions),
                                         reverse=True)
                ][:10]

            print(
                f"Refining {len(selected_regions)} regions in second pass...")
            final_mask = np.zeros(original_size[::-1], dtype=np.float32)
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])

            bbox_image = original_image.copy()
            draw = ImageDraw.Draw(bbox_image)

            shadow_colors = []

            for i, (x1, y1, x2, y2) in enumerate(selected_regions):

                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1), f"R{i+1}", fill="red")

                patch = original_image.crop((x1, y1, x2, y2))
                patch_width, patch_height = patch.size
                patch_np = np.array(patch)

                patch.save(
                    os.path.join(patch_save_dir, f'patch_{i+1}_original.jpg'))

                patch_tensor = transform(patch).unsqueeze(0).to(device)
                with torch.no_grad():
                    patch_output = model(patch_tensor)

                patch_mask = torch.sigmoid(
                    patch_output['logit']).squeeze().cpu().numpy()

                plt.imsave(os.path.join(patch_save_dir,
                                        f'patch_{i+1}_prob_mask.png'),
                           patch_mask,
                           cmap='jet',
                           vmin=0,
                           vmax=1)

                patch_mask_resized = np.array(
                    Image.fromarray(patch_mask).resize(
                        (patch_width, patch_height), Image.BILINEAR))

                high_conf_mask = (patch_mask_resized > 0.9).astype(bool)

                binary_mask = high_conf_mask.astype(np.uint8) * 255
                Image.fromarray(binary_mask).save(
                    os.path.join(patch_save_dir,
                                 f'patch_{i+1}_binary_mask.png'))

                final_mask[y1:y2,
                           x1:x2] = np.maximum(final_mask[y1:y2, x1:x2],
                                               high_conf_mask.astype(float))

                overlay = patch_np.copy()

                overlay[high_conf_mask] = [255, 0, 0]

                overlay_img = Image.blend(Image.fromarray(patch_np),
                                          Image.fromarray(overlay), 0.4)
                overlay_img.save(
                    os.path.join(patch_save_dir, f'patch_{i+1}_overlay.png'))

                if np.any(high_conf_mask):
                    shadow_pixels = patch_np[high_conf_mask]
                    patch_avg_color = np.mean(shadow_pixels, axis=0)
                    print(
                        f"Patch {i+1} ({patch_width}x{patch_height}) - Avg shadow color: {patch_avg_color}"
                    )
                    shadow_colors.append(patch_avg_color)
                else:
                    patch_avg_color = np.empty(3)
                    print(
                        f"Patch {i+1} ({patch_width}x{patch_height}) - No shadow pixels"
                    )
                    shadow_colors.append(patch_avg_color)

            if sensor_type == 'lidar':
                filtered_shadow_colors = filter_road_shadows(
                    selected_regions, shadow_colors, sensor_type, frame_id)

            if len(filtered_shadow_colors) != 0:
                print('final shadow', len(filtered_shadow_colors))
                shadow_color = np.mean(np.array(filtered_shadow_colors),
                                       axis=0)
                self.shadow_color[sensor_type][frame_id] = shadow_color
            else:
                self.shadow_color[sensor_type][frame_id] = np.empty(3)

            binary_mask_np = (final_mask > 0.5).astype(np.uint8) * 255

            bbox_image.save(f'{save_dir}/{base_name}_regions.jpg')
            Image.fromarray(original_np).save(
                f'{save_dir}/{base_name}_input.jpg')
            Image.fromarray(binary_mask_np).save(
                f'{save_dir}/{base_name}_final_mask.png')

            fig, axes = plt.subplots(len(selected_regions),
                                     4,
                                     figsize=(16, len(selected_regions) * 4))
            if len(selected_regions) == 1:
                axes = [axes]

            for i in range(len(selected_regions)):

                patch_img = plt.imread(
                    os.path.join(patch_save_dir, f'patch_{i+1}_original.jpg'))
                axes[i][0].imshow(patch_img)
                axes[i][0].set_title(f'Patch {i+1}')
                axes[i][0].axis('off')

                prob_img = plt.imread(
                    os.path.join(patch_save_dir, f'patch_{i+1}_prob_mask.png'))
                axes[i][1].imshow(prob_img)
                axes[i][1].set_title('Probability')
                axes[i][1].axis('off')

                bin_img = plt.imread(
                    os.path.join(patch_save_dir,
                                 f'patch_{i+1}_binary_mask.png'))
                axes[i][2].imshow(bin_img, cmap='gray')
                axes[i][2].set_title('>90% Confidence')
                axes[i][2].axis('off')

                overlay_img = plt.imread(
                    os.path.join(patch_save_dir, f'patch_{i+1}_overlay.png'))
                axes[i][3].imshow(overlay_img)
                axes[i][3].set_title('Overlay')
                axes[i][3].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{base_name}_patch_grid.jpg'),
                        dpi=200)
            plt.close()

            fig, axs = plt.subplots(2, 2, figsize=(18, 12))

            axs[0, 0].imshow(original_np)
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')

            axs[0, 1].imshow(plt.imread(f'{save_dir}/{base_name}_regions.jpg'))
            axs[0, 1].set_title(f'{len(selected_regions)} Refined Regions')
            axs[0, 1].axis('off')

            axs[1, 0].imshow(binary_mask_np, cmap='gray')
            axs[1, 0].set_title('Final Shadow Mask')
            axs[1, 0].axis('off')

            axs[1, 1].imshow(first_pass_mask, cmap='jet', vmin=0, vmax=1)
            axs[1, 1].set_title('First Pass Probability')
            axs[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(f'{save_dir}/{base_name}_final_results.jpg',
                        dpi=300,
                        bbox_inches='tight')
            plt.close()

            total_time = time.time() - start_time
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Results saved to: {save_dir}/")
            print(f"Patch results saved to: {patch_save_dir}/")

            return final_mask, selected_regions

        for sensor_type, frames in self.frames_with_seg.items():
            if len(frames) == 0:
                continue
            image_dir = f"{self.base_dir}/{self.file_name}/Raw_Images/"
            image_dir_type = f"{image_dir}{sensor_type}"
            if write_raw_images and not os.path.exists(image_dir_type):
                os.makedirs(image_dir_type, exist_ok=True)
            if sensor_type not in self.raw_images:
                if sensor_type == 'camera':
                    self.raw_images[sensor_type] = {
                        camera_name: []
                        for camera_name in self.camera_names
                    }
                else:
                    self.raw_images[sensor_type] = []
            for frame_id, frame in enumerate(frames):
                frame_token = self.frame_map[sensor_type][frame_id]
                if sensor_type == 'camera':
                    for img in frame.images:
                        camera_name = camera_id_to_name(img.name)
                        if camera_name not in self.raw_images[sensor_type]:
                            self.raw_images[sensor_type][camera_name] = []
                        image_np = tf.io.decode_jpeg(img.image).numpy()
                        img_arr = np.array(image_np)
                        self.raw_images[sensor_type][camera_name].append(
                            img_arr)
                        if write_raw_images:
                            camera_dir = f"{image_dir_type}/{camera_name}"
                            os.makedirs(camera_dir, exist_ok=True)
                            image_transfer = f"{camera_dir}/frame_00{frame_token}.png"
                            if not os.path.exists(image_transfer):
                                Image.fromarray(img_arr).save(image_transfer)
                    continue

                image = frame.images[0].image
                image_np = tf.io.decode_jpeg(image).numpy()
                img_arr = np.array(image_np)
                if write_raw_images:
                    im = Image.fromarray(img_arr)
                    image_transfer = f"{image_dir_type}/frame_00{frame_token}.png"
                    if not os.path.exists(image_transfer):
                        im.save(image_transfer)
                self.raw_images[sensor_type].append(img_arr)

                if sensor_type not in self.shadow_color.keys():
                    self.shadow_color[sensor_type] = {}
                if self.camera_data[sensor_type][frame_id][
                        "time_of_day"] == "Night":
                    self.shadow_color[sensor_type][frame_id] = np.empty(3)
                    continue
                if not write_raw_images:
                    continue
                IMAGE_PATH = f"{image_dir_type}/frame_00{frame_token}.jpeg"
                CKPT_PATH = "SDDNet/ckpt/ep_019.ckpt"
                OUTPUT_DIR = f"{self.base_dir}/{self.file_name}/shadow_results_with_two_pass/frame_00{frame_token}"

        self.master[self.file_name]["raw_images"] = image_dir

    def compute_seg_masks(self):
        for sensor_type, frames in self.frames_with_seg.items():
            if len(frames) == 0:
                continue
            save_dir = f'{self.base_dir}/{self.file_name}/Segmentation_Masks'
            save_dir_type = f"{save_dir}/{sensor_type}"
            if not os.path.exists(save_dir_type):
                os.makedirs(save_dir_type, exist_ok=True)
            if sensor_type == 'camera':
                for frame_id, frame in enumerate(frames):
                    semantic_label = self.semantic_labels[frame_id]
                    semantic_label = np.reshape(semantic_label,
                                                np.shape(semantic_label)[0:2])

                    semantic_label_uint8 = semantic_label.astype(np.uint8)
                    semantic_img = Image.fromarray(semantic_label_uint8,
                                                   mode='L')
                    semantic_img.save(
                        f"{save_dir_type}/frame_00{self.frame_map[sensor_type][frame_id]}_sem_label.png"
                    )
            else:
                mmseg_python_executable = os.environ.get(
                    "WAYPOP_MMSEG_PYTHON", "").strip()
                mmseg_script_path = os.environ.get("WAYPOP_MMSEG_SCRIPT",
                                                   "").strip()
                mmseg_config_path = os.environ.get("WAYPOP_MMSEG_CONFIG",
                                                   "").strip()
                mmseg_checkpoint_path = os.environ.get(
                    "WAYPOP_MMSEG_CHECKPOINT", "").strip()
                if not all([
                        mmseg_python_executable, mmseg_script_path,
                        mmseg_config_path, mmseg_checkpoint_path
                ]):
                    raise RuntimeError(
                        "Segmentation mask generation requires "
                        "WAYPOP_MMSEG_PYTHON, WAYPOP_MMSEG_SCRIPT, "
                        "WAYPOP_MMSEG_CONFIG, and WAYPOP_MMSEG_CHECKPOINT.")

                raw_files_path = os.path.abspath(
                    f"{self.base_dir}/{self.file_name}/Raw_Images/{sensor_type}"
                )

                command = [
                    mmseg_python_executable, mmseg_script_path,
                    mmseg_config_path,
                    mmseg_checkpoint_path,
                    '--inference-only', '--show-dir',
                    os.path.join(
                        os.getcwd(),
                        f"{self.base_dir}/{self.file_name}/Segmentation_Masks/{sensor_type}"
                    ), '--cfg-options',
                    f'test_dataloader.dataset.data_root={raw_files_path}',
                    'test_dataloader.dataset.data_prefix.img_path=""',
                    'test_dataloader.dataset.img_suffix=".png"'
                ]

                print(
                    f"Running Segmentation mask prediction for sensor {sensor_type}..."
                )
                try:
                    result = subprocess.run(command,
                                            check=True,
                                            capture_output=True,
                                            text=True,
                                            env=os.environ.copy())
                    print("Segmentation mask prediction successful.")
                    print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(
                        f"Segmentation mask prediction failed with exit code {e.returncode}:"
                    )
                    print(f"Stderr: {e.stderr}")
                    print(f"Stdout: {e.stdout}")
                finally:
                    sys.stdout = sys.__stdout__

    def create_range_map(self):
        print("\033[36mSAMPLING: Saving range images\033[0m")
        """Method to load and process Lidar range images into default image dimensions (1280, 1920).
        After projection range image represents the distance from the camera to an object for every pixel.
        0 if the pixel is not labeled, does not mean distance is 0.
        Append to self.range_images (list of range images (array of shape [1280, 1920] of range (float)))"""

        for frame_id, frame in enumerate(self.frames_with_seg):
            (range_images, camera_projections, segmentation_labels,
             range_image_top_pose
             ) = frame_utils.parse_range_image_and_camera_projection(frame)

            height = 1280
            width = 1920
            camera_image_size = [height, width]

            inclination = None
            for calibration in frame.context.laser_calibrations:
                if calibration.name == 1:
                    extrinsic = calibration.extrinsic
                    extrinsic_matrix = np.array(extrinsic.transform).reshape(
                        (1, 4, 4))
                    inclination = np.array(calibration.beam_inclinations)
                    inclination = inclination.reshape(
                        (1, np.shape(inclination)[0]))

            camera_image = frame.images[0]

            range_images = np.array(range_images[1])
            range_images = range_images[0]
            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_images.data),
                range_images.shape.dims)
            range_image_tensor = range_image_tensor[:, :, 0:1]
            range_image_shape = 1, range_image_tensor.shape[
                0], range_image_tensor.shape[1]
            range_image_tensor = tf.reshape(range_image_tensor,
                                            range_image_shape)
            range_image_polar = range_image_utils.compute_range_image_polar(
                range_image_tensor, extrinsic_matrix, inclination)
            range_image_cartesian = range_image_utils.compute_range_image_cartesian(
                range_image_polar, extrinsic_matrix)

            camera_name = camera_image.name

            calibration = next(cc for cc in frame.context.camera_calibrations
                               if cc.name == camera_image.name)
            extrinsic = tf.reshape(
                tf.constant(list(calibration.extrinsic.transform),
                            dtype=tf.float32), [1, 4, 4])

            camera_projection = camera_projections
            camera_projection = np.array(camera_projection[1])
            camera_projection = camera_projection[0]
            camera_projection_tensor = tf.reshape(
                tf.convert_to_tensor(camera_projection.data),
                camera_projection.shape.dims)
            cam_proj_shape = camera_projection_tensor.shape
            camera_projection_shape = 1, cam_proj_shape[0], cam_proj_shape[
                1], cam_proj_shape[2]
            camera_projection_tensor = tf.reshape(camera_projection_tensor,
                                                  camera_projection_shape)

            camera_depth_image = range_image_utils.build_camera_depth_image(
                range_image_cartesian, extrinsic, camera_projection_tensor,
                camera_image_size, camera_name)
            camera_depth_image = camera_depth_image[0]
            self.range_images.append(camera_depth_image)

    def find_occlusion_map(self, eval=False):
        print("\033[36mSAMPLING: Finding occlusion mask\033[0m")
        """For all objects create a mask of all pixels that occlude the object"""
        panoptic_divisor = int(
            max(1, getattr(self, "panoptic_label_divisor", 1000)))

        def calculate_mask_continuity(mask_array):
            """
            Calculate continuity metrics for a binary mask.
            Returns a continuity score based on:
            1. Number of connected components (fewer is better)
            2. Average component size (larger is better)
            3. Solidity (ratio of mask area to convex hull area)

            Args:
                mask_array: Binary mask array (0-255 or bool)

            Returns:
                continuity_score (0-1): Higher score = better continuity
                metrics (dict): Detailed metrics
            """
            from scipy import ndimage
            from scipy.spatial import ConvexHull

            binary_mask = (mask_array > 128).astype(np.uint8)

            labeled_array, num_components = ndimage.label(binary_mask)

            component_sizes = ndimage.sum(binary_mask, labeled_array,
                                          range(num_components + 1))

            total_area = np.sum(binary_mask)

            if total_area == 0:
                return 0.0, {"num_components": 0, "total_area": 0}

            avg_component_size = total_area / num_components if num_components > 0 else 0

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0.0, {
                    "num_components": num_components,
                    "total_area": total_area
                }

            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)

            if len(largest_contour) >= 3:
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = contour_area / hull_area if hull_area > 0 else 0
            else:
                solidity = 0.0

            component_penalty = 1.0 / (1.0 + num_components - 1)

            size_ratio = min(1.0, avg_component_size / (total_area + 1))

            continuity_score = (0.4 * component_penalty + 0.3 * solidity +
                                0.3 * size_ratio)

            metrics = {
                "num_components": num_components,
                "total_area": total_area,
                "avg_component_size": avg_component_size,
                "solidity": solidity,
                "component_penalty": component_penalty,
                "continuity_score": continuity_score
            }

            return continuity_score, metrics

        if eval:
            ultralytics_dir = f'{self.base_dir}/{self.file_name}/Evaluation/Ultralytics_Segmented_Masks'
            sam_output = f'{self.base_dir}/{self.file_name}/Evaluation/Sam_Masks'
            occlusion_mask_path = f'{self.base_dir}/{self.file_name}/Evaluation/Occlusion_Masks'

        else:
            ultralytics_dir = f'{self.base_dir}/{self.file_name}/Ultralytics_Segmented_Masks'
            sam_output = f'{self.base_dir}/{self.file_name}/Sam_Masks'
            occlusion_mask_path = f'{self.base_dir}/{self.file_name}/Occlusion_Masks'
        if os.path.exists(ultralytics_dir):
            shutil.rmtree(ultralytics_dir)
        if os.path.exists(sam_output):
            shutil.rmtree(sam_output)
        if os.path.exists(occlusion_mask_path):
            shutil.rmtree(occlusion_mask_path)
        os.makedirs(ultralytics_dir, exist_ok=True)
        os.makedirs(sam_output, exist_ok=True)
        os.makedirs(occlusion_mask_path, exist_ok=True)

        for sensor_type, frames in self.frames_with_seg.items():
            print(sensor_type)
            if len(frames) == 0:
                continue

            if sensor_type == 'lidar':
                for ood_id in self.ood_details.keys():
                    if ood_id not in self.occlusion_masks:
                        self.occlusion_masks[ood_id] = {}
                    self.occlusion_masks[ood_id][sensor_type] = [
                        np.zeros((1280, 1920), dtype=np.uint8)
                        for _ in range(len(frames))
                    ]
                print(
                    "\033[33mSkipping lidar image occlusion mask generation.\033[0m"
                )
                continue
            if sensor_type == 'camera' and isinstance(
                    self.raw_images.get(sensor_type), dict):

                for ood_id in self.ood_details.keys():
                    if ood_id not in self.occlusion_masks:
                        self.occlusion_masks[ood_id] = {}
                    if sensor_type not in self.occlusion_masks[ood_id]:
                        self.occlusion_masks[ood_id][sensor_type] = {}
                    for camera_name, frame_list in self.raw_images[
                            sensor_type].items():
                        zero_masks = [
                            np.zeros((1280, 1920), dtype=np.uint8)
                            for _ in range(len(frame_list))
                        ]
                        self.occlusion_masks[ood_id][sensor_type][
                            camera_name] = zero_masks
                print(
                    "\033[33mInitializing camera occlusion masks; per-camera LiDAR occlusion is computed during merge.\033[0m"
                )
                continue
            log_entries = []
            sam_log_entries = []
            for ood_id, details in self.ood_details.items():

                occlusion_masks_for_ood = []
                occlusion_path = f'{occlusion_mask_path}/ood_{ood_id}/{sensor_type}'
                os.makedirs(occlusion_path, exist_ok=True)
                if not self._has_valid_pose(details):
                    if ood_id not in self.occlusion_masks:
                        self.occlusion_masks[ood_id] = {}
                    self.occlusion_masks[ood_id][sensor_type] = [
                        np.zeros((1280, 1920), dtype=np.uint8)
                        for _ in range(len(frames))
                    ]
                    print(f"\033[33mSkipping occlusion map for OOD {ood_id} "
                          "(invalid/missing pose).\033[0m")
                    continue

                for frame_id, frame in enumerate(frames):
                    print(ood_id, frame_id)
                    object_pose_world = np.asarray(details['pose'])
                    object_pose_vehicle = np.linalg.inv(
                        self.vehicle_pose[sensor_type]
                        [frame_id]) @ object_pose_world
                    instance_point = object_pose_vehicle[:3, 3]

                    distance = np.linalg.norm(np.array(instance_point), ord=2)

                    if eval:
                        render_path = f"{self.base_dir}/{self.file_name}/Evaluation/Renders/"
                    else:
                        render_path = f"{self.base_dir}/{self.file_name}/Renders/"
                    no_shadow_file = self._resolve_no_shadow_render_path(
                        ood_id,
                        sensor_type,
                        frame_id,
                        eval=eval,
                    )
                    no_shadow_image = Image.open(no_shadow_file)
                    no_shadow_array = np.array(no_shadow_image)
                    ood_mask = self._extract_alpha_mask_with_fallback(
                        no_shadow_array,
                        hard_alpha_threshold=100,
                        min_pixels=int(max(1,
                                           self.required_render_min_pixels)),
                    )
                    ood_mask_pixels = np.where(ood_mask)

                    rows, columns = np.where(ood_mask)

                    occlusion_mask = np.zeros([1280, 1920])
                    if len(rows) > 0:

                        source_sensor = sensor_type
                        if (sensor_type == 'lidar'
                                and sensor_type not in self.point_cloud_vehicle
                                and 'camera' in self.point_cloud_vehicle):
                            source_sensor = 'camera'
                        if (source_sensor not in self.point_cloud_vehicle
                                or frame_id
                                not in self.point_cloud_vehicle[source_sensor]
                                or source_sensor not in self.cp_actual or
                                frame_id not in self.cp_actual[source_sensor]):
                            print(
                                f"Skipping occlusion computation for {sensor_type} frame {frame_id}: "
                                f"missing point cloud/projections for source '{source_sensor}'."
                            )
                            occlusion_masks_for_ood.append(occlusion_mask)
                            continue

                        points = self.point_cloud_vehicle[source_sensor][
                            frame_id].points

                        cp_points = self.cp_actual[source_sensor][frame_id]

                        distances = np.linalg.norm(np.array(points),
                                                   ord=2,
                                                   axis=1)
                        print('distances', np.max(distances),
                              np.min(distances))
                        probable_occlusion_instances = distances < distance - 0.05
                        cp_occlusion_points = cp_points[
                            probable_occlusion_instances]

                        cp_mask_front_cam = cp_occlusion_points[:, 0] == 1

                        indices_masked = np.where(cp_mask_front_cam)[0]
                        occlusion_instances_pixels = cp_occlusion_points[
                            cp_mask_front_cam][:, 1:3]
                        print('occlusion instances pixels',
                              occlusion_instances_pixels.shape)

                        H, W = 1280, 1920
                        proj = np.rint(occlusion_instances_pixels).astype(int)

                        set_proj_pixels = set(map(tuple, proj))

                        rows, columns = ood_mask_pixels
                        set_ood_pixels = set(zip(columns, rows))

                        if sensor_type == 'camera':
                            common_pixels_set = set_proj_pixels.intersection(
                                set_ood_pixels)
                        else:

                            occ_by_ood = self.ood_occlusion_pixels.get(ood_id)
                            if occ_by_ood is None:
                                if isinstance(ood_id, int):
                                    occ_by_ood = self.ood_occlusion_pixels.get(
                                        str(ood_id))
                                elif isinstance(ood_id,
                                                str) and ood_id.isdigit():
                                    occ_by_ood = self.ood_occlusion_pixels.get(
                                        int(ood_id))

                            if not occ_by_ood or frame_id not in occ_by_ood:
                                print(
                                    f"No precomputed lidar occlusion groups for OOD {ood_id}, "
                                    f"frame {frame_id}; using empty occlusion mask."
                                )
                                common_pixels_set = {}
                            else:
                                common_pixels_set = occ_by_ood[frame_id]
                        unique_panoptic_ids = None

                        if common_pixels_set:
                            print(
                                f"Found {len(common_pixels_set)} common pixels between occluders and the OOD object mask."
                            )
                            if sensor_type == 'camera':
                                panoptic_label = self.panoptic_labels[frame_id]
                                panoptic_label = np.reshape(
                                    panoptic_label,
                                    np.shape(panoptic_label)[0:2])
                                semantic_label = self.semantic_labels[frame_id]
                                semantic_label = np.reshape(
                                    semantic_label,
                                    np.shape(semantic_label)[0:2])

                                common_pixels_array = np.array(
                                    list(common_pixels_set))
                                u_common = common_pixels_array[:, 0]
                                v_common = common_pixels_array[:, 1]

                                panoptic_ids_of_common_pixels = panoptic_label[
                                    v_common, u_common]

                                unique_panoptic_ids = np.unique(
                                    panoptic_ids_of_common_pixels)
                                unique_panoptic_ids = [
                                    p for p in unique_panoptic_ids
                                    if p % panoptic_divisor != 0
                                ]
                                print('unique panoptic ids for camera',
                                      unique_panoptic_ids)
                                for id in unique_panoptic_ids:
                                    occlusion_mask[panoptic_label == id] = 1

                                fig, ax = plt.subplots(figsize=(12, 8))
                                ax.imshow(occlusion_mask, cmap='gray')
                                ax.grid(False)
                                ax.axis('off')
                                fig.tight_layout(pad=0)

                                path = f'{occlusion_mask_path}/ood_{ood_id}/{sensor_type}'
                                os.makedirs(path, exist_ok=True)
                                plt.savefig(
                                    f"{path}/frame_00{self.frame_map[sensor_type][frame_id]}.png",
                                    dpi=160)
                                plt.close()

                                print(
                                    "Panoptic IDs of the occluding objects at the intersection points:",
                                    unique_panoptic_ids)
                            else:

                                print(f"\n{'='*80}")
                                print(
                                    f"🎯 PROCESSING {len(common_pixels_set)} OCCLUSION INSTANCES"
                                )
                                print(f"{'='*80}")
                                print(
                                    f"  OOD ID: {ood_id} | Frame: {frame_id} | Sensor: {sensor_type}"
                                )

                                for occ_inst_id, occ_pixels in common_pixels_set.items(
                                ):
                                    print(f"\n{'='*80}")
                                    print(
                                        f"🔍 PROCESSING OCCLUDER INSTANCE #{occ_inst_id}"
                                    )
                                    print(f"{'='*80}")
                                    print(
                                        f"  Occlusion pixels: {len(occ_pixels)}"
                                    )

                                    common_pixels_array = np.array(
                                        list(occ_pixels))
                                    points_for_intersection = self.occlusion_pixels_from_projection.get(
                                        frame_id, [])
                                    if not points_for_intersection:
                                        print(
                                            f"⚠️  No projected occlusion pixels available for frame {frame_id}; "
                                            f"skipping occluder {occ_inst_id}."
                                        )
                                        continue

                                    common_pixels_set_tuples = set(
                                        map(tuple, common_pixels_array))

                                    intersected_pixels = set()

                                    projection_pixels_set = set(
                                        map(tuple, points_for_intersection))

                                    intersection = common_pixels_set_tuples & projection_pixels_set
                                    intersected_pixels.update(intersection)

                                    if intersected_pixels:
                                        common_pixels_array = np.array(
                                            list(intersected_pixels))
                                    else:
                                        print(
                                            f"⚠️  No intersection found between common pixels and occlusion projections for occluder {occ_inst_id}"
                                        )
                                        continue

                                    u_common = common_pixels_array[:,
                                                                   0].astype(
                                                                       np.int32
                                                                   )
                                    v_common = common_pixels_array[:,
                                                                   1].astype(
                                                                       np.int32
                                                                   )

                                    img_dir = f'{self.base_dir}/{self.file_name}/Raw_Images/{sensor_type}/frame_00{self.frame_map[sensor_type][frame_id]}.png'

                                    image = Image.open(img_dir).convert("RGB")
                                    image_np = np.array(image)

                                    if not self.enable_sam:
                                        print(
                                            f"SAM disabled. Using projection-only fallback occlusion mask "
                                            f"(OOD {ood_id}, frame {frame_id}, occluder {occ_inst_id})."
                                        )
                                        fallback_mask = np.zeros(
                                            image_np.shape[:2], dtype=np.uint8)
                                        valid_pixels = (
                                            (u_common >= 0) &
                                            (u_common < image_np.shape[1]) &
                                            (v_common >= 0) &
                                            (v_common < image_np.shape[0]))
                                        if np.any(valid_pixels):
                                            fallback_mask[
                                                v_common[valid_pixels],
                                                u_common[valid_pixels]] = 255

                                            kernel = np.ones((5, 5), np.uint8)
                                            fallback_mask = cv2.dilate(
                                                fallback_mask,
                                                kernel,
                                                iterations=1)
                                            occlusion_mask[fallback_mask ==
                                                           255] = 1
                                        sam_log_entries.append({
                                            "ood_id":
                                            ood_id,
                                            "frame_id":
                                            self.frame_map[sensor_type]
                                            [frame_id],
                                            "sensor_type":
                                            sensor_type,
                                            "occluder_id":
                                            occ_inst_id,
                                            "best_mask_path":
                                            None,
                                            "continuity_score":
                                            None,
                                            "solidity":
                                            None,
                                            "num_components":
                                            None,
                                            "fallback_mode":
                                            "projection_only"
                                        })
                                        continue

                                    sam_script_path = os.path.join(
                                        os.path.dirname(__file__),
                                        "sam_prediction.py")

                                    sam_output_dir = f'{sam_output}/ood_{ood_id}/{sensor_type}/frame_00{self.frame_map[sensor_type][frame_id]}/occluder_{occ_inst_id}'

                                    os.makedirs(sam_output_dir, exist_ok=True)

                                    temp_image_path = os.path.join(
                                        sam_output_dir, 'temp_image.png')
                                    temp_points_path = os.path.join(
                                        sam_output_dir, 'temp_points.json')

                                    os.makedirs(
                                        os.path.dirname(temp_image_path),
                                        exist_ok=True)

                                    raw_image_path = f'{self.base_dir}/{self.file_name}/Raw_Images/{sensor_type}/frame_00{self.frame_map[sensor_type][frame_id]}.png'

                                    all_points = np.array(
                                        [[int(x) for x in pixel]
                                         for pixel in occ_pixels])

                                    if len(all_points) > 0:

                                        from sklearn.cluster import DBSCAN

                                        dbscan = DBSCAN(eps=7, min_samples=3)
                                        dbscan_labels = dbscan.fit_predict(
                                            all_points)

                                        valid_cluster_mask = dbscan_labels != -1
                                        filtered_points = all_points[
                                            valid_cluster_mask]
                                        outlier_count = np.sum(
                                            ~valid_cluster_mask)

                                        print(
                                            f"📊 DBSCAN Outlier Removal: Kept {len(filtered_points)} points, removed {outlier_count} outliers"
                                        )

                                        outlier_viz_dir = f'{sam_output_dir}/outlier_removal_viz'
                                        os.makedirs(outlier_viz_dir,
                                                    exist_ok=True)

                                        outlier_viz_img = np.array(
                                            image).copy()

                                        for point in all_points:
                                            u, v = int(point[0]), int(point[1])
                                            cv2.circle(outlier_viz_img, (u, v),
                                                       radius=1,
                                                       color=(255, 255, 0),
                                                       thickness=-1)

                                        outlier_points = all_points[
                                            ~valid_cluster_mask]
                                        for point in outlier_points:
                                            u, v = int(point[0]), int(point[1])
                                            cv2.circle(outlier_viz_img, (u, v),
                                                       radius=3,
                                                       color=(255, 0, 0),
                                                       thickness=-1)

                                        for point in filtered_points:
                                            u, v = int(point[0]), int(point[1])
                                            cv2.circle(outlier_viz_img, (u, v),
                                                       radius=2,
                                                       color=(0, 0, 255),
                                                       thickness=-1)

                                        cv2.putText(
                                            outlier_viz_img,
                                            f"Cyan (n={len(all_points)}): All intersection pixels",
                                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6, (0, 255, 255), 1)
                                        cv2.putText(
                                            outlier_viz_img,
                                            f"Red (n={outlier_count}): Outliers (road/background)",
                                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6, (255, 0, 0), 1)
                                        cv2.putText(
                                            outlier_viz_img,
                                            f"Blue (n={len(filtered_points)}): Filtered inliers",
                                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6, (0, 0, 255), 1)

                                        outlier_viz_path = os.path.join(
                                            outlier_viz_dir,
                                            f'01_dbscan_outlier_removal.png')
                                        Image.fromarray(outlier_viz_img).save(
                                            outlier_viz_path)
                                        print(
                                            f"✓ Outlier removal visualization saved: {outlier_viz_path}"
                                        )

                                        if len(filtered_points) > 0:
                                            from sklearn.cluster import KMeans

                                            if len(filtered_points) <= 5:
                                                n_samples = len(
                                                    filtered_points)

                                                sampled_points = filtered_points.tolist(
                                                )
                                                print(
                                                    f"📊 Using all {n_samples} filtered points (small set)"
                                                )
                                            elif len(filtered_points) <= 20:
                                                n_samples = min(
                                                    5, len(filtered_points))
                                            elif len(filtered_points) <= 50:
                                                n_samples = 7
                                            else:
                                                n_samples = 10

                                            if len(filtered_points) > 5:

                                                kmeans = KMeans(
                                                    n_clusters=n_samples,
                                                    random_state=42,
                                                    n_init=10)
                                                kmeans.fit(filtered_points)

                                                sampled_points = []
                                                for cluster_id in range(
                                                        n_samples):
                                                    cluster_mask = kmeans.labels_ == cluster_id
                                                    cluster_points = filtered_points[
                                                        cluster_mask]

                                                    if len(cluster_points) > 0:

                                                        centroid = kmeans.cluster_centers_[
                                                            cluster_id]
                                                        distances = np.linalg.norm(
                                                            cluster_points -
                                                            centroid,
                                                            axis=1)
                                                        closest_idx = np.argmin(
                                                            distances)
                                                        sampled_points.append(
                                                            cluster_points[
                                                                closest_idx].
                                                            tolist())

                                                print(
                                                    f"📊 Spatial grid sampling: Selected {len(sampled_points)} representative points from {len(filtered_points)} filtered points"
                                                )
                                                print(
                                                    f"   → Points distributed across {n_samples} spatial regions for comprehensive coverage"
                                                )

                                            points_list = sampled_points
                                            print(
                                                f"✓ Sampled {len(sampled_points)} points across spatial regions"
                                            )

                                            spatial_viz_img = np.array(
                                                image).copy()

                                            for point in filtered_points:
                                                u, v = int(point[0]), int(
                                                    point[1])
                                                cv2.circle(spatial_viz_img,
                                                           (u, v),
                                                           radius=2,
                                                           color=(128, 128,
                                                                  128),
                                                           thickness=-1)

                                            cluster_colors = [
                                                (255, 0, 0),
                                                (0, 255, 0),
                                                (0, 0, 255),
                                                (255, 255, 0),
                                                (255, 0, 255),
                                                (0, 255, 255),
                                                (255, 128, 0),
                                                (128, 0, 255),
                                                (0, 255, 128),
                                                (255, 0, 128),
                                            ]

                                            if len(filtered_points) > 5:

                                                for cluster_id in range(
                                                        n_samples):
                                                    cluster_mask = kmeans.labels_ == cluster_id
                                                    cluster_points = filtered_points[
                                                        cluster_mask]
                                                    color = cluster_colors[
                                                        cluster_id %
                                                        len(cluster_colors)]

                                                    for point in cluster_points:
                                                        u, v = int(
                                                            point[0]), int(
                                                                point[1])
                                                        cv2.circle(
                                                            spatial_viz_img,
                                                            (u, v),
                                                            radius=3,
                                                            color=color,
                                                            thickness=-1)

                                            for point in points_list:
                                                u, v = int(point[0]), int(
                                                    point[1])
                                                cv2.circle(spatial_viz_img,
                                                           (u, v),
                                                           radius=10,
                                                           color=(255, 255,
                                                                  255),
                                                           thickness=-1)
                                                cv2.circle(spatial_viz_img,
                                                           (u, v),
                                                           radius=10,
                                                           color=(0, 0, 0),
                                                           thickness=2)

                                                cv2.circle(spatial_viz_img,
                                                           (u, v),
                                                           radius=2,
                                                           color=(0, 0, 0),
                                                           thickness=-1)

                                            legend_height = 110
                                            cv2.rectangle(
                                                spatial_viz_img, (5, 5),
                                                (650, legend_height),
                                                (0, 0, 0), -1)
                                            cv2.rectangle(
                                                spatial_viz_img, (5, 5),
                                                (650, legend_height),
                                                (255, 255, 255), 2)

                                            cv2.putText(
                                                spatial_viz_img,
                                                f"Gray dots: All filtered points (n={len(filtered_points)})",
                                                (10, 25),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (255, 255, 255), 1)
                                            cv2.putText(
                                                spatial_viz_img,
                                                f"Colored dots: Spatial clusters (n={n_samples} regions)",
                                                (10, 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (255, 255, 255), 1)
                                            cv2.putText(
                                                spatial_viz_img,
                                                f"White circles: Selected sample points (n={len(points_list)})",
                                                (10, 75),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (255, 255, 255), 1)
                                            cv2.putText(
                                                spatial_viz_img,
                                                f"Method: K-Means spatial grid sampling",
                                                (10, 100),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (255, 255, 255), 1)

                                            spatial_viz_path = os.path.join(
                                                outlier_viz_dir,
                                                f'02_spatial_grid_sampling.png'
                                            )
                                            Image.fromarray(
                                                spatial_viz_img).save(
                                                    spatial_viz_path)
                                            print(
                                                f"✓ Spatial grid sampling visualization saved: {spatial_viz_path}"
                                            )
                                        else:
                                            points_list = []
                                            print(
                                                "⚠️  No valid points after DBSCAN filtering"
                                            )
                                            continue
                                    else:
                                        points_list = []

                                    with open(temp_points_path, 'w') as f:
                                        json.dump(points_list, f)

                                    image_viz = np.array(image).copy()

                                    for point in occ_pixels:
                                        u, v = int(point[0]), int(point[1])

                                        cv2.circle(image_viz, (u, v),
                                                   radius=2,
                                                   color=(255, 255, 0),
                                                   thickness=-1)

                                    for point in points_list:
                                        u, v = int(point[0]), int(point[1])

                                        cv2.circle(image_viz, (u, v),
                                                   radius=6,
                                                   color=(0, 255, 0),
                                                   thickness=-1)
                                        cv2.circle(image_viz, (u, v),
                                                   radius=6,
                                                   color=(255, 255, 255),
                                                   thickness=2)

                                    cv2.putText(
                                        image_viz,
                                        f"Cyan dots (n={len(occ_pixels)}): All intersection pixels",
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (255, 255, 255), 2)
                                    cv2.putText(
                                        image_viz,
                                        f"Green circles (n={len(points_list)}): Sampled representative points",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (255, 255, 255), 2)

                                    sam_python_executable = self.sam_python_executable
                                    sam2_dir = self.sam2_dir
                                    if (not sam_python_executable
                                            or not sam2_dir
                                            or not os.path.isfile(
                                                sam_python_executable)
                                            or not os.path.isdir(sam2_dir)):
                                        print(
                                            "SAM enabled but configuration is invalid. "
                                            "Using projection-only fallback occlusion mask."
                                        )
                                        fallback_mask = np.zeros(
                                            image_np.shape[:2], dtype=np.uint8)
                                        valid_pixels = (
                                            (u_common >= 0) &
                                            (u_common < image_np.shape[1]) &
                                            (v_common >= 0) &
                                            (v_common < image_np.shape[0]))
                                        if np.any(valid_pixels):
                                            fallback_mask[
                                                v_common[valid_pixels],
                                                u_common[valid_pixels]] = 255
                                            kernel = np.ones((5, 5), np.uint8)
                                            fallback_mask = cv2.dilate(
                                                fallback_mask,
                                                kernel,
                                                iterations=1)
                                            occlusion_mask[fallback_mask ==
                                                           255] = 1
                                        sam_log_entries.append({
                                            "ood_id":
                                            ood_id,
                                            "frame_id":
                                            self.frame_map[sensor_type]
                                            [frame_id],
                                            "sensor_type":
                                            sensor_type,
                                            "occluder_id":
                                            occ_inst_id,
                                            "best_mask_path":
                                            None,
                                            "continuity_score":
                                            None,
                                            "solidity":
                                            None,
                                            "num_components":
                                            None,
                                            "fallback_mode":
                                            "projection_only_invalid_sam_config"
                                        })
                                        continue

                                    sam_output_dir = os.path.abspath(
                                        sam_output_dir)
                                    temp_image_path = os.path.abspath(
                                        os.path.join(sam_output_dir,
                                                     'temp_image.png'))
                                    temp_points_path = os.path.abspath(
                                        os.path.join(sam_output_dir,
                                                     'temp_points.json'))

                                    os.makedirs(
                                        os.path.dirname(temp_image_path),
                                        exist_ok=True)

                                    image.save(temp_image_path)
                                    with open(temp_points_path, 'w') as f:
                                        json.dump(points_list, f)

                                    command = [
                                        sam_python_executable, sam_script_path,
                                        '--image_path', temp_image_path,
                                        '--points_path', temp_points_path,
                                        '--output_dir', sam_output_dir
                                    ]

                                    print(
                                        f"Running SAM prediction for frame {frame_id} sensor {sensor_type}..."
                                    )
                                    try:

                                        sam_env = os.environ.copy()
                                        sam_env[
                                            'MKL_SERVICE_FORCE_INTEL'] = '1'
                                        sam_env['MKL_THREADING_LAYER'] = 'GNU'

                                        result = subprocess.run(
                                            command,
                                            check=True,
                                            capture_output=True,
                                            text=True,
                                            cwd=sam2_dir,
                                            env=sam_env)
                                        print("SAM prediction successful.")
                                        print(result.stdout)
                                    except subprocess.CalledProcessError as e:
                                        print(
                                            f"SAM prediction failed with exit code {e.returncode}:"
                                        )
                                        print(f"Stderr: {e.stderr}")
                                        print(f"Stdout: {e.stdout}")
                                    finally:
                                        best_confidence = 0
                                        best_mask_path = None
                                        best_class = ""
                                        best_continuity_score = 0
                                        best_continuity_metrics = {}

                                        if os.path.exists(temp_image_path):
                                            os.remove(temp_image_path)
                                        if os.path.exists(temp_points_path):
                                            os.remove(temp_points_path)

                                        img_dir = f'{self.base_dir}/{self.file_name}/Raw_Images/{sensor_type}/frame_00{self.frame_map[sensor_type][frame_id]}.png'
                                        image = Image.open(img_dir).convert(
                                            "RGB")
                                        image_np = np.array(image)

                                        print(f"\n{'='*80}")
                                        print(
                                            f"🔍 COMBINING ALL SAM MASKS FOR OCCLUDER #{occ_inst_id}"
                                        )
                                        print(f"{'='*80}\n")

                                        combined_mask = np.zeros(
                                            image_np.shape[:2], dtype=np.uint8)
                                        mask_files = []

                                        colors = [
                                            (255, 0, 0),
                                            (0, 255, 0),
                                            (0, 0, 255),
                                            (255, 255, 0),
                                            (255, 0, 255),
                                            (0, 255, 255),
                                            (128, 0, 255),
                                            (255, 128, 0),
                                        ]

                                        visualization = image_np.copy().astype(
                                            np.float32)

                                        for idx, image_file in enumerate(
                                                os.listdir(sam_output_dir)):
                                            sam_mask_path = os.path.join(
                                                sam_output_dir, image_file)

                                            if os.path.isdir(
                                                    sam_mask_path
                                            ) or image_file.startswith(
                                                    'temp_'):
                                                continue

                                            sam_mask_image = Image.open(
                                                sam_mask_path)
                                            sam_mask_array = np.array(
                                                sam_mask_image)
                                            sam_mask_binary = sam_mask_array == 255

                                            combined_mask[
                                                sam_mask_binary] = 255

                                            color = colors[idx % len(colors)]
                                            mask_overlay = np.zeros_like(
                                                image_np, dtype=np.float32)
                                            mask_overlay[
                                                sam_mask_binary] = color

                                            visualization = np.where(
                                                sam_mask_binary[:, :,
                                                                np.newaxis],
                                                visualization * 0.85 +
                                                mask_overlay * 0.15,
                                                visualization)

                                            mask_files.append(image_file)
                                            print(
                                                f"  📊 Mask {idx+1}: {image_file}"
                                            )
                                            print(
                                                f"     Pixels: {np.sum(sam_mask_binary)}"
                                            )
                                            print(f"     Color: RGB{color}")

                                        print(
                                            f"\n  ✅ Combined {len(mask_files)} masks"
                                        )
                                        print(
                                            f"  📊 Total OR-ed pixels: {np.sum(combined_mask == 255)}"
                                        )

                                        if np.sum(combined_mask == 255) > 0:
                                            print(f"\n{'='*80}")
                                            print(
                                                f"🧹 FILTERING SMALL NOISY COMPONENTS"
                                            )
                                            print(f"{'='*80}")

                                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                                                combined_mask, connectivity=8)

                                            component_areas = stats[
                                                1:, cv2.CC_STAT_AREA]

                                            if len(component_areas) > 0:
                                                print(
                                                    f"  📊 Found {len(component_areas)} connected components"
                                                )
                                                print(
                                                    f"  📏 Component areas: {sorted(component_areas, reverse=True)}"
                                                )

                                                largest_area = component_areas.max(
                                                )
                                                area_threshold = max(
                                                    100, largest_area * 0.05)

                                                print(
                                                    f"  🎯 Largest component area: {largest_area} pixels"
                                                )
                                                print(
                                                    f"  🚪 Area threshold: {area_threshold:.0f} pixels (5% of largest, min 100)"
                                                )

                                                filtered_mask = np.zeros_like(
                                                    combined_mask)
                                                components_kept = 0
                                                components_removed = 0

                                                for i in range(1, num_labels):
                                                    area = stats[
                                                        i, cv2.CC_STAT_AREA]
                                                    if area >= area_threshold:
                                                        filtered_mask[labels ==
                                                                      i] = 255
                                                        components_kept += 1
                                                    else:
                                                        components_removed += 1

                                                print(
                                                    f"  ✅ Kept {components_kept} large components"
                                                )
                                                print(
                                                    f"  ❌ Removed {components_removed} small noisy components"
                                                )
                                                print(
                                                    f"  📊 Filtered mask pixels: {np.sum(filtered_mask == 255)} (before: {np.sum(combined_mask == 255)})"
                                                )

                                                combined_mask = filtered_mask
                                            else:
                                                print(
                                                    f"  ⚠️  No components found"
                                                )

                                        combined_mask_path = os.path.join(
                                            sam_output_dir,
                                            'combined_mask.png')
                                        Image.fromarray(combined_mask).save(
                                            combined_mask_path)
                                        best_sam_mask_path = combined_mask_path

                                        visualization_uint8 = visualization.astype(
                                            np.uint8)
                                        visualization_path = os.path.join(
                                            sam_output_dir,
                                            'masks_visualization.png')
                                        Image.fromarray(
                                            visualization_uint8).save(
                                                visualization_path)
                                        print(
                                            f"\n  💾 Saved combined mask: {combined_mask_path}"
                                        )
                                        print(
                                            f"  💾 Saved visualization: {visualization_path}"
                                        )

                                        if best_sam_mask_path and np.sum(
                                                combined_mask == 255) > 0:
                                            print(f"\n{'='*80}")
                                            print(
                                                f"🏆 PROCESSING COMBINED OR-ED MASK"
                                            )
                                            print(f"{'='*80}\n")

                                            best_continuity_score, best_continuity_metrics = calculate_mask_continuity(
                                                combined_mask)
                                            print(
                                                f"  💪 Solidity: {best_continuity_metrics.get('solidity', 0):.4f}"
                                            )
                                            print(
                                                f"  📊 Continuity: {best_continuity_score:.4f}"
                                            )
                                            print(
                                                f"  🔗 Components: {best_continuity_metrics.get('num_components', 0)}\n"
                                            )

                                            rows, cols = np.where(
                                                combined_mask == 255)
                                            raw_img_cropped = np.array([])

                                            if len(rows) > 0 and len(cols) > 0:
                                                y_min, y_max = rows.min(
                                                ), rows.max()
                                                x_min, x_max = cols.min(
                                                ), cols.max()

                                                raw_img_cropped = image_np[
                                                    y_min:y_max + 1,
                                                    x_min:x_max + 1]

                                            if len(raw_img_cropped) > 0:

                                                sam_mask_cropped = combined_mask[
                                                    y_min:y_max + 1,
                                                    x_min:x_max + 1]
                                                crop_height = y_max - y_min + 1
                                                crop_width = x_max - x_min + 1

                                                print(f"\n{'='*80}")
                                                print(
                                                    f"🖼️  COMBINED MASK CROP INFORMATION"
                                                )
                                                print(f"{'='*80}")
                                                print(
                                                    f"  📍 Original Image Size: 1280 x 1920"
                                                )
                                                print(
                                                    f"  📍 Crop Region Bounds:")
                                                print(
                                                    f"     X (columns): [{x_min}, {x_max}]"
                                                )
                                                print(
                                                    f"     Y (rows): [{y_min}, {y_max}]"
                                                )
                                                print(
                                                    f"  📐 Cropped Image Size: {crop_width} x {crop_height}"
                                                )
                                                print(
                                                    f"  🎭 SAM Mask in crop: {np.sum(sam_mask_cropped == 255)} pixels"
                                                )

                                                debug_crop_dir = f"{sam_output_dir}/debug_visualization_combined"
                                                if os.path.exists(
                                                        debug_crop_dir):
                                                    shutil.rmtree(
                                                        debug_crop_dir)
                                                os.makedirs(debug_crop_dir,
                                                            exist_ok=True)

                                                debug_raw_crop = os.path.join(
                                                    debug_crop_dir,
                                                    f'01_raw_crop.jpg')
                                                Image.fromarray(
                                                    raw_img_cropped).save(
                                                        debug_raw_crop)
                                                print(
                                                    f"\n  💾 Saved raw crop visualization: {debug_raw_crop}"
                                                )

                                                debug_mask_overlay = os.path.join(
                                                    debug_crop_dir,
                                                    f'02_sam_mask_overlay.jpg')
                                                overlay_img = raw_img_cropped.copy(
                                                )
                                                overlay_img[sam_mask_cropped ==
                                                            255] = [0, 255, 0]
                                                Image.fromarray(
                                                    overlay_img).save(
                                                        debug_mask_overlay)
                                                print(
                                                    f"  💾 Saved SAM mask overlay: {debug_mask_overlay}"
                                                )

                                                best_mask_path = best_sam_mask_path
                                        else:
                                            print(
                                                f"\n⚠️  NO VALID MASK FOUND FOR OCCLUDER #{occ_inst_id} (all masks below threshold)\n"
                                            )

                                        if best_mask_path:
                                            mask = Image.open(best_mask_path)
                                            mask_array = np.array(mask)
                                            occlusion_mask[mask_array ==
                                                           255] = 1

                                            print(f"\n{'='*80}")
                                            print(
                                                f"🏆 FINAL BEST MASK FOR OCCLUDER #{occ_inst_id}"
                                            )
                                            print(f"{'='*80}")

                                            print(
                                                f"  📊 Continuity: {best_continuity_score:.4f}"
                                            )
                                            print(
                                                f"  🔗 Components: {best_continuity_metrics.get('num_components', 'N/A')}"
                                            )
                                            print(
                                                f"  💪 Solidity: {best_continuity_metrics.get('solidity', 'N/A'):.4f}"
                                                if 'solidity' in
                                                best_continuity_metrics else ""
                                            )
                                            print(
                                                f"  📁 Mask Path: {best_mask_path}"
                                            )
                                            print(f"{'='*80}\n")

                                            sam_log_entry = {
                                                "ood_id":
                                                ood_id,
                                                "frame_id":
                                                self.frame_map[sensor_type]
                                                [frame_id],
                                                "sensor_type":
                                                sensor_type,
                                                "occluder_id":
                                                occ_inst_id,
                                                "best_mask_path":
                                                best_mask_path,
                                                "continuity_score":
                                                float(best_continuity_score),
                                                "solidity":
                                                float(
                                                    best_continuity_metrics.
                                                    get('solidity', 0)),
                                                "num_components":
                                                int(
                                                    best_continuity_metrics.
                                                    get('num_components', 0))
                                            }
                                            sam_log_entries.append(
                                                sam_log_entry)

                                            ind_mask_dir = f'{occlusion_mask_path}/ood_{ood_id}/{sensor_type}/frame_00{self.frame_map[sensor_type][frame_id]}/individual'
                                            os.makedirs(ind_mask_dir,
                                                        exist_ok=True)
                                            ind_path = os.path.join(
                                                ind_mask_dir,
                                                f'occ_{occ_inst_id}_{best_class}.png'
                                            )
                                            best_mask_img = Image.open(
                                                best_mask_path)
                                            best_mask_img.save(ind_path)

                                            viz_img = image_np.copy()
                                            viz_img[mask_array == 255] = [
                                                0, 255, 0
                                            ]
                                            viz_pil = Image.fromarray(viz_img)
                                            draw = ImageDraw.Draw(viz_pil)
                                            draw.text(
                                                (10, 10),
                                                f"Occ #{occ_inst_id}: {best_class} ({best_confidence:.2f})",
                                                fill=(255, 255, 255))
                                            viz_path = os.path.join(
                                                ind_mask_dir,
                                                f'occ_{occ_inst_id}_viz.png')
                                            viz_pil.save(viz_path)
                                            print(f"  💾 Saved: {ind_path}")
                                            print(f"  💾 Saved viz: {viz_path}")

                                print(f"\n{'='*80}")
                                print(
                                    f"💾 FINAL COMBINED MASK - {len(common_pixels_set)} occluders"
                                )
                                print(f"{'='*80}\n")
                                fig, ax = plt.subplots(figsize=(12, 8))
                                ax.imshow(occlusion_mask, cmap='gray')
                                ax.set_title(
                                    f'Combined Mask - OOD {ood_id}, Frame {frame_id}'
                                )
                                ax.axis('off')
                                fig.tight_layout(pad=0)
                                final_dir = f'{occlusion_mask_path}/ood_{ood_id}/{sensor_type}'
                                os.makedirs(final_dir, exist_ok=True)
                                final_file = os.path.join(
                                    final_dir,
                                    f'frame_00{self.frame_map[sensor_type][frame_id]}_FINAL.png'
                                )
                                plt.savefig(final_file, dpi=160)
                                plt.close()
                                print(f"💾 Saved: {final_file}\n")

                                entry = {
                                    "ood_id":
                                    ood_id,
                                    "frame_id":
                                    self.frame_map[sensor_type][frame_id],
                                    "num_occluders":
                                    len(common_pixels_set),
                                    "final_mask_path":
                                    final_file
                                }
                                log_entries.append(entry)

                        else:
                            print(
                                "No common pixels found between projected occluders and the OOD object mask."
                            )

                    occlusion_masks_for_ood.append(occlusion_mask)
                    print("Occlusion mask computed for OOD ID", ood_id,
                          "Frame ID", frame_id, "Sensor", sensor_type)

                if ood_id not in self.occlusion_masks:
                    self.occlusion_masks[ood_id] = {}
                self.occlusion_masks[ood_id][
                    sensor_type] = occlusion_masks_for_ood

            log_key = f"{self.file_name}_{sensor_type}"
            self._update_sam_mask_log(log_key, sam_log_entries)
            print(
                f"✅ SAM log updated with {len(sam_log_entries)} mask selections for {sensor_type}"
            )

    def merge_ood_into_image(self, sensor, eval=False, render_mode='multi'):
        """Method to copy transparent OOD rendering onto the background. Calculate occlusion mask and
        cut the object transparent to create realistic occlusion using the ranger image"""
        print("\033[36mSAMPLER: Merging OOD object into image\033[0m")

        def get_3d_bbox_corners(xmin, ymin, zmin, xmax, ymax, zmax):
            return np.array([[xmin, ymin, zmin], [xmax, ymin, zmin],
                             [xmax, ymax, zmin], [xmin, ymax, zmin],
                             [xmin, ymin, zmax], [xmax, ymin, zmax],
                             [xmax, ymax, zmax], [xmin, ymax, zmax]])

        def project_points(points_3d, frame_id):

            X_cam = -points_3d[:, 1]
            Y_cam = -points_3d[:, 2]
            Z_cam = points_3d[:, 0]

            fx, fy, cx, cy = self.camera_data[sensor_type][frame_id][
                'projection']

            u = (fx * X_cam / Z_cam) + cx
            v = (fy * Y_cam / Z_cam) + cy

            return np.array([u, v]).T

        def get_bbox_properties(corners):
            """
            Compute 3D bounding box properties from its 8 corners.

            Args:
                corners: (8, 3) array of box corners in standard order

            Returns:
                center_x, center_y, center_z, length, width, height, heading
            """

            center = np.mean(corners, axis=0)
            center_x, center_y, center_z = center

            v1 = corners[1] - corners[0]
            v2 = corners[3] - corners[0]
            v3 = corners[4] - corners[0]

            length = np.linalg.norm(v1)
            width = np.linalg.norm(v2)
            height = np.linalg.norm(v3)

            heading = np.arctan2(v1[1], v1[0])

            return center_x, center_y, center_z, length, width, height, heading

        def estimate_alpha_blur_radius(alpha_channel):

            alpha_np = np.array(alpha_channel, dtype=np.float32) / 255.0
            edges = sobel(alpha_np)

            if np.max(edges) < 1e-5:
                return 1

            threshold = 0.05 * np.max(edges)
            edge_mask = edges > threshold

            if not np.any(edge_mask):
                return 1

            edge_values = edges[edge_mask]
            edge_strength = np.percentile(edge_values, 90)

            max_possible = 4 * np.sqrt(2)
            normalized_strength = min(1.0, edge_strength / max_possible)

            blur_radius = 1 + 2 * np.exp(-3 * normalized_strength)
            return max(1, min(3, blur_radius))

        def bbox_based_blur_radius(alpha_channel):
            alpha_np = np.array(alpha_channel)
            y_indices, x_indices = np.where(alpha_np > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                return 1

            width = x_indices.max() - x_indices.min()
            height = y_indices.max() - y_indices.min()
            size = max(width, height)

            return max(1, (size * 0.01))

        def dynamic_blur_radius(alpha_channel):
            edge_based = estimate_alpha_blur_radius(alpha_channel)
            size_based = bbox_based_blur_radius(alpha_channel)
            weight = 0.75

            return int(weight * edge_based + (1 - weight) * size_based)

        pipeline_dataset = None
        for sensor_type in self.frames_with_seg.keys():
            print(sensor_type, len(self.frames_with_seg[sensor_type]))

            if len(self.frames_with_seg[sensor_type]) == 0:
                continue
            if sensor_type == 'camera' and isinstance(
                    self.raw_images.get(sensor_type), dict):

                continue
            if self.train:
                train_valid = "Train"
            else:
                train_valid = "Valid"
            base_image_path = self.image_path if self.image_path else f"Pipeline_2_{train_valid}"

            if not eval:
                pipeline_dataset = f"{base_image_path}/{self.file_name}/merged_directly_shadow1_resized/"
                pipeline_dataset_type = f"{pipeline_dataset}{sensor_type}"
                pipeline_dataset_pr = f"{base_image_path}/{self.file_name}/merged_without_occlusion/"
                pipeline_dataset_pr_type = f"{pipeline_dataset_pr}{sensor_type}"
            else:
                pipeline_dataset = f"{base_image_path}/{self.file_name}/Evaluation/merged_directly_shadow1_resized/"
                pipeline_dataset_type = f"{pipeline_dataset}{sensor_type}"
                pipeline_dataset_pr = f"{base_image_path}/{self.file_name}/Evaluation/merged_without_occlusion/"
                pipeline_dataset_pr_type = f"{pipeline_dataset_pr}{sensor_type}"
            if render_mode == 'single':
                pipeline_dataset = f"{self.base_dir}/{self.file_name}/Single_Rendered_Images/"
                pipeline_dataset_type = f"{pipeline_dataset}{sensor_type}/"
                pipeline_dataset_pr = f"{self.base_dir}/{self.file_name}/Single_Rendered_Images_No_Occlusion/"
                pipeline_dataset_pr_type = f"{pipeline_dataset_pr}{sensor_type}/"

            os.makedirs(pipeline_dataset_type, exist_ok=True)
            os.makedirs(pipeline_dataset_pr_type, exist_ok=True)

            if sensor_type not in self.merged_images:
                self.merged_images[sensor_type] = []
            for frame_id, frame in enumerate(self.raw_images[sensor_type]):
                print(f'Merging frame_id {frame_id} of sensor {sensor_type}')
                back_image_direct = Image.fromarray(frame)
                back_image_wo_occlusion = Image.fromarray(frame)

                camera_point = [
                    p[3] for p in self.camera_data[sensor_type][frame_id]
                    ["camera_pose"][0:3]
                ]

                instance_points_dict = {
                    ood_id: [p[3] for p in details['pose'][0:3]]
                    for ood_id, details in self.ood_details.items()
                    if self._has_valid_pose(details)
                }

                distances_dict = {
                    ood_id:
                    np.linalg.norm(np.array(inst_p) - np.array(camera_point))
                    for ood_id, inst_p in instance_points_dict.items()
                }

                sorted_indices = sorted(
                    distances_dict.keys(),
                    key=lambda ood_id: distances_dict[ood_id],
                    reverse=True)

                all_ood_mask = np.zeros([1280, 1920], dtype=bool)
                if sensor_type == 'camera':
                    semantic_label = self.semantic_labels[frame_id]
                    instance_label = self.instance_labels[frame_id]
                for ood_id in sorted_indices:

                    if eval:
                        render_file = f"{self.base_dir}/{self.file_name}/Evaluation/Renders/ood_{ood_id}/{sensor_type}/frame_00{self.frame_map[sensor_type][frame_id]}.png"
                    else:
                        render_file = f"{self.base_dir}/{self.file_name}/Renders/ood_{ood_id}/{sensor_type}/frame_00{self.frame_map[sensor_type][frame_id]}.png"
                    ood_image = Image.open(render_file)

                    back_image_wo_occlusion.paste(ood_image, (0, 0), ood_image)
                    self.merged_images[sensor_type].append(
                        np.array(back_image_wo_occlusion))

                    bbox_3d = self.ood_details.get(ood_id, {}).get('bbox_3d')
                    if (isinstance(bbox_3d, dict) and ('min' in bbox_3d)
                            and ('max' in bbox_3d)):
                        bbox_min = np.array(bbox_3d['min'])
                        bbox_max = np.array(bbox_3d['max'])

                        extrinsic_gl = self.vehicle_pose[sensor_type][
                            frame_id] @ self.camera_data[sensor_type][
                                frame_id]['extrinsic']
                        bbox_min = (np.linalg.inv(extrinsic_gl) @ np.vstack(
                            (bbox_min.reshape(3, 1), [[1]]))).flatten()[:3]
                        bbox_max = (np.linalg.inv(extrinsic_gl) @ np.vstack(
                            (bbox_max.reshape(3, 1), [[1]]))).flatten()[:3]

                        corners_3d = get_3d_bbox_corners(*bbox_min, *bbox_max)

                        if 'camera_labels' not in self.ood_details[
                                ood_id].keys():
                            self.ood_details[ood_id]['camera_labels'] = {}
                        self.ood_details[ood_id]['camera_labels'][
                            frame_id] = {}

                        center_x, center_y, center_z, width, length, height, heading = get_bbox_properties(
                            corners_3d)
                        self.ood_details[ood_id]['camera_labels'][frame_id][
                            'center_x'] = center_x
                        self.ood_details[ood_id]['camera_labels'][frame_id][
                            'center_y'] = center_y
                        self.ood_details[ood_id]['camera_labels'][frame_id][
                            'center_z'] = center_z
                        self.ood_details[ood_id]['camera_labels'][frame_id][
                            'width'] = width
                        self.ood_details[ood_id]['camera_labels'][frame_id][
                            'length'] = length
                        self.ood_details[ood_id]['camera_labels'][frame_id][
                            'height'] = height
                        self.ood_details[ood_id]['camera_labels'][frame_id][
                            'heading'] = heading

                    corners_2d = project_points(corners_3d, frame_id)

                    corners_2d = corners_2d.astype(int)

                back_image_wo_occlusion = back_image_wo_occlusion.convert(
                    'RGB')
                back_image_wo_occlusion.save(
                    f"{pipeline_dataset_pr_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}.jpg"
                )

    def merge_ood_into_all_camera_images(self,
                                         eval=False,
                                         render_mode='multi',
                                         ood_semantic_class=None):
        """Merge rendered OOD layers into all camera streams and emit per-camera labels."""
        if 'camera' not in self.raw_images or not isinstance(
                self.raw_images['camera'], dict):
            return None
        if ood_semantic_class is None:
            ood_semantic_class = self.ood_semantic_class

        if self.train:
            train_valid = "Train"
        else:
            train_valid = "Valid"
        base_image_path = self.image_path if self.image_path else f"Pipeline_2_{train_valid}"

        if not eval:
            pipeline_dataset = f"{base_image_path}/{self.file_name}/merged_directly_shadow1_resized/"
            pipeline_dataset_pr = f"{base_image_path}/{self.file_name}/merged_without_occlusion/"
        else:
            pipeline_dataset = f"{base_image_path}/{self.file_name}/Evaluation/merged_directly_shadow1_resized/"
            pipeline_dataset_pr = f"{base_image_path}/{self.file_name}/Evaluation/merged_without_occlusion/"
        if render_mode == 'single':
            pipeline_dataset = f"{self.base_dir}/{self.file_name}/Single_Rendered_Images/"
            pipeline_dataset_pr = f"{self.base_dir}/{self.file_name}/Single_Rendered_Images_No_Occlusion/"

        if 'camera' not in self.merged_images or not isinstance(
                self.merged_images.get('camera'), dict):
            self.merged_images['camera'] = {}

        def get_bbox_properties(corners):
            center = np.mean(corners, axis=0)
            center_x, center_y, center_z = center
            v1 = corners[1] - corners[0]
            v2 = corners[3] - corners[0]
            v3 = corners[4] - corners[0]
            length = np.linalg.norm(v1)
            width = np.linalg.norm(v2)
            height = np.linalg.norm(v3)
            heading = np.arctan2(v1[1], v1[0])
            return center_x, center_y, center_z, width, length, height, heading

        def get_3d_bbox_corners(xmin, ymin, zmin, xmax, ymax, zmax):
            return np.array([[xmin, ymin, zmin], [xmax, ymin, zmin],
                             [xmax, ymax, zmin], [xmin, ymax, zmin],
                             [xmin, ymin, zmax], [xmax, ymin, zmax],
                             [xmax, ymax, zmax], [xmin, ymax, zmax]])

        def export_shadow_debug(camera_name, frame_token, ood_id, raw_rgb,
                                full_rgba, no_shadow_rgba):
            if not self.enable_shadow_debug_exports:
                return
            metrics = self._shadow_metrics_from_rgba(
                full_rgba=full_rgba,
                no_shadow_rgba=no_shadow_rgba,
                dilate_px=self.shadow_debug_contact_dilate_px,
            )
            if metrics is None:
                return

            if not eval:
                debug_root = Path(
                    self.base_dir) / self.file_name / "Shadow_Debug"
            else:
                debug_root = Path(
                    self.base_dir
                ) / self.file_name / "Evaluation" / "Shadow_Debug"
            debug_dir = debug_root / f"ood_{int(ood_id)}" / "camera" / camera_name
            debug_dir.mkdir(parents=True, exist_ok=True)

            object_mask = (metrics["object_mask"].astype(np.uint8) * 255)
            shadow_mask = (metrics["shadow_only"].astype(np.uint8) * 255)
            full_alpha = (full_rgba[:, :, 3].astype(np.uint8))
            Image.fromarray(object_mask).save(
                debug_dir / f"frame_00{frame_token}_object_mask.png")
            Image.fromarray(shadow_mask).save(
                debug_dir / f"frame_00{frame_token}_shadow_only.png")
            Image.fromarray(full_alpha).save(
                debug_dir / f"frame_00{frame_token}_full_alpha.png")

            overlay = np.array(raw_rgb, copy=True)
            if overlay.shape[:2] != metrics["object_mask"].shape:
                overlay = cv2.resize(
                    overlay,
                    (metrics["object_mask"].shape[1],
                     metrics["object_mask"].shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            overlay_obj = overlay.copy()
            overlay_obj[metrics["object_mask"]] = [0, 255, 0]
            overlay_obj[metrics["shadow_only"]] = [255, 32, 32]
            overlay = cv2.addWeighted(overlay, 0.55, overlay_obj, 0.45, 0.0)
            Image.fromarray(overlay.astype(np.uint8)).save(
                debug_dir / f"frame_00{frame_token}_overlay.png")

        for camera_name, frame_images in self.raw_images['camera'].items():
            camera_output_dir = f"{pipeline_dataset}camera/{camera_name}/"
            camera_output_dir_pr = f"{pipeline_dataset_pr}camera/{camera_name}/"
            os.makedirs(camera_output_dir, exist_ok=True)
            os.makedirs(camera_output_dir_pr, exist_ok=True)
            self.merged_images['camera'][camera_name] = []
            frame_count = len(frame_images)
            render_bundle_cache = {}

            def _get_camera_render_paths(ood_id, frame_idx):
                frame_token = self.frame_map['camera'][frame_idx]
                if eval:
                    render_file_local = (
                        f"{self.base_dir}/{self.file_name}/Evaluation/Renders/ood_{ood_id}/camera/{camera_name}/"
                        f"frame_00{frame_token}.png")
                    no_shadow_file_local = (
                        f"{self.base_dir}/{self.file_name}/Evaluation/Renders/ood_{ood_id}/camera/{camera_name}/"
                        f"no_shadow_frame_00{frame_token}.png")
                else:
                    render_file_local = (
                        f"{self.base_dir}/{self.file_name}/Renders/ood_{ood_id}/camera/{camera_name}/"
                        f"frame_00{frame_token}.png")
                    no_shadow_file_local = (
                        f"{self.base_dir}/{self.file_name}/Renders/ood_{ood_id}/camera/{camera_name}/"
                        f"no_shadow_frame_00{frame_token}.png")
                return render_file_local, no_shadow_file_local

            def _load_render_bundle_for_frame(ood_id, frame_idx, image_h,
                                              image_w):
                key = (int(ood_id), int(frame_idx), int(image_h), int(image_w))
                if key in render_bundle_cache:
                    return render_bundle_cache[key]

                render_file, no_shadow_file = _get_camera_render_paths(
                    ood_id, frame_idx)
                if not os.path.exists(render_file) or not os.path.exists(
                        no_shadow_file):
                    render_bundle_cache[key] = None
                    return None

                no_shadow_image_local = Image.open(no_shadow_file)
                ood_no_shadow_array_local = np.array(no_shadow_image_local)
                ood_mask_local = self._extract_alpha_mask_with_fallback(
                    ood_no_shadow_array_local,
                    hard_alpha_threshold=100,
                    min_pixels=int(max(1, self.required_render_min_pixels)),
                )
                if ood_mask_local.shape != (image_h, image_w):
                    ood_mask_local = cv2.resize(
                        ood_mask_local.astype(np.uint8), (image_w, image_h),
                        interpolation=cv2.INTER_NEAREST) > 0
                else:
                    ood_mask_local = ood_mask_local.astype(bool)

                occlusion_mask_local = np.zeros((image_h, image_w),
                                                dtype=np.uint8)
                sensor_masks_local = self.occlusion_masks.get(ood_id,
                                                              {}).get('camera')
                if isinstance(sensor_masks_local, dict):
                    camera_masks_local = sensor_masks_local.get(camera_name)
                    if camera_masks_local is not None and frame_idx < len(
                            camera_masks_local):
                        occlusion_mask_local = np.asarray(
                            camera_masks_local[frame_idx])
                elif isinstance(sensor_masks_local,
                                list) and frame_idx < len(sensor_masks_local):
                    occlusion_mask_local = np.asarray(
                        sensor_masks_local[frame_idx])
                if np.sum(occlusion_mask_local) == 0:
                    occlusion_mask_local = self._compute_camera_occlusion_mask_from_lidar(
                        ood_id,
                        frame_idx,
                        camera_name,
                        ood_mask_local,
                    )

                ood_image_local = Image.open(render_file)
                if ood_image_local.size != (image_w, image_h):
                    ood_image_local = ood_image_local.resize(
                        (image_w, image_h), resample=Image.BILINEAR)
                ood_array_local = np.array(ood_image_local)
                if ood_no_shadow_array_local.shape[:
                                                   2] != ood_array_local.shape[:
                                                                               2]:
                    resized_no_shadow_local = cv2.resize(
                        ood_no_shadow_array_local,
                        (ood_array_local.shape[1], ood_array_local.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    resized_no_shadow_local = ood_no_shadow_array_local
                if occlusion_mask_local.shape != ood_array_local.shape[:2]:
                    occlusion_mask_local = cv2.resize(
                        occlusion_mask_local.astype(np.uint8),
                        (ood_array_local.shape[1], ood_array_local.shape[0]),
                        interpolation=cv2.INTER_NEAREST)

                occlusion_mask_local = (occlusion_mask_local > 0).astype(
                    np.uint8)
                visible_mask_local = ood_mask_local & (occlusion_mask_local
                                                       == 0)
                src_count_local = int(np.count_nonzero(ood_mask_local))
                vis_count_local = int(np.count_nonzero(visible_mask_local))
                vis_ratio_local = (
                    float(vis_count_local) /
                    float(src_count_local)) if src_count_local > 0 else 0.0

                bundle = {
                    "ood_image": ood_image_local,
                    "ood_array": ood_array_local,
                    "resized_no_shadow": resized_no_shadow_local,
                    "ood_mask": ood_mask_local,
                    "occlusion_mask": occlusion_mask_local,
                    "src_count": src_count_local,
                    "vis_count": vis_count_local,
                    "vis_ratio": vis_ratio_local,
                }
                render_bundle_cache[key] = bundle
                return bundle

            def _find_temporal_support(ood_id, center_frame_idx, image_h,
                                       image_w):
                src_min = int(self.camera_occlusion_temporal_min_source_px)
                drop_ratio = float(self.camera_occlusion_temporal_drop_ratio)
                support_ratio = float(
                    self.camera_occlusion_temporal_support_ratio)
                max_gap = int(self.camera_occlusion_temporal_max_gap)
                center_bundle = _load_render_bundle_for_frame(
                    ood_id, center_frame_idx, image_h, image_w)
                if center_bundle is None:
                    return False, None
                if center_bundle["src_count"] < src_min:
                    return False, None
                if center_bundle["vis_ratio"] > drop_ratio:
                    return False, None

                max_hops = max_gap + 1
                prev_idx = None
                next_idx = None

                for hop in range(1, max_hops + 1):
                    idx = center_frame_idx - hop
                    if idx < 0:
                        continue
                    cand = _load_render_bundle_for_frame(
                        ood_id, idx, image_h, image_w)
                    if cand is None:
                        continue
                    if cand["src_count"] >= src_min and cand[
                            "vis_ratio"] >= support_ratio:
                        prev_idx = idx
                        break

                for hop in range(1, max_hops + 1):
                    idx = center_frame_idx + hop
                    if idx >= frame_count:
                        continue
                    cand = _load_render_bundle_for_frame(
                        ood_id, idx, image_h, image_w)
                    if cand is None:
                        continue
                    if cand["src_count"] >= src_min and cand[
                            "vis_ratio"] >= support_ratio:
                        next_idx = idx
                        break

                if prev_idx is None or next_idx is None:
                    return False, None

                reason = (
                    f"vis_ratio={center_bundle['vis_ratio']:.3f} <= drop={drop_ratio:.3f}, "
                    f"support_frames={self.frame_map['camera'][prev_idx]} and {self.frame_map['camera'][next_idx]}"
                )
                return True, reason

            for frame_id, frame_rgb in enumerate(frame_images):
                back_image_direct = Image.fromarray(frame_rgb)
                back_image_wo_occlusion = Image.fromarray(frame_rgb)
                image_h, image_w = frame_rgb.shape[:2]
                all_ood_mask = np.zeros((image_h, image_w), dtype=bool)

                frame_camera_data = self._get_camera_entries_for_frame(
                    'camera', frame_id)
                camera_meta = frame_camera_data.get(camera_name)
                if camera_meta is None:
                    continue
                camera_point = [p[3] for p in camera_meta["camera_pose"][0:3]]

                instance_points_dict = {
                    ood_id: [p[3] for p in details['pose'][0:3]]
                    for ood_id, details in self.ood_details.items()
                    if self._has_valid_pose(details)
                }
                distances_dict = {
                    ood_id:
                    np.linalg.norm(np.array(inst_p) - np.array(camera_point))
                    for ood_id, inst_p in instance_points_dict.items()
                }
                sorted_indices = sorted(
                    distances_dict.keys(),
                    key=lambda ood_id: distances_dict[ood_id],
                    reverse=True)
                if self.max_objects_per_frame is not None and len(
                        sorted_indices) > self.max_objects_per_frame:

                    nearest_ids = sorted(
                        distances_dict.keys(),
                        key=lambda ood_id: distances_dict[ood_id],
                    )[:self.max_objects_per_frame]
                    sorted_indices = sorted(
                        nearest_ids,
                        key=lambda ood_id: distances_dict[ood_id],
                        reverse=True)

                if camera_name in self.semantic_labels_by_camera and frame_id < len(
                        self.semantic_labels_by_camera[camera_name]):
                    semantic_label = np.array(
                        self.semantic_labels_by_camera[camera_name][frame_id],
                        copy=True)
                    instance_label = np.array(
                        self.instance_labels_by_camera[camera_name][frame_id],
                        copy=True)
                else:
                    semantic_label = np.array(self.semantic_labels[frame_id],
                                              copy=True)
                    instance_label = np.array(self.instance_labels[frame_id],
                                              copy=True)

                for ood_id in sorted_indices:
                    bundle = _load_render_bundle_for_frame(
                        ood_id, frame_id, image_h, image_w)
                    if bundle is None:
                        continue

                    ood_image = bundle["ood_image"]
                    ood_array = np.array(bundle["ood_array"], copy=True)
                    resized_no_shadow = bundle["resized_no_shadow"]
                    ood_mask = np.array(bundle["ood_mask"], copy=True)
                    occlusion_mask = np.array(bundle["occlusion_mask"],
                                              copy=True)

                    if self.camera_occlusion_temporal_bridge:
                        should_bridge, bridge_reason = _find_temporal_support(
                            ood_id=ood_id,
                            center_frame_idx=frame_id,
                            image_h=image_h,
                            image_w=image_w,
                        )
                        if should_bridge:
                            frame_token = self.frame_map['camera'][frame_id]
                            print(
                                f"\033[33mTemporal occlusion bridge applied: "
                                f"camera={camera_name} frame={frame_token} ood={ood_id} "
                                f"({bridge_reason})\033[0m")
                            occlusion_mask = np.zeros_like(occlusion_mask,
                                                           dtype=np.uint8)

                    export_shadow_debug(
                        camera_name=camera_name,
                        frame_token=self.frame_map['camera'][frame_id],
                        ood_id=ood_id,
                        raw_rgb=frame_rgb,
                        full_rgba=ood_array,
                        no_shadow_rgba=resized_no_shadow,
                    )
                    back_image_wo_occlusion.paste(ood_image, (0, 0), ood_image)
                    ood_array[(occlusion_mask == 1) |
                              (occlusion_mask == 255)] = [0.0, 0.0, 0.0, 0.0]
                    masked_ood_image = Image.fromarray(
                        ood_array.astype(np.uint8))
                    back_image_direct.paste(masked_ood_image, (0, 0),
                                            masked_ood_image)

                    bbox_3d = self.ood_details.get(ood_id, {}).get('bbox_3d')
                    if (isinstance(bbox_3d, dict) and ('min' in bbox_3d)
                            and ('max' in bbox_3d)):
                        bbox_min = np.array(bbox_3d['min'])
                        bbox_max = np.array(bbox_3d['max'])
                        extrinsic_gl = self.vehicle_pose['camera'][
                            frame_id] @ np.asarray(camera_meta['extrinsic'])
                        bbox_min = (np.linalg.inv(extrinsic_gl) @ np.vstack(
                            (bbox_min.reshape(3, 1), [[1]]))).flatten()[:3]
                        bbox_max = (np.linalg.inv(extrinsic_gl) @ np.vstack(
                            (bbox_max.reshape(3, 1), [[1]]))).flatten()[:3]
                        corners_3d = get_3d_bbox_corners(*bbox_min, *bbox_max)
                        center_x, center_y, center_z, width, length, height, heading = get_bbox_properties(
                            corners_3d)

                        if 'camera_labels' not in self.ood_details[ood_id]:
                            self.ood_details[ood_id]['camera_labels'] = {}
                        if frame_id not in self.ood_details[ood_id][
                                'camera_labels']:
                            self.ood_details[ood_id]['camera_labels'][
                                frame_id] = {}
                        self.ood_details[ood_id]['camera_labels'][frame_id][
                            camera_name] = {
                                'center_x': center_x,
                                'center_y': center_y,
                                'center_z': center_z,
                                'width': width,
                                'length': length,
                                'height': height,
                                'heading': heading,
                            }

                    if occlusion_mask.shape != all_ood_mask.shape:
                        occlusion_mask = cv2.resize(
                            occlusion_mask.astype(np.uint8),
                            (image_w, image_h),
                            interpolation=cv2.INTER_NEAREST)
                    ood_mask[occlusion_mask == 1] = False
                    all_ood_mask = all_ood_mask | ood_mask
                    label_ood_mask = ood_mask
                    label_shape = semantic_label.shape[:2]
                    if label_shape != ood_mask.shape:
                        label_ood_mask = cv2.resize(
                            ood_mask.astype(np.uint8),
                            (label_shape[1], label_shape[0]),
                            interpolation=cv2.INTER_NEAREST) > 0
                    semantic_label[label_ood_mask] = ood_semantic_class
                    instance_label[label_ood_mask] = ood_id

                back_image_direct = back_image_direct.convert('RGB')
                back_image_direct.save(
                    f"{camera_output_dir}final_frame_00{self.frame_map['camera'][frame_id]}.jpg"
                )
                back_image_wo_occlusion = back_image_wo_occlusion.convert(
                    'RGB')
                back_image_wo_occlusion.save(
                    f"{camera_output_dir_pr}final_frame_00{self.frame_map['camera'][frame_id]}.jpg"
                )
                self.merged_images['camera'][camera_name].append(
                    np.array(back_image_direct))

                Image.fromarray((all_ood_mask * 255).astype(np.uint8)).save(
                    f"{camera_output_dir}final_frame_00{self.frame_map['camera'][frame_id]}_mask.png"
                )

                if semantic_label.ndim == 3 and semantic_label.shape[-1] == 1:
                    semantic_label = semantic_label.squeeze(-1)
                if instance_label.ndim == 3 and instance_label.shape[-1] == 1:
                    instance_label = instance_label.squeeze(-1)

                semantic_img = Image.fromarray(semantic_label.astype(np.uint8),
                                               mode='L')
                semantic_img.save(
                    f"{camera_output_dir}final_frame_00{self.frame_map['camera'][frame_id]}_sem_label.png"
                )
                instance_img = Image.fromarray(instance_label.astype(
                    np.uint16),
                                               mode='I;16')
                instance_img.save(
                    f"{camera_output_dir}final_frame_00{self.frame_map['camera'][frame_id]}_inst_label.png"
                )
                panoptic_divisor = int(
                    max(1, getattr(self, "panoptic_label_divisor", 1000)))
                panoptic_label = compose_export_camera_panoptic_label(
                    semantic_label,
                    instance_label,
                    divisor=panoptic_divisor,
                )
                if int(np.max(panoptic_label)) > np.iinfo(np.uint16).max:
                    print(
                        f"\033[33mWarning: camera panoptic export exceeds uint16 range "
                        f"(max={int(np.max(panoptic_label))}, divisor={panoptic_divisor}) for "
                        f"{camera_name} frame {self.frame_map['camera'][frame_id]}. "
                        f"Use semantic + instance labels or *_panoptic_id.png as source of truth.\033[0m"
                    )
                Image.fromarray(
                    panoptic_label.astype(np.uint16), mode='I;16'
                ).save(
                    f"{camera_output_dir}final_frame_00{self.frame_map['camera'][frame_id]}_panoptic_label.png"
                )
                Image.fromarray(
                    encode_export_panoptic_ids_to_rgb(panoptic_label),
                    mode='RGB',
                ).save(
                    f"{camera_output_dir}final_frame_00{self.frame_map['camera'][frame_id]}_panoptic_id.png"
                )
                color_map = camera_segmentation_utils._generate_color_map(
                    SEGMENTATION_COLOR_MAP)
                semantic_seg = camera_segmentation_utils.semantic_label_to_rgb(
                    semantic_label, color_map=color_map)
                Image.fromarray(semantic_seg).save(
                    f"{camera_output_dir}final_frame_00{self.frame_map['camera'][frame_id]}_seg_map.png"
                )
                panoptic_seg = camera_segmentation_utils.panoptic_label_to_rgb(
                    semantic_label,
                    instance_label.astype(np.int32),
                    color_map=color_map)
                Image.fromarray(panoptic_seg).save(
                    f"{camera_output_dir}final_frame_00{self.frame_map['camera'][frame_id]}_panoptic_seg_map.png"
                )

        return pipeline_dataset

    def merge_ood_into_image_improved(self,
                                      sensor,
                                      eval=False,
                                      render_mode='multi'):
        print('SAMPLER: Merging OOD object into image improved')
        ood_semantic_class = self.ood_semantic_class

        def get_3d_bbox_corners(xmin, ymin, zmin, xmax, ymax, zmax):
            return np.array([[xmin, ymin, zmin], [xmax, ymin, zmin],
                             [xmax, ymax, zmin], [xmin, ymax, zmin],
                             [xmin, ymin, zmax], [xmax, ymin, zmax],
                             [xmax, ymax, zmax], [xmin, ymax, zmax]])

        def project_points(points_3d, frame_id):

            X_cam = -points_3d[:, 1]
            Y_cam = -points_3d[:, 2]
            Z_cam = points_3d[:, 0]

            fx, fy, cx, cy = self.camera_data[sensor_type][frame_id][
                'projection']

            u = (fx * X_cam / Z_cam) + cx
            v = (fy * Y_cam / Z_cam) + cy

            return np.array([u, v]).T

        def draw_3d_bbox(image, corners_2d, color=(0, 255, 0), thickness=1):

            edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
                     (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

            for start, end in edges:
                pt1 = tuple(corners_2d[start])
                pt2 = tuple(corners_2d[end])
                cv2.line(image, pt1, pt2, color, thickness)

            return image

        def draw_2d_bbox(image,
                         center_x,
                         center_y,
                         length,
                         width,
                         color=(0, 255, 0),
                         thickness=1):

            x1 = int(center_x - 0.5 * length)
            y1 = int(center_y - 0.5 * width)
            x2 = int(center_x + 0.5 * length)
            y2 = int(center_y + 0.5 * width)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            return image

        def get_bbox_properties(corners):
            """
            Compute 3D bounding box properties from its 8 corners.

            Args:
                corners: (8, 3) array of box corners in standard order

            Returns:
                center_x, center_y, center_z, length, width, height, heading
            """

            center = np.mean(corners, axis=0)
            center_x, center_y, center_z = center

            v1 = corners[1] - corners[0]
            v2 = corners[3] - corners[0]
            v3 = corners[4] - corners[0]

            length = np.linalg.norm(v1)
            width = np.linalg.norm(v2)
            height = np.linalg.norm(v3)

            heading = np.arctan2(v1[1], v1[0])

            return center_x, center_y, center_z, length, width, height, heading

        def estimate_alpha_blur_radius(alpha_channel):

            alpha_np = np.array(alpha_channel, dtype=np.float32) / 255.0
            edges = sobel(alpha_np)

            if np.max(edges) < 1e-5:
                return 1

            threshold = 0.05 * np.max(edges)
            edge_mask = edges > threshold

            if not np.any(edge_mask):
                return 1

            edge_values = edges[edge_mask]
            edge_strength = np.percentile(edge_values, 90)

            max_possible = 4 * np.sqrt(2)
            normalized_strength = min(1.0, edge_strength / max_possible)

            blur_radius = 1 + 2 * np.exp(-3 * normalized_strength)
            return max(1, min(3, blur_radius))

        def bbox_based_blur_radius(alpha_channel):
            alpha_np = np.array(alpha_channel)
            y_indices, x_indices = np.where(alpha_np > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                return 1

            width = x_indices.max() - x_indices.min()
            height = y_indices.max() - y_indices.min()
            size = max(width, height)

            return max(1, (size * 0.01))

        def dynamic_blur_radius(alpha_channel):
            edge_based = estimate_alpha_blur_radius(alpha_channel)
            size_based = bbox_based_blur_radius(alpha_channel)
            weight = 0.75

            return int(weight * edge_based + (1 - weight) * size_based)

        pipeline_dataset = None
        for sensor_type in self.frames_with_seg.keys():
            print(sensor_type, len(self.frames_with_seg[sensor_type]))

            if len(self.frames_with_seg[sensor_type]) == 0:
                continue

            if sensor_type == 'lidar' and sensor != 'lidar':
                print("\033[33mSkipping lidar image merge stage.\033[0m")
                continue
            if sensor_type == 'camera' and not isinstance(
                    self.raw_images.get(sensor_type), dict):
                recovered_camera_images = self._recover_camera_raw_images_from_disk(
                )
                if recovered_camera_images is not None:
                    self.raw_images[sensor_type] = recovered_camera_images
            if sensor_type == 'camera' and isinstance(
                    self.raw_images.get(sensor_type), dict):
                camera_pipeline = self.merge_ood_into_all_camera_images(
                    eval=eval,
                    render_mode=render_mode,
                    ood_semantic_class=ood_semantic_class,
                )
                if camera_pipeline is not None:
                    pipeline_dataset = camera_pipeline
                continue
            if self.train:
                train_valid = "Train"
            else:
                train_valid = "Valid"
            base_image_path = self.image_path if self.image_path else f"Pipeline_2_{train_valid}"

            print(
                f"\033[32m✓ [merge_ood_into_image_improved] base_image_path={base_image_path}\033[0m"
            )

            if not eval:
                pipeline_dataset = f"{base_image_path}/{self.file_name}/merged_directly_shadow1_resized/"
                pipeline_dataset_type = f"{pipeline_dataset}{sensor_type}"
                pipeline_dataset_pr = f"{base_image_path}/{self.file_name}/merged_without_occlusion/"
                pipeline_dataset_pr_type = f"{pipeline_dataset_pr}{sensor_type}"
            else:
                pipeline_dataset = f"{base_image_path}/{self.file_name}/Evaluation/merged_directly_shadow1_resized/"
                pipeline_dataset_type = f"{pipeline_dataset}{sensor_type}"
                pipeline_dataset_pr = f"{base_image_path}/{self.file_name}/Evaluation/merged_without_occlusion/"
                pipeline_dataset_pr_type = f"{pipeline_dataset_pr}{sensor_type}"
            if render_mode == 'single':
                pipeline_dataset = f"{self.base_dir}/{self.file_name}/Single_Rendered_Images/"
                pipeline_dataset_type = f"{pipeline_dataset}{sensor_type}/"
                pipeline_dataset_pr = f"{self.base_dir}/{self.file_name}/Single_Rendered_Images_No_Occlusion/"
                pipeline_dataset_pr_type = f"{pipeline_dataset_pr}{sensor_type}/"

            os.makedirs(pipeline_dataset_type, exist_ok=True)
            os.makedirs(pipeline_dataset_pr_type, exist_ok=True)

            if sensor_type not in self.merged_images:
                self.merged_images[sensor_type] = []
            for frame_id, frame in enumerate(self.raw_images[sensor_type]):
                print(f'Merging frame_id {frame_id} of sensor {sensor_type}')
                back_image_direct = Image.fromarray(frame)
                back_image_wo_occlusion = Image.fromarray(frame)

                camera_point = [
                    p[3] for p in self.camera_data[sensor_type][frame_id]
                    ["camera_pose"][0:3]
                ]

                instance_points_dict = {
                    ood_id: [p[3] for p in details['pose'][0:3]]
                    for ood_id, details in self.ood_details.items()
                    if self._has_valid_pose(details)
                }

                distances_dict = {
                    ood_id:
                    np.linalg.norm(np.array(inst_p) - np.array(camera_point))
                    for ood_id, inst_p in instance_points_dict.items()
                }

                sorted_indices = sorted(
                    distances_dict.keys(),
                    key=lambda ood_id: distances_dict[ood_id],
                    reverse=True)

                all_ood_mask = np.zeros([1280, 1920], dtype=bool)
                if sensor_type == 'camera':
                    semantic_label = np.array(self.semantic_labels[frame_id],
                                              copy=True)
                    instance_label = np.array(self.instance_labels[frame_id],
                                              copy=True)
                for ood_id in sorted_indices:

                    occlusion_masks = self.occlusion_masks[ood_id][sensor_type]
                    if eval:
                        render_file = f"{self.base_dir}/{self.file_name}/Evaluation/Renders/ood_{ood_id}/{sensor_type}/frame_00{self.frame_map[sensor_type][frame_id]}.png"
                    else:
                        render_file = f"{self.base_dir}/{self.file_name}/Renders/ood_{ood_id}/{sensor_type}/frame_00{self.frame_map[sensor_type][frame_id]}.png"
                    ood_image = Image.open(render_file)
                    print(
                        f'Pasting OOD {ood_id} into frame {frame_id} of sensor {sensor_type}'
                    )
                    ood_array = np.array(ood_image)

                    occlusion_mask = occlusion_masks[frame_id]

                    occluded_pixels = np.sum((occlusion_mask == 1)
                                             | (occlusion_mask == 255))
                    ood_pixels = np.sum(ood_array[:, :, 3] > 100)
                    print(
                        f'  OOD {ood_id}: {ood_pixels} OOD pixels, {occluded_pixels} occluded pixels'
                    )

                    back_image_wo_occlusion.paste(ood_image, (0, 0), ood_image)

                    ood_array[(occlusion_mask == 1) |
                              (occlusion_mask == 255)] = [0.0, 0.0, 0.0, 0.0]

                    remaining_ood_pixels = np.sum(ood_array[:, :, 3] > 100)
                    print(
                        f'  After occlusion: {remaining_ood_pixels} visible OOD pixels remain'
                    )

                    ood_image = Image.fromarray(ood_array.astype(np.uint8))
                    back_image_direct.paste(ood_image, (0, 0), ood_image)

                    bbox_min = np.array(
                        self.ood_details[ood_id]['bbox_3d']['min'])
                    bbox_max = np.array(
                        self.ood_details[ood_id]['bbox_3d']['max'])

                    extrinsic_gl = self.vehicle_pose[sensor_type][
                        frame_id] @ self.camera_data[sensor_type][frame_id][
                            'extrinsic']
                    bbox_min = (np.linalg.inv(extrinsic_gl) @ np.vstack(
                        (bbox_min.reshape(3, 1), [[1]]))).flatten()[:3]
                    bbox_max = (np.linalg.inv(extrinsic_gl) @ np.vstack(
                        (bbox_max.reshape(3, 1), [[1]]))).flatten()[:3]

                    corners_3d = get_3d_bbox_corners(*bbox_min, *bbox_max)

                    if 'camera_labels' not in self.ood_details[ood_id].keys():
                        self.ood_details[ood_id]['camera_labels'] = {}
                    self.ood_details[ood_id]['camera_labels'][frame_id] = {}

                    center_x, center_y, center_z, width, length, height, heading = get_bbox_properties(
                        corners_3d)
                    self.ood_details[ood_id]['camera_labels'][frame_id][
                        'center_x'] = center_x
                    self.ood_details[ood_id]['camera_labels'][frame_id][
                        'center_y'] = center_y
                    self.ood_details[ood_id]['camera_labels'][frame_id][
                        'center_z'] = center_z
                    self.ood_details[ood_id]['camera_labels'][frame_id][
                        'width'] = width
                    self.ood_details[ood_id]['camera_labels'][frame_id][
                        'length'] = length
                    self.ood_details[ood_id]['camera_labels'][frame_id][
                        'height'] = height
                    self.ood_details[ood_id]['camera_labels'][frame_id][
                        'heading'] = heading

                    corners_2d = project_points(corners_3d, frame_id)

                    corners_2d = corners_2d.astype(int)

                    no_shadow_file = self._resolve_no_shadow_render_path(
                        ood_id,
                        sensor_type,
                        frame_id,
                        eval=eval,
                    )
                    no_shadow_image = Image.open(no_shadow_file)
                    ood_no_shadow_array = np.array(no_shadow_image)
                    ood_mask = np.zeros([1280, 1920], dtype=int)
                    ood_mask[ood_no_shadow_array[:, :, 3] != 0] = 1
                    ood_mask = self._extract_alpha_mask_with_fallback(
                        ood_no_shadow_array,
                        hard_alpha_threshold=100,
                        min_pixels=int(max(1,
                                           self.required_render_min_pixels)),
                    )

                    ood_mask[occlusion_mask == 1] = False

                    individual_ood_mask = ood_mask.copy()

                    all_ood_mask = all_ood_mask | individual_ood_mask

                    all_ood_mask_img = Image.fromarray(
                        (all_ood_mask * 255).astype(np.uint8))
                    all_ood_mask_img.save(
                        f"{pipeline_dataset_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}_mask.png"
                    )

                    if sensor_type == 'camera':
                        semantic_label[ood_mask] = ood_semantic_class
                        instance_label[ood_mask] = ood_id
                back_image_direct = back_image_direct.convert('RGB')
                back_image_direct.save(
                    f"{pipeline_dataset_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}.jpg"
                )
                back_image_wo_occlusion = back_image_wo_occlusion.convert(
                    'RGB')
                back_image_wo_occlusion.save(
                    f"{pipeline_dataset_pr_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}.jpg"
                )
                self.merged_images[sensor_type].append(
                    np.array(back_image_direct))

                output_path = f"{pipeline_dataset_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}_3d_bbox.png"

                if sensor_type == 'camera':
                    if semantic_label.ndim == 3 and semantic_label.shape[
                            -1] == 1:
                        semantic_label = semantic_label.squeeze(-1)
                    if instance_label.ndim == 3 and instance_label.shape[
                            -1] == 1:
                        instance_label = instance_label.squeeze(-1)

                    semantic_label_uint8 = semantic_label.astype(np.uint8)
                    semantic_img = Image.fromarray(semantic_label_uint8,
                                                   mode='L')
                    semantic_img.save(
                        f"{pipeline_dataset_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}_sem_label.png"
                    )

                    instance_label_u16 = instance_label.astype(np.uint16)
                    instance_img = Image.fromarray(instance_label_u16,
                                                   mode='I;16')
                    instance_img.save(
                        f"{pipeline_dataset_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}_inst_label.png"
                    )

                    panoptic_divisor = int(
                        max(1, getattr(self, "panoptic_label_divisor", 1000)))
                    panoptic_label = compose_export_camera_panoptic_label(
                        semantic_label,
                        instance_label,
                        divisor=panoptic_divisor,
                    )
                    if int(np.max(panoptic_label)) > np.iinfo(np.uint16).max:
                        print(
                            f"\033[33mWarning: camera panoptic export exceeds uint16 range "
                            f"(max={int(np.max(panoptic_label))}, divisor={panoptic_divisor}) for "
                            f"{sensor_type} frame {self.frame_map[sensor_type][frame_id]}. "
                            f"Use semantic + instance labels or *_panoptic_id.png as source of truth.\033[0m"
                        )
                    panoptic_img = Image.fromarray(panoptic_label.astype(
                        np.uint16),
                                                   mode='I;16')
                    panoptic_img.save(
                        f"{pipeline_dataset_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}_panoptic_label.png"
                    )
                    Image.fromarray(
                        encode_export_panoptic_ids_to_rgb(panoptic_label),
                        mode='RGB',
                    ).save(
                        f"{pipeline_dataset_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}_panoptic_id.png"
                    )

                    COLOR_MAP = camera_segmentation_utils._generate_color_map(
                        SEGMENTATION_COLOR_MAP)
                    semantic_seg = camera_segmentation_utils.semantic_label_to_rgb(
                        semantic_label, color_map=COLOR_MAP)
                    Image.fromarray(semantic_seg).save(
                        f"{pipeline_dataset_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}_seg_map.png"
                    )

                    panoptic_seg = camera_segmentation_utils.panoptic_label_to_rgb(
                        semantic_label,
                        instance_label.astype(np.int32),
                        color_map=COLOR_MAP)
                    Image.fromarray(panoptic_seg).save(
                        f"{pipeline_dataset_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}_panoptic_seg_map.png"
                    )

        self.master[self.file_name]["ood_merged_images"] = pipeline_dataset

        if self.enable_pct:
            self.improve_merged_images(eval=eval, render_mode=render_mode)
        else:
            print(
                "\033[33mSkipping PCT-Net harmonization (--enable_pct False)\033[0m"
            )

    def improve_merged_images(self, eval=False, render_mode='multi'):
        """
        Improve merged images using PCT-Net harmonization.
        Runs image harmonization models (ViT_pct and CNN_pct) on the merged images.
        """
        if not self.enable_pct:
            print(
                "\033[33mPCT-Net harmonization disabled. Returning early.\033[0m"
            )
            return

        probe_cmd = [
            self.pct_python_executable,
            "-c",
            ("import json, torch; "
             "cuda=bool(torch.cuda.is_available()); "
             "count=int(torch.cuda.device_count() if cuda else 0); "
             "print(json.dumps({'cuda_available': cuda, 'visible_gpu_count': count}))"
             ),
        ]
        try:
            probe_result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
                cwd="./PCT-Net-Image-Harmonization",
            )
            probe_lines = [
                line.strip() for line in probe_result.stdout.splitlines()
                if line.strip()
            ]
            probe_payload = json.loads(probe_lines[-1]) if probe_lines else {}
        except Exception as probe_exc:
            raise RuntimeError(
                f"Failed to probe PCT-Net torch/CUDA environment via subprocess: {probe_exc}"
            ) from probe_exc

        cuda_available = bool(probe_payload.get("cuda_available", False))
        visible_gpu_count = int(probe_payload.get("visible_gpu_count", 0))
        print(
            f"\033[36mPCT-Net device config: requested={self.pct_gpu_device}, "
            f"cuda_available={cuda_available}, visible_gpus={visible_gpu_count}\033[0m"
        )
        if self.pct_require_gpu and not cuda_available:
            raise RuntimeError(
                "PCT-Net is configured to require GPU, but torch.cuda.is_available() is False."
            )

        print(
            "\033[36mSAMPLER: Improving merged images with PCT-Net harmonization\033[0m"
        )

        if self.train:
            train_valid = "Train"
        else:
            train_valid = "Valid"

        base_image_path = self.image_path if self.image_path else f"Pipeline_2_{train_valid}"

        if not eval:
            base_path = f"{base_image_path}/{self.file_name}/merged_directly_shadow1_resized"
        else:
            base_path = f"{base_image_path}/{self.file_name}/Evaluation/merged_directly_shadow1_resized"
        if render_mode == 'single':
            base_path = f"{self.base_dir}/{self.file_name}/Single_Rendered_Images"

        if not os.path.exists(base_path):
            print(
                f"\033[33mWarning: Base path does not exist: {base_path}\033[0m"
            )
            return

        sensors_to_process = ['camera']

        model_specs = []
        if self.pct_model in {'vit', 'both'}:
            model_specs.append(
                ('ViT_pct', "./pretrained_models/PCTNet_ViT.pth"))
        if self.pct_model in {'cnn', 'both'}:
            model_specs.append(
                ('CNN_pct', "./pretrained_models/PCTNet_CNN.pth"))
        print(
            f"\033[36mPCT model selection: {self.pct_model} ({[m[0] for m in model_specs]})\033[0m"
        )

        for sensor in sensors_to_process:
            sensor_path = os.path.join(base_path, sensor)

            if not os.path.isdir(sensor_path):
                print(f"\033[33mSkipping {sensor}: directory not found\033[0m")
                continue

            processing_targets = []
            top_level_images = [
                name for name in os.listdir(sensor_path)
                if name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if top_level_images:
                processing_targets.append((sensor, sensor_path, False))
            else:
                camera_subdirs = []
                for entry in sorted(os.listdir(sensor_path)):
                    entry_path = os.path.join(sensor_path, entry)
                    if not os.path.isdir(entry_path):
                        continue
                    has_images = any(name.lower().endswith(('.jpg', '.jpeg',
                                                            '.png'))
                                     for name in os.listdir(entry_path))
                    if has_images:
                        camera_subdirs.append(entry)
                if camera_subdirs:
                    processing_targets.append(
                        (f"{sensor}/*", sensor_path, True))
            if not processing_targets:
                print(f"\033[33mSkipping {sensor}: no images found\033[0m")
                continue

            for model, weights in model_specs:

                if not eval:
                    save_dir = f"{base_image_path}/{self.file_name}/merged_directly_shadow1_resized_{model}/{sensor}"
                else:
                    save_dir = f"{base_image_path}/{self.file_name}/Evaluation/merged_directly_shadow1_resized_{model}/{sensor}"
                if render_mode == 'single':
                    save_dir = f"{self.base_dir}/{self.file_name}/Single_Rendered_Images_{model}/{sensor}"

                if os.path.exists(save_dir):
                    print(
                        f"\033[33mCleaning existing directory: {save_dir}\033[0m"
                    )
                    for filename in os.listdir(save_dir):
                        file_path = os.path.join(save_dir, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(
                                    file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            print(
                                f'\033[31mFailed to delete {file_path}. Reason: {e}\033[0m'
                            )
                else:
                    os.makedirs(save_dir, exist_ok=True)

                for target_label, target_path, recursive in processing_targets:
                    print(
                        f"\033[32mProcessing {target_label} with {model}\033[0m"
                    )

                    python_path = self.pct_python_executable

                    sensor_path_abs = os.path.abspath(target_path)
                    save_dir_abs = os.path.abspath(save_dir)

                    command = [
                        python_path, 'scripts/evaluate_folder.py',
                        sensor_path_abs, model, '--weights', weights, '--gpu',
                        self.pct_gpu_device, '--require_gpu', '--save_dir',
                        save_dir_abs
                    ]
                    if recursive:
                        command.append('--recursive')

                    try:
                        print(f"  Running command: {' '.join(command)}")
                        result = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            check=True,
                            cwd='./PCT-Net-Image-Harmonization')

                        print(
                            f"\033[32m  {model} processing completed successfully\033[0m"
                        )
                        if result.stdout:
                            print(f"  STDOUT: {result.stdout[:500]}")

                    except subprocess.CalledProcessError as e:
                        print(f"\033[31m  Error executing {model}: {e}\033[0m")
                        print(f"  STDOUT: {e.stdout}")
                        print(f"  STDERR: {e.stderr}")

                    except FileNotFoundError:
                        print(
                            f"\033[31m  Error: Python executable or script not found\033[0m"
                        )
                        print(f"  Python path: {python_path}")
                        print(
                            f"  Working directory: ./PCT-Net-Image-Harmonization"
                        )

                if sensor == 'camera' and self.pct_temporal_consistency != 'off':
                    try:
                        print(
                            f"\033[36mApplying PCT temporal consistency "
                            f"(mode={self.pct_temporal_consistency}) on {save_dir}\033[0m"
                        )
                        self._apply_pct_temporal_consistency_to_folder(
                            harmonized_camera_root=save_dir,
                            eval=eval,
                        )
                    except Exception as temporal_exc:
                        print(
                            f"\033[33mWarning: temporal PCT consistency failed for {save_dir}: "
                            f"{temporal_exc}\033[0m")
                if sensor == 'camera' and self.pct_stage2_global_match:
                    try:
                        print(
                            f"\033[36mApplying PCT stage-2 global color match "
                            f"(strength={self.pct_stage2_strength:.2f}) on {save_dir}\033[0m"
                        )
                        self._apply_pct_stage2_global_match_to_folder(
                            harmonized_camera_root=save_dir,
                            eval=eval,
                        )
                    except Exception as stage2_exc:
                        print(
                            f"\033[33mWarning: PCT stage-2 global color match failed for {save_dir}: "
                            f"{stage2_exc}\033[0m")

        print("\033[32mFinished improving images\033[0m")

    @staticmethod
    def _pct_extract_frame_token_from_filename(filename):
        """Extract numeric frame token from harmonized image filename."""
        stem = os.path.basename(filename)
        match = re.match(
            r"^final_frame_00(\d+)_harm\.(jpg|jpeg|png)$",
            stem,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        token = match.group(1)
        try:
            return str(int(token))
        except Exception:
            token = token.lstrip("0")
            return token if token else "0"

    def _pct_discover_camera_harmonized_dirs(self, harmonized_camera_root):
        """Return camera_name -> list[(frame_idx, frame_token, image_path)] for harmonized outputs."""
        token_to_idx = {}
        camera_tokens = self.frame_map.get('camera', [])
        for idx, token in enumerate(camera_tokens):
            try:
                norm = str(int(token))
            except Exception:
                norm = str(token).strip()
            if norm:
                token_to_idx[norm] = idx
            token_to_idx[str(token)] = idx

        camera_dirs = {}
        root = Path(harmonized_camera_root)
        if not root.exists():
            return camera_dirs

        def collect_entries(camera_name, folder_path):
            entries = []
            for name in os.listdir(folder_path):
                token = self._pct_extract_frame_token_from_filename(name)
                if token is None:
                    continue
                frame_idx = token_to_idx.get(token)
                if frame_idx is None:

                    try:
                        frame_idx = token_to_idx.get(str(int(token)))
                    except Exception:
                        frame_idx = None
                if frame_idx is None:
                    continue
                entries.append(
                    (int(frame_idx), token, os.path.join(folder_path, name)))
            entries.sort(key=lambda x: x[0])
            if entries:
                camera_dirs[camera_name] = entries

        subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
        if subdirs:
            for subdir in subdirs:
                collect_entries(subdir.name, str(subdir))
            if camera_dirs:
                return camera_dirs

        collect_entries("front", str(root))
        return camera_dirs

    def _pct_get_visible_ood_mask(
        self,
        ood_id,
        camera_name,
        frame_idx,
        image_h,
        image_w,
        eval=False,
        mask_cache=None,
    ):
        """Get per-ood visible mask (no-shadow alpha minus occlusion) for one camera frame."""
        cache = mask_cache if mask_cache is not None else {}
        cache_key = (
            int(ood_id),
            str(camera_name),
            int(frame_idx),
            int(image_h),
            int(image_w),
            bool(eval),
        )
        if cache_key in cache:
            return cache[cache_key]

        if frame_idx < 0 or frame_idx >= len(self.frame_map.get('camera', [])):
            cache[cache_key] = None
            return None

        frame_token = self.frame_map['camera'][frame_idx]
        base_render_dir = (
            f"{self.base_dir}/{self.file_name}/Evaluation/Renders/ood_{ood_id}/camera"
            if eval else
            f"{self.base_dir}/{self.file_name}/Renders/ood_{ood_id}/camera")
        no_shadow_name = f"no_shadow_frame_00{frame_token}.png"
        candidate_paths = [
            os.path.join(base_render_dir, camera_name, no_shadow_name),
            os.path.join(base_render_dir, no_shadow_name),
        ]
        no_shadow_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                no_shadow_path = path
                break
        if no_shadow_path is None:
            cache[cache_key] = None
            return None

        try:
            ood_no_shadow_array = np.asarray(Image.open(no_shadow_path))
        except Exception:
            cache[cache_key] = None
            return None

        ood_mask = self._extract_alpha_mask_with_fallback(
            ood_no_shadow_array,
            hard_alpha_threshold=100,
            min_pixels=int(max(1, self.required_render_min_pixels)),
        )
        if ood_mask.shape != (image_h, image_w):
            ood_mask = cv2.resize(
                ood_mask.astype(np.uint8),
                (image_w, image_h),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        else:
            ood_mask = ood_mask.astype(bool)

        occlusion_mask = np.zeros((image_h, image_w), dtype=np.uint8)
        sensor_masks = self.occlusion_masks.get(ood_id, {}).get('camera')
        if isinstance(sensor_masks, dict):
            camera_masks = sensor_masks.get(camera_name)
            if camera_masks is not None and frame_idx < len(camera_masks):
                occlusion_mask = np.asarray(camera_masks[frame_idx])
        elif isinstance(sensor_masks, list) and frame_idx < len(sensor_masks):
            occlusion_mask = np.asarray(sensor_masks[frame_idx])

        if np.count_nonzero(occlusion_mask) == 0:
            occlusion_mask = self._compute_camera_occlusion_mask_from_lidar(
                ood_id,
                frame_idx,
                camera_name,
                ood_mask,
            )
        if occlusion_mask.shape != (image_h, image_w):
            occlusion_mask = cv2.resize(
                occlusion_mask.astype(np.uint8),
                (image_w, image_h),
                interpolation=cv2.INTER_NEAREST,
            )
        visible_mask = ood_mask & (occlusion_mask == 0)
        cache[cache_key] = visible_mask
        return visible_mask

    @staticmethod
    def _pct_compute_lab_stats(image_bgr, mask_bool):
        """Compute LAB mean/std for pixels inside mask."""
        if image_bgr is None or mask_bool is None or np.count_nonzero(
                mask_bool) == 0:
            return None
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels = lab[mask_bool]
        if pixels.size == 0:
            return None
        mean = np.mean(pixels, axis=0).astype(np.float32)
        std = np.std(pixels, axis=0).astype(np.float32)
        std = np.maximum(std, 1.0)
        return mean, std

    def _pct_apply_lab_affine_with_feather(self, image_bgr, mask_bool, gain,
                                           bias):
        """Apply LAB affine correction on masked region with feathered blending."""
        if image_bgr is None or mask_bool is None or np.count_nonzero(
                mask_bool) == 0:
            return image_bgr
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        corrected_lab = lab.copy()
        for channel_idx in range(3):
            channel = corrected_lab[:, :, channel_idx]
            channel_vals = channel[mask_bool]
            if channel_vals.size == 0:
                continue
            channel[mask_bool] = channel_vals * float(
                gain[channel_idx]) + float(bias[channel_idx])
        corrected_lab = np.clip(corrected_lab, 0.0, 255.0).astype(np.uint8)
        corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

        sigma = float(self.pct_temporal_feather_sigma)
        if sigma <= 0.0:
            out = image_bgr.copy()
            out[mask_bool] = corrected_bgr[mask_bool]
            return out

        alpha = cv2.GaussianBlur(mask_bool.astype(np.float32), (0, 0),
                                 sigmaX=sigma,
                                 sigmaY=sigma)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        blended = (image_bgr.astype(np.float32) * (1.0 - alpha) +
                   corrected_bgr.astype(np.float32) * alpha)
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def _apply_pct_temporal_consistency_to_folder(self,
                                                  harmonized_camera_root,
                                                  eval=False):
        """Temporal color-consistency pass for harmonized camera outputs."""
        camera_tracks = self._pct_discover_camera_harmonized_dirs(
            harmonized_camera_root)
        if not camera_tracks:
            print(
                f"\033[33mNo harmonized camera frames found under {harmonized_camera_root}. Skipping temporal PCT consistency.\033[0m"
            )
            return

        valid_ood_ids = [
            int(ood_id) for ood_id, details in self.ood_details.items()
            if self._has_valid_pose(details)
        ]
        valid_ood_ids = sorted(valid_ood_ids)
        if len(valid_ood_ids) == 0:
            print(
                "\033[33mNo valid OOD poses available. Skipping temporal PCT consistency.\033[0m"
            )
            return

        min_pixels = int(self.pct_temporal_min_pixels)
        ema = float(self.pct_temporal_ema)
        gain_min = float(self.pct_temporal_gain_clip_min)
        gain_max = float(self.pct_temporal_gain_clip_max)
        mode = self.pct_temporal_consistency
        mask_cache = {}
        debug_summary = {
            "mode": mode,
            "min_pixels": min_pixels,
            "ema": ema,
            "gain_clip": [gain_min, gain_max],
            "cameras": {},
            "global_refs": {},
        }

        image_cache = {}

        def load_image(path):
            if path in image_cache:
                return image_cache[path]
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image_cache[path] = image
            return image

        global_refs = {}
        if mode == 'global':
            for ood_id in valid_ood_ids:
                best = None
                for camera_name, entries in camera_tracks.items():
                    for frame_idx, frame_token, image_path in entries:
                        image = load_image(image_path)
                        if image is None:
                            continue
                        image_h, image_w = image.shape[:2]
                        visible_mask = self._pct_get_visible_ood_mask(
                            ood_id=ood_id,
                            camera_name=camera_name,
                            frame_idx=frame_idx,
                            image_h=image_h,
                            image_w=image_w,
                            eval=eval,
                            mask_cache=mask_cache,
                        )
                        if visible_mask is None:
                            continue
                        pixel_count = int(np.count_nonzero(visible_mask))
                        if pixel_count < min_pixels:
                            continue
                        stats = self._pct_compute_lab_stats(
                            image, visible_mask)
                        if stats is None:
                            continue
                        if best is None or pixel_count > best["pixel_count"]:
                            best = {
                                "mean": stats[0],
                                "std": stats[1],
                                "pixel_count": pixel_count,
                                "camera": camera_name,
                                "frame_token": frame_token,
                            }
                if best is not None:
                    global_refs[ood_id] = best
                    debug_summary["global_refs"][str(ood_id)] = {
                        "camera": best["camera"],
                        "frame_token": best["frame_token"],
                        "pixel_count": int(best["pixel_count"]),
                    }

        for camera_name, entries in camera_tracks.items():
            camera_debug = {
                "frames": len(entries),
                "ood_refs": {},
                "corrected_frames": 0,
                "corrections": 0,
            }

            per_camera_refs = {}
            if mode == 'per_camera':
                for ood_id in valid_ood_ids:
                    best = None
                    for frame_idx, frame_token, image_path in entries:
                        image = load_image(image_path)
                        if image is None:
                            continue
                        image_h, image_w = image.shape[:2]
                        visible_mask = self._pct_get_visible_ood_mask(
                            ood_id=ood_id,
                            camera_name=camera_name,
                            frame_idx=frame_idx,
                            image_h=image_h,
                            image_w=image_w,
                            eval=eval,
                            mask_cache=mask_cache,
                        )
                        if visible_mask is None:
                            continue
                        pixel_count = int(np.count_nonzero(visible_mask))
                        if pixel_count < min_pixels:
                            continue
                        stats = self._pct_compute_lab_stats(
                            image, visible_mask)
                        if stats is None:
                            continue
                        if best is None or pixel_count > best["pixel_count"]:
                            best = {
                                "mean": stats[0],
                                "std": stats[1],
                                "pixel_count": pixel_count,
                                "frame_token": frame_token,
                            }
                    if best is not None:
                        per_camera_refs[ood_id] = best
                        camera_debug["ood_refs"][str(ood_id)] = {
                            "frame_token": best["frame_token"],
                            "pixel_count": int(best["pixel_count"]),
                        }

            ema_state = {}
            for frame_idx, frame_token, image_path in entries:
                image = load_image(image_path)
                if image is None:
                    continue
                image_h, image_w = image.shape[:2]
                corrected = image.copy()
                frame_corrections = 0

                for ood_id in valid_ood_ids:
                    if mode == 'global':
                        ref_bundle = global_refs.get(ood_id)
                    else:
                        ref_bundle = per_camera_refs.get(ood_id)
                    if ref_bundle is None:
                        continue

                    visible_mask = self._pct_get_visible_ood_mask(
                        ood_id=ood_id,
                        camera_name=camera_name,
                        frame_idx=frame_idx,
                        image_h=image_h,
                        image_w=image_w,
                        eval=eval,
                        mask_cache=mask_cache,
                    )
                    if visible_mask is None:
                        continue
                    pixel_count = int(np.count_nonzero(visible_mask))
                    if pixel_count < min_pixels:
                        continue

                    cur_stats = self._pct_compute_lab_stats(
                        corrected, visible_mask)
                    if cur_stats is None:
                        continue
                    cur_mean, cur_std = cur_stats
                    ref_mean = ref_bundle["mean"]
                    ref_std = ref_bundle["std"]

                    gain_raw = ref_std / np.maximum(cur_std, 1.0)
                    gain_raw = np.clip(gain_raw, gain_min,
                                       gain_max).astype(np.float32)
                    bias_raw = ref_mean - gain_raw * cur_mean

                    prev_state = ema_state.get(ood_id)
                    if prev_state is None:
                        gain = gain_raw
                        bias = bias_raw
                    else:
                        gain = (ema * prev_state[0] +
                                (1.0 - ema) * gain_raw).astype(np.float32)
                        bias = (ema * prev_state[1] +
                                (1.0 - ema) * bias_raw).astype(np.float32)
                    ema_state[ood_id] = (gain, bias)

                    corrected = self._pct_apply_lab_affine_with_feather(
                        corrected,
                        visible_mask,
                        gain,
                        bias,
                    )
                    frame_corrections += 1
                    camera_debug["corrections"] += 1

                if frame_corrections > 0:
                    camera_debug["corrected_frames"] += 1
                    cv2.imwrite(image_path, corrected)
                    image_cache[image_path] = corrected

            debug_summary["cameras"][camera_name] = camera_debug

        if self.pct_temporal_save_debug:
            summary_path = os.path.join(
                harmonized_camera_root,
                "pct_temporal_consistency_summary.json")
            try:
                with open(summary_path, "w") as f:
                    json.dump(debug_summary, f, indent=2)
                print(
                    f"\033[36mSaved PCT temporal consistency debug summary: {summary_path}\033[0m"
                )
            except Exception as exc:
                print(
                    f"\033[33mWarning: failed to save PCT temporal consistency summary: {exc}\033[0m"
                )

    def _pct_get_union_visible_mask_for_frame(
        self,
        camera_name,
        frame_idx,
        image_h,
        image_w,
        valid_ood_ids,
        eval=False,
        mask_cache=None,
        union_cache=None,
    ):
        """Return union visible mask of all valid OODs for one camera frame."""
        cache = union_cache if union_cache is not None else {}
        cache_key = (
            str(camera_name),
            int(frame_idx),
            int(image_h),
            int(image_w),
            bool(eval),
            tuple(valid_ood_ids),
        )
        if cache_key in cache:
            return cache[cache_key]

        union_mask = np.zeros((image_h, image_w), dtype=bool)
        for ood_id in valid_ood_ids:
            visible_mask = self._pct_get_visible_ood_mask(
                ood_id=ood_id,
                camera_name=camera_name,
                frame_idx=frame_idx,
                image_h=image_h,
                image_w=image_w,
                eval=eval,
                mask_cache=mask_cache,
            )
            if visible_mask is None:
                continue
            union_mask |= visible_mask.astype(bool)
        cache[cache_key] = union_mask
        return union_mask

    def _apply_pct_stage2_global_match_to_folder(self,
                                                 harmonized_camera_root,
                                                 eval=False):
        """Stage-2 sequence-level color stabilization on union visible OOD masks."""
        camera_tracks = self._pct_discover_camera_harmonized_dirs(
            harmonized_camera_root)
        if not camera_tracks:
            print(
                f"\033[33mNo harmonized camera frames found under {harmonized_camera_root}. Skipping PCT stage-2 global match.\033[0m"
            )
            return

        valid_ood_ids = [
            int(ood_id) for ood_id, details in self.ood_details.items()
            if self._has_valid_pose(details)
        ]
        valid_ood_ids = sorted(valid_ood_ids)
        if len(valid_ood_ids) == 0:
            print(
                "\033[33mNo valid OOD poses available. Skipping PCT stage-2 global match.\033[0m"
            )
            return

        min_pixels = int(self.pct_stage2_min_pixels)
        strength = float(self.pct_stage2_strength)
        gain_min = float(self.pct_stage2_gain_clip_min)
        gain_max = float(self.pct_stage2_gain_clip_max)
        if strength <= 0.0:
            print(
                "\033[33mPCT stage-2 strength is 0. Skipping global match.\033[0m"
            )
            return

        mask_cache = {}
        union_cache = {}
        debug_summary = {
            "min_pixels": min_pixels,
            "strength": strength,
            "gain_clip": [gain_min, gain_max],
            "reference_pixels": 0,
            "cameras": {},
        }

        total_pixels = 0
        lab_sum = np.zeros(3, dtype=np.float64)
        lab_sumsq = np.zeros(3, dtype=np.float64)
        for camera_name, entries in camera_tracks.items():
            for frame_idx, _frame_token, image_path in entries:
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                image_h, image_w = image.shape[:2]
                union_mask = self._pct_get_union_visible_mask_for_frame(
                    camera_name=camera_name,
                    frame_idx=frame_idx,
                    image_h=image_h,
                    image_w=image_w,
                    valid_ood_ids=valid_ood_ids,
                    eval=eval,
                    mask_cache=mask_cache,
                    union_cache=union_cache,
                )
                pixel_count = int(np.count_nonzero(union_mask))
                if pixel_count < min_pixels:
                    continue
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
                pixels = lab[union_mask]
                if pixels.size == 0:
                    continue
                total_pixels += int(pixels.shape[0])
                lab_sum += np.sum(pixels, axis=0, dtype=np.float64)
                lab_sumsq += np.sum(np.square(pixels, dtype=np.float64),
                                    axis=0,
                                    dtype=np.float64)

        if total_pixels < min_pixels:
            print(
                "\033[33mInsufficient visible OOD pixels for PCT stage-2 reference. "
                "Skipping global match.\033[0m")
            return

        ref_mean = (lab_sum / float(total_pixels)).astype(np.float32)
        ref_var = (lab_sumsq / float(total_pixels)) - np.square(
            ref_mean, dtype=np.float64)
        ref_var = np.maximum(ref_var, 1.0)
        ref_std = np.sqrt(ref_var).astype(np.float32)
        debug_summary["reference_pixels"] = int(total_pixels)
        debug_summary["reference_mean_lab"] = [
            float(x) for x in ref_mean.tolist()
        ]
        debug_summary["reference_std_lab"] = [
            float(x) for x in ref_std.tolist()
        ]

        for camera_name, entries in camera_tracks.items():
            camera_debug = {
                "frames": len(entries),
                "corrected_frames": 0,
                "skipped_low_pixels": 0,
            }
            for frame_idx, frame_token, image_path in entries:
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                image_h, image_w = image.shape[:2]
                union_mask = self._pct_get_union_visible_mask_for_frame(
                    camera_name=camera_name,
                    frame_idx=frame_idx,
                    image_h=image_h,
                    image_w=image_w,
                    valid_ood_ids=valid_ood_ids,
                    eval=eval,
                    mask_cache=mask_cache,
                    union_cache=union_cache,
                )
                pixel_count = int(np.count_nonzero(union_mask))
                if pixel_count < min_pixels:
                    camera_debug["skipped_low_pixels"] += 1
                    continue

                cur_stats = self._pct_compute_lab_stats(image, union_mask)
                if cur_stats is None:
                    camera_debug["skipped_low_pixels"] += 1
                    continue
                cur_mean, cur_std = cur_stats
                gain = ref_std / np.maximum(cur_std, 1.0)
                gain = np.clip(gain, gain_min, gain_max).astype(np.float32)
                bias = ref_mean - gain * cur_mean

                corrected = self._pct_apply_lab_affine_with_feather(
                    image_bgr=image,
                    mask_bool=union_mask,
                    gain=gain,
                    bias=bias,
                )
                if strength < 1.0:
                    corrected = np.clip(
                        image.astype(np.float32) * (1.0 - strength) +
                        corrected.astype(np.float32) * strength,
                        0.0,
                        255.0,
                    ).astype(np.uint8)
                cv2.imwrite(image_path, corrected)
                camera_debug["corrected_frames"] += 1
            debug_summary["cameras"][camera_name] = camera_debug

        if self.pct_stage2_save_debug:
            summary_path = os.path.join(
                harmonized_camera_root, "pct_stage2_global_match_summary.json")
            try:
                with open(summary_path, "w") as f:
                    json.dump(debug_summary, f, indent=2)
                print(
                    f"\033[36mSaved PCT stage-2 global match summary: {summary_path}\033[0m"
                )
            except Exception as exc:
                print(
                    f"\033[33mWarning: failed to save PCT stage-2 summary: {exc}\033[0m"
                )

    def get_consistent_colors(self):
        """Apply color consistency correction to harmonized images across all frames from both sensors."""
        print("Starting color consistency processing...")

        train_valid = "Train" if self.train else "Valid"

        base_image_path = self.image_path if self.image_path else f"Pipeline_2_{train_valid}"

        vit_folder = f"{base_image_path}/{self.file_name}/merged_directly_shadow1_resized_ViT_pct"
        mask_folder = f"{base_image_path}/{self.file_name}/merged_directly_shadow1_resized"

        all_object_pixels = []
        sensor_data = {}

        for sensor in ['camera', 'lidar']:
            sensor_path = os.path.join(vit_folder, sensor)
            mask_path = os.path.join(mask_folder, sensor)

            if not os.path.exists(sensor_path):
                print(f"Warning: Path does not exist: {sensor_path}")
                continue

            if not os.path.exists(mask_path):
                print(f"Warning: Mask path does not exist: {mask_path}")
                continue

            print(f"Collecting pixels from sensor: {sensor}")

            try:
                image_paths = [
                    os.path.join(sensor_path, img)
                    for img in sorted(os.listdir(sensor_path))
                ]
                mask_paths = [
                    os.path.join(mask_path,
                                 img.replace('_harm.jpg', '_mask.png'))
                    for img in sorted(os.listdir(sensor_path))
                ]
            except FileNotFoundError as e:
                print(f"File not found: {e}")
                continue

            sensor_data[sensor] = {
                'image_paths': image_paths,
                'mask_paths': mask_paths,
                'sensor_path': sensor_path,
                'mask_path': mask_path
            }

            if len(image_paths) > 0 and len(mask_paths) > 0:
                for i in range(len(image_paths)):
                    img = cv2.imread(image_paths[i])
                    mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
                    if img is None or mask is None:
                        print(
                            f"Warning: Could not read image or mask: {image_paths[i]} or {mask_paths[i]}"
                        )
                        continue

                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

                    object_pixels = img[mask == 255]
                    if object_pixels.size > 0:
                        all_object_pixels.append(object_pixels)

        if not all_object_pixels:
            print(
                f"No object pixels found in any frame across all sensors. Copying images as-is."
            )

            for sensor, data in sensor_data.items():
                output_dir = f"{base_image_path}/{self.file_name}/merged_directly_shadow1_resized_ViT_pct_matched_colors/{sensor}"
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)

                for img_path in data['image_paths']:
                    shutil.copy2(
                        img_path,
                        os.path.join(output_dir, os.path.basename(img_path)))
                print(f"\033[32mCopied images to: {output_dir}\033[0m")

            print("Color consistency processing finished.")
            return

        combined_pixels = np.concatenate(all_object_pixels, axis=0)
        reference_mean = np.mean(combined_pixels, axis=0)
        reference_std = np.std(combined_pixels, axis=0)
        print(
            f"Global reference stats (across all sensors) - Mean: {reference_mean}, Std: {reference_std}"
        )

        for sensor, data in sensor_data.items():
            print(f"Applying color correction to sensor: {sensor}")

            output_dir = f"{base_image_path}/{self.file_name}/merged_directly_shadow1_resized_ViT_pct_matched_colors/{sensor}"
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            image_paths = data['image_paths']
            mask_paths = data['mask_paths']

            for i in range(len(image_paths)):
                target_img = cv2.imread(image_paths[i])
                target_mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)

                _, target_mask = cv2.threshold(target_mask, 127, 255,
                                               cv2.THRESH_BINARY)
                target_mask = cv2.resize(
                    target_mask, (target_img.shape[1], target_img.shape[0]))

                corrected_img = target_img.copy()

                target_pixels = target_img[target_mask == 255]

                if target_pixels.size > 0:

                    target_mean = np.mean(target_pixels, axis=0)
                    target_std = np.std(target_pixels, axis=0)

                    corrected_pixels = (target_pixels - target_mean) * (
                        reference_std / (target_std + 1e-6)) + reference_mean

                    corrected_pixels = np.clip(corrected_pixels, 0,
                                               255).astype(np.uint8)

                    corrected_img[target_mask == 255] = corrected_pixels

                output_path = f"{output_dir}/{os.path.basename(image_paths[i])}"

                cv2.imwrite(output_path, corrected_img)

            print(
                f"\033[32mCreated folder with color-matched images: {output_dir}\033[0m"
            )

        print("Color consistency processing finished.")

    def sample_random_models(self, eval=False):
        train_valid = "train" * self.train + "valid" * (not self.train)
        if not eval:
            objects_json_path = self.objects_train_json if self.train else self.objects_valid_json
        else:
            objects_json_path = self.objects_eval_json

        with open(objects_json_path, "r") as json_file:
            ood_objects = json.load(json_file)
            if eval:
                ood_objects = ood_objects[:10]

        print(f"Reading OOD object pool from {objects_json_path}")

        if eval and isinstance(ood_objects, list):
            car_objects = []
            for obj in ood_objects:
                if isinstance(obj, dict) and obj.get('lvis') in ['bicycle']:
                    car_objects.append(obj)
            if car_objects:
                ood_objects = car_objects
            else:
                print(
                    "  Warning: No car objects found in evaluation set, using all objects"
                )
        if not isinstance(ood_objects, list) or len(ood_objects) == 0:
            raise ValueError(
                f"Object pool is empty or invalid: {objects_json_path}")

        target_count = int(self.num_objects)
        eligible_entries = []
        available_entries = []
        missing_or_unavailable = 0
        rejected_description = 0
        invalid_entries = 0

        for pool_idx, model in enumerate(ood_objects):
            if not isinstance(model, dict):
                invalid_entries += 1
                continue
            raw_path = model.get("path")
            if not raw_path:
                missing_or_unavailable += 1
                continue
            raw_path = os.path.expandvars(str(raw_path).strip())
            if os.path.isabs(os.path.expanduser(raw_path)):
                path = os.path.expanduser(raw_path)
            else:
                path = os.path.abspath(
                    os.path.join(os.path.dirname(objects_json_path), raw_path))
            if not os.path.exists(path):
                missing_or_unavailable += 1
                continue
            available_entries.append((pool_idx, model, path))
            lvis = model.get("lvis", model.get("LVIS", None))
            reject_reason = self._description_reject_reason(
                model.get("description", ""),
                lvis=lvis,
            )
            if reject_reason is not None:
                rejected_description += 1
                continue
            eligible_entries.append((pool_idx, model, path))

        if (self.enable_description_reject_filter
                and len(eligible_entries) < target_count
                and len(available_entries) >= target_count):
            print(
                f"\033[33mDescription reject filter left only {len(eligible_entries)} eligible objects "
                f"(< target_count={target_count}). Falling back to path-valid pool of "
                f"{len(available_entries)} objects.\033[0m")
            eligible_entries = available_entries

        if len(eligible_entries) == 0:
            raise FileNotFoundError(
                "No eligible objects available after path/existence and description filters. "
                f"pool={objects_json_path}, total={len(ood_objects)}, "
                f"missing_or_unavailable={missing_or_unavailable}, invalid={invalid_entries}, "
                f"rejected_description={rejected_description}.")

        selected_models = []
        self.sampled_model_metadata = []
        seen_pool_indices = set()
        attempts = 0
        max_attempts = max(1000, target_count * 300)

        def append_selected_entry(pool_idx, model, path):
            lvis = model.get("lvis", model.get("LVIS", None))
            file_identifier = model.get("fileIdentifier",
                                        model.get("sha", None))
            scale = float(model.get("scale", 1.0))
            prior = self._resolve_model_size_prior(model)
            preserve_manual_scale = self._should_preserve_manual_reviewed_scale_at_render(
                model)
            selected_models.append((file_identifier, lvis, path, scale))
            self.sampled_model_metadata.append({
                "pool_index":
                int(pool_idx),
                "candidate_id":
                model.get("candidate_id"),
                "source_uid":
                model.get("source_uid"),
                "file_identifier":
                file_identifier,
                "lvis":
                lvis,
                "path":
                path,
                "base_scale":
                float(scale),
                "original_scale":
                model.get("original_scale"),
                "manual_scale_review_source":
                model.get("manual_scale_review_source"),
                "preserve_manual_scale_at_render":
                bool(preserve_manual_scale),
                "description":
                prior["description"],
                "target_ref_dim_m":
                (float(prior["target_ref_dim_m"])
                 if prior["target_ref_dim_m"] is not None else None),
                "target_ref_dim_source":
                prior.get("target_ref_dim_source"),
                "target_ref_dim_raw":
                prior.get("target_ref_dim_raw"),
                "target_max_dim_m": (None if preserve_manual_scale else
                                     model.get("target_max_dim_m")),
                "scale_policy":
                prior["scale_policy"],
                "size_bucket":
                prior["size_bucket"],
                "placement_slot_index":
                model.get("placement_slot_index"),
                "placement_candidate_id":
                model.get("placement_candidate_id"),
                "placement_point_xyz":
                model.get("placement_point_xyz"),
                "placement_distance_m":
                model.get("placement_distance_m"),
                "placement_distance_q":
                model.get("placement_distance_q"),
                "placement_visibility_score":
                model.get("placement_visibility_score"),
                "placement_visible_camera_count":
                model.get("placement_visible_camera_count"),
                "placement_plan":
                model.get("placement_plan"),
            })

        ordered_slot_entries = []
        exact_slot_pool = False
        try:
            for pool_idx, model, path in eligible_entries:
                slot_index = int(model.get("placement_slot_index"))
                ordered_slot_entries.append(
                    (slot_index, int(pool_idx), model, path))
            ordered_slot_entries.sort(key=lambda item: (item[0], item[1]))
            exact_slot_pool = len(ordered_slot_entries) == len(
                eligible_entries) and len(eligible_entries) > 0
        except Exception:
            ordered_slot_entries = []
            exact_slot_pool = False

        if exact_slot_pool and len(eligible_entries) != target_count:
            print(
                f"\033[36mExact placement-slot pool detected in {objects_json_path}. "
                f"Overriding num_objects from {target_count} to {len(eligible_entries)} "
                "and preserving pool rows as-is.\033[0m")
            target_count = int(len(eligible_entries))
            self.num_objects = target_count
            max_attempts = max(1000, target_count * 300)

        replace = len(eligible_entries) < target_count
        if replace and not exact_slot_pool:
            print(
                f"\033[33mWarning: requested {target_count} objects but eligible pool has only "
                f"{len(eligible_entries)} entries. Sampling with replacement.\033[0m"
            )

        preserve_explicit_slot_order = exact_slot_pool and len(
            ordered_slot_entries) == target_count

        if preserve_explicit_slot_order:
            print(
                f"\033[36mPreserving explicit placement-slot order from {objects_json_path} "
                f"for {target_count} objects.\033[0m")
            for _slot_index, pool_idx, model, path in ordered_slot_entries:
                append_selected_entry(pool_idx, model, path)
        else:
            while len(selected_models
                      ) < target_count and attempts < max_attempts:
                attempts += 1
                local_idx = int(self.seed.integers(0, len(eligible_entries)))
                pool_idx, model, path = eligible_entries[local_idx]
                if not replace:
                    if pool_idx in seen_pool_indices:
                        continue
                    seen_pool_indices.add(pool_idx)

                append_selected_entry(pool_idx, model, path)

        if len(selected_models) < target_count:
            raise FileNotFoundError(
                f"Could not sample {target_count} objects from pool {objects_json_path}. "
                f"Only found {len(selected_models)} after {attempts} attempts "
                f"(eligible={len(eligible_entries)}, missing_or_unavailable={missing_or_unavailable}, "
                f"invalid={invalid_entries}, rejected_description={rejected_description})."
            )

        print(
            f"\033[36mObject pool filtering: eligible={len(eligible_entries)}, "
            f"rejected_description={rejected_description}, missing={missing_or_unavailable}, "
            f"invalid={invalid_entries}.\033[0m")
        self.models.extend(selected_models)

    def extract_lidar_parameters(self):

        for sensor_type, frames in self.frames_with_seg.items():
            if len(frames) == 0:
                continue
            if sensor_type not in self.lidar_parameters:
                self.lidar_parameters[sensor_type] = {}

            lidar_par_dir = f"{self.base_dir}/{self.file_name}/lidar_data/"
            lidar_par_dir_type = f"{lidar_par_dir}{sensor_type}"

            if not os.path.exists(lidar_par_dir_type):
                os.makedirs(lidar_par_dir_type, exist_ok=True)

            for frame_id, frame in enumerate(frames):

                laser = frame.context.laser_calibrations[
                    -1] if frame.context.laser_calibrations else None

                if laser:

                    extrinsic_matrix = np.array(
                        laser.extrinsic.transform).reshape(4, 4)

                    beam_inclinations = laser.beam_inclinations

                    pose_matrix = np.array(frame.pose.transform).reshape(4, 4)

                    extrinsic_global = pose_matrix @ extrinsic_matrix

                    if extrinsic_global.size > 0 and pose_matrix.size > 0:

                        self.lidar_parameters[sensor_type][frame_id] = {
                            "pose": extrinsic_matrix,
                            "pose_global": extrinsic_global,
                            "inclinations": np.asarray(beam_inclinations)
                        }
                        lidar_parameters = {
                            "pose": extrinsic_matrix.tolist(),
                            "pose_global": extrinsic_global.tolist(),
                            "inclinations":
                            np.asarray(beam_inclinations).tolist()
                        }
                        with open(
                                f"{lidar_par_dir_type}/lidar_data_frame_00{self.frame_map[sensor_type][frame_id]}.json",
                                "w") as lidar_file:
                            json.dump(lidar_parameters, lidar_file, indent=4)

        print(f"\033[32mSaving {sensor_type} LiDAR file.\033[0m")
        self.master[self.file_name]['lidar_data'] = lidar_par_dir

        np.save(
            f"{self.base_dir}/{self.file_name}/lidar_data/lidar_parameters.npy",
            self.lidar_parameters,
            allow_pickle=True)

    def visualize_ood_locations_per_frame_in_global_frame(self):
        output_dir = f"{self.base_dir}/{self.file_name}/Point_Cloud/"
        dataset_world_to_blender_world = np.array([[0, -1, 0, 0], [1, 0, 0, 0],
                                                   [0, 0, 1, 0], [0, 0, 0, 1]])
        try:
            active_lidar_sensor_types = self._get_active_lidar_pipeline_sensor_types(
            )
            serializable_frames_with_seg = {
                sensor_type: len(frames)
                for sensor_type, frames in self.frames_with_seg.items()
                if len(frames) > 0
            }
            if (self.raycast_lidar_only
                    and len(active_lidar_sensor_types) > 0):
                serializable_frames_with_seg = {
                    sensor_type: serializable_frames_with_seg[sensor_type]
                    for sensor_type in active_lidar_sensor_types
                    if serializable_frames_with_seg.get(sensor_type, 0) > 0
                }

            valid_ood_details = {
                ood_id: details
                for ood_id, details in self.ood_details.items()
                if self._has_valid_pose(details)
            }
            skipped_for_pose = len(self.ood_details) - len(valid_ood_details)
            if skipped_for_pose > 0:
                print(
                    f"\033[33mSkipping {skipped_for_pose} OOD entries with invalid/missing pose "
                    "for LiDAR raycast reconstruction.\033[0m")
            blender_data = {
                'file_name': self.file_name,
                'frames_with_seg': serializable_frames_with_seg,
                'ood_details': valid_ood_details,
                'file_dir': self.base_dir,
                'lidar_import_mode': self.lidar_import_mode,
            }

            with tempfile.NamedTemporaryFile(
                    mode='w+', delete=False,
                    suffix='.json') as temp_payload_file:
                json.dump(blender_data, temp_payload_file)
                payload_json_path = temp_payload_file.name

            with tempfile.NamedTemporaryFile(mode='w+',
                                             delete=False) as temp_json_file:
                json_output_path = temp_json_file.name

            command = [
                self.blender_path, "--background", "--python",
                self.script_path, "--",
                "visualize_ood_locations_and_get_data_for_pcd",
                f"@{payload_json_path}", json_output_path
            ]

            result = subprocess.run(command,
                                    check=True,
                                    capture_output=True,
                                    text=True,
                                    cwd=os.getcwd())

            print('Raycasting completed')

            with open(json_output_path, 'r') as f:
                blender_output = json.load(f)

            manifest_path = self._postprocess_lidar_raycast_results(
                raycast_output_path=json_output_path,
                output_root=f"{self.base_dir}/{self.file_name}",
            )
            self._load_lidar_raycast_manifest(manifest_path)

        except subprocess.CalledProcessError as e:
            print(f"Blender subprocess failed with exit code {e.returncode}:")
            print(f"Stdout:\n{e.stdout}")
            print(f"Stderr:\n{e.stderr}")
            raise RuntimeError(
                "visualize_ood_locations_per_frame_in_global_frame failed in Blender subprocess."
            ) from e

        finally:
            if os.path.exists(json_output_path):
                os.remove(json_output_path)
            if os.path.exists(payload_json_path):
                os.remove(payload_json_path)
            self.master[self.file_name]['ood_point_cloud'] = output_dir

    def _postprocess_lidar_raycast_results(self, raycast_output_path,
                                           output_root):
        helper_script = Path(__file__).resolve(
        ).parent / "scripts" / "postprocess_lidar_raycast.py"
        if not helper_script.exists():
            raise FileNotFoundError(
                f"Missing LiDAR postprocess helper: {helper_script}")

        config = {
            "raycast_output_path": str(raycast_output_path),
            "output_root": str(output_root),
            "state_root": str(Path(output_root) / "Exact_Raycast_Point_Cloud"),
            "lidar_import_mode": self.lidar_import_mode,
            "color_dict": {str(k): v
                           for k, v in self.color_dict.items()},
            "vehicle_pose": {
                sensor_type: {
                    str(frame_id):
                    np.asarray(self.vehicle_pose[sensor_type][frame_id],
                               dtype=np.float64).tolist()
                    for frame_id in range(len(frames))
                }
                for sensor_type, frames in self.frames_with_seg.items()
                if len(frames) > 0 and (
                    not self.raycast_lidar_only or sensor_type in
                    self._get_active_lidar_pipeline_sensor_types())
            },
            "frame_map": {
                sensor_type: {
                    str(frame_id): int(self.frame_map[sensor_type][frame_id])
                    for frame_id in range(len(frames))
                }
                for sensor_type, frames in self.frames_with_seg.items()
                if len(frames) > 0 and (
                    not self.raycast_lidar_only or sensor_type in
                    self._get_active_lidar_pipeline_sensor_types())
            },
            "num_objects": int(self.num_objects),
        }

        with tempfile.NamedTemporaryFile(mode='w+',
                                         delete=False,
                                         suffix='.json') as temp_config_file:
            json.dump(config, temp_config_file)
            config_path = temp_config_file.name

        try:
            command = [
                sys.executable,
                str(helper_script), "--config", config_path
            ]
            result = subprocess.run(command,
                                    check=True,
                                    capture_output=True,
                                    text=True,
                                    cwd=os.getcwd())
            stdout_lines = [
                line.strip() for line in result.stdout.splitlines()
                if line.strip()
            ]
            manifest_path = Path(stdout_lines[-1]) if stdout_lines else Path(
                config["state_root"]) / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"LiDAR postprocess manifest not found: {manifest_path}")
            return manifest_path
        except subprocess.CalledProcessError as e:
            print(
                f"LiDAR postprocess helper failed with exit code {e.returncode}:"
            )
            print(f"Stdout:\n{e.stdout}")
            print(f"Stderr:\n{e.stderr}")
            raise RuntimeError("LiDAR postprocess helper failed.") from e
        finally:
            if os.path.exists(config_path):
                os.remove(config_path)

    def _load_lidar_raycast_manifest(self, manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        for sensor_type, frames in manifest.get("sensors", {}).items():
            if sensor_type not in self.point_cloud:
                self.point_cloud[sensor_type] = {}
            if sensor_type not in self.point_cloud_without_axes:
                self.point_cloud_without_axes[sensor_type] = {}
            if sensor_type not in self.intensity:
                self.intensity[sensor_type] = {}
            if sensor_type not in self.framewise_ood_details:
                self.framewise_ood_details[sensor_type] = {}
            if sensor_type not in self.point_cloud_dict:
                self.point_cloud_dict[sensor_type] = {}

            for frame_id_str, entry in frames.items():
                frame_id = int(frame_id_str)
                self.point_cloud[sensor_type][
                    frame_id] = o3d.io.read_point_cloud(entry["merged_pcd"])
                self.point_cloud_without_axes[sensor_type][
                    frame_id] = o3d.io.read_point_cloud(entry["ood_only_pcd"])
                self.intensity[sensor_type][frame_id] = np.load(
                    entry["intensity_npy"])

                with open(entry["framewise_details_json"],
                          "r") as details_file:
                    details = json.load(details_file)
                self.framewise_ood_details[sensor_type][frame_id] = {
                    int(ood_id): values
                    for ood_id, values in details.items()
                }
                self.point_cloud_dict[sensor_type][frame_id] = {
                    int(ood_id): []
                    for ood_id in details.keys()
                }

    def convert_point_cloud_to_vehicle_frame(self, pcd_points, sensor_type,
                                             frame_id):

        blender_world_to_dataset_world = np.array([[0, 1, 0, 0], [-1, 0, 0, 0],
                                                   [0, 0, 1, 0], [0, 0, 0, 1]])

        if pcd_points.ndim == 1:
            points = pcd_points.reshape(1, -1)
        else:
            points = np.asarray(pcd_points)

        if np.array_equal(pcd_points, np.array([[0.0, 0.0, 0.0]])):
            print('returned')
            return np.array([[0.0, 0.0, 0.0]])

        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

        points_dataset_world = blender_world_to_dataset_world @ points_homogeneous.T

        transformation = np.asarray(self.vehicle_pose[sensor_type][frame_id])

        dataset_world_to_dataset_vehicle = (
            np.linalg.inv(transformation) @ points_dataset_world).T

        return dataset_world_to_dataset_vehicle[:, :3]

    def convert_dataset_world_point_cloud_to_vehicle_frame(
            self, pcd_points, sensor_type, frame_id):
        if pcd_points.ndim == 1:
            points = pcd_points.reshape(1, -1)
        else:
            points = np.asarray(pcd_points)

        if np.array_equal(pcd_points, np.array([[0.0, 0.0, 0.0]])):
            return np.array([[0.0, 0.0, 0.0]])

        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        transformation = np.asarray(self.vehicle_pose[sensor_type][frame_id])
        dataset_world_to_dataset_vehicle = (
            np.linalg.inv(transformation) @ points_homogeneous.T).T
        return dataset_world_to_dataset_vehicle[:, :3]

    def convert_point_cloud_to_lidar_frame(self, pcd_points, frame_id):

        points = np.asarray(pcd_points)

        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

        transformation = np.asarray(self.lidar_parameters[frame_id]['pose'])

        dataset_vehicle_to_dataset_lidar = (
            np.linalg.inv(transformation) @ points_homogeneous.T).T

        return dataset_vehicle_to_dataset_lidar[:, :3]

    def point_cloud_to_range_image(self):
        """
        Convert a point cloud (N x 3) to a range image (H x W).
        Assumes Waymo's 64-beam LiDAR parameters.
        """
        print("\033[31mCreating Range Image from Point Cloud Merged\033[0m")
        num_rows = 64
        num_cols = 2650

        method_list = ['vehicle', 'lidar', 'vehicle_var', 'lidar_var']
        lidar_list = ['fixed', 'variable']

        for lidar in lidar_list:
            for method in method_list:

                frame_ids = list(range(10))
                points_np = [
                    self.point_cloud_exclusive[method][fid]
                    for fid in frame_ids
                ]
                colors_np = self.colors_vehicle_frame

                num_points = [pts.shape[0] for pts in points_np]

                max_points = max(num_points)

                padded_points = []
                for pts in points_np:
                    padding = max_points - pts.shape[0]
                    padded = np.pad(pts, [(0, padding), (0, 0)],
                                    mode='constant')
                    padded_points.append(padded)

                points_vehicle_frame = tf.convert_to_tensor(padded_points,
                                                            dtype=tf.float32)

                num_points = tf.convert_to_tensor(num_points, dtype=tf.int32)

                extrinsics = [
                    np.asarray(self.lidar_parameters[fid]['pose'])
                    for fid in frame_ids
                ]

                extrinsic = tf.convert_to_tensor(extrinsics, dtype=tf.float32)

                inclinations = [
                    np.asarray(
                        self.lidar_parameters[fid]['inclinations'])[::-1]
                    for fid in frame_ids
                ]

                inclination = tf.convert_to_tensor(inclinations,
                                                   dtype=tf.float32)

                with tf.device('/CPU:0'):
                    range_images, ri_indices, _ = range_image_utils.build_range_image_from_point_cloud(
                        points_vehicle_frame,
                        num_points,
                        extrinsic,
                        inclination, [num_rows, num_cols],
                        point_features=None,
                        dtype=tf.float32)

                for frame_id in range(range_images.shape[0]):

                    mrg_range_image = range_images[frame_id]

                    mrg_range_image = tf.where(mrg_range_image == 0,
                                               -tf.ones_like(mrg_range_image),
                                               mrg_range_image)

                    depth_min = tf.reduce_min(
                        mrg_range_image[mrg_range_image > 0])
                    depth_max = tf.reduce_max(
                        mrg_range_image[mrg_range_image > 0])

                    normalized_merged_depth = tf.where(
                        depth_max > depth_min, (mrg_range_image - depth_min) /
                        (depth_max - depth_min),
                        tf.zeros_like(mrg_range_image))

                    self.merged_range_image[method][frame_id] = mrg_range_image

                    org_range_images, _, _, _ = frame_utils.parse_range_image_and_camera_projection(
                        self.frames_with_seg[frame_id])
                    range_image_top = org_range_images[1]

                    range_image_tf = []
                    for layer in range_image_top:
                        if hasattr(layer, 'data'):
                            layer_data = tf.convert_to_tensor(layer.data,
                                                              dtype=tf.float32)
                            range_image_tf.append(layer_data)

                    range_image_tf = tf.stack(range_image_tf, axis=0)
                    range_image_top = tf.reshape(range_image_tf[0],
                                                 (64, 2650, 4))
                    org_range_image = range_image_top[:, :, 0]
                    org_range_image_intensity = range_image_top[:, :, 1]

                    flattened_values = tf.reshape(org_range_image_intensity,
                                                  [-1]).numpy()

                    depth_min = tf.reduce_min(
                        org_range_image[org_range_image > 0])
                    depth_max = tf.reduce_max(
                        org_range_image[org_range_image > 0])

                    normalized_org_depth = tf.where(
                        depth_max > depth_min, (org_range_image - depth_min) /
                        (depth_max - depth_min),
                        tf.zeros_like(org_range_image))

                    depth_min = tf.reduce_min(org_range_image_intensity[
                        org_range_image_intensity > 0])
                    depth_max = tf.reduce_max(org_range_image_intensity[
                        org_range_image_intensity > 0])

                    normalized_org_intensity = tf.where(
                        depth_max > depth_min,
                        (org_range_image_intensity - depth_min) /
                        (depth_max - depth_min),
                        tf.zeros_like(org_range_image_intensity))

                    diff = tf.abs(org_range_image - mrg_range_image)

                    log_diff = tf.math.log1p(diff)
                    depth_min = tf.reduce_min(log_diff)
                    depth_max = tf.reduce_max(log_diff)

                    normalized_log_diff = tf.where(depth_max > depth_min,
                                                   (log_diff - depth_min) /
                                                   (depth_max - depth_min),
                                                   tf.zeros_like(log_diff))

                    cmaps = [
                        'gray', 'gray_r', 'viridis', 'inferno', 'jet', 'turbo',
                        'coolwarm', 'hot'
                    ]

                    for cmap in cmaps:
                        fig, ax = plt.subplots(4, 1, figsize=(25, 16))

                        range_folder = f"{self.base_dir}/{self.file_name}/Range_Images_from_Point_Cloud_Combined/{cmap}"
                        os.makedirs(range_folder, exist_ok=True)

                        im0 = ax[0].imshow(mrg_range_image,
                                           cmap=cmap,
                                           aspect='auto')
                        ax[0].set_title('Range Image with object')
                        ax[0].axis('off')
                        im1 = ax[1].imshow(org_range_image,
                                           cmap=cmap,
                                           aspect='auto')
                        ax[1].set_title('Range Image without object')
                        ax[1].axis('off')
                        im2 = ax[2].imshow(normalized_log_diff,
                                           cmap=cmap,
                                           aspect='auto')
                        ax[2].set_title('Difference Log Absolute')
                        ax[2].axis('off')

                        im3 = ax[3].hist(flattened_values,
                                         bins=100,
                                         color='skyblue',
                                         edgecolor='black')
                        ax[3].set_title('Histogram of Range Image Intensity')
                        ax[3].set_xlabel('Intensity Value')
                        ax[3].set_ylabel('Frequency')

                        cbar = fig.colorbar(im0,
                                            ax=ax[0],
                                            orientation='vertical',
                                            label='Distance(m)')
                        cbar = fig.colorbar(im1,
                                            ax=ax[1],
                                            orientation='vertical',
                                            label='Distance(m)')
                        cbar = fig.colorbar(im2,
                                            ax=ax[2],
                                            orientation='vertical')

                        plt.savefig(
                            f'{range_folder}/frame_00{self.frame_map[sensor_type][frame_id]}',
                            bbox_inches='tight',
                            pad_inches=0)
                        plt.close()

    @ResourceLogger.log_resources_during_execution()
    def ood_point_cloud_to_range_image(self):
        """
        Convert a point cloud (N x 3) to a range image (H x W).
        Assumes Waymo's 64-beam LiDAR parameters.
        """
        print("\033[31mCreating Range Image from Object Point Cloud\033[0m")
        num_rows = 64
        num_cols = 2650

        lidar_list = ['variable']
        method_list = ['lidar']

        def remove_repeated_indices_tf(tensor):
            frame_id, N, _ = tensor.shape
            filtered_indices = []
            keep_masks = []

            for fid in range(frame_id):
                indices = tensor[fid]

                keys = indices[:, 0] * tf.cast(N, tf.int32) + indices[:, 1]

                unique_keys, _, counts = tf.unique_with_counts(keys)
                duplicated_keys = unique_keys[counts > 1]

                dup_mask = tf.reduce_any(tf.equal(tf.expand_dims(keys, 1),
                                                  duplicated_keys),
                                         axis=1)
                keep_mask = tf.logical_not(dup_mask)

                filtered = tf.boolean_mask(indices, keep_mask)
                filtered_indices.append(filtered)
                keep_masks.append(keep_mask)

            return filtered_indices, keep_masks

        for sensor_type in self._get_active_lidar_pipeline_sensor_types():
            if len(self.frames_with_seg.get(sensor_type, [])) == 0:
                continue
            dir_range = f"{self.base_dir}/{self.file_name}/OOD_Merged_Range_Image/"
            dir_intensity = f"{self.base_dir}/{self.file_name}/OOD_Merged_Intensity/"
            dir_range_type = f"{dir_range}/{sensor_type}"
            dir_intensity_type = f"{dir_intensity}/{sensor_type}"

            if sensor_type not in self.ood_labels_to_indices.keys():
                self.ood_labels_to_indices[sensor_type] = {}
                self.ood_labels_to_ranges[sensor_type] = {}

            if not os.path.exists(dir_range_type):
                os.makedirs(dir_range_type, exist_ok=True)
                os.makedirs(dir_intensity_type, exist_ok=True)

            frame_ids = list(range(len(self.frames_with_seg[sensor_type])))

            sensor_point_clouds = self.point_cloud_without_axes.get(
                sensor_type, {})
            if not sensor_point_clouds:
                print(
                    f"Skipping ood_point_cloud_to_range_image for sensor '{sensor_type}': "
                    "no reconstructed point clouds available.")
                continue

            point_clouds = [
                sensor_point_clouds[fid] for fid in frame_ids
                if fid in sensor_point_clouds
            ]

            if not point_clouds:
                print(
                    f"Skipping ood_point_cloud_to_range_image for sensor '{sensor_type}': "
                    "reconstructed point cloud list is empty.")
                continue

            print(f"\033[41m{point_clouds}\033[0m")

            points_np = [np.asarray(pcd.points) for pcd in point_clouds]
            colors_np = [np.asarray(pcd.colors) for pcd in point_clouds]

            color_to_label = {
                tuple(np.round(np.array(v) / 255.0, 6)): k
                for k, v in self.color_dict.items()
            }

            labels_np = [
                np.array([
                    color_to_label.get(tuple(np.round(color, 6)), -1)
                    for color in np.asarray(pcd_colors)
                ]) for pcd_colors in colors_np
            ]

            sensor_intensity = self.intensity.get(sensor_type, {})
            if not sensor_intensity:
                print(
                    f"Skipping ood_point_cloud_to_range_image for sensor '{sensor_type}': "
                    "missing intensity values from raycast stage.")
                continue

            intensities = [
                sensor_intensity[fid] for fid in frame_ids
                if fid in sensor_point_clouds and fid in sensor_intensity
            ]

            if not intensities:
                print(
                    f"Skipping ood_point_cloud_to_range_image for sensor '{sensor_type}': "
                    "no aligned intensity entries for reconstructed points.")
                continue

            extrinsics_np = [
                np.asarray(self.lidar_parameters[sensor_type][fid]['pose'])
                for fid in frame_ids
            ]

            sanitized_points_np = []
            sanitized_labels_np = []
            sanitized_intensities = []
            for frame_id, (pts, labels, inten, ext) in enumerate(
                    zip(points_np, labels_np, intensities, extrinsics_np)):
                pts_sanitized, (
                    labels_sanitized, inten_sanitized
                ) = self._sanitize_range_image_builder_frame_inputs(
                    points_np=pts,
                    aligned_arrays=[labels, inten],
                    extrinsic=ext,
                    sensor_type=sensor_type,
                    local_frame_id=frame_id,
                    num_cols=num_cols,
                    context="ood_point_cloud_to_range_image",
                )
                sanitized_points_np.append(pts_sanitized)
                sanitized_labels_np.append(labels_sanitized)
                sanitized_intensities.append(inten_sanitized)
            points_np = sanitized_points_np
            labels_np = sanitized_labels_np
            intensities = sanitized_intensities

            num_points = [pts.shape[0] for pts in points_np]

            max_points = max(num_points)

            padded_points = []
            padded_intensities = []
            padded_labels = []

            org_range_images_list = []
            org_intensity_list = []
            for frame_id, frame in enumerate(
                    self.frames_with_seg[sensor_type]):

                top_lidar_first_return = self._get_cached_top_lidar_first_return(
                    sensor_type, frame_id, frame)
                range_image = tf.convert_to_tensor(top_lidar_first_return,
                                                   dtype=tf.float32)

                self.range_images[frame_id] = range_image

                range_ch = range_image[..., 0]
                intensity = range_image[..., 1]

                org_range_images_list.append(range_ch)
                org_intensity_list.append(intensity)

            org_range_images = tf.stack(org_range_images_list, axis=0)
            org_intensity = tf.stack(org_intensity_list, axis=0)

            self.original_range_images = org_range_images

            for frame_id, (pts, inten, labels) in enumerate(
                    zip(points_np, intensities, labels_np)):

                pad_count = max_points - pts.shape[0]
                if pad_count > 0:
                    safe_pad_point = self._make_safe_range_image_padding_point(
                        extrinsics_np[frame_id])
                    pad_pts = np.vstack([
                        pts,
                        np.repeat(safe_pad_point[None, :], pad_count, axis=0)
                    ])
                else:
                    pad_pts = pts

                log_intensity = tf.where(
                    org_intensity[frame_id] > 0,
                    tf.math.log1p(org_intensity[frame_id]),
                    tf.zeros_like(org_intensity[frame_id]))

                intensity_max = tf.reduce_max(log_intensity)
                intensity_min = tf.reduce_min(log_intensity)

                divisions = 20
                range_div = intensity_max / divisions

                base_intensity_mask = log_intensity < range_div
                base_intensity = tf.reduce_mean(
                    tf.boolean_mask(log_intensity, base_intensity_mask))

                inten *= base_intensity

                inten = np.expm1(inten)
                pad_inten = np.pad(inten, (0, max_points - pts.shape[0]),
                                   mode='constant')

                padded_points.append(pad_pts)
                padded_intensities.append(pad_inten)

                pad_labels = np.pad(labels, (0, max_points - len(labels)),
                                    mode='constant',
                                    constant_values=-1)

                padded_labels.append(pad_labels)

            points_vehicle_frame = tf.convert_to_tensor(padded_points,
                                                        dtype=tf.float32)

            intensity = tf.expand_dims(tf.convert_to_tensor(padded_intensities,
                                                            dtype=tf.float32),
                                       axis=-1)

            zeros = tf.zeros_like(intensity)
            point_features = tf.concat([intensity, zeros], axis=-1)

            num_points = tf.convert_to_tensor(num_points, dtype=tf.int32)

            extrinsics = extrinsics_np

            extrinsic = tf.convert_to_tensor(extrinsics, dtype=tf.float32)

            inclinations = [
                np.asarray(self.lidar_parameters[sensor_type][fid]
                           ['inclinations'])[::-1] for fid in frame_ids
            ]

            inclination = tf.convert_to_tensor(inclinations, dtype=tf.float32)

            print([
                tf.reduce_max(points_vehicle_frame[fid])
                for fid in range(len(points_vehicle_frame))
            ])

            with tf.device('/CPU:0'):
                range_images, ri_indices_unfiltered, ri_ranges = range_image_utils.build_range_image_from_point_cloud(
                    points_vehicle_frame,
                    num_points,
                    extrinsic,
                    inclination, [num_rows, num_cols],
                    point_features=point_features,
                    dtype=tf.float32)
            ri_indices_filtered, keep_masks = remove_repeated_indices_tf(
                ri_indices_unfiltered)

            ri_ranges_filtered = tf.ragged.boolean_mask(ri_ranges, keep_masks)

            range_image_labels = tf.ones([len(frame_ids), num_rows, num_cols],
                                         dtype=tf.int32) * -1

            scatter_indices = []
            scatter_labels = []

            for fid in range(len(frame_ids)):

                filtered_indices = ri_indices_filtered[fid]

                keep_mask = keep_masks[fid]

                valid_labels = tf.boolean_mask(padded_labels[fid], keep_mask)
                valid_labels = tf.cast(valid_labels, tf.int32)

                batch_indices = tf.fill([tf.shape(filtered_indices)[0], 1],
                                        fid)
                full_indices = tf.concat([batch_indices, filtered_indices],
                                         axis=1)

                scatter_indices.append(full_indices)
                scatter_labels.append(valid_labels)

            scatter_indices = tf.concat(scatter_indices, axis=0)
            scatter_labels = tf.concat(scatter_labels, axis=0)

            range_image_labels = tf.tensor_scatter_nd_update(
                range_image_labels, scatter_indices, scatter_labels)

            range_channel = range_images[..., 0]
            intensity_channel = range_images[..., 1]

            range_images = tf.where(range_channel == 0,
                                    -tf.ones_like(range_channel),
                                    range_channel)

            self.range_image_ood_exclusive = range_images

            if org_range_images.shape == range_images.shape:

                valid_ood = range_images != -1
                valid_original = org_range_images != -1

                ood_priority = tf.where(valid_ood, range_images,
                                        org_range_images)

                ood_range_images = tf.minimum(ood_priority, org_range_images)

                indices_to_remove_for_final_pc = tf.where(
                    ood_range_images != ood_priority)

                for frame_idx in range(ood_range_images.shape[0]):

                    frame_ood = ood_range_images[frame_idx]
                    frame_labels = range_image_labels[frame_idx]
                    frame_valid_ood = valid_ood[frame_idx]
                    frame_filtered_ranges = ri_ranges_filtered[frame_idx]
                    filtered_indices_frame = ri_indices_filtered[frame_idx]

                    remove_mask = tf.logical_and(
                        tf.equal(frame_ood, ood_priority[frame_idx]),
                        frame_valid_ood)

                    if not tf.reduce_any(remove_mask):
                        self.ood_labels_to_indices[sensor_type][frame_idx] = {}
                        self.ood_labels_to_ranges[sensor_type][frame_idx] = {}
                        continue

                    removal_indices = tf.where(remove_mask)

                    M = tf.shape(filtered_indices_frame)[0]
                    if M == 0:
                        self.ood_labels_to_indices[sensor_type][frame_idx] = {}
                        self.ood_labels_to_ranges[sensor_type][frame_idx] = {}
                        continue

                    index_map = tf.fill([num_rows, num_cols], -1)
                    indices_to_update = tf.cast(filtered_indices_frame,
                                                tf.int32)
                    values = tf.range(M, dtype=tf.int32)
                    index_map = tf.tensor_scatter_nd_update(
                        index_map, indices_to_update, values)

                    removal_indices_int32 = tf.cast(removal_indices, tf.int32)
                    positions_in_filtered = tf.gather_nd(
                        index_map, removal_indices_int32)

                    valid_mask = positions_in_filtered >= 0
                    valid_positions = tf.boolean_mask(positions_in_filtered,
                                                      valid_mask)
                    valid_removal_indices = tf.boolean_mask(
                        removal_indices, valid_mask)

                    if tf.size(valid_positions) == 0:
                        self.ood_labels_to_indices[sensor_type][frame_idx] = {}
                        self.ood_labels_to_ranges[sensor_type][frame_idx] = {}
                        continue

                    removal_labels = tf.gather_nd(frame_labels,
                                                  valid_removal_indices)
                    ranges_for_removal = tf.gather(frame_filtered_ranges,
                                                   valid_positions)

                    unique_labels, group_idx = tf.unique(removal_labels)
                    num_groups = tf.reduce_max(group_idx) + 1
                    partitioned_indices = tf.dynamic_partition(
                        valid_removal_indices, group_idx, num_groups)
                    partitioned_ranges = tf.dynamic_partition(
                        ranges_for_removal, group_idx, num_groups)

                    label_to_indices = {}
                    label_to_ranges = {}
                    for i, label in enumerate(unique_labels.numpy()):
                        int_label = int(label)
                        if int_label > 0:
                            label_to_indices[int_label] = partitioned_indices[
                                i].numpy()
                            label_to_ranges[int_label] = partitioned_ranges[
                                i].numpy()

                    self.ood_labels_to_indices[sensor_type][
                        frame_idx] = label_to_indices
                    self.ood_labels_to_ranges[sensor_type][
                        frame_idx] = label_to_ranges

                indices_per_frame = [
                    tuple((int(i), int(j))
                          for i, j in indices_to_remove_for_final_pc.numpy()[
                              tf.where(indices_to_remove_for_final_pc[:, 0] ==
                                       frame_id)[:, 0].numpy()][:, 1:])
                    for frame_id in range(
                        len(self.frames_with_seg[sensor_type]))
                ]

                if sensor_type not in self.range_images_ood.keys():
                    self.range_images_ood[sensor_type] = {}

                self.range_images_ood[sensor_type] = ood_range_images

                intensity_mask = ood_range_images != org_range_images

                ood_intensity = tf.where(intensity_mask, intensity_channel,
                                         org_intensity)

                ood_intensity_filtered = tf.where(ood_intensity > 0,
                                                  ood_intensity,
                                                  tf.zeros_like(ood_intensity))
                if sensor_type not in self.ood_intensity.keys():
                    self.ood_intensity[sensor_type] = {}

                self.ood_intensity[sensor_type] = ood_intensity

            filtered_ri_ranges_per_frame = []
            filtered_ri_indices_per_frame = []

            for frame_id in range(len(self.frames_with_seg[sensor_type])):

                frame_indices = ri_indices_filtered[frame_id].numpy()
                frame_ranges = ri_ranges[frame_id].numpy()

                remove_indices = set(indices_per_frame[frame_id])

                all_indices = tf.range(tf.shape(frame_ranges)[0])

                keep_indices = []
                for idx, (i, j) in enumerate(frame_indices):
                    if (i, j) not in remove_indices:
                        keep_indices.append(idx)
                keep_indices = tf.constant(keep_indices, dtype=tf.int32)

                filtered_indices = tf.gather(frame_indices, keep_indices)
                filtered_ranges = tf.gather(frame_ranges, keep_indices)

                filtered_ri_indices_per_frame.append(filtered_indices)
                filtered_ri_ranges_per_frame.append(filtered_ranges)

                range_image = range_images[frame_id, :, :]

                ood_range_image = ood_range_images[frame_id, :, :]
                ood_int = ood_intensity[frame_id]

                ood_range_image_np = ood_range_image.numpy().astype(np.float32)
                ood_int_np = ood_int.numpy().astype(np.float32)

                ood_range_image_np.tofile(
                    f"{dir_range_type}/frame_00{self.frame_map[sensor_type][frame_id]}.bin"
                )
                ood_int_np.tofile(
                    f"{dir_intensity_type}/frame_00{self.frame_map[sensor_type][frame_id]}.bin"
                )

                self.ood_range_image[frame_id] = range_image

                org_range_image = org_range_images[frame_id, :, :]

                if self.save_range_image_viz:
                    diff = tf.abs(ood_range_image - org_range_image)
                    log_diff = tf.math.log1p(diff)
                    depth_min = tf.reduce_min(log_diff)
                    depth_max = tf.reduce_max(log_diff)

                    normalized_log_diff = tf.where(depth_max > depth_min,
                                                   (log_diff - depth_min) /
                                                   (depth_max - depth_min),
                                                   tf.zeros_like(log_diff))

                    cmaps = ['jet']
                    for cmap in cmaps:
                        fig, ax = plt.subplots(4, 1, figsize=(30, 16))

                        range_folder = f"{self.base_dir}/{self.file_name}/Range Images from Point Cloud/{sensor_type}/"
                        os.makedirs(range_folder, exist_ok=True)

                        im0 = ax[0].imshow(range_image,
                                           cmap=cmap,
                                           aspect='auto',
                                           vmax=np.max(ood_range_image))
                        ax[0].set_title('Range Image of object')
                        ax[0].axis('off')
                        im1 = ax[1].imshow(ood_range_image,
                                           cmap=cmap,
                                           aspect='auto')
                        ax[1].set_title('Range Image Combined')
                        ax[1].axis('off')
                        im2 = ax[2].imshow(org_range_image,
                                           cmap=cmap,
                                           aspect='auto')
                        ax[2].set_title('Range Image Original')
                        ax[2].axis('off')
                        im3 = ax[3].imshow(normalized_log_diff,
                                           cmap=cmap,
                                           aspect='auto',
                                           vmin=0,
                                           vmax=1)
                        ax[3].set_title('Difference Log Absolute')
                        ax[3].axis('off')
                        fig.colorbar(im0,
                                     ax=ax[0],
                                     orientation='vertical',
                                     label='Distance(m)')
                        fig.colorbar(im1,
                                     ax=ax[1],
                                     orientation='vertical',
                                     label='Distance(m)')
                        fig.colorbar(im2,
                                     ax=ax[2],
                                     orientation='vertical',
                                     label='Distance(m)')
                        fig.colorbar(im3, ax=ax[3], orientation='vertical')

                        plt.savefig(
                            f'{range_folder}/frame_00{self.frame_map[sensor_type][frame_id]}',
                            bbox_inches='tight',
                            pad_inches=0,
                        )
                        plt.close()
            if sensor_type not in self.ri_ranges_ood.keys():
                self.ri_ranges_ood[sensor_type] = {}

            self.ri_ranges_ood[sensor_type] = tf.ragged.stack(
                filtered_ri_ranges_per_frame)

            if sensor_type not in self.range_images_ood_indices.keys():
                self.range_images_ood_indices[sensor_type] = {}

            indices = tf.ragged.stack(filtered_ri_indices_per_frame)

            new_flat_values = tf.reshape(indices.flat_values, [-1, 2])

            new_indices = tf.RaggedTensor.from_row_splits(
                values=new_flat_values, row_splits=indices.row_splits)

            row_lengths = new_indices.row_lengths()

            self.range_images_ood_indices[sensor_type] = new_indices

        self.master[self.file_name]['ood_merged_range_image'] = dir_range
        self.master[self.file_name]['ood_merged_intensity'] = dir_intensity

    def get_top_lidar_point_cloud(self):

        sensor_type = 'camera'
        if 'camera' not in self.point_cloud_vehicle.keys():
            self.point_cloud_vehicle['camera'] = {}
            self.cp_actual['camera'] = {}

        cache_name = "top_lidar_camera"
        cache_meta = {
            "dataset_path":
            self._dataset_cache_signature.get("dataset_path"),
            "dataset_size":
            self._dataset_cache_signature.get("dataset_size"),
            "dataset_mtime_ns":
            self._dataset_cache_signature.get("dataset_mtime_ns"),
            "num_frames":
            int(self.num_frames),
            "frame_map_camera":
            tuple(int(v) for v in self.frame_map.get(sensor_type, [])),
        }
        cached_payload = self._load_precompute_cache(cache_name,
                                                     expected_meta=cache_meta)
        if isinstance(cached_payload, dict):
            cached_frames = cached_payload.get("frames", {})
            for frame_key in sorted(cached_frames.keys(),
                                    key=lambda x: int(x)):
                frame_payload = cached_frames[frame_key]
                frame_id = int(frame_key)
                points_np = np.asarray(frame_payload.get("points", []),
                                       dtype=np.float32)
                cp_np = np.asarray(frame_payload.get("cp_points", []))
                if points_np.ndim != 2 or points_np.shape[0] == 0:
                    continue
                pcd_lidar = o3d.geometry.PointCloud()
                pcd_lidar.points = o3d.utility.Vector3dVector(points_np)
                pcd_lidar.paint_uniform_color([0.7, 0.7, 0.7])
                self.point_cloud_vehicle['camera'][frame_id] = pcd_lidar
                self.cp_actual['camera'][frame_id] = cp_np
            print("[CACHE] Reused top LiDAR point cloud precompute.",
                  flush=True)
            return

        cached_frames = {}
        for frame_id, frame in enumerate(self.frames_with_seg['camera']):

            range_images, camera_projections, seg_labels, range_image_top_pose = (
                self._get_cached_range_parse(sensor_type, frame_id, frame))

            points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=0,
                keep_polar_features=False)

            pcd_lidar = o3d.geometry.PointCloud()
            pcd_lidar.points = o3d.utility.Vector3dVector(points[0])

            pcd_lidar.paint_uniform_color([0.7, 0.7, 0.7])

            self.point_cloud_vehicle['camera'][frame_id] = pcd_lidar
            self.cp_actual['camera'][frame_id] = cp_points[0]
            cached_frames[int(frame_id)] = {
                "points": np.asarray(points[0], dtype=np.float32),
                "cp_points": np.asarray(cp_points[0]),
            }

            file_path = f'{self.base_dir}/{self.file_name}/Actual_Top_Lidar_Point_Cloud'
            file_path_dir = os.path.join(file_path, sensor_type)
            if not os.path.exists(file_path_dir):
                os.makedirs(file_path_dir, exist_ok=True)

            file_path = os.path.join(
                file_path_dir,
                f'frame_00{self.frame_map[sensor_type][frame_id]}.pcd')
            o3d.io.write_point_cloud(file_path, pcd_lidar)

        self._save_precompute_cache(
            cache_name,
            {"frames": cached_frames},
            meta=cache_meta,
        )

    def get_color_top_lidar_point_cloud(self):
        """
        Extracts the top LiDAR point cloud from the frames and assigns colors based on the segmentation labels.
        """
        def get_instance_perturbation(instance_id):
            """Generate stronger deterministic color perturbation for an instance ID"""
            primes = [73856093, 19349669, 83492791]
            perturbation_range = 80
            instance_id = int(instance_id)

            offsets = []
            for prime in primes:

                val = (instance_id * prime) % 101

                offset = int((val - 50) * (perturbation_range / 50))
                offsets.append(offset)
            return offsets

        sensor_type = 'lidar'
        if 'lidar' not in self.point_cloud_vehicle.keys():
            self.point_cloud_vehicle['lidar'] = {}
            self.cp_actual['lidar'] = {}

        current_dir = os.getcwd()
        file_dir = Path(
            os.path.join(
                current_dir,
                f'{self.base_dir}/{self.file_name}/lidar_label_metadata'))

        points_dir = os.path.join(file_dir, 'velodyne')

        labels_dir = os.path.join(file_dir, 'labels')

        proj_dir = os.path.join(file_dir, 'projections')
        if os.path.isdir(points_dir) and os.path.isdir(labels_dir):
            label_files = os.listdir(labels_dir)
            points_files = os.listdir(points_dir)

            proj_files = os.listdir(proj_dir)
            label_files = sorted(label_files)[:self.num_frames]

            points_files = sorted(points_files)[:self.num_frames]
            proj_files = sorted(proj_files)[:self.num_frames]

            if len(label_files) == len(points_files):
                for frame_id, (label_file, points_file,
                               proj_file) in enumerate(
                                   zip(sorted(label_files),
                                       sorted(points_files),
                                       sorted(proj_files))):
                    if label_file.endswith('.label') and points_file.endswith(
                            '.bin'):
                        label_path = os.path.join(labels_dir, label_file)
                        points_path = os.path.join(points_dir, points_file)
                        proj_path = os.path.join(proj_dir, proj_file)

                        points = np.fromfile(points_path,
                                             dtype=np.float32).reshape(-1, 4)
                        print(points.shape, points_path)
                        labels = np.fromfile(label_path, dtype=np.uint32)
                        print(labels.shape, label_path)
                        if labels.shape[0] == points.shape[0]:

                            projections = np.fromfile(proj_path,
                                                      dtype=np.uint32).reshape(
                                                          -1, 6)
                            print(projections.shape, proj_path)
                            self.points_lidar[frame_id] = points[:, :3]
                            self.lidar_labels[frame_id] = labels

                            front_cam_mask = (projections[:, 0]
                                              == 1) | (projections[:, 3] == 1)
                            labels_front_cam = labels[front_cam_mask]

                            projections_filtered = projections[front_cam_mask]

                            road_mask = np.isin(labels_front_cam // 1000,
                                                [17, 18, 19])

                            projections_final = projections_filtered[road_mask]

                            projection_array = projections_final[:, [1, 2]]
                            self.road_projections_lidar[
                                frame_id] = projection_array

                            x0, y0, z0, intensity = points.T

                            semantic_labels = labels // 1000

                            self.lidar_sematic_labels.append(semantic_labels)
                            instance_labels = labels % 1000
                            print('Unique instance labels:', np.unique(labels))

                            base_colors = self.color_array[semantic_labels]
                            base_colors_int = base_colors.astype(np.int32)

                            offsets = np.zeros_like(base_colors_int)

                            non_zero_mask = instance_labels != 0
                            unique_instances = np.unique(
                                instance_labels[non_zero_mask])

                            perturbation_dict = {}
                            for inst_id in unique_instances:
                                perturbation_dict[
                                    inst_id] = get_instance_perturbation(
                                        inst_id)

                            for inst_id in unique_instances:
                                mask = instance_labels == inst_id
                                offsets[mask] = perturbation_dict[inst_id]

                            new_colors = base_colors_int + offsets
                            new_colors = np.clip(new_colors, 0,
                                                 255).astype(np.uint8)

                            nonzero_inst_mask = instance_labels != 0
                            nonzero_indices = np.where(
                                nonzero_inst_mask)[0][:10]

                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(
                                np.vstack([x0, y0, z0]).T)
                            pcd.colors = o3d.utility.Vector3dVector(
                                new_colors.astype(np.float32) / 255.0)

                            stored_colors_check = (np.asarray(pcd.colors) *
                                                   255).astype(np.uint8)

                            nonzero_check = np.where(
                                instance_labels != 0)[0][:5]

                            self.point_cloud_vehicle['lidar'][frame_id] = pcd
                            self.cp_actual['lidar'][frame_id] = projections

                            retrieved_pcd = self.point_cloud_vehicle['lidar'][
                                frame_id]
                            retrieved_colors = (
                                np.asarray(retrieved_pcd.colors) * 255).astype(
                                    np.uint8)

                            file_path = f'{self.base_dir}/{self.file_name}/Actual_Top_Lidar_Point_Cloud'
                            file_path_dir = os.path.join(
                                file_path, sensor_type)
                            if not os.path.exists(file_path_dir):
                                os.makedirs(file_path_dir, exist_ok=True)

                            file_path_f = os.path.join(
                                file_path_dir,
                                f'frame_00{self.frame_map[sensor_type][frame_id]}.pcd'
                            )
                            o3d.io.write_point_cloud(file_path_f, pcd)
                        else:
                            raise ValueError(
                                f"Labels and points do not match in shape for frame {frame_id}"
                            )
                self.master[self.file_name]['actual_point_cloud'] = file_path

    def _compute_valid_range_image_column_mask(self, points_vehicle_frame,
                                               extrinsic, num_cols):
        points_vehicle_frame = np.asarray(points_vehicle_frame,
                                          dtype=np.float32)
        if points_vehicle_frame.size == 0:
            return np.zeros((0, ), dtype=bool)

        finite_mask = np.all(np.isfinite(points_vehicle_frame), axis=1)
        if not np.any(finite_mask):
            return finite_mask

        extrinsic = np.asarray(extrinsic, dtype=np.float32)
        valid_indices = np.flatnonzero(finite_mask)
        points_subset = points_vehicle_frame[valid_indices]

        with tf.device('/CPU:0'):
            points_tf = tf.convert_to_tensor(points_subset[None, ...],
                                             dtype=tf.float32)
            extrinsic_tf = tf.convert_to_tensor(extrinsic[None, ...],
                                                dtype=tf.float32)

            vehicle_to_laser = tf.linalg.inv(extrinsic_tf)
            rotation = vehicle_to_laser[:, 0:3, 0:3]
            translation = tf.expand_dims(vehicle_to_laser[:, 0:3, 3], 1)
            sensor_points = tf.einsum('bij,bkj->bik', points_tf,
                                      rotation) + translation

            az_correction = tf.expand_dims(
                tf.atan2(extrinsic_tf[..., 1, 0], extrinsic_tf[..., 0, 0]), -1)
            point_azimuth = tf.atan2(sensor_points[..., 1],
                                     sensor_points[..., 0]) + az_correction
            point_azimuth = point_azimuth - tf.cast(point_azimuth > np.pi,
                                                    tf.float32) * (2.0 * np.pi)
            point_azimuth = point_azimuth + tf.cast(point_azimuth < -np.pi,
                                                    tf.float32) * (2.0 * np.pi)
            point_ri_col_indices = tf.cast(
                tf.round(num_cols - 1.0 + 0.5 - (point_azimuth + np.pi) /
                         (2.0 * np.pi) * num_cols),
                tf.int32,
            )

            sensor_points_np = sensor_points.numpy()[0]
            point_azimuth_np = point_azimuth.numpy()[0]
            point_ri_col_indices_np = point_ri_col_indices.numpy()[0]

        valid_subset = np.all(np.isfinite(sensor_points_np), axis=1)
        valid_subset &= (point_ri_col_indices_np >=
                         0) & (point_ri_col_indices_np < num_cols)

        boundary_eps = np.finfo(np.float32).eps
        seam_mask = np.abs(np.abs(point_azimuth_np) - np.pi) <= boundary_eps
        valid_subset &= ~seam_mask

        valid_mask = np.zeros_like(finite_mask, dtype=bool)
        valid_mask[valid_indices] = valid_subset
        return valid_mask

    def _sanitize_range_image_builder_frame_inputs(
        self,
        *,
        points_np,
        aligned_arrays,
        extrinsic,
        sensor_type,
        local_frame_id,
        num_cols,
        context,
    ):
        valid_mask = self._compute_valid_range_image_column_mask(
            points_np, extrinsic, num_cols)
        dropped = int(valid_mask.shape[0] - np.count_nonzero(valid_mask))
        if dropped:
            frame_token = self._frame_token_for_sensor_index(
                sensor_type, local_frame_id)
            print(
                f"[RI SANITIZE] {context}: dropped {dropped}/{valid_mask.shape[0]} invalid "
                f"range-image boundary points for sensor '{sensor_type}' "
                f"frame_id={local_frame_id} token={frame_token}.")

        sanitized_points = np.asarray(points_np)[valid_mask]
        sanitized_arrays = [
            np.asarray(arr)[valid_mask] for arr in aligned_arrays
        ]
        return sanitized_points, sanitized_arrays

    def _make_safe_range_image_padding_point(self, extrinsic):
        extrinsic = np.asarray(extrinsic)
        vehicle_to_laser = np.linalg.inv(extrinsic)
        rotation = vehicle_to_laser[0:3, 0:3]
        translation = vehicle_to_laser[0:3, 3]
        safe_sensor_point = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        safe_vehicle_point = (safe_sensor_point - translation) @ rotation
        return safe_vehicle_point.astype(np.float32)

    @ResourceLogger.log_resources_during_execution()
    def get_range_image_indices_of_point_cloud(self):

        num_rows = 64
        num_cols = 2650

        for sensor_type in self._get_active_lidar_pipeline_sensor_types():
            if len(self.frames_with_seg.get(sensor_type, [])) == 0:
                continue
            source_sensor = sensor_type
            if (sensor_type not in self.point_cloud_vehicle
                    or len(self.point_cloud_vehicle[sensor_type]) == 0):
                if (sensor_type == 'lidar'
                        and 'camera' in self.point_cloud_vehicle
                        and len(self.point_cloud_vehicle['camera']) > 0):
                    source_sensor = 'camera'
                    print(
                        "RI index extraction fallback: using camera-top LiDAR point clouds "
                        "to populate lidar indices.")
                else:
                    print(
                        f"Skipping ri index extraction for sensor '{sensor_type}': "
                        "no point clouds available.")
                    continue
            if (source_sensor not in self.cp_actual
                    or len(self.cp_actual[source_sensor]) == 0):
                print(
                    f"Skipping ri index extraction for sensor '{sensor_type}': "
                    f"missing camera projections for source '{source_sensor}'."
                )
                continue
            if sensor_type not in self.ri_indices.keys():
                self.ri_indices[sensor_type] = None
                self.ri_ranges[sensor_type] = None
            if sensor_type not in self.points_vehicle_frame.keys():
                self.points_vehicle_frame[sensor_type] = {}
                self.colors_vehicle_frame[sensor_type] = {}
                self.projections_vehicle_frame[sensor_type] = {}
            frame_ids = list(range(len(self.frames_with_seg[sensor_type])))
            missing_fids = [
                fid for fid in frame_ids
                if fid not in self.point_cloud_vehicle[source_sensor]
                or fid not in self.cp_actual[source_sensor]
                or fid not in self.lidar_parameters[source_sensor]
            ]
            if missing_fids:
                print(
                    f"Skipping ri index extraction for sensor '{sensor_type}': "
                    f"source '{source_sensor}' is missing frame IDs {missing_fids}."
                )
                continue

            point_clouds = [
                self.point_cloud_vehicle[source_sensor][fid]
                for fid in frame_ids
                if fid in self.point_cloud_vehicle[source_sensor]
            ]

            points_np = [np.asarray(pcd.points) for pcd in point_clouds]
            colors_np = [np.asarray(pcd.colors) for pcd in point_clouds]

            if len(colors_np) > 0:

                sample_colors_stored = (colors_np[0][:10] * 255).astype(
                    np.uint8)

                color_61 = (colors_np[0][61] * 255).astype(np.uint8)

                all_colors_frame0 = (colors_np[0] * 255).astype(np.uint8)
                vegetation_base = np.array([107, 142, 35], dtype=np.uint8)
                perturbations_all = all_colors_frame0 - vegetation_base
                nonzero_pert = np.sum(np.any(perturbations_all != 0, axis=1))

            projections_np = [
                np.asarray(self.cp_actual[source_sensor][fid])
                for fid in frame_ids
            ]
            extrinsics_np = [
                np.asarray(self.lidar_parameters[source_sensor][fid]['pose'])
                for fid in frame_ids
            ]

            sanitized_points_np = []
            sanitized_colors_np = []
            sanitized_projections_np = []
            for frame_id, (pts, clrs, pjcs, ext) in enumerate(
                    zip(points_np, colors_np, projections_np, extrinsics_np)):
                pts_sanitized, (
                    clrs_sanitized, pjcs_sanitized
                ) = self._sanitize_range_image_builder_frame_inputs(
                    points_np=pts,
                    aligned_arrays=[clrs, pjcs],
                    extrinsic=ext,
                    sensor_type=sensor_type,
                    local_frame_id=frame_id,
                    num_cols=num_cols,
                    context="get_range_image_indices_of_point_cloud",
                )
                sanitized_points_np.append(pts_sanitized)
                sanitized_colors_np.append(clrs_sanitized)
                sanitized_projections_np.append(pjcs_sanitized)
            points_np = sanitized_points_np
            colors_np = sanitized_colors_np
            projections_np = sanitized_projections_np

            num_points = [pts.shape[0] for pts in points_np]

            if len(num_points) == 0:
                continue

            max_points = max(num_points)

            padded_points = []
            padded_colors = []
            padded_projections = []

            for frame_id, pts in enumerate(points_np):

                pad_count = max_points - pts.shape[0]
                if pad_count > 0:
                    safe_pad_point = self._make_safe_range_image_padding_point(
                        extrinsics_np[frame_id])
                    pad_pts = np.vstack([
                        pts,
                        np.repeat(safe_pad_point[None, :], pad_count, axis=0)
                    ])
                else:
                    pad_pts = pts
                pad_clrs = np.pad(colors_np[frame_id],
                                  [(0, max_points - pts.shape[0]), (0, 0)],
                                  mode='constant')
                pad_pjcs = np.pad(projections_np[frame_id],
                                  [(0, max_points - pts.shape[0]), (0, 0)],
                                  mode='constant')
                padded_points.append(pad_pts)
                padded_colors.append(pad_clrs)
                padded_projections.append(pad_pjcs)

            print(
                np.asarray(padded_points).shape,
                np.asarray(padded_colors).shape,
                np.asarray(padded_projections).shape)

            points_vehicle_frame = tf.convert_to_tensor(padded_points,
                                                        dtype=tf.float32)
            colors_vehicle_frame = tf.convert_to_tensor(padded_colors,
                                                        dtype=tf.float32)
            self.projections_vehicle_frame[sensor_type] = tf.convert_to_tensor(
                padded_projections, dtype=tf.int32)

            num_points = tf.convert_to_tensor(num_points, dtype=tf.int32)

            extrinsics = extrinsics_np

            extrinsic = tf.convert_to_tensor(extrinsics, dtype=tf.float32)

            inclinations = [
                np.asarray(self.lidar_parameters[source_sensor][fid]
                           ['inclinations'])[::-1] for fid in frame_ids
            ]

            inclination = tf.convert_to_tensor(inclinations, dtype=tf.float32)

            print([
                tf.reduce_max(points_vehicle_frame[fid])
                for fid in range(len(points_vehicle_frame))
            ])

            with tf.device('/CPU:0'):
                range_images, ri_indices, ri_ranges = range_image_utils.build_range_image_from_point_cloud(
                    points_vehicle_frame,
                    num_points,
                    extrinsic,
                    inclination, [num_rows, num_cols],
                    point_features=None,
                    dtype=tf.float32)

            self.ri_indices[sensor_type] = ri_indices
            self.ri_ranges[sensor_type] = ri_ranges
            self.points_vehicle_frame[sensor_type] = points_vehicle_frame
            self.colors_vehicle_frame[sensor_type] = colors_vehicle_frame
            self.projections_vehicle_frame[sensor_type] = tf.convert_to_tensor(
                padded_projections, dtype=tf.int32)

    def check_road_points_using_range_image(self):

        sensor_type = 'lidar'
        points_vehicle_frame = self.points_vehicle_frame[sensor_type]
        colors_vehicle_frame = self.colors_vehicle_frame[sensor_type]
        ri_indices_all = self.ri_indices[sensor_type]
        ri_ranges_all = self.ri_ranges[sensor_type]

        road_colors_1d_list = []
        points_list = []
        indices_list = []
        ranges_list = []

        for frame_id, frame in enumerate(self.frames_with_seg[sensor_type]):
            colors = colors_vehicle_frame[frame_id]
            road_colors = colors == [128 / 255.0, 64 / 255.0, 128 / 255.0]
            road_colors_1d = tf.reduce_all(road_colors, axis=-1)
            road_colors_1d_list.append(road_colors_1d)

            road_points = points_vehicle_frame[frame_id][road_colors_1d]
            road_ranges = ri_ranges_all[frame_id][road_colors_1d]
            road_indices = ri_indices_all[frame_id][road_colors_1d]

            points_list.append(road_points)
            ranges_list.append(road_ranges)
            indices_list.append(road_indices)

        road_range_selections = []
        for frame_id in range(len(self.frames_with_seg[sensor_type])):

            road_indices = indices_list[frame_id]
            road_ranges = ranges_list[frame_id]
            road_points = points_list[frame_id]
            full_indices = ri_indices_all[frame_id]
            full_ranges = ri_ranges_all[frame_id]
            road_colors_1d = road_colors_1d_list[frame_id]

            non_road_mask = tf.logical_not(road_colors_1d)
            non_road_indices = tf.boolean_mask(full_indices, non_road_mask)
            non_road_ranges = tf.boolean_mask(full_ranges, non_road_mask)

            max_ring = tf.reduce_max(full_indices[:, 0]) + 1
            max_index = tf.reduce_max(full_indices[:, 1]) + 1
            grid_shape = [max_ring, max_index]

            grid_non_road = tf.fill(
                grid_shape, tf.constant(float('inf'), dtype=tf.float32))

            if tf.size(non_road_ranges) > 0:

                non_road_indices = tf.cast(non_road_indices, tf.int64)
                grid_non_road = tf.tensor_scatter_nd_min(
                    grid_non_road, non_road_indices, non_road_ranges)

            road_min_non_road = tf.gather_nd(grid_non_road,
                                             tf.cast(road_indices, tf.int64))

            selection_mask = road_ranges < road_min_non_road
            selected_points = tf.boolean_mask(road_points, selection_mask)
            road_range_selections.append(selected_points)

        road_range_selections = tf.ragged.stack(road_range_selections)

        points_all = []
        points_road_filtered = []
        for frame_id, pcd in self.point_cloud_vehicle[sensor_type].items():
            points_vehicle = np.asarray(pcd.points)

            colors_vehicle = np.asarray(pcd.colors)

            labels = np.asarray(self.lidar_sematic_labels[frame_id])
            labels = labels[labels != 0]

            mask = ~np.all(colors_vehicle == [0, 0, 0], axis=1)
            points_vehicle_labeled = points_vehicle[mask]

            x = points_vehicle_labeled[:, 0]
            y = points_vehicle_labeled[:, 1]

            transformation = self.vehicle_pose[sensor_type][frame_id]

            x = points_vehicle_labeled[:, 0]
            points_to_consider = points_vehicle[x > 0]
            n = points_to_consider.shape[0]
            points_vehicle_homogenous = np.hstack(
                (points_to_consider, np.ones((n, 1))))
            transformation = self.vehicle_pose[sensor_type][frame_id]
            points_global = (
                transformation @ points_vehicle_homogenous.T).T[:, :3]

            points_all.append(points_global)

            road_points = np.array(road_range_selections[frame_id].to_list(),
                                   dtype=object)
            x = road_points[:, 0]
            road_points_to_consider = road_points[x > 0]
            n = road_points_to_consider.shape[0]
            road_points_vehicle_homogenous = np.hstack(
                (road_points_to_consider, np.ones((n, 1))))
            road_points_global = (
                transformation @ road_points_vehicle_homogenous.T).T[:, :3]
            points_road_filtered.append(road_points_global)
        points = np.vstack(points_all)
        road_points = np.vstack(points_road_filtered)

        pcd_points = o3d.geometry.PointCloud()
        pcd_points.points = o3d.utility.Vector3dVector(points)
        pcd_points.paint_uniform_color([1, 1, 1])

        pcd_road_points = o3d.geometry.PointCloud()
        pcd_road_points.points = o3d.utility.Vector3dVector(road_points)
        pcd_road_points.paint_uniform_color([1, 1, 0])

        merged_pcd = pcd_points + pcd_road_points

        file_path = f"{self.base_dir}/{self.file_name}/scenario.pcd"
        o3d.io.write_point_cloud(file_path, merged_pcd)

    def combine_point_clouds(self):
        print('Combining point clouds')

        lidar_list = ['variable']
        method_list = ['lidar']

        for sensor_type in self._get_active_lidar_pipeline_sensor_types():
            if len(self.frames_with_seg.get(sensor_type, [])) == 0:
                continue
            if sensor_type not in self.point_cloud or sensor_type not in self.point_cloud_without_axes:
                print(
                    f"Skipping combine_point_clouds for sensor '{sensor_type}': missing reconstructed point clouds."
                )
                continue
            if sensor_type not in self.point_cloud_vehicle:
                print(
                    f"Skipping combine_point_clouds for sensor '{sensor_type}': missing vehicle point clouds."
                )
                continue

            for frame_id in self.point_cloud_vehicle[sensor_type].keys():
                if frame_id not in self.point_cloud[sensor_type]:
                    print(
                        f"Skipping frame {frame_id} for sensor '{sensor_type}': no reconstructed point cloud."
                    )
                    continue
                if frame_id not in self.point_cloud_without_axes[sensor_type]:
                    print(
                        f"Skipping frame {frame_id} for sensor '{sensor_type}': no reconstructed point cloud (no axes)."
                    )
                    continue

                pcd = self.point_cloud[sensor_type][frame_id]
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)

                pcd_without_axes = self.point_cloud_without_axes[sensor_type][
                    frame_id]
                points_without_axes = np.asarray(pcd_without_axes.points)
                colors_without_axes = np.asarray(pcd_without_axes.colors)

                pcd_lidar = self.point_cloud_vehicle[sensor_type][frame_id]
                points_lidar = np.asarray(pcd_lidar.points)
                colors_lidar = np.asarray(pcd_lidar.colors)

                combined_points = np.vstack((points, points_lidar))
                combined_colors = np.vstack((colors, colors_lidar))

                pcd_combined = o3d.geometry.PointCloud()
                pcd_combined.points = o3d.utility.Vector3dVector(
                    combined_points)

                pcd_combined.colors = o3d.utility.Vector3dVector(
                    combined_colors)

                output_dir = f"{self.base_dir}/{self.file_name}/Point_Cloud_Combined/{sensor_type}"
                os.makedirs(output_dir, exist_ok=True)

                file_path = f"{output_dir}/frame_00{self.frame_map[sensor_type][frame_id]}.pcd"
                o3d.io.write_point_cloud(file_path, pcd_combined)

    def get_point_cloud_from_range_image_ood(self, precision=False):
        def convert_transformation_to_lidar_pose_range_image(
            transformation_matrices: tf.Tensor, ) -> tf.Tensor:
            """Converts transformation matrices back to per-pixel rotation and translation.

            Args:
                transformation_matrices: A [H, W, 4, 4] tensor of transformation matrices.

            Returns:
                A [H, W, 6] tensor where the last dimension is [roll, pitch, yaw, x, y, z].
            """

            R = transformation_matrices[..., :3, :3]

            pitch = tf.asin(-R[..., 2, 0])

            yaw = tf.atan2(R[..., 1, 0], R[..., 0, 0])

            roll = tf.atan2(R[..., 2, 1], R[..., 2, 2])

            translation = transformation_matrices[..., :3, 3]

            lidar_pose_range_image = tf.stack([
                roll,
                pitch,
                yaw,
                translation[..., 0],
                translation[..., 1],
                translation[..., 2],
            ],
                                              axis=-1)

            return lidar_pose_range_image

        frame_ids = list(range(len(self.frames_with_seg)))

        extrinsics = [
            np.asarray(self.lidar_parameters[fid]['pose']) for fid in frame_ids
        ]

        extrinsic = tf.convert_to_tensor(extrinsics, dtype=tf.float32)

        inclinations = [
            np.asarray(self.lidar_parameters[fid]['inclinations'])[::-1]
            for fid in frame_ids
        ]

        inclination = tf.convert_to_tensor(inclinations, dtype=tf.float32)

        range_image_top_pose_list = []
        pose_constant_list = []
        is_identity_list = []
        pose_constant_sph_list = []

        print("\033[31mCreating Point Cloud from Range Image\033[0m")
        for frame_id, frame in enumerate(self.frames_with_seg):

            range_image_top_pose_tf = tf.convert_to_tensor(
                self.lidar_range_pose[sensor_type][frame_id], dtype=tf.float32)

            range_image_top_pose_tensor = tf.cast(range_image_top_pose_tf,
                                                  dtype=tf.float32)

            pose_sph = convert_transformation_to_lidar_pose_range_image(
                range_image_top_pose_tf)

            pose_constant = tf.cast(
                tf.convert_to_tensor(self.vehicle_pose[frame_id],
                                     dtype=tf.float32), tf.float32)

            pose_expanded = tf.expand_dims(tf.expand_dims(pose_constant,
                                                          axis=0),
                                           axis=0)

            pose_constant = tf.tile(pose_expanded, [64, 2650, 1, 1])

            identity_matrix = tf.eye(4, dtype=tf.float32)
            identity_matrix = tf.expand_dims(tf.expand_dims(identity_matrix,
                                                            axis=0),
                                             axis=0)
            identity_matrix = tf.tile(identity_matrix, [64, 2650, 1, 1])

            is_identity = tf.reduce_all(tf.equal(range_image_top_pose_tf,
                                                 identity_matrix),
                                        axis=[2, 3])

            is_identity = tf.expand_dims(tf.expand_dims(is_identity, axis=-1),
                                         axis=-1)

            pose_constant_tensor = tf.where(is_identity,
                                            range_image_top_pose_tensor,
                                            pose_constant)

            range_image_top_pose_list.append(range_image_top_pose_tensor)

            pose_constant_list.append(pose_constant_tensor)

            pose_constant_sph_list.append(pose_sph)

            is_identity_list.append(is_identity)

        pose_dict = {}

        pixel_pose = tf.convert_to_tensor(range_image_top_pose_list,
                                          dtype=tf.float32)

        frame_pose = tf.convert_to_tensor(self.vehicle_pose, dtype=tf.float32)

        pixel_pose_constant = tf.convert_to_tensor(pose_constant_list,
                                                   dtype=tf.float32)
        pose_constant_sph_tensor = tf.convert_to_tensor(pose_constant_sph_list,
                                                        dtype=tf.float32)
        pose_constant_sph_tensor = tf.reshape(
            pose_constant_sph_tensor, [len(self.frames_with_seg), 64, 2650, 6])

        lidar_list = ['variable']
        method_list = ['lidar']

        for lidar in lidar_list:
            for method in method_list:
                for frame_id, frame in enumerate(self.frames_with_seg):
                    constant_pose = self.vehicle_pose[frame_id]
                    constant_pose_sph = convert_transformation_to_lidar_pose_range_image(
                        constant_pose)

                    points = np.asarray(
                        self.point_cloud_vehicle[frame_id].points)
                    points = np.copy(points)
                    pcd_original = o3d.geometry.PointCloud()
                    pcd_original.points = o3d.utility.Vector3dVector(points)
                    pcd_original.paint_uniform_color([0, 1, 0])

                    range_images, camera_projections, seg_labels, range_image_top_pose = (
                        frame_utils.parse_range_image_and_camera_projection(
                            frame))

                    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                        frame,
                        range_images,
                        camera_projections,
                        range_image_top_pose,
                        ri_index=0,
                        keep_polar_features=False)

                    self.cp_actual[frame_id] = cp_points[0]

                    original_ri = range_images[open_dataset.LaserName.TOP][0]

                    h = original_ri.shape.dims[0]
                    w = original_ri.shape.dims[1]
                    original_array = np.array(original_ri.data).reshape(
                        h, w, 4)

                    original_array[:, :, 0] = self.range_images_ood[lidar][
                        method][frame_id]

                    del original_ri.data[:]
                    original_ri.data.extend(
                        original_array.astype(np.float64).flatten().tolist())

                    pose_constant_sph = open_dataset.MatrixFloat()
                    pose_constant_sph.data.extend(
                        pose_constant_sph_tensor[frame_id].numpy().flatten(
                        ).tolist())
                    pose_constant_sph.shape.dims.extend([h, w, 6])

                    if lidar == 'variable':

                        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                            frame,
                            range_images,
                            camera_projections,
                            range_image_top_pose,
                            ri_index=0,
                            keep_polar_features=False)

                    elif lidar == 'fixed':

                        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                            frame,
                            range_images,
                            camera_projections,
                            pose_constant_sph,
                            ri_index=0,
                            keep_polar_features=False)

                    if lidar not in self.pc_reconstructed.keys():
                        self.pc_reconstructed[lidar] = {}
                    if method not in self.pc_reconstructed[lidar].keys():
                        self.pc_reconstructed[lidar][method] = {}
                    points_re = np.copy(points[0])
                    self.pc_reconstructed[lidar][method][
                        frame_id] = points_re.astype(np.float64)

                    original_array[:, :, 0] = self.range_image_ood_exclusive[
                        lidar][method][frame_id]

                    del original_ri.data[:]
                    original_ri.data.extend(
                        original_array.astype(np.float64).flatten().tolist())

                    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                        frame,
                        range_images,
                        camera_projections,
                        range_image_top_pose,
                        ri_index=0,
                        keep_polar_features=False)

                    if lidar not in self.cp_reconstructed.keys():
                        self.cp_reconstructed[lidar] = {}
                    if method not in self.cp_reconstructed[lidar].keys():
                        self.cp_reconstructed[lidar][method] = {}
                    self.cp_reconstructed[lidar][method][frame_id] = cp_points[
                        0].astype(np.float64)

                    pcd_modified_frame = o3d.geometry.PointCloud()
                    pcd_modified_frame.points = o3d.utility.Vector3dVector(
                        points_re)
                    pcd_modified_frame.paint_uniform_color([1, 0.5, 0])

                    output_dir = f"{self.base_dir}/{self.file_name}/Reconstructed Point Cloud/{lidar}/{method}"
                    os.makedirs(output_dir, exist_ok=True)
                    merged_pcd = pcd_original + pcd_modified_frame + self.point_cloud_without_axes[
                        lidar][method][frame_id]

                    file_path = f"{output_dir}/frame_00{self.frame_map[sensor_type][frame_id]}.pcd"
                    o3d.io.write_point_cloud(file_path, merged_pcd)

    def compare_point_cloud(self):

        lidar_list = ['variable']
        method_list = ['lidar']

        for sensor_type, frames in self.frames_with_seg.items():
            if len(frames) == 0 or sensor_type == 'camera':
                continue

            for frame_id, frame in enumerate(frames):

                pcd_ood = self.point_cloud_without_axes[sensor_type][frame_id]
                pcd_ood.paint_uniform_color([0, 1, 0])

                output_dir = f"{self.base_dir}/{self.file_name}/Comparison/{sensor_type}"
                os.makedirs(output_dir, exist_ok=True)

                pcd_filtered = self.filtered_points[sensor_type][frame_id]

                merged_pcd = self.pc_reconstructed_indices[sensor_type][
                    frame_id] + pcd_ood + pcd_filtered

                file_path = f"{output_dir}/frame_00{self.frame_map[sensor_type][frame_id]}.pcd"
                o3d.io.write_point_cloud(file_path, merged_pcd)

    def get_vehicle_pose(self, render_mode='multi'):
        for sensor_type, frames in self.frames_with_seg.items():
            if len(frames) == 0:
                continue
            self.vehicle_pose[sensor_type] = []
            for frame in frames:
                vehicle_pose = np.array(frame.pose.transform).reshape(4, 4)
                self.vehicle_pose[sensor_type].append(vehicle_pose)

        if render_mode == 'multi':
            reference_sensor = None
            if (not self.enable_lidar_insertion) and len(
                    self.vehicle_pose.get('camera', [])) >= 2:
                reference_sensor = 'camera'
            elif len(self.vehicle_pose.get('lidar', [])) >= 2:
                reference_sensor = 'lidar'
            elif len(self.vehicle_pose.get('camera', [])) >= 2:
                reference_sensor = 'camera'

            dist = None
            if reference_sensor is not None:
                ref_poses = self.vehicle_pose[reference_sensor]
                dist = np.linalg.norm(ref_poses[-1][:3, -1] -
                                      ref_poses[0][:3, -1])

            self.num_objects = self._resolve_num_objects_for_sequence(
                distance_m=None if dist is None else float(dist))
            self._ensure_ood_color_capacity(self.num_objects)
            self._initialize_sequence_placement_constraints()
            if dist is None:
                print(
                    f"\033[36mOBJECT_COUNT: policy={self.object_count_policy}, "
                    f"distance=n/a (single-frame fallback) -> "
                    f"num_objects={self.num_objects}\033[0m")
            else:
                print(
                    f"\033[36mOBJECT_COUNT: policy={self.object_count_policy}, "
                    f"distance={float(dist):.2f}m (sensor={reference_sensor}) -> "
                    f"num_objects={self.num_objects}\033[0m")

    def check_render_points_for_lidar(
        self,
        ood_id,
        scale,
        sensor='lidar',
        render_mode='multi',
        eval=False,
        dist=None,
        request_only=False,
    ):
        points_all = []
        points_ground_all = []
        points_non_ground_all = []
        points_non_ground_by_frame = defaultdict(list)
        sensor_type = 'lidar'
        pose_sensor = 'lidar'
        visibility_sensor = 'lidar'

        lidar_points = self.point_cloud_vehicle.get('lidar', {})
        has_lidar_semantic = (len(lidar_points) > 0
                              and len(self.lidar_sematic_labels) > 0)
        frame_items = list(lidar_points.items())
        use_extended_camera_fallback = False

        if not has_lidar_semantic:
            pose_sensor = 'camera'
            visibility_sensor = 'camera'
            if len(self.camera_accumulation_points_vehicle) > 0:
                use_extended_camera_fallback = True
            else:
                camera_points = self.point_cloud_vehicle.get('camera', {})
                frame_items = list(camera_points.items())
            if use_extended_camera_fallback or frame_items:
                print(
                    "\033[33mPlacement fallback: lidar semantic labels unavailable; "
                    "using geometric ground heuristic from camera-top LiDAR points.\033[0m"
                )

        if (len(self.camera_data.get('camera', [])) > 0
                and len(self.vehicle_pose.get('camera', [])) > 0):
            visibility_sensor = 'camera'

        if pose_sensor not in self.vehicle_pose:
            return np.zeros((3, ))
        if has_lidar_semantic and len(frame_items) == 0:
            return np.zeros((3, ))
        if (not has_lidar_semantic) and (
                not use_extended_camera_fallback) and len(frame_items) == 0:
            return np.zeros((3, ))

        def append_accumulated_points(points_vehicle_labeled, labels_labeled,
                                      transformation, frame_token):
            if points_vehicle_labeled.shape[0] == 0:
                return

            ground_mask, non_ground_mask = self._fused_ground_mask(
                points_vehicle_labeled,
                labels_labeled=labels_labeled,
            )
            points_vehicle_ground = points_vehicle_labeled[ground_mask]
            points_vehicle_non_ground = points_vehicle_labeled[non_ground_mask]

            points_vehicle_homogenous = np.hstack(
                (points_vehicle_labeled,
                 np.ones((points_vehicle_labeled.shape[0], 1))))
            points_global = (
                transformation @ points_vehicle_homogenous.T).T[:, :3]
            points_all.append(points_global)

            if points_vehicle_non_ground.shape[0] > 0:
                points_non_ground_h = np.hstack(
                    (points_vehicle_non_ground,
                     np.ones((points_vehicle_non_ground.shape[0], 1))))
                points_non_ground_global = (
                    transformation @ points_non_ground_h.T).T[:, :3]
                points_non_ground_all.append(points_non_ground_global)
                try:
                    token = int(frame_token)
                    points_non_ground_by_frame[token].append(
                        points_non_ground_global)
                except Exception:
                    pass

            if points_vehicle_ground.shape[0] == 0:
                return
            points_ground_h = np.hstack(
                (points_vehicle_ground,
                 np.ones((points_vehicle_ground.shape[0], 1))))
            points_ground_global = (
                transformation @ points_ground_h.T).T[:, :3]
            points_ground_all.append(points_ground_global)

        if use_extended_camera_fallback:
            for frame_global_id in sorted(
                    self.camera_accumulation_points_vehicle.keys()):
                points_vehicle = np.asarray(
                    self.camera_accumulation_points_vehicle[frame_global_id],
                    dtype=np.float32,
                )
                if points_vehicle.ndim != 2 or points_vehicle.shape[0] == 0:
                    continue
                transformation = np.asarray(
                    self.camera_accumulation_pose_world.get(frame_global_id),
                    dtype=np.float64,
                )
                if transformation.shape != (4, 4):
                    continue
                points_vehicle_labeled = points_vehicle[:, :3]
                if render_mode == 'single' and dist is not None:
                    radial = np.linalg.norm(points_vehicle_labeled[:, :2],
                                            axis=1)
                    within_dist = radial < float(dist)
                    points_vehicle_labeled = points_vehicle_labeled[
                        within_dist]
                    if points_vehicle_labeled.shape[0] == 0:
                        continue
                append_accumulated_points(
                    points_vehicle_labeled,
                    None,
                    transformation,
                    frame_global_id,
                )
        else:
            for frame_id, pcd in frame_items:
                points_vehicle = np.asarray(pcd.points)
                if points_vehicle.size == 0:
                    continue

                if has_lidar_semantic:
                    if frame_id >= len(self.lidar_sematic_labels):
                        continue
                    colors_vehicle = np.asarray(pcd.colors)
                    labels = np.asarray(self.lidar_sematic_labels[frame_id])
                    if labels.size == 0:
                        continue

                    mask = ~np.all(colors_vehicle == [0, 0, 0], axis=1)
                    points_vehicle_labeled = points_vehicle[mask]
                    labels_labeled = labels[mask]
                else:
                    points_vehicle_labeled = points_vehicle
                    labels_labeled = None

                if points_vehicle_labeled.shape[0] == 0:
                    continue

                if render_mode == 'single' and dist is not None:
                    radial = np.linalg.norm(points_vehicle_labeled[:, :2],
                                            axis=1)
                    within_dist = radial < float(dist)
                    points_vehicle_labeled = points_vehicle_labeled[
                        within_dist]
                    if labels_labeled is not None:
                        labels_labeled = labels_labeled[within_dist]
                    if points_vehicle_labeled.shape[0] == 0:
                        continue

                transformation = self.vehicle_pose[pose_sensor][frame_id]
                frame_token = self._frame_token_for_sensor_index(
                    pose_sensor, frame_id)
                append_accumulated_points(
                    points_vehicle_labeled,
                    labels_labeled,
                    transformation,
                    frame_token,
                )

        if not points_all or not points_ground_all:
            return np.zeros((3, ))

        points = np.vstack(points_all)
        points_ground = np.vstack(points_ground_all)
        points_non_ground = (np.vstack(points_non_ground_all)
                             if points_non_ground_all else np.empty(
                                 (0, 3), dtype=np.float32))
        if points_ground.shape[0] == 0:
            return np.zeros((3, ))

        self._export_sampling_accumulated_cloud(points, eval=eval)

        self._export_sampling_pool_cloud(points_ground, eval=eval)

        num_candidates = min(points_ground.shape[0],
                             int(self.placement_candidate_budget))
        if num_candidates <= 0:
            return np.zeros((3, ))
        candidate_rng = self._get_policy_candidate_rng(
            visibility_sensor=visibility_sensor,
            render_mode=render_mode,
            eval=eval,
            dist=dist,
        )
        random_indices = candidate_rng.choice(points_ground.shape[0],
                                              num_candidates,
                                              replace=False)
        random_points = points_ground[random_indices]
        ground_tree = cKDTree(points_ground)
        all_xy_tree = cKDTree(points[:, :2])

        non_ground_points_by_token = {}
        non_ground_xy_tree_by_token = {}
        for frame_token, chunks in points_non_ground_by_frame.items():
            if len(chunks) == 0:
                continue
            merged_chunks = np.vstack(chunks)
            if merged_chunks.shape[0] == 0:
                continue
            token = int(frame_token)
            non_ground_points_by_token[token] = merged_chunks
            non_ground_xy_tree_by_token[token] = cKDTree(merged_chunks[:, :2])
        sorted_non_ground_tokens = sorted(non_ground_points_by_token.keys())

        candidate_visibility_cache = []
        camera_frame_candidate_pool = defaultdict(lambda: defaultdict(int))
        if visibility_sensor == 'camera' and self.enable_fov_candidate_pools:

            draw_budget = min(
                points_ground.shape[0],
                max(
                    int(num_candidates),
                    int(num_candidates * 20),
                ),
            )
            draw_indices = candidate_rng.choice(points_ground.shape[0],
                                                draw_budget,
                                                replace=False)
            selected_points = []
            selected_visibility_cache = []
            visible_pool_cap = min(
                int(draw_budget),
                max(int(num_candidates), int(num_candidates * 4)),
            )
            for draw_idx in draw_indices:
                point = points_ground[int(draw_idx)]
                (
                    score,
                    hits,
                    visible_by_camera,
                    visible_frames,
                    visible_camera_frames,
                ) = self._collect_world_point_visibility_detail(
                    point,
                    sensor_type=visibility_sensor,
                )
                if int(hits) < int(self.min_fov_hits):
                    continue
                visible_frame_tokens = sorted({
                    int(
                        self._frame_token_for_sensor_index(
                            visibility_sensor, frame_idx))
                    for frame_idx in visible_frames
                })
                frame_camera_tokens = {}
                for frame_idx, camera_names in visible_camera_frames.items():
                    frame_token = int(
                        self._frame_token_for_sensor_index(
                            visibility_sensor, frame_idx))
                    frame_camera_tokens[frame_token] = list(camera_names)
                    for camera_name in camera_names:
                        camera_frame_candidate_pool[camera_name][
                            frame_token] += 1
                selected_points.append(np.asarray(point, dtype=np.float64)[:3])
                selected_visibility_cache.append({
                    "visibility_score":
                    float(score),
                    "visibility_hits":
                    int(hits),
                    "visible_by_camera":
                    dict(visible_by_camera),
                    "visible_frames":
                    list(visible_frames),
                    "visible_frame_tokens":
                    visible_frame_tokens,
                    "frame_camera_tokens":
                    frame_camera_tokens,
                })
                if len(selected_points) >= int(visible_pool_cap):
                    break
            if len(selected_points) > 0:
                selected_points_arr = np.asarray(selected_points,
                                                 dtype=np.float64)
                if len(selected_points) > int(num_candidates):
                    _vis_scores, _vis_hits, _cam_counts, rank_scores = self._compute_visibility_rank_arrays(
                        selected_visibility_cache,
                        len(selected_points),
                    )
                    keep_idx = self._select_diverse_candidate_indices(
                        points_xyz=selected_points_arr[:, :3],
                        scores=rank_scores,
                        k=int(num_candidates),
                        diversity_weight=0.55,
                    )
                    random_points = selected_points_arr[keep_idx]
                    candidate_visibility_cache = [
                        copy.deepcopy(selected_visibility_cache[int(idx)])
                        for idx in keep_idx
                    ]
                else:
                    random_points = selected_points_arr
                    candidate_visibility_cache = selected_visibility_cache
                if len(selected_points) < int(num_candidates):
                    print(
                        f"\033[33mFOV-conditioned sampling: collected {len(selected_points)}/{num_candidates} "
                        f"camera-visible candidates (draw_budget={draw_budget}).\033[0m"
                    )
            else:

                random_points = points_ground[draw_indices[:num_candidates]]
                candidate_visibility_cache = []
                camera_frame_candidate_pool = defaultdict(
                    lambda: defaultdict(int))
                print(
                    "\033[33mFOV-conditioned sampling found no visible candidates; "
                    "falling back to unfiltered random candidate subset.\033[0m"
                )

            self._export_fov_conditioned_candidates_cloud(random_points,
                                                          eval=eval)
            self._export_fov_candidate_pool_summary(
                visibility_sensor=visibility_sensor,
                random_points=random_points,
                candidate_visibility_cache=candidate_visibility_cache,
                camera_frame_pool=camera_frame_candidate_pool,
                eval=eval,
            )

        if request_only:
            if self.placement_policy == 'gpt_oneshot' and visibility_sensor == 'camera':
                strong_request_points, strong_request_visibility_cache = (
                    self._prefilter_request_only_camera_candidates(
                        candidate_points=random_points,
                        candidate_visibility_cache=candidate_visibility_cache,
                        accumulated_points=points,
                        placement_pool_points=points_ground,
                        points_non_ground_by_frame=points_non_ground_by_frame,
                        visibility_sensor=visibility_sensor,
                        pose_sensor=pose_sensor,
                        has_lidar_semantic=has_lidar_semantic,
                        eval=eval,
                    ))
                self._ensure_oneshot_plan_cache(
                    accumulated_points=points,
                    placement_pool_points=points_ground,
                    candidate_points=strong_request_points,
                    candidate_visibility_cache=strong_request_visibility_cache,
                    points_non_ground_by_frame=points_non_ground_by_frame,
                    visibility_sensor=visibility_sensor,
                    pose_sensor=pose_sensor,
                    eval=eval,
                )
            else:
                print(
                    "\033[33mREQUEST-ONLY: placement_policy is not gpt_oneshot/camera-visible; "
                    "skipped one-shot request generation.\033[0m")
            return np.zeros((3, ))

        target_bucket_idx = self.ood_target_distance_bucket.get(int(ood_id))
        best_point = None
        best_score = -np.inf
        best_patch = None
        best_visible_frames = []
        best_visible_by_camera = {}
        best_min_distance = np.inf
        placement_reject_stats = defaultdict(int)
        if self.enable_occlusion_aware_sampling and visibility_sensor == 'camera':
            self._prepare_sampling_occlusion_cache()
        temporal_target_frame = self._get_temporal_target_frame(
            ood_id, visibility_sensor)
        temporal_total_frames = len(
            self.vehicle_pose.get(visibility_sensor, []))
        temporal_max_delta = self._resolve_temporal_spread_max_delta(
            temporal_total_frames)
        enforce_temporal = (self.enable_temporal_spread_sampling
                            and temporal_target_frame is not None
                            and temporal_total_frames >= 2)

        def evaluate_candidates(
            enforce_bucket=False,
            use_occlusion_filter=True,
            enforce_temporal_spread=True,
            candidate_points_subset=None,
        ):
            local_best_point = None
            local_best_score = -np.inf
            local_best_patch = None
            local_best_visible_frames = []
            local_best_visible_by_camera = {}
            local_best_min_distance = np.inf

            if candidate_points_subset is None:
                eval_points = random_points
            else:
                eval_points = np.asarray(candidate_points_subset,
                                         dtype=np.float64)
                if eval_points.ndim != 2 or eval_points.shape[
                        0] == 0 or eval_points.shape[1] < 3:
                    return (
                        local_best_point,
                        local_best_score,
                        local_best_patch,
                        local_best_visible_frames,
                        local_best_visible_by_camera,
                        local_best_min_distance,
                    )
                eval_points = eval_points[:, :3]

            for cand_idx, point in enumerate(eval_points):

                min_distance_to_pose = self._min_distance_to_pose(
                    point, pose_sensor)
                dist_limit = self.placement_max_distance_eval if eval else self.placement_max_distance
                if np.isfinite(
                        dist_limit
                ) and dist_limit > 0 and min_distance_to_pose >= dist_limit:
                    placement_reject_stats["distance_cap"] += 1
                    continue

                if enforce_bucket and target_bucket_idx is not None:
                    if not self._distance_in_bucket(min_distance_to_pose,
                                                    target_bucket_idx):
                        placement_reject_stats["bucket_mismatch"] += 1
                        continue

                neighborhood_idx = ground_tree.query_ball_point(
                    point, self.ground_plane_radius)
                if len(neighborhood_idx) < self.ground_plane_min_neighbors:
                    placement_reject_stats["insufficient_neighbors"] += 1
                    continue
                neighborhood = points_ground[neighborhood_idx]
                centered = neighborhood - np.mean(
                    neighborhood, axis=0, keepdims=True)
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                normal = vh[-1]
                if abs(float(normal[2])) < self.ground_plane_min_normal_z:
                    placement_reject_stats["slope_reject"] += 1
                    continue
                if (np.max(neighborhood[:, 2]) -
                        np.min(neighborhood[:, 2])) > 0.35:
                    placement_reject_stats["height_variation_reject"] += 1
                    continue

                candidate = np.array(point, copy=True)
                candidate[-1] = float(np.median(neighborhood[:, 2]))

                local_floor_idx = all_xy_tree.query_ball_point(
                    candidate[:2], self.local_floor_radius)
                if len(local_floor_idx) >= self.local_floor_min_neighbors:
                    local_floor_z = float(
                        np.percentile(
                            points[local_floor_idx, 2],
                            self.local_floor_percentile,
                        ))
                    if (float(candidate[2]) - local_floor_z) > float(
                            self.local_floor_max_offset):
                        placement_reject_stats["elevated_floor_reject"] += 1
                        continue

                if self._candidate_in_ego_swept_volume(candidate, pose_sensor):
                    placement_reject_stats["ego_swept_reject"] += 1
                    continue

                use_cached_visibility = (
                    candidate_points_subset is None
                    and visibility_sensor == 'camera'
                    and cand_idx < len(candidate_visibility_cache))
                if use_cached_visibility:
                    cached_visibility = candidate_visibility_cache[cand_idx]
                    visibility_score = float(
                        cached_visibility["visibility_score"])
                    visibility_hits = int(cached_visibility["visibility_hits"])
                    visible_by_camera = dict(
                        cached_visibility["visible_by_camera"])
                    visible_frames = list(cached_visibility["visible_frames"])
                    visible_frame_tokens = list(
                        cached_visibility["visible_frame_tokens"])
                else:
                    (
                        visibility_score,
                        visibility_hits,
                        visible_by_camera,
                        visible_frames,
                    ) = self._collect_world_point_visibility_stats(
                        candidate,
                        sensor_type=visibility_sensor,
                    )
                    visible_frame_tokens = [
                        int(
                            self._frame_token_for_sensor_index(
                                visibility_sensor, frame_idx))
                        for frame_idx in visible_frames
                    ]

                if (self.placement_fast_collision_radius > 0
                        and len(self.accepted_object_positions_world) > 0):
                    collide = False
                    for prev_point in self.accepted_object_positions_world:
                        dxy = np.linalg.norm(candidate[:2] - prev_point[:2])
                        if dxy < self.placement_fast_collision_radius:
                            if abs(
                                    float(candidate[2]) - float(prev_point[2])
                            ) <= self.placement_fast_collision_height_tol:
                                collide = True
                                break
                    if collide:
                        placement_reject_stats["collision_reject"] += 1
                        continue

                if len(non_ground_xy_tree_by_token) > 0:
                    seed_tokens = [int(t) for t in visible_frame_tokens]
                    if len(seed_tokens
                           ) == 0 and len(sorted_non_ground_tokens) > 0:
                        seed_tokens = [
                            int(sorted_non_ground_tokens[
                                len(sorted_non_ground_tokens) // 2])
                        ]
                    local_tokens = self._expand_temporal_token_window(
                        seed_tokens,
                        sorted_non_ground_tokens,
                        int(self.occupancy_temporal_window_frames),
                    )
                    if len(local_tokens) == 0:
                        local_tokens = list(sorted_non_ground_tokens)

                    occupied_points_count = 0
                    for token in local_tokens:
                        token = int(token)
                        tree = non_ground_xy_tree_by_token.get(token)
                        if tree is None:
                            continue
                        nearby_idx = tree.query_ball_point(
                            candidate[:2], self.occupied_xy_radius)
                        if not nearby_idx:
                            continue
                        nearby_points = non_ground_points_by_token[token][
                            nearby_idx]
                        vertical_mask = (
                            (nearby_points[:, 2] >
                             (candidate[2] + self.occupied_min_height))
                            & (nearby_points[:, 2] <
                               (candidate[2] + self.occupied_max_height)))
                        occupied_points_count += int(
                            np.count_nonzero(vertical_mask))
                        if occupied_points_count >= self.occupied_min_points:
                            break
                    if occupied_points_count >= self.occupied_min_points:
                        placement_reject_stats["occupied_reject"] += 1
                        continue

                if visibility_hits < self.min_fov_hits:
                    placement_reject_stats["fov_hits_reject"] += 1
                    continue

                if enforce_temporal_spread and enforce_temporal:
                    representative_frame = self._representative_frame_id(
                        visible_frames)
                    if representative_frame is None:
                        placement_reject_stats[
                            "temporal_no_visible_frame"] += 1
                        continue
                    if abs(
                            int(representative_frame) -
                            int(temporal_target_frame)) > int(
                                temporal_max_delta):
                        placement_reject_stats["temporal_spread_reject"] += 1
                        continue

                if (use_occlusion_filter
                        and self.enable_occlusion_aware_sampling
                        and visibility_sensor == 'camera'):
                    occ_stats = self._collect_world_point_occlusion_stats(
                        candidate,
                        sensor_type='camera',
                    )
                    clear_hits = int(occ_stats["clear_visible_hits"])
                    blocked_ratio = float(occ_stats["blocked_ratio"])
                    clear_by_camera = occ_stats["clear_by_camera"]
                    clear_camera_count = sum(1
                                             for v in clear_by_camera.values()
                                             if int(v) > 0)

                    if clear_hits < self.min_clear_hits:
                        placement_reject_stats["low_clear_hits_reject"] += 1
                        continue
                    if clear_camera_count < self.min_clear_cameras:
                        placement_reject_stats["low_clear_cameras_reject"] += 1
                        continue
                    if blocked_ratio > self.max_blocked_ratio:
                        placement_reject_stats["blocked_ratio_reject"] += 1
                        continue

                    visible_frames = occ_stats["clear_frames"]
                    visible_by_camera = clear_by_camera
                    visibility_hits = clear_hits
                    visibility_score = float(clear_hits) + (
                        self.side_camera_bonus * float(
                            clear_by_camera.get("front_left", 0) +
                            clear_by_camera.get("front_right", 0) +
                            clear_by_camera.get("side_left", 0) +
                            clear_by_camera.get("side_right", 0)))

                if (self.required_visible_cameras
                        and visibility_sensor == 'camera'):
                    missing_required = [
                        cam_name for cam_name in self.required_visible_cameras
                        if int(visible_by_camera.get(cam_name, 0)) <
                        self.required_visible_camera_min_hits
                    ]
                    if missing_required:
                        placement_reject_stats["required_camera_reject"] += 1
                        continue
                if (self.enable_adjacent_camera_overlap_filter
                        and visibility_sensor == 'camera'):
                    mismatch_pairs = self._adjacent_camera_overlap_mismatches(
                        visible_by_camera)
                    if mismatch_pairs:
                        placement_reject_stats["adjacent_overlap_reject"] += 1
                        continue

                if self.enable_camera_ground_anchor_filter and visibility_sensor == 'camera':
                    enforce_ground_anchor = True
                    if self.camera_ground_anchor_only_fallback and has_lidar_semantic:
                        enforce_ground_anchor = False
                    if enforce_ground_anchor:
                        frame_filter = list(visible_frames) if len(
                            visible_frames) > 0 else None
                        camera_filter = [
                            cam_name for cam_name, cam_hits in
                            visible_by_camera.items() if int(cam_hits) > 0
                        ]
                        if len(camera_filter) == 0:
                            camera_filter = None
                        ground_stats = self._collect_world_point_ground_label_stats(
                            candidate,
                            sensor_type='camera',
                            frame_ids=frame_filter,
                            camera_names=camera_filter,
                        )
                        if ground_stats[
                                "label_hits"] < self.camera_ground_min_hits:
                            placement_reject_stats[
                                "ground_anchor_low_label_hits"] += 1
                            continue
                        if ground_stats[
                                "ground_hits"] < self.camera_ground_min_hits:
                            placement_reject_stats[
                                "ground_anchor_low_hits"] += 1
                            continue
                        if ground_stats[
                                "ground_camera_count"] < self.camera_ground_min_cameras:
                            placement_reject_stats[
                                "ground_anchor_low_cameras"] += 1
                            continue
                        if float(ground_stats["ground_ratio"]
                                 ) < self.camera_ground_min_ratio:
                            placement_reject_stats[
                                "ground_anchor_low_ratio"] += 1
                            continue
                        if float(ground_stats["non_ground_ratio"]
                                 ) > self.camera_non_ground_max_ratio:
                            placement_reject_stats[
                                "ground_anchor_high_non_ground_ratio"] += 1
                            continue

                if self.max_objects_per_frame is not None:
                    frame_overflow = False
                    for visible_frame in visible_frames:
                        if self.frame_visibility_load.get(
                                int(visible_frame),
                                0) >= self.max_objects_per_frame:
                            frame_overflow = True
                            break
                    if frame_overflow:
                        placement_reject_stats["frame_load_reject"] += 1
                        continue

                dominant_camera = None
                dominant_hits = 0
                for cam_name, cam_hits in visible_by_camera.items():
                    if int(cam_hits) > dominant_hits:
                        dominant_hits = int(cam_hits)
                        dominant_camera = cam_name
                if dominant_camera is not None and dominant_hits > 0:
                    quota_limit = self._get_camera_quota_limit(dominant_camera)
                    if quota_limit is not None:
                        if self.camera_fov_counts.get(dominant_camera,
                                                      0) >= int(quota_limit):
                            placement_reject_stats["camera_quota_reject"] += 1
                            continue

                num_visible_cameras = sum(1
                                          for v in visible_by_camera.values()
                                          if v > 0)
                candidate_score = visibility_score + (0.25 *
                                                      num_visible_cameras)
                if candidate_score > local_best_score:
                    local_best_score = candidate_score
                    local_best_point = candidate
                    local_best_patch = neighborhood
                    local_best_visible_frames = list(visible_frames)
                    local_best_visible_by_camera = dict(visible_by_camera)
                    local_best_min_distance = float(min_distance_to_pose)

            return (
                local_best_point,
                local_best_score,
                local_best_patch,
                local_best_visible_frames,
                local_best_visible_by_camera,
                local_best_min_distance,
            )

        enforce_bucket = bool(target_bucket_idx is not None
                              and self.enable_distance_buckets)
        lock_primary_active = (self.placement_policy == 'gpt_oneshot'
                               and self.placement_gpt_lock_primary
                               and visibility_sensor == 'camera')

        policy_candidate_subset = self._build_policy_candidate_subset(
            ood_id=ood_id,
            accumulated_points=points,
            placement_pool_points=points_ground,
            candidate_points=random_points,
            candidate_visibility_cache=candidate_visibility_cache,
            points_non_ground_by_frame=points_non_ground_by_frame,
            visibility_sensor=visibility_sensor,
            pose_sensor=pose_sensor,
            eval=eval,
        )

        if lock_primary_active and (policy_candidate_subset is None
                                    or len(policy_candidate_subset) == 0):
            print(
                f"\033[33mPrimary placement: no candidate available for OOD {ood_id}; "
                "skipping object.\033[0m")
            return np.zeros((3, ))

        if policy_candidate_subset is not None and len(
                policy_candidate_subset) > 0:
            if lock_primary_active:

                forced_point = np.asarray(policy_candidate_subset[0],
                                          dtype=np.float64)[:3]
                (
                    visibility_score,
                    _visibility_hits,
                    visible_by_camera,
                    visible_frames,
                ) = self._collect_world_point_visibility_stats(
                    forced_point,
                    sensor_type=visibility_sensor,
                )
                best_point = forced_point
                best_score = float(visibility_score)
                best_patch = None
                best_visible_frames = list(visible_frames)
                best_visible_by_camera = dict(visible_by_camera)
                best_min_distance = float(
                    self._min_distance_to_pose(forced_point, pose_sensor))
                print(
                    f"\033[36mPrimary placement: bypassing placement filters for OOD {ood_id}; "
                    "using primary candidate directly.\033[0m")
            else:
                (
                    best_point,
                    best_score,
                    best_patch,
                    best_visible_frames,
                    best_visible_by_camera,
                    best_min_distance,
                ) = evaluate_candidates(
                    enforce_bucket=False,
                    use_occlusion_filter=True,
                    enforce_temporal_spread=False,
                    candidate_points_subset=policy_candidate_subset,
                )
        else:
            best_point = None
            best_score = -np.inf
            best_patch = None
            best_visible_frames = []
            best_visible_by_camera = {}
            best_min_distance = np.inf

        if best_point is None:
            (
                best_point,
                best_score,
                best_patch,
                best_visible_frames,
                best_visible_by_camera,
                best_min_distance,
            ) = evaluate_candidates(
                enforce_bucket=enforce_bucket,
                use_occlusion_filter=True,
                enforce_temporal_spread=True,
            )

        if best_point is None and enforce_bucket:
            if self.strict_spread_controls:
                print(
                    f"\033[33mStrict spread: distance-bucket gate kept for OOD {ood_id}; "
                    "not relaxing bucket filter.\033[0m")
            else:
                bucket_bounds = self._bucket_bounds(target_bucket_idx)
                if bucket_bounds is not None:
                    lo, hi = bucket_bounds
                    print(
                        f"\033[33mPlacement fallback for OOD {ood_id}: no candidates in distance bucket "
                        f"[{lo:.1f}, {hi:.1f}); retrying without bucket filter.\033[0m"
                    )
                (
                    best_point,
                    best_score,
                    best_patch,
                    best_visible_frames,
                    best_visible_by_camera,
                    best_min_distance,
                ) = evaluate_candidates(
                    enforce_bucket=False,
                    use_occlusion_filter=True,
                    enforce_temporal_spread=True,
                )

        if best_point is None and enforce_temporal:
            if self.strict_spread_controls:
                print(
                    f"\033[33mStrict spread: temporal gate kept for OOD {ood_id}; "
                    "not relaxing temporal spread filter.\033[0m")
            else:
                print(
                    f"\033[33mPlacement fallback for OOD {ood_id}: "
                    "temporal spread filter rejected all candidates; retrying without temporal gate.\033[0m"
                )
                (
                    best_point,
                    best_score,
                    best_patch,
                    best_visible_frames,
                    best_visible_by_camera,
                    best_min_distance,
                ) = evaluate_candidates(
                    enforce_bucket=False,
                    use_occlusion_filter=True,
                    enforce_temporal_spread=False,
                )

        if (best_point is None and self.enable_occlusion_aware_sampling
                and visibility_sensor == 'camera'):
            print(
                f"\033[33mPlacement fallback for OOD {ood_id}: "
                "occlusion-aware filters rejected all candidates; retrying without occlusion gate.\033[0m"
            )
            (
                best_point,
                best_score,
                best_patch,
                best_visible_frames,
                best_visible_by_camera,
                best_min_distance,
            ) = evaluate_candidates(
                enforce_bucket=False,
                use_occlusion_filter=False,
                enforce_temporal_spread=False,
            )

        if best_point is not None:
            if self.enable_occlusion_aware_sampling:
                summary = ", ".join(
                    f"{k}={int(v)}"
                    for k, v in sorted(placement_reject_stats.items()))
                if summary:
                    print(
                        f"\033[36mPlacement filter stats for OOD {ood_id}: {summary}\033[0m"
                    )
            placement_surface_stats = self._empty_ground_label_stats()
            placement_visible_frame_tokens = []
            if visibility_sensor == 'camera':
                best_camera_filter = [
                    cam_name
                    for cam_name, cam_hits in best_visible_by_camera.items()
                    if int(cam_hits) > 0
                ]
                if len(best_camera_filter) == 0:
                    best_camera_filter = None
                placement_surface_stats = self._collect_world_point_ground_label_stats(
                    best_point,
                    sensor_type='camera',
                    frame_ids=best_visible_frames
                    if len(best_visible_frames) > 0 else None,
                    camera_names=best_camera_filter,
                )
                placement_visible_frame_tokens = sorted({
                    int(
                        self._frame_token_for_sensor_index(
                            visibility_sensor, frame_idx))
                    for frame_idx in best_visible_frames
                })
                if self.enable_camera_ground_anchor_filter:
                    print(
                        f"\033[36mPlacement ground-anchor stats for OOD {ood_id}: "
                        f"ground_hits={placement_surface_stats['ground_hits']}/"
                        f"{placement_surface_stats['label_hits']} "
                        f"(ratio={placement_surface_stats['ground_ratio']:.2f}), "
                        f"ground_cameras={placement_surface_stats['ground_camera_count']}, "
                        f"non_ground_ratio={placement_surface_stats['non_ground_ratio']:.2f}\033[0m"
                    )
                if ood_id not in self.ood_details:
                    self.ood_details[ood_id] = {}
                self.ood_details[ood_id]['placement_surface_type'] = str(
                    placement_surface_stats.get("surface_type", "unknown"))
                self.ood_details[ood_id][
                    'placement_dominant_surface_label'] = str(
                        placement_surface_stats.get("dominant_semantic_label",
                                                    "unknown"))
                self.ood_details[ood_id][
                    'placement_dominant_surface_semantic_id'] = (
                        placement_surface_stats.get("dominant_semantic_class",
                                                    None))
                self.ood_details[ood_id]['placement_surface_label_hits'] = int(
                    placement_surface_stats.get("label_hits", 0))
                self.ood_details[ood_id][
                    'placement_surface_ground_hits'] = int(
                        placement_surface_stats.get("ground_hits", 0))
                self.ood_details[ood_id][
                    'placement_surface_ground_ratio'] = float(
                        placement_surface_stats.get("ground_ratio", 0.0))
                self.ood_details[ood_id][
                    'placement_surface_non_ground_ratio'] = float(
                        placement_surface_stats.get("non_ground_ratio", 0.0))
                self.ood_details[ood_id][
                    'placement_surface_label_counts'] = dict(
                        placement_surface_stats.get("semantic_label_counts",
                                                    {}))
                self.ood_details[ood_id]['placement_plane_type_counts'] = dict(
                    placement_surface_stats.get("surface_type_counts", {}))
                self.ood_details[ood_id]['placement_visible_frames'] = [
                    int(v) for v in best_visible_frames
                ]
                self.ood_details[ood_id]['placement_visible_frame_tokens'] = [
                    int(v) for v in placement_visible_frame_tokens
                ]
                self.ood_details[ood_id]['placement_visible_by_camera'] = dict(
                    best_visible_by_camera)
                self.ood_details[ood_id]['placement_visible_cameras'] = [
                    cam_name
                    for cam_name, cam_hits in best_visible_by_camera.items()
                    if int(cam_hits) > 0
                ]
                self.ood_details[ood_id][
                    'placement_runtime_visibility_score'] = float(best_score)
            self.ood_location[
                ood_id] = best_patch if best_patch is not None else np.zeros(
                    (0, 3))

            self._register_accepted_placement(
                ood_id=ood_id,
                point_world=best_point,
                visibility_sensor=visibility_sensor,
            )
            if self.enable_distance_buckets:
                bucket_idx = self.ood_target_distance_bucket.get(int(ood_id))
                if bucket_idx is None or not self._distance_in_bucket(
                        best_min_distance, bucket_idx):
                    bucket_idx = self._distance_to_bucket_index(
                        best_min_distance)
                if bucket_idx is not None:
                    self.ood_resolved_distance_bucket[int(ood_id)] = int(
                        bucket_idx)

            search_location_path = self._get_search_locations_dir(eval=eval)
            output_dir = f"{search_location_path}/ood_{ood_id}/{sensor}"
            os.makedirs(output_dir, exist_ok=True)
            try:
                debug_pcd = o3d.geometry.PointCloud()
                debug_pcd.points = o3d.utility.Vector3dVector(best_patch)
                debug_pcd.paint_uniform_color([0, 1, 0])
                self._write_point_cloud_dual_formats(
                    f"{output_dir}/best_candidate_ground_patch.pcd",
                    debug_pcd,
                )
            except Exception:
                pass

            self._export_sampling_selected_spread(eval=eval, finalize=False)
            return best_point
        if self.enable_occlusion_aware_sampling and placement_reject_stats:
            summary = ", ".join(
                f"{k}={int(v)}"
                for k, v in sorted(placement_reject_stats.items()))
            print(
                f"\033[33mPlacement rejection summary for OOD {ood_id}: {summary}\033[0m"
            )
        return np.zeros((3, ))

    def check_points_on_road_by_ransac(self, pcd_points, max_planes=3):
        """
        Uses RANSAC to find a dominant plane and samples points near a target z-value.
        """

        remaining_points = pcd_points.copy()
        all_plane_models = []
        all_plane_inliers = []

        z = pcd_points[:, 2]
        z_range = np.percentile(z, 95) - np.percentile(z, 5)
        tolerance = 0.01 * z_range
        ransac_n = 6
        num_iterations = int(np.clip(len(pcd_points) / 100, 100, 3000))

        for i in range(max_planes):
            if len(remaining_points) < 500:
                break

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(remaining_points)

            plane_model, inliers = pcd.segment_plane(
                distance_threshold=tolerance,
                ransac_n=ransac_n,
                num_iterations=num_iterations)

            inlier_points = remaining_points[inliers]
            normal = np.array(plane_model[:3])
            normal /= np.linalg.norm(normal)

            if np.abs(normal[2]) < 0.9:
                break

            all_plane_models.append(plane_model)
            all_plane_inliers.append(inlier_points)

            mask = np.ones(len(remaining_points), dtype=bool)
            mask[inliers] = False
            remaining_points = remaining_points[mask]

        road_points = np.vstack(
            all_plane_inliers) if all_plane_inliers else np.empty((0, 3))
        return all_plane_models, road_points

    def sample_points_by_z(self, pcd_points, tolerance=0.1, grid_size=3):

        target_z = np.min(pcd_points[:, -1])
        min_points_per_grid = len(pcd_points) // 10

        x_bins = np.floor(pcd_points[:, 0] / grid_size)
        y_bins = np.floor(pcd_points[:, 1] / grid_size)

        grid_dict = {}
        for i, (x, y, z) in enumerate(zip(x_bins, y_bins, pcd_points[:, 2])):
            key = (x, y)
            if key not in grid_dict:
                grid_dict[key] = []
            grid_dict[key].append(i)

        selected_indices = []
        for indices in grid_dict.values():
            if len(indices) < min_points_per_grid:
                continue

            avg_z = np.mean(pcd_points[indices, 2])
            if abs(avg_z - target_z) <= tolerance:
                selected_indices.extend(indices)

        if len(selected_indices) == 0:
            return np.empty((0, 3))

        num_samples = int(len(selected_indices))
        sampled_indices = self.seed.random.choice(selected_indices,
                                                  num_samples,
                                                  replace=False)

        return pcd_points[sampled_indices]

    def get_range_image_top_pose(self):
        active_lidar_sensor_types = self._get_active_lidar_pipeline_sensor_types(
        )
        if self.raycast_lidar_only and len(active_lidar_sensor_types) > 0:
            sensor_types = active_lidar_sensor_types
        else:
            sensor_types = [
                sensor_type
                for sensor_type, frames in self.frames_with_seg.items()
                if len(frames) > 0
            ]

        for sensor_type in sensor_types:
            frames = self.frames_with_seg.get(sensor_type, [])
            if len(frames) == 0:
                continue
            if sensor_type not in self.lidar_range_pose.keys():
                self.lidar_range_pose[sensor_type] = {}
            for frame_id, frame in enumerate(frames):
                _, _, _, range_image_top_pose = self._get_cached_range_parse(
                    sensor_type, frame_id, frame)

                pose_shape = list(range_image_top_pose.shape.dims)
                pose_np = np.array(range_image_top_pose.data,
                                   dtype=np.float64).reshape(pose_shape)

                pose_range_image = _v2_lidar.PoseRangeImage(
                    values=pose_np.flatten().tolist(), shape=[64, 2650, 6])

                range_image_top_pose_tf = lidar_utils.convert_lidar_pose_range_image_to_transformation(
                    pose_range_image)
                self.lidar_range_pose[sensor_type][
                    frame_id] = range_image_top_pose_tf.numpy()

        np.save(
            f"{self.base_dir}/{self.file_name}/lidar_data/lidar_range_pose.npy",
            self.lidar_range_pose,
            allow_pickle=True)

    def save_ood_details(self, eval=False):
        def _to_json_value(value):
            if value is None:
                return None
            if isinstance(value, np.ndarray):
                return value.tolist()
            return value

        def _serialize_pose(value):
            pose_arr = self._normalize_pose_4x4(value)
            return pose_arr.tolist() if pose_arr is not None else None

        print(f"Saving ood_details for {len(self.ood_details)} objects")
        ood_details_serializable = {}
        for ood_id, details in self.ood_details.items():
            sampled_position = _to_json_value(details.get("position"))
            if isinstance(sampled_position, np.ndarray):
                sampled_position = sampled_position.tolist()
            entry = {
                "file_identifier":
                details["file_identifier"]
                if "file_identifier" in details else None,
                "lvis":
                details["lvis"],
                "path":
                details["path"],
                "scale":
                float(details["scale"]) if "scale" in details else 1.0,
                "sampled_position":
                sampled_position,
                "pose":
                _serialize_pose(details.get("pose")),
            }
            for extra_key in [
                    "base_scale",
                    "effective_scale",
                    "render_pose_override",
                    "shadow_plane_point_override",
                    "shadow_plane_normal_override",
                    "ground_contact_epsilon_override",
                    "shadow_catcher_size_override",
                    "distance_to_ego_m",
                    "distance_scale_multiplier",
                    "target_ref_dim_m",
                    "target_ref_dim_source",
                    "target_ref_dim_raw",
                    "target_max_dim_m",
                    "scale_policy",
                    "size_bucket",
                    "description",
                    "candidate_id",
                    "source_uid",
                    "pool_index",
                    "placement_slot_index",
                    "placement_candidate_id",
                    "placement_point_xyz",
                    "placement_distance_m",
                    "placement_distance_q",
                    "placement_visibility_score",
                    "placement_visible_camera_count",
                    "placement_plan",
                    "placement_surface_type",
                    "placement_dominant_surface_label",
                    "placement_dominant_surface_semantic_id",
                    "placement_surface_label_hits",
                    "placement_surface_ground_hits",
                    "placement_surface_ground_ratio",
                    "placement_surface_non_ground_ratio",
                    "placement_surface_label_counts",
                    "placement_plane_type_counts",
                    "placement_visible_frames",
                    "placement_visible_frame_tokens",
                    "placement_visible_by_camera",
                    "placement_visible_cameras",
                    "placement_runtime_visibility_score",
            ]:
                if extra_key in details:
                    entry[extra_key] = _to_json_value(details.get(extra_key))
            ood_details_serializable[ood_id] = entry

        if not eval:
            json_path = f"{self.base_dir}/{self.file_name}/ood_details.json"
        else:
            json_path = f"{self.base_dir}/{self.file_name}/Evaluation/ood_details.json"
        with open(json_path, "w") as json_file:
            json.dump(ood_details_serializable, json_file, indent=4)

    def _normalize_pose_4x4(self, pose_value):
        """Return pose as float64 (4,4) array, or None when invalid."""
        if pose_value is None:
            return None
        try:
            pose_arr = np.asarray(pose_value, dtype=np.float64)
        except Exception:
            return None
        if pose_arr.ndim == 0:
            return None
        if pose_arr.ndim == 1:
            if pose_arr.size == 16:
                pose_arr = pose_arr.reshape(4, 4)
            else:
                return None
        if pose_arr.shape != (4, 4):
            return None
        if not np.isfinite(pose_arr).all():
            return None

        linear = pose_arr[:3, :3]
        if float(np.linalg.norm(linear)) < 1e-8:
            repaired = np.eye(4, dtype=np.float64)
            repaired[:3, 3] = pose_arr[:3, 3]
            return repaired
        return pose_arr

    def _has_valid_pose(self, details):
        return self._normalize_pose_4x4(details.get('pose')) is not None

    def _resolve_lidar_insertion_sensor_type(self):
        preferred = self.lidar_insertion_frame_source
        camera_available = len(self.frames_with_seg.get('camera', [])) > 0
        lidar_available = len(self.frames_with_seg.get('lidar', [])) > 0

        if preferred == 'camera':
            if camera_available:
                return 'camera'
            if lidar_available:
                return 'lidar'
            return None

        if preferred == 'lidar':
            if lidar_available:
                return 'lidar'
            if camera_available:
                return 'camera'
            return None

        if camera_available:
            return 'camera'
        if lidar_available:
            return 'lidar'
        return None

    def _get_active_lidar_pipeline_sensor_types(self):
        sensor_type = getattr(self, 'lidar_insertion_sensor_type', None)
        if sensor_type and len(self.frames_with_seg.get(sensor_type, [])) > 0:
            return [sensor_type]
        return [
            sensor_name
            for sensor_name, frames in self.frames_with_seg.items()
            if len(frames) > 0 and sensor_name != 'camera'
        ]

    def save_framewise_ood_details(self):

        print(self.framewise_ood_details)
        output_root = f'{self.base_dir}/{self.file_name}/Point_Cloud_Metadata'
        for sensor_type, framewise_data in self.framewise_ood_details.items():
            if sensor_type not in self._get_active_lidar_pipeline_sensor_types(
            ):
                continue
            for frame_id, ood_data in framewise_data.items():
                data = self.framewise_ood_details[sensor_type][frame_id]

                ood_details_serializable = {
                    ood_id: {
                        "num_points":
                        details["num_points"]
                        if "num_points" in details else None,
                        "num_points_reconstructed":
                        details["num_points_reconstructed"]
                        if "num_points_reconstructed" in details else None,
                        "center_x":
                        details["center_x"] if "center_x" in details else None,
                        "center_y":
                        details["center_y"] if "center_y" in details else None,
                        "center_z":
                        details["center_z"] if "center_z" in details else None,
                        "length":
                        details["length"] if "length" in details else None,
                        "width":
                        details["width"] if "width" in details else None,
                        "height":
                        details["height"] if "height" in details else None,
                        "heading":
                        details["heading"] if "heading" in details else None,
                    }
                    for ood_id, details in data.items()
                }

                dir_path = os.path.join(output_root, sensor_type)
                os.makedirs(dir_path, exist_ok=True)

                file_path = os.path.join(
                    dir_path,
                    f"frame_id_00{self.frame_map[sensor_type][frame_id]}.json")

                with open(file_path, "w") as f:
                    json.dump(ood_details_serializable, f, indent=4)

    def get_color_point_cloud(self, frame, i):

        save_dir = f'{self.base_dir}/Domain/{self.file_name}/Point Cloud Segmentation Data/'

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'velodyne'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'labels'), exist_ok=True)
        if Waymo2SemanticKITTI is None:
            raise RuntimeError(
                "waymo_to_semanticKITTI is required only for point-cloud color export."
            )
        converter = Waymo2SemanticKITTI(save_dir)
        converter.convert_one(frame, file_idx=f'frame_{i + 1}')

    def project_points_to_image_plane(self, eval=False):

        method = 'lidar'
        lidar = 'variable'

        for sensor_type, frames in self.frames_with_seg.items():
            if sensor_type == 'camera':
                continue
            if len(frames) == 0:
                continue
            for frame_id, frame in enumerate(frames):

                pose_matrix = np.array(
                    self.vehicle_pose[sensor_type][frame_id])
                vehicle_points = np.array(
                    self.pc_reconstructed_indices[sensor_type]
                    [frame_id].points)
                vehicle_colors = np.array(
                    self.pc_reconstructed_indices[sensor_type]
                    [frame_id].colors)

                world_points = (pose_matrix @ np.hstack(
                    (vehicle_points, np.ones(
                        (vehicle_points.shape[0], 1)))).T).T[:, :3]

                panoptic_labels = self.reconstructed_panoptic_labels[
                    sensor_type][frame_id]
                print('unique panoptic labels',
                      np.unique(np.asarray(panoptic_labels)))
                print(
                    f'\033[33mInitial shapes - vehicle_points: {vehicle_points.shape}, panoptic_labels: {panoptic_labels.shape}\033[0m'
                )

                calibration = frame.context.camera_calibrations[0]

                extrinsic = tf.reshape(
                    tf.constant(list(calibration.extrinsic.transform),
                                dtype=tf.float32), [4, 4])

                extrinsic_global = pose_matrix @ extrinsic.numpy()
                extrinsic_global = tf.constant(extrinsic_global,
                                               dtype=tf.float32)
                intrinsic = tf.constant(list(calibration.intrinsic),
                                        dtype=tf.float32)
                metadata = tf.constant(
                    [
                        calibration.width,
                        calibration.height,
                        calibration.rolling_shutter_direction,
                    ],
                    dtype=tf.int32,
                )
                image = frame.images[0]

                camera_image_metadata = list(image.pose.transform) + [
                    image.velocity.v_x,
                    image.velocity.v_y,
                    image.velocity.v_z,
                    image.velocity.w_x,
                    image.velocity.w_y,
                    image.velocity.w_z,
                    image.pose_timestamp,
                    image.shutter,
                    image.camera_trigger_time,
                    image.camera_readout_done_time,
                ]

                if py_camera_model_ops is None:

                    points_camera_space = np.zeros((world_points.shape[0], 3),
                                                   dtype=np.float32)
                    points_camera_space[:, 2] = -1.0

                    points_vehicle_h = np.hstack(
                        (vehicle_points, np.ones(
                            (vehicle_points.shape[0], 1))))
                    camera_extrinsic = np.asarray(
                        self.camera_data[sensor_type][frame_id]
                        ['extrinsic']).reshape(4, 4)
                    points_camera = (np.linalg.inv(camera_extrinsic)
                                     @ points_vehicle_h.T).T[:, :3]

                    valid_forward = points_camera[:, 0] > 1e-6
                    if np.any(valid_forward):
                        X_cam = -points_camera[valid_forward, 1]
                        Y_cam = -points_camera[valid_forward, 2]
                        Z_cam = points_camera[valid_forward, 0]
                        fx, fy, cx, cy = self.camera_data[sensor_type][
                            frame_id]['projection']

                        u = (fx * X_cam / Z_cam) + cx
                        v = (fy * Y_cam / Z_cam) + cy

                        valid_indices = np.where(valid_forward)[0]
                        points_camera_space[valid_indices, 0] = u
                        points_camera_space[valid_indices, 1] = v
                        points_camera_space[valid_indices, 2] = 1.0
                else:
                    points_camera_space = py_camera_model_ops.world_to_image(
                        extrinsic, intrinsic, metadata, camera_image_metadata,
                        world_points).numpy()
                print(
                    f'\033[33mProjected points shape: {points_camera_space.shape}\033[0m'
                )
                points_camera_space_int = np.rint(points_camera_space).astype(
                    int)
                image_width = 1920
                image_height = 1280
                u = points_camera_space_int[:, 0]
                v = points_camera_space_int[:, 1]
                valid_uv = ((u >= 0) & (u < image_width) & (v >= 0) &
                            (v < image_height) &
                            (points_camera_space_int[:, 2] == 1))
                points_camera_space_int = points_camera_space_int[valid_uv]
                point_colors_camera_space = vehicle_colors[valid_uv]
                panoptic_labels_valid = panoptic_labels[valid_uv]

                print(
                    f'\033[33mAfter valid_uv filter - points: {points_camera_space_int.shape}, labels: {panoptic_labels_valid.shape}\033[0m'
                )
                print('unique panoptic labels after uv filter:',
                      np.unique(panoptic_labels_valid))

                points_camera_space_uv = points_camera_space_int[:, :2]

                points_camera_space_int_set = set(
                    zip(points_camera_space_uv[:, 0],
                        points_camera_space_uv[:, 1]))

                pixel_to_panoptic_map = {
                    tuple(point): label
                    for point, label in zip(points_camera_space_uv,
                                            panoptic_labels_valid)
                }
                pixel_to_color_map = {
                    tuple(point): color
                    for point, color in zip(points_camera_space_uv,
                                            point_colors_camera_space)
                }

                print(
                    f'final shapes: {world_points.shape}, {points_camera_space_int.shape}'
                )

                all_occlusion_pixels_to_visualize = []
                if sensor_type == 'lidar':
                    for ood_id in self.ood_details.keys():
                        if ood_id not in self.ood_occlusion_pixels.keys():
                            self.ood_occlusion_pixels[ood_id] = {}
                        if frame_id not in self.ood_occlusion_pixels[
                                ood_id].keys():
                            self.ood_occlusion_pixels[ood_id][frame_id] = {}
                        if eval:
                            render_path = f"{self.base_dir}/{self.file_name}/Evaluation/Renders/ood_{ood_id}/{sensor_type}/no_shadow_frame_00{self.frame_map[sensor_type][frame_id]}.png"
                        else:
                            render_path = f"{self.base_dir}/{self.file_name}/Renders/ood_{ood_id}/{sensor_type}/no_shadow_frame_00{self.frame_map[sensor_type][frame_id]}.png"
                        if os.path.exists(render_path):
                            ood_img = Image.open(render_path)
                            ood_array = np.array(ood_img)
                            ood_mask = self._extract_alpha_mask_with_fallback(
                                ood_array,
                                hard_alpha_threshold=100,
                                min_pixels=int(
                                    max(1, self.required_render_min_pixels)),
                            )
                            ood_mask_pixels = np.where(ood_mask)
                            ood_mask_pixel_set = set(
                                zip(ood_mask_pixels[1], ood_mask_pixels[0]))
                            ood_occlusion_pixels = points_camera_space_int_set.intersection(
                                ood_mask_pixel_set)

                            ood_occlusion_panoptics = [
                                pixel_to_panoptic_map[pixel]
                                for pixel in ood_occlusion_pixels
                                if pixel in pixel_to_panoptic_map
                            ]

                            panoptic_divisor = int(
                                max(
                                    1,
                                    getattr(self, "panoptic_label_divisor",
                                            1000)))
                            unique_occlusion_panoptics = set([
                                p for p in ood_occlusion_panoptics
                                if p % panoptic_divisor != 0
                            ])
                            ood_panoptic_start = self.ood_semantic_class * panoptic_divisor
                            ood_panoptics = set(
                                range(ood_panoptic_start,
                                      ood_panoptic_start + panoptic_divisor))
                            unique_occlusion_panoptics = unique_occlusion_panoptics - ood_panoptics
                            print(
                                f"Unique occlusion panoptics: {unique_occlusion_panoptics}"
                            )

                            for panoptic_id in unique_occlusion_panoptics:
                                pixels_for_panoptic = [
                                    pixel for pixel, label in
                                    pixel_to_panoptic_map.items()
                                    if label == panoptic_id
                                ]
                                if pixels_for_panoptic:
                                    self.ood_occlusion_pixels[ood_id][
                                        frame_id][int(
                                            panoptic_id)] = pixels_for_panoptic
                                    all_occlusion_pixels_to_visualize.extend(
                                        pixels_for_panoptic)

                    render_file = f"{self.image_path}/{self.file_name}/merged_without_occlusion/{sensor_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}.jpg"
                else:
                    render_file = f"{self.image_path}/{self.file_name}/Evaluation/merged_without_occlusion/{sensor_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}.jpg"
                ood_image = Image.open(render_file)
                ood_array = np.array(ood_image)

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(ood_array)

                ax.scatter(points_camera_space_int[:, 0],
                           points_camera_space_int[:, 1],
                           c=point_colors_camera_space,
                           s=0.4,
                           alpha=0.6,
                           marker='x',
                           label='Reconstructed Points')

                if all_occlusion_pixels_to_visualize:
                    self.occlusion_pixels_from_projection[
                        frame_id] = all_occlusion_pixels_to_visualize
                    occlusion_pixels_np = np.array(
                        all_occlusion_pixels_to_visualize)

                    import matplotlib.cm as cm

                    panoptic_ids = [
                        pixel_to_panoptic_map.get(tuple(p))
                        for p in occlusion_pixels_np
                    ]
                    unique_panoptics = list(
                        set([pid for pid in panoptic_ids if pid is not None]))

                    print(
                        f"Frame {frame_id}: Visualizing {len(occlusion_pixels_np)} occlusion pixels. Panoptic IDs found: {unique_panoptics}"
                    )

                    colors = cm.rainbow(
                        np.linspace(0, 1, len(unique_panoptics)))

                    for pid, color in zip(unique_panoptics, colors):

                        mask = [pid == p_id for p_id in panoptic_ids]
                        pix_np = occlusion_pixels_np[mask]

                        if len(pix_np) > 0:
                            ax.scatter(pix_np[:, 0],
                                       pix_np[:, 1],
                                       color=color,
                                       s=5,
                                       marker='o',
                                       alpha=0.5,
                                       label=f'Panoptic {pid}')

                            center = np.mean(pix_np, axis=0)
                            ax.text(center[0],
                                    center[1],
                                    str(pid),
                                    color='white',
                                    fontsize=8,
                                    fontweight='bold',
                                    bbox=dict(facecolor='black',
                                              alpha=0.5,
                                              pad=1))

                else:
                    self.occlusion_pixels_from_projection[frame_id] = []

                ax.grid(False)
                ax.axis('off')
                fig.tight_layout(pad=0)
                path = f'{self.base_dir}/{self.file_name}/Points_Projected_in_Image/{sensor_type}'
                os.makedirs(path, exist_ok=True)
                plt.savefig(
                    f"{path}/frame_00{self.frame_map[sensor_type][frame_id]}.png",
                    dpi=160,
                    bbox_inches='tight',
                    pad_inches=0)
                plt.close()

    def project_points_to_image_plane_via_formula(self):
        print('In method projection')

        lidar_list = ['variable']
        method_list = ['lidar']

        for sensor_type in self.frames_with_seg.keys():
            if len(self.frames_with_seg[sensor_type]) == 0:
                continue
            for lidar in lidar_list:
                for method in method_list:
                    print(f'Processing {sensor_type}, {lidar}, {method}')
                    for frame_id, projections in self.cp_actual[
                            sensor_type].items():

                        points_vehicle_filtered = np.asarray(
                            self.reconstructed_ood_points[sensor_type]
                            [frame_id].points)

                        N = points_vehicle_filtered.shape[0]
                        points_vehicle_homogeneous = np.hstack(
                            (points_vehicle_filtered, np.ones((N, 1))))

                        transformation = np.linalg.inv(
                            np.asarray(self.camera_data[sensor_type][frame_id]
                                       ['extrinsic']).reshape(4, 4))

                        points_camera_reconstructed = (
                            transformation
                            @ points_vehicle_homogeneous.T).T[:, :3]

                        valid = points_camera_reconstructed[:, 0] > 0
                        X_cam = -points_camera_reconstructed[valid, 1]
                        Y_cam = -points_camera_reconstructed[valid, 2]
                        Z_cam = points_camera_reconstructed[valid, 0]

                        fx, fy, cx, cy = self.camera_data[sensor_type][
                            frame_id]['projection']

                        u = (fx * X_cam / Z_cam) + cx
                        v = (fy * Y_cam / Z_cam) + cy

                        image_width = 1920
                        image_height = 1280
                        valid_uv = (u >= 0) & (u < image_width) & (v >= 0) & (
                            v < image_height)

                        final_valid = valid.copy()
                        final_valid[valid] = valid_uv

                        u_ood_reconstructed_indices = u[valid_uv]
                        v_ood_reconstructed_indices = v[valid_uv]

                        points_vehicle_filtered = np.asarray(
                            self.filtered_points[sensor_type][frame_id].points)
                        colors_vehicle_filtered = np.asarray(
                            self.filtered_points[sensor_type][frame_id].colors)

                        N = points_vehicle_filtered.shape[0]
                        points_vehicle_homogeneous = np.hstack(
                            (points_vehicle_filtered, np.ones((N, 1))))

                        transformation = np.linalg.inv(
                            np.asarray(self.camera_data[sensor_type][frame_id]
                                       ['extrinsic']).reshape(4, 4))

                        points_camera_reconstructed = (
                            transformation
                            @ points_vehicle_homogeneous.T).T[:, :3]

                        valid = points_camera_reconstructed[:, 0] > 0
                        X_cam = -points_camera_reconstructed[valid, 1]
                        Y_cam = -points_camera_reconstructed[valid, 2]
                        Z_cam = points_camera_reconstructed[valid, 0]

                        u = (fx * X_cam / Z_cam) + cx
                        v = (fy * Y_cam / Z_cam) + cy

                        image_width = 1920
                        image_height = 1280
                        valid_uv = (u >= 0) & (u < image_width) & (v >= 0) & (
                            v < image_height)

                        final_valid = valid.copy()
                        final_valid[valid] = valid_uv

                        u_filtered_reconstructed_indices = u[valid_uv]
                        v_filtered_reconstructed_indices = v[valid_uv]
                        colors_filtered_reconstructed_indices = colors_vehicle_filtered[
                            final_valid]

                        if self.train:
                            train_valid = 'Train'
                        else:
                            train_valid = 'Valid'
                        base_image_path = self.image_path if self.image_path else f"Pipeline_2_{train_valid}"
                        final_img_dir = f"{base_image_path}/{self.file_name}/merged_directly_shadow1_resized/"
                        render_file = f"{final_img_dir}{sensor_type}/final_frame_00{self.frame_map[sensor_type][frame_id]}.jpg"
                        ood_image = Image.open(render_file)
                        ood_array = np.array(ood_image)

                        fig, ax = plt.subplots(figsize=(12, 8))
                        ax.imshow(ood_array)
                        if sensor_type == 'camera':
                            ax.scatter(u_filtered_reconstructed_indices,
                                       v_filtered_reconstructed_indices,
                                       c='green',
                                       s=0.4,
                                       alpha=0.6,
                                       marker='o')
                        else:
                            ax.scatter(u_filtered_reconstructed_indices,
                                       v_filtered_reconstructed_indices,
                                       c=colors_filtered_reconstructed_indices,
                                       s=0.4,
                                       alpha=0.6,
                                       marker='.')

                        ax.scatter(u_ood_reconstructed_indices,
                                   v_ood_reconstructed_indices,
                                   c='orange',
                                   s=0.4,
                                   alpha=0.6,
                                   marker='x')

                        ax.grid(False)
                        ax.axis('off')
                        fig.tight_layout(pad=0)
                        path = f'{self.base_dir}/{self.file_name}/Points Projected in Image via Formula/{sensor_type}'
                        os.makedirs(path, exist_ok=True)
                        plt.savefig(
                            f"{path}/frame_00{self.frame_map[sensor_type][frame_id]}.png",
                            dpi=160,
                            bbox_inches='tight',
                            pad_inches=0)
                        plt.close()

    def check_intensity(self):

        import matplotlib.colors

        for sensor_type, frames in self.frames_with_seg.items():
            if len(frames) == 0:
                continue
            for frame_id, frame in enumerate(frames):
                fig = plt.figure(figsize=(12, 24), dpi=300)
                gs = GridSpec(8,
                              2,
                              figure=fig,
                              height_ratios=[1.5, 1.5, 1.5, 1, 1, 1, 1, 1],
                              hspace=0.5)

                ax0 = fig.add_subplot(gs[0, :])
                ax1 = fig.add_subplot(gs[1, 0])
                ax2 = fig.add_subplot(gs[1, 1])
                ax3 = fig.add_subplot(gs[2, 0])
                ax4 = fig.add_subplot(gs[2, 1])
                ax_range = fig.add_subplot(gs[3, :])
                ax_intensity = fig.add_subplot(gs[4, :])
                ax_hist_low = fig.add_subplot(gs[5, :])
                ax_hist_middle = fig.add_subplot(gs[6, :])
                ax_hist_high = fig.add_subplot(gs[7, :])

                def get_camera_points(cp_points_array, camera_name):
                    """Filter points for specific camera and valid projections"""
                    valid_points = []
                    for point in cp_points_array:

                        if point[0] == camera_name:
                            valid_points.append([point[1], point[2]])

                        if point[3] == camera_name:
                            valid_points.append([point[4], point[5]])
                    return np.array(valid_points)

                intensity_array = self.range_images[frame_id][:, :, 1]

                mask = intensity_array > 0
                intensity_array = np.where(mask, intensity_array, 0)

                filtered_intensities = np.log1p(intensity_array)
                a = filtered_intensities[10:34, 1310:1340]

                high_intensity_indices = np.where((filtered_intensities > 3),
                                                  True, False)

                high_ranges = np.where(high_intensity_indices,
                                       self.range_images[frame_id][:, :,
                                                                   0], -1.0)

                (range_images, camera_projections, seg_labels,
                 range_image_top_pose) = (
                     frame_utils.parse_range_image_and_camera_projection(frame)
                 )

                original_top_ri = range_images[open_dataset.LaserName.TOP][0]
                h, w = original_top_ri.shape.dims[
                    0], original_top_ri.shape.dims[1]
                original_top_data = np.array(original_top_ri.data).reshape(
                    h, w, 4)

                modified_top_data = original_top_data.copy()

                modified_top_data[:, :, 0] = high_ranges

                new_top_ri = open_dataset.MatrixFloat()
                new_top_ri.shape.dims.extend([h, w, 4])
                new_top_ri.data.extend(
                    modified_top_data.astype(np.float64).flatten().tolist())

                temp_range_images = copy.deepcopy(range_images)

                temp_range_images[open_dataset.LaserName.TOP][0] = new_top_ri

                points, _ = frame_utils.convert_range_image_to_point_cloud(
                    frame,
                    temp_range_images,
                    camera_projections,
                    range_image_top_pose,
                    ri_index=0,
                    keep_polar_features=False)

                pcd_high_intensity = o3d.geometry.PointCloud()
                pcd_high_intensity.points = o3d.utility.Vector3dVector(
                    points[0])
                pcd_high_intensity.paint_uniform_color([1, 0, 0])

                merged_pcd = self.point_cloud_vehicle[sensor_type][
                    frame_id] + pcd_high_intensity

                intensity_max, intensity_min = np.max(
                    filtered_intensities), np.min(filtered_intensities)
                intensity_mid = (intensity_max + intensity_min) / 2
                intensity_less_than_mid = filtered_intensities[
                    filtered_intensities < intensity_mid]
                intensity_greater_than_mid = filtered_intensities[
                    filtered_intensities >= intensity_mid]

                divisions = 20
                alpha_values = np.linspace(0.01, 0.5, divisions)
                size = np.linspace(0.01, 0.5, divisions)

                norm = mpl.colors.Normalize(vmin=intensity_min,
                                            vmax=intensity_max)

                norm = mpl.colors.Normalize(vmin=intensity_min,
                                            vmax=intensity_max)

                cmap = plt.get_cmap('jet')

                normalized_values = np.linspace(0, 1, divisions)

                colors = cmap(normalized_values)
                colors[:, -1] = alpha_values

                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])

                range_div = intensity_max / divisions
                mask_dict = {}

                start = 0
                for i in range(divisions):

                    mask_dict[f'mask{i}'] = (intensity_array >
                                             start) & (intensity_array <
                                                       (start + range_div))
                    start = start + range_div

                (range_images, camera_projections, seg_labels,
                 range_image_top_pose) = (
                     frame_utils.parse_range_image_and_camera_projection(frame)
                 )

                original_top_ri = range_images[open_dataset.LaserName.TOP][0]
                h, w = original_top_ri.shape.dims[
                    0], original_top_ri.shape.dims[1]
                original_top_data = np.array(original_top_ri.data).reshape(
                    h, w, 4)

                cp_points_list = []

                for mask in mask_dict.values():

                    modified_top_data = original_top_data.copy()

                    modified_top_data[:, :,
                                      0] = np.where(mask,
                                                    original_top_data[:, :,
                                                                      0], -1)

                    new_top_ri = open_dataset.MatrixFloat()
                    new_top_ri.shape.dims.extend([h, w, 4])
                    new_top_ri.data.extend(
                        modified_top_data.astype(
                            np.float64).flatten().tolist())

                    temp_range_images = copy.deepcopy(range_images)

                    temp_range_images[
                        open_dataset.LaserName.TOP][0] = new_top_ri

                    _, cp_points = frame_utils.convert_range_image_to_point_cloud(
                        frame,
                        temp_range_images,
                        camera_projections,
                        range_image_top_pose,
                        ri_index=0,
                        keep_polar_features=False)
                    if len(cp_points[0]) > 0:
                        cp_points_list.append(cp_points[0])

                    else:
                        cp_points_list.append([[-1, 0, 0, -1, 0, 0]])

                front_img = frame.images[0]
                image = tf.image.decode_jpeg(front_img.image).numpy()
                ax0.imshow(image)

                for i in range(len(cp_points_list)):
                    front_points = get_camera_points(
                        np.asarray(cp_points_list[i]), 1)
                    if front_points.shape[0] > 0:
                        im_f = ax0.scatter(front_points[:, 0],
                                           front_points[:, 1],
                                           cmap=cmap,
                                           s=size[i])
                ax0.set_title('FRONT_CAMERA')
                ax0.axis('off')

                camera_axes = {
                    'FRONT_LEFT': (ax1, 2),
                    'FRONT_RIGHT': (ax2, 3),
                    'SIDE_LEFT': (ax3, 4),
                    'SIDE_RIGHT': (ax4, 5)
                }

                for img in frame.images[1:5]:
                    cam_name = open_dataset.CameraName.Name.Name(img.name)
                    ax, cam_id = camera_axes[cam_name]
                    image = tf.image.decode_jpeg(img.image).numpy()
                    ax.imshow(image)

                    for i in range(len(cp_points_list)):
                        cam_points = get_camera_points(
                            np.asarray(cp_points_list[i]), cam_id)
                        if cam_points.shape[0] > 0:
                            im_s = ax.scatter(cam_points[:, 0],
                                              cam_points[:, 1],
                                              cmap=cmap,
                                              s=size[i])
                    ax.set_title(f'{cam_name}')
                    ax.axis('off')

                im_r = ax_range.imshow(self.range_images[frame_id][:, :,
                                                                   0].numpy(),
                                       cmap='jet',
                                       aspect='auto')
                ax_range.set_title('Range channel')
                ax_range.axis('off')

                im_i = ax_intensity.imshow(filtered_intensities,
                                           cmap='jet',
                                           aspect='auto')
                ax_intensity.set_title('Intensity channel')
                ax_intensity.axis('off')

                _, bins, patches = ax_hist_low.hist(intensity_less_than_mid,
                                                    bins=divisions // 2,
                                                    edgecolor='black')
                ax_hist_low.set_title('Less Intensity Histogram')
                ax_hist_low.set_xlabel('Intensity')
                ax_hist_low.set_ylabel('Frequency')

                for i, (patch, left_edge, right_edge) in enumerate(
                        zip(patches, bins[:-1], bins[1:])):

                    bin_midpoint = (left_edge + right_edge) / 2

                    normalized_value = norm(bin_midpoint)

                    color = cmap(normalized_value)

                    patch.set_facecolor(color)

                right_edge_first_bin = bins[1]
                mask = (filtered_intensities > right_edge_first_bin) & (
                    filtered_intensities < intensity_mid)
                intensities_middle = filtered_intensities[mask]
                _, bins, patches = ax_hist_middle.hist(intensities_middle,
                                                       bins=(divisions // 2) -
                                                       1,
                                                       edgecolor='black')
                ax_hist_middle.set_title('Lower Intensity Histogram')
                ax_hist_middle.set_xlabel('Intensity')
                ax_hist_middle.set_ylabel('Frequency')
                for i, (patch, left_edge, right_edge) in enumerate(
                        zip(patches, bins[:-1], bins[1:])):

                    bin_midpoint = (left_edge + right_edge) / 2

                    normalized_value = norm(bin_midpoint)

                    color = cmap(normalized_value)

                    patch.set_facecolor(color)

                _, bins, patches = ax_hist_high.hist(
                    intensity_greater_than_mid,
                    bins=divisions // 2,
                    edgecolor='black')
                ax_hist_high.set_title('Greater Intensity Histogram')
                ax_hist_high.set_xlabel('Intensity')
                ax_hist_high.set_ylabel('Frequency')

                for i, (patch, left_edge, right_edge) in enumerate(
                        zip(patches, bins[:-1], bins[1:])):

                    bin_midpoint = (left_edge + right_edge) / 2

                    normalized_value = norm(bin_midpoint)

                    color = cmap(normalized_value)

                    patch.set_facecolor(color)

                cbar = fig.colorbar(im_r,
                                    ax=ax_range,
                                    orientation='vertical',
                                    label='Distance(m)')
                cbar = fig.colorbar(im_i,
                                    ax=ax_intensity,
                                    orientation='vertical',
                                    label='Log Intensity')
                cbar = fig.colorbar(sm,
                                    ax=ax0,
                                    orientation='vertical',
                                    label='Log Intensity')
                cbar = fig.colorbar(sm,
                                    ax=[ax1, ax2],
                                    orientation='vertical',
                                    label='Log Intensity')

                cbar = fig.colorbar(sm,
                                    ax=[ax3, ax4],
                                    orientation='vertical',
                                    label='Log Intensity')

                path = f'{self.base_dir}/{self.file_name}/Intensity/{sensor_type}'
                os.makedirs(path, exist_ok=True)
                o3d.io.write_point_cloud(
                    f"{path}/frame_00{self.frame_map[sensor_type][frame_id]}.pcd",
                    merged_pcd)
                plt.savefig(
                    f"{path}/frame_00{self.frame_map[sensor_type][frame_id]}.png",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.25)
                plt.close()

    def get_intensity(self):
        def get_camera_points(cp_points_array, camera_name):
            """Filter points for specific camera and valid projections"""
            valid_points = []
            for point in cp_points_array:

                if point[0] == camera_name:
                    valid_points.append([point[1], point[2]])

                if point[3] == camera_name:
                    valid_points.append([point[4], point[5]])
            return np.array(valid_points)

        for sensor_type, frames in self.frames_with_seg.items():
            if sensor_type not in self._get_active_lidar_pipeline_sensor_types(
            ):
                continue
            if len(frames) == 0:
                continue
            for frame_id, frame in enumerate(frames):
                fig = plt.figure(figsize=(12, 24), dpi=300)
                gs = GridSpec(6,
                              2,
                              figure=fig,
                              height_ratios=[1.5, 1, 1, 1, 1, 1],
                              hspace=0.5)

                ax0 = fig.add_subplot(gs[0, :])

                ax_range = fig.add_subplot(gs[1, :])
                ax_intensity = fig.add_subplot(gs[2, :])
                ax_hist_low = fig.add_subplot(gs[3, :])
                ax_hist_middle = fig.add_subplot(gs[4, :])
                ax_hist_high = fig.add_subplot(gs[5, :])

                intensity_array = self.ood_intensity[sensor_type][frame_id]

                mask = intensity_array > 0
                intensity_array = np.where(mask, intensity_array, 0)

                filtered_intensities = np.log1p(intensity_array)

                intensity_max, intensity_min = np.max(
                    filtered_intensities), np.min(filtered_intensities)
                intensity_mid = (intensity_max + intensity_min) / 2
                intensity_less_than_mid = filtered_intensities[
                    filtered_intensities < intensity_mid]
                intensity_greater_than_mid = filtered_intensities[
                    filtered_intensities >= intensity_mid]

                divisions = 20
                alpha_values = np.linspace(0.01, 0.5, divisions)
                size = np.linspace(0.01, 0.5, divisions)

                norm = mpl.colors.Normalize(vmin=intensity_min,
                                            vmax=intensity_max)

                norm = mpl.colors.Normalize(vmin=intensity_min,
                                            vmax=intensity_max)

                cmap = plt.get_cmap('jet')

                normalized_values = np.linspace(0, 1, divisions)

                colors = cmap(normalized_values)
                colors[:, -1] = alpha_values

                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])

                range_div = (intensity_max - intensity_min) / divisions
                mask_dict = {}

                start = 0
                for i in range(divisions):
                    mask_dict[f'mask{i}'] = (intensity_array >
                                             start) & (intensity_array <
                                                       (start + range_div))
                    start = start + range_div

                (range_images, camera_projections, seg_labels,
                 range_image_top_pose) = (
                     frame_utils.parse_range_image_and_camera_projection(frame)
                 )

                original_top_ri = range_images[open_dataset.LaserName.TOP][0]
                h, w = original_top_ri.shape.dims[
                    0], original_top_ri.shape.dims[1]
                original_top_data = np.array(original_top_ri.data).reshape(
                    h, w, 4)

                cp_points_list = []

                for mask in mask_dict.values():

                    modified_top_data = original_top_data.copy()

                    modified_top_data[:, :, 0] = np.where(
                        mask,
                        self.range_images_ood[sensor_type][frame_id].numpy(),
                        -1)

                    new_top_ri = open_dataset.MatrixFloat()
                    new_top_ri.shape.dims.extend([h, w, 4])
                    new_top_ri.data.extend(
                        modified_top_data.astype(
                            np.float64).flatten().tolist())

                    temp_range_images = copy.deepcopy(range_images)

                    temp_range_images[
                        open_dataset.LaserName.TOP][0] = new_top_ri

                    _, cp_points = frame_utils.convert_range_image_to_point_cloud(
                        frame,
                        temp_range_images,
                        camera_projections,
                        range_image_top_pose,
                        ri_index=0,
                        keep_polar_features=False)
                    if len(cp_points[0]) > 0:
                        cp_points_list.append(cp_points[0])

                    else:
                        cp_points_list.append([[0, 0, 0, 0, 0, 0]])

                image = self.merged_images[sensor_type][frame_id]
                ax0.imshow(image)

                for i in range(len(cp_points_list)):
                    front_points = get_camera_points(
                        np.asarray(cp_points_list[i]), 1)
                    if front_points.shape[0] > 0:
                        im_f = ax0.scatter(front_points[:, 0],
                                           front_points[:, 1],
                                           cmap=cmap,
                                           s=size[i])
                ax0.set_title('FRONT_CAMERA')
                ax0.axis('off')

                im_r = ax_range.imshow(
                    self.range_images_ood[sensor_type][frame_id].numpy(),
                    cmap='jet',
                    aspect='auto')
                ax_range.set_title('Range channel')
                ax_range.axis('off')

                im_i = ax_intensity.imshow(filtered_intensities,
                                           cmap='jet',
                                           aspect='auto')
                ax_intensity.set_title('Intensity channel')
                ax_intensity.axis('off')

                _, bins, patches = ax_hist_low.hist(intensity_less_than_mid,
                                                    bins=divisions // 2,
                                                    edgecolor='black')
                ax_hist_low.set_title('Less Intensity Histogram')
                ax_hist_low.set_xlabel('Intensity')
                ax_hist_low.set_ylabel('Frequency')

                for i, (patch, left_edge, right_edge) in enumerate(
                        zip(patches, bins[:-1], bins[1:])):

                    bin_midpoint = (left_edge + right_edge) / 2

                    normalized_value = norm(bin_midpoint)

                    color = cmap(normalized_value)

                    patch.set_facecolor(color)

                right_edge_first_bin = bins[1]
                mask = (filtered_intensities > right_edge_first_bin) & (
                    filtered_intensities < intensity_mid)
                intensities_middle = filtered_intensities[mask]
                _, bins, patches = ax_hist_middle.hist(intensities_middle,
                                                       bins=(divisions // 2) -
                                                       1,
                                                       edgecolor='black')
                ax_hist_middle.set_title('Lower Intensity Histogram')
                ax_hist_middle.set_xlabel('Intensity')
                ax_hist_middle.set_ylabel('Frequency')
                for i, (patch, left_edge, right_edge) in enumerate(
                        zip(patches, bins[:-1], bins[1:])):

                    bin_midpoint = (left_edge + right_edge) / 2

                    normalized_value = norm(bin_midpoint)

                    color = cmap(normalized_value)

                    patch.set_facecolor(color)

                _, bins, patches = ax_hist_high.hist(
                    intensity_greater_than_mid,
                    bins=divisions // 2,
                    edgecolor='black')
                ax_hist_high.set_title('Greater Intensity Histogram')
                ax_hist_high.set_xlabel('Intensity')
                ax_hist_high.set_ylabel('Frequency')

                for i, (patch, left_edge, right_edge) in enumerate(
                        zip(patches, bins[:-1], bins[1:])):

                    bin_midpoint = (left_edge + right_edge) / 2

                    normalized_value = norm(bin_midpoint)

                    color = cmap(normalized_value)

                    patch.set_facecolor(color)

                cbar = fig.colorbar(im_r,
                                    ax=ax_range,
                                    orientation='vertical',
                                    label='Distance(m)')
                cbar = fig.colorbar(im_i,
                                    ax=ax_intensity,
                                    orientation='vertical',
                                    label='Log Intensity')
                cbar = fig.colorbar(sm,
                                    ax=ax0,
                                    orientation='vertical',
                                    label='Log Intensity')

                path = f'{self.base_dir}/{self.file_name}/OOD_Intensity'
                os.makedirs(path, exist_ok=True)
                plt.savefig(
                    f"{path}/frame_00{self.frame_map[sensor_type][frame_id]}.png",
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.25)
                plt.close()

    @ResourceLogger.log_resources_during_execution()
    def get_rays_az_inc(self):
        active_lidar_sensor_types = self._get_active_lidar_pipeline_sensor_types(
        )
        if self.raycast_lidar_only and len(active_lidar_sensor_types) > 0:
            sensor_types = active_lidar_sensor_types
        else:
            sensor_types = [
                sensor_type
                for sensor_type, frames in self.frames_with_seg.items()
                if len(frames) > 0
            ]

        if not sensor_types:
            return

        for sensor_type in sensor_types:
            frames = self.frames_with_seg.get(sensor_type, [])
            if len(frames) == 0:
                continue

            range_channel_list = []
            extrinsic_list = []
            inclinations_list = []

            for frame_id, frame in enumerate(frames):
                original_top_data = self._get_cached_top_lidar_first_return(
                    sensor_type,
                    frame_id,
                    frame,
                )
                range_channel_list.append(original_top_data[:, :, 0])

                extrinsic = np.asarray(self.lidar_parameters[sensor_type]
                                       [frame_id]['pose']).reshape(4, 4)
                extrinsic_list.append(extrinsic)
                inclinations = np.asarray(self.lidar_parameters[sensor_type]
                                          [frame_id]['inclinations'][::-1])
                inclinations_list.append(inclinations)

            with tf.device('/CPU:0'):
                range_channel_tensor = tf.convert_to_tensor(range_channel_list,
                                                            dtype=tf.float32)
                extrinsic_tensor = tf.convert_to_tensor(extrinsic_list,
                                                        dtype=tf.float32)
                inclinations_tensor = tf.convert_to_tensor(inclinations_list,
                                                           dtype=tf.float32)

                range_image_polar = range_image_utils.compute_range_image_polar(
                    range_image=range_channel_tensor,
                    extrinsic=extrinsic_tensor,
                    inclination=inclinations_tensor,
                    dtype=tf.float32,
                )

                azimuth, inclination, _ = tf.unstack(range_image_polar,
                                                     axis=-1)
                self.azimuth[sensor_type] = tf.cast(azimuth,
                                                    tf.float32).numpy()
                self.inclination[sensor_type] = tf.cast(
                    inclination, tf.float32).numpy()

        np.save(f"{self.base_dir}/{self.file_name}/lidar_data/azimuth.npy",
                self.azimuth,
                allow_pickle=True)
        np.save(f"{self.base_dir}/{self.file_name}/lidar_data/inclination.npy",
                self.inclination,
                allow_pickle=True)

    @ResourceLogger.log_resources_during_execution()
    def get_point_cloud_from_range_image_via_indices(self,
                                                     eval=False,
                                                     render_mode='multi',
                                                     master_save=False):
        def format_time(seconds):
            """Convert seconds to human-readable format (hours, minutes, seconds, milliseconds)"""
            if seconds < 0.001:
                return f"{seconds * 1_000_000:.2f}µs"
            elif seconds < 1:
                return f"{seconds * 1000:.2f}ms"
            elif seconds < 60:
                return f"{seconds:.2f}s"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = seconds % 60
                return f"{minutes}m {secs:.2f}s"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = seconds % 60
                return f"{hours}h {minutes}m {secs:.2f}s"

        def simple_remove_points(original_indices, all_remove_indices, points,
                                 colors, projections):
            """
            Optimized NumPy-based solution to remove points per frame.
            """
            B = len(original_indices)
            filtered_points = [None] * B
            filtered_colors = [None] * B
            removed_points = [None] * B
            removed_projections = [None] * B

            for b in range(B):

                remove_set = {tuple(idx) for idx in all_remove_indices[b]}

                indices_tuples = [tuple(idx) for idx in original_indices[b]]
                mask = np.array(
                    [idx not in remove_set for idx in indices_tuples])

                filtered_points[b] = points[b][mask]
                filtered_colors[b] = colors[b][mask]
                removed_points[b] = points[b][~mask]
                removed_projections[b] = projections[b][~mask]

            return filtered_points, filtered_colors, removed_points, removed_projections

        def reconstruct_points_from_ragged_indices(ri_indices,
                                                   ri_ranges,
                                                   inclination,
                                                   extrinsic,
                                                   range_image_size,
                                                   dtype=tf.float32):
            """
            Properly handles ragged indices using TensorFlow's RaggedTensor API
            """

            with tf.device('/CPU:0'):

                ri_indices = ri_indices.with_row_splits_dtype(tf.int32)
                ri_ranges = ri_ranges.with_row_splits_dtype(tf.int32)

                H, W = range_image_size

                flat_values = tf.reshape(ri_indices.flat_values, [-1, 2])
                ri_indices = tf.RaggedTensor.from_row_splits(
                    flat_values,
                    ri_indices.row_splits).with_row_splits_dtype(tf.int32)

                row_splits = ri_indices.row_splits
                flat_indices = ri_indices.flat_values
                rows_flat = flat_indices[:, 0]
                cols_flat = flat_indices[:, 1]

                rows = tf.RaggedTensor.from_row_splits(
                    rows_flat, row_splits).with_row_splits_dtype(tf.int32)
                cols = tf.RaggedTensor.from_row_splits(
                    cols_flat, row_splits).with_row_splits_dtype(tf.int32)

                phi = tf.gather(inclination, rows, batch_dims=1)
                phi = phi.with_row_splits_dtype(tf.int32) if isinstance(
                    phi, tf.RaggedTensor) else phi

                theta_corrected = (
                    (tf.cast(W, dtype) - 0.5 - tf.cast(cols, dtype)) *
                    (2 * np.pi) / tf.cast(W, dtype)) - np.pi

                az_correction = tf.atan2(extrinsic[..., 1, 0], extrinsic[...,
                                                                         0, 0])
                az_correction = tf.expand_dims(az_correction, axis=-1)

                theta_sensor = theta_corrected - az_correction

                r = tf.cast(ri_ranges, dtype)
                x = r * tf.cos(theta_sensor) * tf.cos(phi)
                y = r * tf.sin(theta_sensor) * tf.cos(phi)
                z = r * tf.sin(phi)
                points_sensor = tf.stack([x, y, z], axis=-1)

                R = extrinsic[:, 0:3, 0:3]
                t = extrinsic[:, 0:3, 3]

                points_vehicle = (
                    tf.einsum('bij,bnj->bni', R, points_sensor.to_tensor()) +
                    tf.expand_dims(t, 1))

                return tf.RaggedTensor.from_tensor(
                    points_vehicle, lengths=ri_ranges.row_lengths())

        def get_ragged_tensors_for_label(sensor_type, label):
            """
            Creates ragged tensors for indices and ranges for a specific label across frames.
            """
            indices_list = []
            ranges_list = []

            for frame_idx in self.ood_labels_to_indices[sensor_type].keys():

                if (frame_idx in self.ood_labels_to_indices.get(
                        sensor_type, {})
                        and frame_idx in self.ood_labels_to_ranges.get(
                            sensor_type, {}) and label
                        in self.ood_labels_to_indices[sensor_type][frame_idx]
                        and label
                        in self.ood_labels_to_ranges[sensor_type][frame_idx]):

                    indices_list.append(self.ood_labels_to_indices[sensor_type]
                                        [frame_idx][label])
                    ranges_list.append(self.ood_labels_to_ranges[sensor_type]
                                       [frame_idx][label])
                else:

                    indices_list.append(np.empty((0, 2)))
                    ranges_list.append(np.empty((0, 1)))

            ri_indices_ragged = tf.ragged.constant(indices_list,
                                                   dtype=tf.int32)
            ri_ranges_ragged = tf.ragged.constant(ranges_list,
                                                  dtype=tf.float32)

            return ri_indices_ragged, ri_ranges_ragged

        def get_instance_perturbation(instance_id):
            """Generate stronger deterministic color perturbation for an instance ID"""
            primes = [73856093, 19349669, 83492791]
            perturbation_range = 80
            instance_id = int(instance_id)

            offsets = []
            for prime in primes:

                val = (instance_id * prime) % 101

                offset = int((val - 50) * (perturbation_range / 50))
                offsets.append(offset)
            return offsets

        def remove_padded_points(points_np, colors_np, original_indices_np,
                                 projections_np):
            """Efficiently remove padded points."""
            B = points_np.shape[0]
            valid_points = [None] * B
            valid_colors = [None] * B
            valid_indices = [None] * B
            valid_projections = [None] * B

            for b in range(B):
                zero_mask = np.all(points_np[b] == 0, axis=-1)
                valid_mask = ~zero_mask

                valid_points[b] = points_np[b][valid_mask]
                valid_colors[b] = colors_np[b][valid_mask]
                valid_indices[b] = original_indices_np[b][valid_mask]
                valid_projections[b] = projections_np[b][valid_mask]

            return valid_points, valid_colors, valid_indices, valid_projections

        def create_ood_point_cloud(oodwise_points, oodwise_colors, frame_id):
            """Create OOD point cloud from all OOD points."""
            all_points = []
            all_colors = []

            for ood_id, points in oodwise_points.items():
                frame_points = points[frame_id].numpy()
                if frame_points.shape[0] > 0:
                    frame_colors = oodwise_colors[ood_id][frame_id].numpy()
                    all_points.append(frame_points)
                    all_colors.append(frame_colors)

            pcd_vehicle = o3d.geometry.PointCloud()
            if all_points:
                stacked_points = np.vstack(all_points)
                stacked_colors = np.vstack(all_colors)
                pcd_vehicle.points = o3d.utility.Vector3dVector(stacked_points)
                pcd_vehicle.colors = o3d.utility.Vector3dVector(stacked_colors)

            return pcd_vehicle

        def assign_semantic_labels_vectorized(rec_colors_int,
                                              color_array,
                                              debug=False):
            """Vectorized semantic label assignment."""
            start_time = time.time()

            color_array_int = color_array.astype(np.uint8)
            rec_colors_int = rec_colors_int.astype(np.uint8)

            if debug:
                print(f"\n[Semantic Label Assignment]")
                print(f"  Input points: {len(rec_colors_int)}")
                print(f"  Color array shape: {color_array_int.shape}")
                print(f"  Sample input colors: {rec_colors_int[:3]}")

            distances = np.linalg.norm(color_array_int[:, np.newaxis, :] -
                                       rec_colors_int[np.newaxis, :, :],
                                       axis=2)
            semantic_labels = np.argmin(distances, axis=0)

            elapsed_time = time.time() - start_time

            if debug:
                unique_labels, counts = np.unique(semantic_labels,
                                                  return_counts=True)
                print(f"  Unique semantic labels: {unique_labels}")
                print(
                    f"  Label distribution: {dict(zip(unique_labels, counts))}"
                )
                print(
                    f"  Min distance: {np.min(distances):.2f}, Max distance: {np.max(distances):.2f}"
                )
                print(f"  Time taken: {elapsed_time:.4f}s")

            return semantic_labels, elapsed_time

        def assign_instance_labels_vectorized(rec_colors_int,
                                              rec_semantic_labels,
                                              color_array,
                                              get_instance_perturbation_func,
                                              instance_labels,
                                              debug=False):
            """Vectorized instance label assignment."""
            start_time = time.time()

            unique_instances = np.unique(instance_labels)
            rec_instance_labels = np.zeros(len(rec_colors_int), dtype=np.int32)

            rec_colors_int = rec_colors_int.astype(np.uint8)
            color_array_int = color_array.astype(np.uint8)

            if debug:
                print(f"\n[Instance Label Assignment]")
                print(f"  Input points: {len(rec_colors_int)}")
                print(
                    f"  Unique instances in original: {unique_instances[:10]}... (total: {len(unique_instances)})"
                )
                print(f"  Sample point analysis:")

                for idx in range(min(5, len(rec_colors_int))):
                    sem_lbl = rec_semantic_labels[idx]
                    base_col = color_array_int[sem_lbl]
                    actual_col = rec_colors_int[idx]
                    pert = actual_col.astype(np.int32) - base_col.astype(
                        np.int32)
                    print(
                        f"    Point {idx}: semantic={sem_lbl}, color={actual_col}, base={base_col}, perturbation={pert}"
                    )

            matched_count = 0
            unmatched_count = 0
            instance_stats = {}
            perturbation_mismatches = []

            for i, color in enumerate(rec_colors_int):
                semantic_label = rec_semantic_labels[i]
                base_color = color_array_int[semantic_label]
                perturbation = color.astype(np.int32) - base_color.astype(
                    np.int32)

                found = False
                for inst_id in unique_instances:
                    if inst_id == 0:
                        continue
                    expected_perturbation = np.array(
                        get_instance_perturbation_func(inst_id),
                        dtype=np.int32)
                    if np.array_equal(perturbation, expected_perturbation):
                        rec_instance_labels[i] = inst_id
                        found = True
                        matched_count += 1
                        instance_stats[inst_id] = instance_stats.get(
                            inst_id, 0) + 1
                        break

                if not found:
                    rec_instance_labels[i] = 0
                    unmatched_count += 1

                    if debug and len(perturbation_mismatches) < 3:

                        min_dist = float('inf')
                        closest_inst = None
                        for inst_id in unique_instances[:20]:
                            if inst_id == 0:
                                continue
                            exp_pert = np.array(
                                get_instance_perturbation_func(inst_id),
                                dtype=np.int32)
                            dist = np.linalg.norm(perturbation - exp_pert)
                            if dist < min_dist:
                                min_dist = dist
                                closest_inst = (inst_id, exp_pert)
                        perturbation_mismatches.append({
                            'point_idx':
                            i,
                            'color':
                            color,
                            'base_color':
                            base_color,
                            'perturbation':
                            perturbation,
                            'semantic_label':
                            semantic_label,
                            'closest_inst_id':
                            closest_inst[0] if closest_inst else None,
                            'closest_perturbation':
                            closest_inst[1] if closest_inst else None,
                            'distance':
                            min_dist
                        })

            elapsed_time = time.time() - start_time

            if debug:
                print(f"  Matched instances: {matched_count}")
                print(f"  Unmatched points: {unmatched_count}")
                print(f"  Recovered {len(instance_stats)} unique instances")
                print(
                    f"  Sample instance distribution: {dict(list(instance_stats.items())[:5])}"
                )
                print(f"  Time taken: {elapsed_time:.4f}s")

                if perturbation_mismatches:
                    print(f"\n  Sample Perturbation Mismatches:")
                    for mm in perturbation_mismatches:
                        print(f"    Point {mm['point_idx']}:")
                        print(f"      Semantic label: {mm['semantic_label']}")
                        print(
                            f"      Color: {mm['color']}, Base: {mm['base_color']}"
                        )
                        print(
                            f"      Actual perturbation: {mm['perturbation']}")
                        print(
                            f"      Closest instance: {mm['closest_inst_id']}")
                        print(
                            f"      Expected perturbation: {mm['closest_perturbation']}"
                        )
                        print(f"      Distance: {mm['distance']:.2f}")

                print(f"\n  Sample Expected Perturbations:")
                for inst_id in list(unique_instances[1:6]):
                    exp_pert = get_instance_perturbation_func(inst_id)
                    print(f"    Instance {inst_id}: {exp_pert}")

            return rec_instance_labels, elapsed_time, instance_stats

        def compute_ood_panoptic_labels(pcd_vehicle,
                                        color_dict,
                                        num_objects,
                                        debug=False):
            """Compute OOD panoptic labels efficiently."""
            start_time = time.time()

            if len(pcd_vehicle.points) == 0:
                return np.array([], dtype=np.int32), 0.0, {}

            ood_colors = np.asarray(pcd_vehicle.colors) * 255.0
            ood_class = self.ood_semantic_class
            tolerance = 1

            if debug:
                print(f"\n[OOD Panoptic Label Assignment]")
                print(f"  OOD points: {len(ood_colors)}")
                print(f"  Number of OOD objects: {num_objects}")

            ood_panoptic_labels = np.zeros(ood_colors.shape[0], dtype=np.int32)
            ood_stats = {}
            unmatched_ood = 0

            for i, color in enumerate(ood_colors):
                matched = False
                for ood_id in range(1, num_objects + 1):
                    target_color = np.array(color_dict[ood_id],
                                            dtype=np.float32)
                    distance = np.linalg.norm(color - target_color)
                    if distance < tolerance:
                        ood_panoptic_labels[i] = 1000 * ood_class + ood_id
                        ood_stats[ood_id] = ood_stats.get(ood_id, 0) + 1
                        matched = True
                        break
                if not matched:
                    unmatched_ood += 1

            elapsed_time = time.time() - start_time

            if debug:
                print(f"  OOD label distribution: {ood_stats}")
                print(f"  Unmatched OOD points: {unmatched_ood}")
                print(f"  Time taken: {elapsed_time:.4f}s")

            return ood_panoptic_labels, elapsed_time, ood_stats

        def append_to_json_file(filename, new_data):
            """Merge dictionary data to JSON file."""
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = {}
            else:
                data = {}

            data.update(new_data)

            with open(filename, "w") as f:
                json.dump(data, f, indent=4)

        for sensor_type in self._get_active_lidar_pipeline_sensor_types():
            if len(self.frames_with_seg.get(sensor_type, [])) == 0:
                continue

            frame_ids = np.arange(len(self.frames_with_seg[sensor_type]))

            extrinsics = [
                np.asarray(self.lidar_parameters[sensor_type][fid]['pose'])
                for fid in frame_ids
            ]
            extrinsic = tf.convert_to_tensor(extrinsics, dtype=tf.float32)

            inclinations = [
                np.asarray(self.lidar_parameters[sensor_type][fid]
                           ['inclinations'])[::-1] for fid in frame_ids
            ]
            inclination = tf.convert_to_tensor(inclinations, dtype=tf.float32)

            if sensor_type not in self.pc_reconstructed_indices:
                self.pc_reconstructed_indices[sensor_type] = {}
                self.cp_reconstructed_indices[sensor_type] = {}
                self.filtered_points[sensor_type] = []
                self.removed_points[sensor_type] = []
                self.removed_projections[sensor_type] = []
                self.reconstructed_ood_points[sensor_type] = {}
                self.reconstructed_panoptic_labels[sensor_type] = {}

                dir_range_type = f'{self.base_dir}/{self.file_name}/OOD_Merged_Panoptic_Labels/{sensor_type}'
                os.makedirs(dir_range_type, exist_ok=True)

            original_indices_np = self.ri_indices[sensor_type].numpy()
            points_np = self.points_vehicle_frame[sensor_type].numpy()
            colors_np = self.colors_vehicle_frame[sensor_type].numpy()
            projections_np = self.projections_vehicle_frame[sensor_type].numpy(
            )

            valid_points, valid_colors, valid_indices, valid_projections = remove_padded_points(
                points_np, colors_np, original_indices_np, projections_np)

            remove_indices_list = [
                self.range_images_ood_indices[sensor_type][frame_id].numpy()
                for frame_id in range(len(self.frames_with_seg[sensor_type]))
            ]

            filtered_points, filtered_colors, removed_points, removed_projections = simple_remove_points(
                valid_indices, remove_indices_list, valid_points, valid_colors,
                valid_projections)

            oodwise_points = {}
            oodwise_colors = {}

            color_dict_tensor = {}
            for ood_id in range(1, self.num_objects + 1):
                color_dict_tensor[ood_id] = tf.constant(
                    np.array(self.color_dict[ood_id]) / 255.0,
                    dtype=tf.float32)

            for ood_id in range(1, self.num_objects + 1):
                indices_ragged, ranges_ragged = get_ragged_tensors_for_label(
                    sensor_type, ood_id)

                points_vehicle = reconstruct_points_from_ragged_indices(
                    indices_ragged,
                    ranges_ragged,
                    inclination,
                    extrinsic,
                    (64, 2650),
                )

                color = color_dict_tensor[ood_id]
                colors_vehicle = points_vehicle.with_flat_values(
                    tf.tile(tf.expand_dims(color, 0),
                            [tf.shape(points_vehicle.flat_values)[0], 1]))

                oodwise_points[ood_id] = points_vehicle
                oodwise_colors[ood_id] = colors_vehicle

            output_dir = f"{self.base_dir}/{self.file_name}/Reconstructed_Point_Cloud_via_Indices/{sensor_type}"
            dir_range_type = f"{self.base_dir}/{self.file_name}/OOD_Merged_Panoptic_Labels/{sensor_type}"
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(dir_range_type, exist_ok=True)

            base_colors_255 = np.array(
                [self.color_map[i] for i in range(len(self.color_map))],
                dtype=np.uint8)

            debug_frames = [0]

            frame_timing_stats = []

            for frame_id, frame in enumerate(
                    self.frames_with_seg[sensor_type]):
                frame_start_time = time.time()
                debug_mode = frame_id in debug_frames

                if debug_mode:
                    print(f"\n{'='*80}")
                    print(
                        f"Processing Frame {frame_id} (frame_00{self.frame_map[sensor_type][frame_id]})"
                    )
                    print(f"{'='*80}")

                pcd_rec = o3d.geometry.PointCloud()
                pcd_rec.points = o3d.utility.Vector3dVector(
                    filtered_points[frame_id][:, :3])
                pcd_rec.colors = o3d.utility.Vector3dVector(
                    filtered_colors[frame_id][:, :3])

                self.filtered_points[sensor_type].append(pcd_rec)
                self.removed_points[sensor_type].append(
                    removed_points[frame_id][:, :3])
                self.removed_projections[sensor_type].append(
                    removed_projections[frame_id][:, :6])

                pcd_vehicle = create_ood_point_cloud(oodwise_points,
                                                     oodwise_colors, frame_id)
                self.reconstructed_ood_points[sensor_type][
                    frame_id] = pcd_vehicle

                merged_pcd = pcd_rec + pcd_vehicle
                self.pc_reconstructed_indices[sensor_type][
                    frame_id] = merged_pcd
                self.cp_reconstructed_indices[sensor_type][
                    frame_id] = np.array(pcd_vehicle.points)

                o3d.io.write_point_cloud(
                    f"{output_dir}/frame_00{self.frame_map[sensor_type][frame_id]}.pcd",
                    merged_pcd)

                label_recovery_enabled = (
                    self.enable_lidar_panoptic_label_recovery
                    and sensor_type == 'lidar')
                if not label_recovery_enabled:
                    non_padded_mask = ~np.all(points_np[frame_id] == 0,
                                              axis=-1)
                    org_panoptic_labels = np.zeros(
                        int(np.count_nonzero(non_padded_mask)),
                        dtype=np.int32,
                    )
                    if debug_mode:
                        print(
                            f"  INFO: LiDAR panoptic label recovery disabled for "
                            f"{sensor_type} frame {frame_id}; using zero background labels."
                        )
                elif frame_id in self.lidar_labels:
                    org_panoptic_labels = self.lidar_labels[frame_id]
                elif (isinstance(self.lidar_labels, dict)
                      and sensor_type in self.lidar_labels
                      and isinstance(self.lidar_labels[sensor_type], dict)
                      and frame_id in self.lidar_labels[sensor_type]):
                    org_panoptic_labels = self.lidar_labels[sensor_type][
                        frame_id]
                else:

                    non_padded_mask = ~np.all(points_np[frame_id] == 0,
                                              axis=-1)
                    org_panoptic_labels = np.zeros(
                        int(np.count_nonzero(non_padded_mask)),
                        dtype=np.int32,
                    )
                    if debug_mode:
                        print(
                            f"  WARNING: Missing lidar labels for frame {frame_id}; "
                            "using zero background panoptic labels.")

                max_points = points_np.shape[1]
                num_original_points = len(org_panoptic_labels)

                if debug_mode or num_original_points != max_points:
                    print(f"\n{'='*80}")
                    print(
                        f"DEBUG: Label Size Analysis for {sensor_type} frame {frame_id}"
                    )
                    print(f"{'='*80}")
                    print(f"  Sensor type: {sensor_type}")
                    print(f"  Frame ID: {frame_id}")
                    print(f"  max_points (from padded array): {max_points}")
                    print(
                        f"  num_original_points (label count): {num_original_points}"
                    )
                    print(f"  Difference: {num_original_points - max_points}")
                    print(f"  points_np shape: {points_np.shape}")
                    print(
                        f"  org_panoptic_labels shape: {org_panoptic_labels.shape}"
                    )
                    if num_original_points > max_points:
                        print(
                            f"  ⚠️  WARNING: Labels have MORE elements than max_points!"
                        )
                        print(
                            f"  This suggests the original point cloud for this frame had more points"
                        )
                        print(
                            f"  than the padded array expects. Trimming labels to match."
                        )
                    elif num_original_points < max_points:
                        print(
                            f"  ℹ️  INFO: Labels have FEWER elements, padding with zeros"
                        )
                    print(f"{'='*80}\n")

                if num_original_points < max_points:

                    padded_labels = np.pad(
                        org_panoptic_labels,
                        (0, max_points - num_original_points),
                        mode='constant',
                        constant_values=0)
                elif num_original_points > max_points:

                    padded_labels = org_panoptic_labels[:max_points]
                    print(
                        f"  🔧 Trimmed labels from {num_original_points} to {max_points}"
                    )
                else:

                    padded_labels = org_panoptic_labels

                zero_mask = np.all(points_np[frame_id] == 0, axis=-1)
                valid_mask = ~zero_mask
                labels_after_padding_removal = padded_labels[valid_mask]

                remove_indices_frame = remove_indices_list[frame_id]

                if len(remove_indices_frame) > 0:

                    original_indices_frame = valid_indices[frame_id]
                    remove_set = {tuple(idx) for idx in remove_indices_frame}
                    indices_tuples = [
                        tuple(idx) for idx in original_indices_frame
                    ]
                    keep_mask = np.array(
                        [idx not in remove_set for idx in indices_tuples])

                    rec_panoptic_labels = labels_after_padding_removal[
                        keep_mask]
                else:

                    rec_panoptic_labels = labels_after_padding_removal

                rec_semantic_labels = rec_panoptic_labels // 1000
                rec_instance_labels = rec_panoptic_labels % 1000

                sem_time = 0.0
                inst_time = 0.0
                inst_stats = {}
                for inst_id in np.unique(rec_instance_labels):
                    if inst_id > 0:
                        inst_stats[inst_id] = np.sum(
                            rec_instance_labels == inst_id)

                if debug_mode:
                    print(
                        f"\n  Direct Label Assignment (synchronized with point filtering):"
                    )
                    print(
                        f"    Original panoptic labels shape: {org_panoptic_labels.shape}"
                    )
                    print(f"    Padded labels shape: {padded_labels.shape}")
                    print(
                        f"    After padding removal: {len(labels_after_padding_removal)}"
                    )
                    print(f"    After OOD removal: {len(rec_panoptic_labels)}")
                    print(
                        f"    Filtered points: {len(filtered_points[frame_id])}"
                    )
                    print(
                        f"    Labels-points alignment: {'✓ MATCH' if len(rec_panoptic_labels) == len(filtered_points[frame_id]) else '✗ MISMATCH'}"
                    )
                    print(
                        f"    Unique recovered panoptic: {np.unique(rec_panoptic_labels)}"
                    )
                    print(
                        f"    Unique recovered semantic: {np.unique(rec_semantic_labels)}"
                    )
                    print(
                        f"    Unique recovered instance: {np.unique(rec_instance_labels)}"
                    )
                    print(
                        f"    Sample panoptic labels: {rec_panoptic_labels[:10]}"
                    )

                    if 2061 in rec_panoptic_labels:
                        truck_count = np.sum(rec_panoptic_labels == 2061)
                        print(
                            f"    ✓ Truck label 2061 found: {truck_count} points"
                        )

                if label_recovery_enabled:
                    ood_panoptic_labels, ood_time, ood_stats = compute_ood_panoptic_labels(
                        pcd_vehicle,
                        self.color_dict,
                        self.num_objects,
                        debug=debug_mode)
                else:
                    ood_panoptic_labels = np.zeros(len(pcd_vehicle.points),
                                                   dtype=np.int32)
                    ood_time = 0.0
                    ood_stats = {}

                final_panoptic_labels = np.concatenate(
                    (rec_panoptic_labels, ood_panoptic_labels))

                unique_panoptic = np.unique(final_panoptic_labels)
                print(
                    f'\nFrame {frame_id} - Unique panoptic labels: {unique_panoptic}'
                )
                print(f'  Total unique labels: {len(unique_panoptic)}')
                print(
                    f'  Recovered points: {len(rec_panoptic_labels)}, OOD points: {len(ood_panoptic_labels)}'
                )

                frame_total_time = time.time() - frame_start_time

                instance_accuracy = 100.0

                timing_data = {
                    'frame_id': frame_id,
                    'frame_map_id': self.frame_map[sensor_type][frame_id],
                    'total_time_readable': format_time(frame_total_time),
                    'instance_time_readable': format_time(inst_time),
                    'ood_time_readable': format_time(ood_time),
                    'num_ood_points': len(pcd_vehicle.points),
                    'num_total_points': len(merged_pcd.points),
                    'instance_accuracy': instance_accuracy,
                    'num_recovered_instances': len(inst_stats),
                    'num_ood_objects': len(ood_stats)
                }

                for ood_id, count in ood_stats.items():
                    timing_data[f'ood_{ood_id}_points'] = count

                frame_timing_stats.append(timing_data)

                if debug_mode:
                    print(f"\nLabel Assignment Results:")

                    print(f"  Instance accuracy: {instance_accuracy:.2f}%")
                    print(f"  Total time: {format_time(frame_total_time)}")
                    print(f"    - Semantic: {format_time(sem_time)}")
                    print(f"    - Instance: {format_time(inst_time)}")
                    print(f"    - OOD: {format_time(ood_time)}")

                if np.asarray(
                        self.pc_reconstructed_indices[sensor_type][frame_id].
                        points).shape[0] == final_panoptic_labels.shape[0]:
                    self.reconstructed_panoptic_labels[sensor_type][
                        frame_id] = final_panoptic_labels
                    final_panoptic_labels.tofile(
                        f"{dir_range_type}/frame_00{self.frame_map[sensor_type][frame_id]}.bin"
                    )
                elif debug_mode:
                    print(f"\n⚠️ WARNING: Size mismatch!")
                    print(
                        f"  Point cloud size: {np.asarray(self.pc_reconstructed_indices[sensor_type][frame_id].points).shape[0]}"
                    )
                    print(
                        f"  Label array size: {final_panoptic_labels.shape[0]}"
                    )

            if len(frame_timing_stats) > 0 and ResourceLogger.wandb_enabled:
                avg_instance_acc = np.mean(
                    [s['instance_accuracy'] for s in frame_timing_stats])

                wandb_log_data = {
                    f'{sensor_type}/label_assignment/avg_instance_accuracy':
                    avg_instance_acc,
                    f'{sensor_type}/label_assignment/total_frames':
                    len(frame_timing_stats)
                }

                timing_table = wandb.Table(columns=[
                    'Frame', 'Total Time', 'Instance Time', 'OOD Time',
                    'Instance Acc'
                ],
                                           data=[[
                                               s['frame_id'],
                                               s['total_time_readable'],
                                               s['instance_time_readable'],
                                               s['ood_time_readable'],
                                               s['instance_accuracy']
                                           ] for s in frame_timing_stats])
                wandb_log_data[
                    f'{sensor_type}/label_assignment/frame_timing_table'] = timing_table

                wandb.log(wandb_log_data)

                print(f"\n{'='*80}")
                print(f"Label Assignment Summary for {sensor_type}:")
                print(f"  Average instance accuracy: {avg_instance_acc:.2f}%")
                print(f"  Total frames processed: {len(frame_timing_stats)}")
                print(f"{'='*80}\n")

        self.master[self.file_name][
            'ood_merged_point_cloud'] = f"{self.base_dir}/{self.file_name}/Reconstructed_Point_Cloud_via_Indices"
        self.master[self.file_name][
            'ood_merged_panoptic_labels'] = f"{self.base_dir}/{self.file_name}/OOD_Merged_Panoptic_Labels"

        if not eval and render_mode == 'multi' and master_save:
            append_to_json_file("master1.json", self.master)

    def get_actual_projections(self):

        for sensor_type, frames in self.frames_with_seg.items():
            if len(frames) == 0:
                continue
            if sensor_type not in self.cp_actual.keys():
                self.cp_actual[sensor_type] = {}

            for frame_id, frame in enumerate(frames):

                range_images, camera_projections, seg_labels, range_image_top_pose = (
                    frame_utils.parse_range_image_and_camera_projection(frame))

                _, cp_points = frame_utils.convert_range_image_to_point_cloud(
                    frame,
                    range_images,
                    camera_projections,
                    range_image_top_pose,
                    ri_index=0,
                    keep_polar_features=False)

                cp_points_all = np.concatenate(cp_points, axis=0)
                self.cp_actual[sensor_type][frame_id] = cp_points_all

    def cleanup(self, file_name):
        current_dir = os.getcwd()
        check_dir = Path(
            os.path.join(current_dir, f'{self.base_dir}/{file_name}'))

        if not check_dir.exists():
            print(f"Directory {check_dir} does not exist.")
            return

        contents = os.listdir(check_dir)

        if "Point Cloud" not in contents:

            for item in check_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

            check_dir.rmdir()

            print(f"Cleaned up: {check_dir}")

    def batch_search_points(self, tensor, target_points, tol=1e-6):

        expanded_tensor = tensor[None, :, :, :]
        expanded_targets = target_points[:, None, None, :]

        diff = np.abs(expanded_tensor - expanded_targets)
        matches = np.all(diff < tol, axis=-1)

        M, B, N = matches.shape

    def visualize_ood_bbox_in_scene(self):
        method = 'lidar'
        lidar = 'variable'

        def create_bbox_lines(corners):
            lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                     [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(corners),
                lines=o3d.utility.Vector2iVector(lines))
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]] *
                                                         len(lines))
            return line_set

        for sensor_type, frames in self.frames_with_seg.items():
            if len(frames) == 0:
                continue
            else:
                for frame_id, frame in enumerate(frames):

                    vis = o3d.visualization.Visualizer()
                    vis.create_window()

                    lidar_synced_boxes = [
                        lbl.box for lbl in frame.laser_labels
                    ]

                    scale_factor = 10.0

                    pcd_all = self.pc_reconstructed_indices[sensor_type][
                        lidar][method][frame_id]
                    pcd_all.normals = o3d.utility.Vector3dVector()
                    points = np.array(pcd_all.points)

                    points = points * scale_factor

                    scaled_pcd = o3d.geometry.PointCloud()
                    scaled_pcd.points = o3d.utility.Vector3dVector(points)

                    if pcd_all.has_colors():
                        colors = np.asarray(pcd_all.colors)
                        scaled_pcd.colors = o3d.utility.Vector3dVector(colors)

                    print(np.max(points[:, 0]), np.max(points[:, 1]),
                          np.max(points[2]))

                    all_geometries = [pcd_all]

                    in_box_mask_global = np.zeros(points.shape[0], dtype=bool)

                    for a in lidar_synced_boxes:

                        dx = a.length / 2
                        dy = a.width / 2
                        dz = a.height / 2

                        heading = a.heading

                        local_corners = np.array([
                            [-dx, -dy, -dz],
                            [-dx, dy, -dz],
                            [dx, dy, -dz],
                            [dx, -dy, -dz],
                            [-dx, -dy, dz],
                            [-dx, dy, dz],
                            [dx, dy, dz],
                            [dx, -dy, dz],
                        ])

                        cos_h = np.cos(heading)
                        sin_h = np.sin(heading)
                        R = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0],
                                      [0, 0, 1]])

                        rotated_corners = (R @ local_corners.T).T
                        box_center = np.array(
                            [a.center_x, a.center_y, a.center_z])
                        global_corners = rotated_corners + box_center

                        bbox_lines = create_bbox_lines(global_corners)
                        all_geometries.append(bbox_lines)

                        heading_vec_local = np.array([1.5 * dx, 0, 0])
                        heading_vec_global = R @ heading_vec_local

                        arrow_line = o3d.geometry.LineSet(
                            points=o3d.utility.Vector3dVector(
                                [box_center, box_center + heading_vec_global]),
                            lines=o3d.utility.Vector2iVector([[0, 1]]))
                        arrow_line.colors = o3d.utility.Vector3dVector(
                            [[0, 0, 1]])
                        all_geometries.append(arrow_line)

                        shifted_points = points - box_center
                        rotated_points = (R.T @ shifted_points.T).T

                        mask = ((rotated_points[:, 0] >= -dx) &
                                (rotated_points[:, 0] <= dx) &
                                (rotated_points[:, 1] >= -dy) &
                                (rotated_points[:, 1] <= dy) &
                                (rotated_points[:, 2] >= -dz) &
                                (rotated_points[:, 2] <= dz))

                        in_box_mask_global |= mask

                        points_in_box = points[mask]

                        if len(points_in_box) > 0:
                            pcd_in_box = o3d.geometry.PointCloud()
                            pcd_in_box.points = o3d.utility.Vector3dVector(
                                points_in_box)
                            pcd_in_box.paint_uniform_color([1, 0, 0])
                            all_geometries.append(pcd_in_box)

                    scale_factor = 10.0
                    for ood_id, framewise_values in self.framewise_ood_details[
                            sensor_type].items():
                        f_values = framewise_values[frame_id]

                        if f_values['num_points'] == 0:
                            continue

                        dx = (f_values['length'] * scale_factor)
                        dy = (f_values['width'] * scale_factor)
                        dz = (f_values['height'] * scale_factor)

                        heading = f_values['heading']

                        local_corners = np.array([
                            [-dx, -dy, -dz],
                            [-dx, dy, -dz],
                            [dx, dy, -dz],
                            [dx, -dy, -dz],
                            [-dx, -dy, dz],
                            [-dx, dy, dz],
                            [dx, dy, dz],
                            [dx, -dy, dz],
                        ])

                        cos_h = np.cos(heading)
                        sin_h = np.sin(heading)
                        R = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0],
                                      [0, 0, 1]])

                        rotated_corners = (R @ local_corners.T).T
                        box_center = np.array([
                            f_values['center_x'] * scale_factor,
                            f_values['center_y'] * scale_factor,
                            f_values['center_z'] * scale_factor
                        ])
                        global_corners = rotated_corners + box_center

                        bbox_lines = create_bbox_lines(global_corners)
                        all_geometries.append(bbox_lines)

                        heading_vec_local = np.array([1.5 * dx, 0, 0])
                        heading_vec_global = R @ heading_vec_local

                        arrow_line = o3d.geometry.LineSet(
                            points=o3d.utility.Vector3dVector(
                                [box_center, box_center + heading_vec_global]),
                            lines=o3d.utility.Vector2iVector([[0, 1]]))
                        arrow_line.colors = o3d.utility.Vector3dVector(
                            [[1, 0, 0]])
                        all_geometries.append(arrow_line)

                        shifted_points = points - box_center
                        rotated_points = (R.T @ shifted_points.T).T

                        mask = ((rotated_points[:, 0] >= -dx) &
                                (rotated_points[:, 0] <= dx) &
                                (rotated_points[:, 1] >= -dy) &
                                (rotated_points[:, 1] <= dy) &
                                (rotated_points[:, 2] >= -dz) &
                                (rotated_points[:, 2] <= dz))

                        in_box_mask_global |= mask

                        points_in_box = points[mask]

                        if len(points_in_box) > 0:
                            pcd_in_box = o3d.geometry.PointCloud()
                            pcd_in_box.points = o3d.utility.Vector3dVector(
                                points_in_box)
                            pcd_in_box.paint_uniform_color([0, 1, 0])
                            all_geometries.append(pcd_in_box)

                    all_geometries = [scaled_pcd] + all_geometries[1:]

                    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=2.0 * scale_factor, origin=[0, 0, 0])
                    all_geometries.append(origin)

                    for geom in all_geometries:
                        vis.add_geometry(geom)

                    view_ctl = vis.get_view_control()

                    view_ctl.set_up([0, -1, 0])
                    view_ctl.set_front([0, 0, -1])
                    view_ctl.set_lookat([0, 0, 0])

                    cam_params = view_ctl.convert_to_pinhole_camera_parameters(
                    )
                    cam_params.extrinsic = np.array(
                        [[1, 0, 0, 3 * scale_factor], [0, 1, 0, 0],
                         [0, 0, 1, 1 * scale_factor], [0, 0, 0, 1]])
                    view_ctl.convert_from_pinhole_camera_parameters(cam_params)

                    opt = vis.get_render_option()
                    opt.background_color = np.array([0, 0, 0])
                    opt.point_size = 5.0
                    opt.line_width = 3.0

                    print("Running the visualizer.")

                    vis.run()

                    vis.destroy_window()

    def get_ood_bbox(self):
        def recover_box_params(corners, heading):

            center = np.mean(corners, axis=0)
            center_x, center_y, center_z = center

            cos_h = np.cos(-heading)
            sin_h = np.sin(-heading)
            R_inv = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0,
                                                                      1]])

            local_corners = (R_inv @ (corners - center).T).T

            min_corner = local_corners.min(axis=0)
            max_corner = local_corners.max(axis=0)

            length = max_corner[0] - min_corner[0]
            width = max_corner[1] - min_corner[1]
            height = max_corner[2] - min_corner[2]

            return {
                "center_x": float(center_x),
                "center_y": float(center_y),
                "center_z": float(center_z),
                "length": float(length),
                "width": float(width),
                "height": float(height),
                "heading": float(heading)
            }

        def compute_3d_bounding_box(points):
            if len(points) == 0:
                return None

            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')
            points = np.array(points)

            for point in points:

                x, y, z = point[0], point[1], point[2]
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_z = min(min_z, z)
                max_z = max(max_z, z)

            return (min_x, min_y, min_z), (max_x, max_y, max_z)

        def compute_bbox_corners(min_corner, max_corner, color=(1, 0, 0)):
            """Create a wireframe bounding box for visualization"""
            min_bound = np.array(min_corner)
            max_bound = np.array(max_corner)

            points = [[min_bound[0], min_bound[1], min_bound[2]],
                      [max_bound[0], min_bound[1], min_bound[2]],
                      [max_bound[0], max_bound[1], min_bound[2]],
                      [min_bound[0], max_bound[1], min_bound[2]],
                      [min_bound[0], min_bound[1], max_bound[2]],
                      [max_bound[0], min_bound[1], max_bound[2]],
                      [max_bound[0], max_bound[1], max_bound[2]],
                      [min_bound[0], max_bound[1], max_bound[2]]]

            return points

        def create_bounding_box_geometry(min_corner,
                                         max_corner,
                                         color=(1, 0, 0)):
            """Create a wireframe bounding box for visualization"""
            min_bound = np.array(min_corner)
            max_bound = np.array(max_corner)

            points = [[min_bound[0], min_bound[1], min_bound[2]],
                      [max_bound[0], min_bound[1], min_bound[2]],
                      [max_bound[0], max_bound[1], min_bound[2]],
                      [min_bound[0], max_bound[1], min_bound[2]],
                      [min_bound[0], min_bound[1], max_bound[2]],
                      [max_bound[0], min_bound[1], max_bound[2]],
                      [max_bound[0], max_bound[1], max_bound[2]],
                      [min_bound[0], max_bound[1], max_bound[2]]]

            lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                     [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color(color)
            return line_set

        lidar = 'variable'
        method = 'lidar'

        def create_heading_line(center, heading, length=1.5, color=(0, 1, 0)):
            """
            Create a line indicating heading direction in the XY plane.
            Starts from the center and extends in heading direction.
            """
            start = np.array(center)
            direction = np.array([np.cos(heading), np.sin(heading), 0])
            end = start + direction * length

            points = [start, end]
            lines = [[0, 1]]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color(color)
            return line_set

        for sensor_type, frames in self.frames_with_seg.items():
            if sensor_type not in self._get_active_lidar_pipeline_sensor_types(
            ):
                continue
            if len(frames) == 0:
                continue
            if sensor_type not in self.point_cloud_rec_dict.keys():
                self.point_cloud_rec_dict[sensor_type] = {}
            for frame_id, frame in enumerate(frames):
                if frame_id not in self.point_cloud_rec_dict[sensor_type].keys(
                ):
                    self.point_cloud_rec_dict[sensor_type][frame_id] = {}
                reconstructed_ood_pc = self.reconstructed_ood_points[
                    sensor_type][frame_id]
                reconstructed_ood_points = np.array(
                    reconstructed_ood_pc.points)
                reconstructed_ood_colors = np.array(
                    reconstructed_ood_pc.colors)
                unique_colors = np.unique(reconstructed_ood_colors, axis=0)

                for color in unique_colors:

                    mask = np.all(reconstructed_ood_colors == color, axis=1)
                    ood_points = reconstructed_ood_points[mask]

                    tolerance = 1

                    ood_id = None
                    for k, v in self.color_dict.items():
                        dist = np.linalg.norm(color * 255.0 -
                                              np.array(v, dtype=np.float32))
                        if dist < tolerance:
                            ood_id = k
                            break

                    if ood_id not in self.point_cloud_rec_dict[sensor_type][
                            frame_id].keys():
                        self.point_cloud_rec_dict[sensor_type][frame_id][
                            ood_id] = []
                    self.point_cloud_rec_dict[sensor_type][frame_id][
                        ood_id] = ood_points
                    self.framewise_ood_details[sensor_type][frame_id][ood_id][
                        'num_points_reconstructed'] = len(ood_points)

                    min_corner, max_corner = compute_3d_bounding_box(
                        ood_points)
                    corners = compute_bbox_corners(min_corner,
                                                   max_corner,
                                                   color=(1, 0, 0))

                    details = self.ood_details[ood_id]
                    pose_4x4 = self._normalize_pose_4x4(details.get('pose'))
                    if pose_4x4 is None:
                        continue

                    obj_rot_world = pose_4x4[:3, :3]

                    x_axis_world = obj_rot_world[:, 0]

                    vehicle_rot = np.array(
                        self.vehicle_pose[sensor_type][frame_id])[:3, :3]
                    world_to_vehicle_rot = np.linalg.inv(vehicle_rot)

                    x_axis_vehicle = world_to_vehicle_rot @ x_axis_world

                    dir_vec = x_axis_vehicle[:2] / np.linalg.norm(
                        x_axis_vehicle[:2])
                    heading = np.arctan2(dir_vec[1], dir_vec[0])

                    params = recover_box_params(corners, heading)

                    for key, value in params.items():
                        self.framewise_ood_details[sensor_type][frame_id][
                            ood_id][key] = value

                for ood_id, points in self.point_cloud_rec_dict[sensor_type][
                        frame_id].items():

                    min_corner, max_corner = compute_3d_bounding_box(points)

                    bbox = create_bounding_box_geometry(min_corner,
                                                        max_corner,
                                                        color=(1, 0, 0))

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
                    pcd.paint_uniform_color([0.7, 0.7, 0.7])

                    heading = self.framewise_ood_details[sensor_type][
                        frame_id][ood_id]['heading']
                    center = [
                        self.framewise_ood_details[sensor_type][frame_id]
                        [ood_id]['center_x'],
                        self.framewise_ood_details[sensor_type][frame_id]
                        [ood_id]['center_y'],
                        self.framewise_ood_details[sensor_type][frame_id]
                        [ood_id]['center_z']
                    ]

                    heading_line = create_heading_line(center,
                                                       heading,
                                                       length=1.5,
                                                       color=(0, 1, 0))

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(np.array(points))
                    pcd.paint_uniform_color([0.3, 0.3, 0.3])

    def go(self,
           file_dir,
           sensor,
           eval=False,
           render_mode='multi',
           dist=None,
           size_multiplier=1.0,
           retry=False,
           save_path=None,
           image_path=None,
           master_save=False):
        try:
            print(f"\n[GO METHOD] Starting go() for {self.file_name}",
                  flush=True)
            print(
                f"[GO METHOD] file_dir={file_dir}, sensor={sensor}, render_mode={render_mode}",
                flush=True)
            self._update_run_progress(
                stage="go",
                status="starting",
                sensor=sensor,
                render_mode=render_mode,
                eval=bool(eval),
            )
            stage_timing_summary = {}

            def run_stage(stage_name, fn, *args, **kwargs):
                print(f"[GO STAGE] START {stage_name}", flush=True)
                start_time = time.time()
                self._update_run_progress(stage=stage_name, status="running")
                try:
                    result = fn(*args, **kwargs)
                except Exception:
                    elapsed_s = time.time() - start_time
                    self._update_run_progress(
                        stage=stage_name,
                        status="failed",
                        elapsed_s=round(float(elapsed_s), 3),
                    )
                    raise
                elapsed_s = time.time() - start_time
                print(f"[GO STAGE] END   {stage_name} ({elapsed_s:.2f}s)",
                      flush=True)
                self._update_run_progress(
                    stage=stage_name,
                    status="completed",
                    elapsed_s=round(float(elapsed_s), 3),
                )
                stage_timing_summary.setdefault(stage_name, []).append(
                    round(float(elapsed_s), 3))
                return result

            if image_path:
                self.image_path = image_path

            elif not self.image_path:
                train_valid = "Train" if self.train else "Valid"
                self.image_path = f"Pipeline_2_{train_valid}"

            effective_sensor_for_frame_selection = sensor
            if self.lidar_only_resume:
                if sensor in {
                        "both", "camera"
                } and self.lidar_insertion_frame_source in {"camera", "auto"}:
                    effective_sensor_for_frame_selection = 'both'
                else:
                    effective_sensor_for_frame_selection = 'lidar'
            print(
                "[GO METHOD] Starting find_frames_with_seg... "
                f"(selection_sensor={effective_sensor_for_frame_selection})",
                flush=True,
            )
            self.find_frames_with_seg(file_dir,
                                      effective_sensor_for_frame_selection)
            print(
                f"[GO METHOD] find_frames_with_seg completed. "
                f"Found {len(self.frames_with_seg.get('lidar', []))} lidar frames",
                flush=True,
            )

            camera_pipeline_requested = (sensor in {
                "camera", "both"
            }) and (not self.lidar_only_resume)
            lidar_pipeline_requested = (sensor in {"camera", "lidar", "both"}
                                        and self.enable_lidar_insertion)
            self.lidar_insertion_sensor_type = (
                self._resolve_lidar_insertion_sensor_type()
                if lidar_pipeline_requested else None)
            required_lidar_frames = bool(sensor == "lidar"
                                         or lidar_pipeline_requested)
            strict_lidar_frame_requirement = bool(
                sensor == "lidar"
                and self.lidar_insertion_sensor_type != 'camera')
            render_sensor_mode = 'lidar' if self.lidar_only_resume else sensor

            if lidar_pipeline_requested and self.lidar_insertion_sensor_type is None:
                print(
                    "\033[31mSAMPLER: No valid frame stream available for LiDAR insertion.\033[0m"
                )
                self._update_run_progress(stage="go",
                                          status="failed",
                                          reason="no_lidar_frames")
                return
            if lidar_pipeline_requested:
                print(
                    f"\033[36mLiDAR insertion will use sensor stream "
                    f"'{self.lidar_insertion_sensor_type}' "
                    f"(preference={self.lidar_insertion_frame_source}).\033[0m"
                )
            required_insertion_frames = (len(
                self.frames_with_seg.get(self.lidar_insertion_sensor_type, []))
                                         if self.lidar_insertion_sensor_type
                                         is not None else len(
                                             self.frames_with_seg['lidar']))
            if required_lidar_frames and required_insertion_frames == 0:
                if self.lidar_insertion_sensor_type == 'camera':
                    print(
                        "\033[31mSAMPLER: No camera frames available for LiDAR insertion.\033[0m"
                    )
                    self._update_run_progress(
                        stage="go",
                        status="failed",
                        reason="no_camera_frames_for_lidar")
                else:
                    print(
                        "\033[31mSAMPLER: No frames with lidar segmentation labels found.\033[0m"
                    )
                    self._update_run_progress(stage="go",
                                              status="failed",
                                              reason="no_lidar_frames")
                return
            if camera_pipeline_requested and len(
                    self.frames_with_seg['camera']) == 0:
                print(
                    "\033[31mSAMPLER: No frames with camera panoptic labels found.\033[0m"
                )
                self._update_run_progress(stage="go",
                                          status="failed",
                                          reason="no_camera_frames")
                return
            if required_lidar_frames and required_insertion_frames < self.num_frames:
                if strict_lidar_frame_requirement:
                    print(
                        "\033[31mNot sufficient LiDAR frames found for rendering.\033[0m"
                    )
                    self._update_run_progress(
                        stage="go",
                        status="failed",
                        reason="insufficient_lidar_frames",
                        found=int(required_insertion_frames),
                        required=int(self.num_frames),
                    )
                    return
                print(
                    "\033[33mSparse LiDAR insertion frame availability detected: "
                    f"using {required_insertion_frames} frames from "
                    f"'{self.lidar_insertion_sensor_type}' for the LiDAR pipeline, "
                    f"while keeping {self.num_frames} camera frames.\033[0m")
            if camera_pipeline_requested and len(
                    self.frames_with_seg['camera']) < self.num_frames:
                print(
                    "\033[31mNot sufficient camera frames found for rendering.\033[0m"
                )
                self._update_run_progress(
                    stage="go",
                    status="failed",
                    reason="insufficient_camera_frames",
                    found=int(len(self.frames_with_seg['camera'])),
                    required=int(self.num_frames),
                )
                return

            if camera_pipeline_requested or required_lidar_frames:

                run_stage("collect_global_bounding_boxes",
                          self.collect_global_bounding_boxes)
                run_stage("get_vehicle_pose",
                          self.get_vehicle_pose,
                          render_mode=render_mode)

                ood_file = None
                ood_details = None

                load_saved_ood_details = bool(retry
                                              or self.load_saved_ood_details)
                if load_saved_ood_details:
                    if eval:
                        ood_file = os.path.join(self.base_dir, self.file_name,
                                                "Evaluation1",
                                                "ood_details.json")
                    else:
                        ood_file = os.path.join(self.base_dir, self.file_name,
                                                "ood_details.json")
                    ood_file = os.path.abspath(os.path.expanduser(ood_file))

                if ood_file and os.path.exists(ood_file):
                    print(f"File exists: {ood_file}")
                    with open(ood_file, 'r') as f:
                        ood_details = json.load(f)

                    print(f"Loaded ood_details: {ood_details}")
                    print(
                        f"Type: {type(ood_details)}, Length: {len(ood_details)}"
                    )

                    if ood_details and isinstance(
                            ood_details, dict) and len(ood_details) > 0:
                        if self.require_exact_placement:
                            saved_ood_ids = checked_saved_object_ids(
                                ood_details, self._exact_assignment_rows())
                        else:
                            saved_ood_ids = [
                                int(key) for key in ood_details if int(key) != 0
                            ]
                        self.models = []
                        self.num_objects = len(saved_ood_ids)
                        print("Num objects from file:", self.num_objects)
                        self._ensure_ood_color_capacity(self.num_objects)
                        self._initialize_sequence_placement_constraints()
                        for ood_id in saved_ood_ids:
                            ood_id_str = str(ood_id)
                            if ood_id != 0:
                                file_identifier = ood_details[ood_id_str][
                                    'file_identifier'] if 'file_identifier' in ood_details[
                                        ood_id_str] else None
                                lvis = ood_details[ood_id_str]['lvis']
                                path = ood_details[ood_id_str]['path']
                                scale = float(ood_details[ood_id_str].get(
                                    'scale', 1.0))
                                sampled_position = ood_details[ood_id_str][
                                    'sampled_position'] if 'sampled_position' in ood_details[
                                        ood_id_str] else None

                                print("Loading object:", lvis, path)

                                self.models.append(
                                    (file_identifier, lvis, path, scale))

                                if sampled_position is not None:

                                    if ood_id not in self.ood_details:
                                        self.ood_details[ood_id] = {}
                                    self.ood_details[ood_id][
                                        'position'] = sampled_position

                                    visibility_sensor = 'camera' if len(
                                        self.camera_data.get(
                                            'camera', [])) > 0 else 'lidar'
                                    self._register_accepted_placement(
                                        ood_id=ood_id,
                                        point_world=sampled_position,
                                        visibility_sensor=visibility_sensor,
                                    )

                    else:
                        print("ood_details is empty or invalid")
                elif (load_saved_ood_details and self.skip_blender_render
                      and ood_file):

                    sampled_positions_json = os.path.join(
                        self._get_placement_debug_dir(eval=eval),
                        "sampled_positions_only.json",
                    )
                    print(
                        f"[RECOVERY] Missing ood_details.json for retry+skip path. "
                        f"Trying sampled placements from: {sampled_positions_json}"
                    )
                    if not self.models:
                        print("Num objects:", self.num_objects)
                        self.sample_random_models(eval=eval)
                    recovered_ood_details = {}
                    if os.path.exists(sampled_positions_json):
                        with open(sampled_positions_json, "r") as f:
                            sampled_entries = json.load(f)
                        sampled_by_order = {}
                        if isinstance(sampled_entries, list):
                            for idx_entry, entry in enumerate(sampled_entries,
                                                              start=1):
                                if not isinstance(entry, dict):
                                    continue
                                raw_order = entry.get("order", idx_entry)
                                try:
                                    order = int(raw_order)
                                except Exception:
                                    order = idx_entry
                                if not all(k in entry
                                           for k in ("x", "y", "z")):
                                    continue
                                sampled_by_order[order] = [
                                    [0.0, 0.0, 0.0,
                                     float(entry["x"])],
                                    [0.0, 0.0, 0.0,
                                     float(entry["y"])],
                                    [0.0, 0.0, 0.0,
                                     float(entry["z"])],
                                    [0.0, 0.0, 0.0, 1.0],
                                ]
                        for ood_id in range(1, int(self.num_objects) + 1):
                            model_info = self.models[ood_id -
                                                     1] if ood_id - 1 < len(
                                                         self.models) else (
                                                             None, "unknown",
                                                             None, 1.0)
                            if len(model_info) >= 4:
                                file_identifier, lvis, path, scale = model_info[
                                    0], model_info[1], model_info[2], float(
                                        model_info[3])
                            else:
                                file_identifier, lvis, path = model_info[
                                    0], model_info[1], model_info[2]
                                scale = 1.0
                            recovered_entry = {
                                "file_identifier": file_identifier,
                                "lvis": lvis,
                                "path": path,
                                "scale": float(scale),
                                "sampled_position":
                                sampled_by_order.get(ood_id),
                            }
                            if sampled_by_order.get(ood_id) is not None:
                                recovered_entry["pose"] = sampled_by_order.get(
                                    ood_id)
                            recovered_ood_details[str(
                                ood_id)] = recovered_entry
                    if recovered_ood_details:
                        ood_details = recovered_ood_details
                        os.makedirs(os.path.dirname(ood_file), exist_ok=True)
                        with open(ood_file, "w") as f:
                            json.dump(ood_details, f, indent=4)
                        print(
                            f"[RECOVERY] Rebuilt ood_details.json with {len(ood_details)} objects: {ood_file}"
                        )
                        self.models = []
                        self._ensure_ood_color_capacity(self.num_objects)
                        self._initialize_sequence_placement_constraints()
                        for ood_id_str in ood_details.keys():
                            ood_id = int(ood_id_str)
                            if ood_id != 0:
                                file_identifier = ood_details[ood_id_str][
                                    'file_identifier'] if 'file_identifier' in ood_details[
                                        ood_id_str] else None
                                lvis = ood_details[ood_id_str]['lvis']
                                path = ood_details[ood_id_str]['path']
                                scale = float(ood_details[ood_id_str].get(
                                    'scale', 1.0))
                                sampled_position = ood_details[ood_id_str][
                                    'sampled_position'] if 'sampled_position' in ood_details[
                                        ood_id_str] else None
                                print("Loading object:", lvis, path)
                                self.models.append(
                                    (file_identifier, lvis, path, scale))
                                if sampled_position is not None:
                                    if ood_id not in self.ood_details:
                                        self.ood_details[ood_id] = {}
                                    self.ood_details[ood_id][
                                        'position'] = sampled_position
                                    visibility_sensor = 'camera' if len(
                                        self.camera_data.get(
                                            'camera', [])) > 0 else 'lidar'
                                    self._register_accepted_placement(
                                        ood_id=ood_id,
                                        point_world=sampled_position,
                                        visibility_sensor=visibility_sensor,
                                    )
                    else:
                        raise FileNotFoundError(
                            f"Retry+skip_blender_render requested, but recovery failed: "
                            f"could not build ood_details from {sampled_positions_json}"
                        )
                else:

                    if not self.models:
                        print("Num objects:", self.num_objects)
                        if self.request_only:

                            print(
                                "\033[36mREQUEST-ONLY: skipping object path sampling/existence checks.\033[0m"
                            )
                        else:
                            self.sample_random_models(eval=eval)
                    else:
                        print(
                            f"Using pre-configured models: {len(self.models)} objects"
                        )
                    self._ensure_ood_color_capacity(self.num_objects)
                ood_id = 0
                times_failed = 0
                skip_rest = False
                lidar_pipeline_requested = (sensor in {"lidar", "both"}
                                            and self.enable_lidar_insertion)
                lidar_aux_required = (self.compute_rays_aux
                                      or lidar_pipeline_requested
                                      or self.enable_projection_debug
                                      or self.enable_intensity_debug)
                if lidar_aux_required:
                    run_stage("extract_lidar_parameters",
                              self.extract_lidar_parameters)
                    if not self.compute_rays_aux and lidar_pipeline_requested:
                        print(
                            "[GO STAGE] FORCE get_rays_az_inc "
                            "(required for lidar insertion pipeline).",
                            flush=True,
                        )
                    run_stage("get_rays_az_inc", self.get_rays_az_inc)
                else:
                    print(
                        "[GO STAGE] SKIP extract_lidar_parameters/get_rays_az_inc "
                        "(no lidar insertion and no lidar debug/aux requested).",
                        flush=True,
                    )
                lidar_metadata_dir = Path(
                    os.path.join(
                        os.getcwd(),
                        f"{self.base_dir}/{self.file_name}/lidar_label_metadata"
                    ))
                if self.export_lidar_label_metadata or lidar_metadata_dir.exists(
                ):
                    run_stage("get_color_top_lidar_point_cloud",
                              self.get_color_top_lidar_point_cloud)
                else:
                    print(
                        "[GO STAGE] SKIP get_color_top_lidar_point_cloud "
                        "(no lidar_label_metadata; placement will use geometric fallback).",
                        flush=True,
                    )
                run_stage("get_top_lidar_point_cloud",
                          self.get_top_lidar_point_cloud)
                if camera_pipeline_requested:
                    run_stage(
                        "prepare_camera_accumulation_points",
                        self._prepare_camera_accumulation_points,
                    )
                if lidar_aux_required:
                    run_stage("get_range_image_indices_of_point_cloud",
                              self.get_range_image_indices_of_point_cloud)
                else:
                    print(
                        "[GO STAGE] SKIP get_range_image_indices_of_point_cloud "
                        "(no lidar insertion and no lidar debug/aux requested).",
                        flush=True,
                    )
                if self.request_only:
                    run_stage(
                        "request_only_generate_placement_artifacts",
                        self.check_render_points_for_lidar,
                        ood_id=1,
                        scale=1.0,
                        sensor='lidar',
                        render_mode=render_mode,
                        eval=eval,
                        dist=dist,
                        request_only=True,
                    )
                    print(
                        "[GO STAGE] REQUEST-ONLY mode: generated placement request/BEV artifacts "
                        "and exiting before OOD loop.",
                        flush=True,
                    )
                    self._update_run_progress(
                        stage="go",
                        status="completed",
                        mode="request_only",
                    )
                    return True

                blender_parallel_enabled = (
                    (not self.stop_after_sampling)
                    and (not self.skip_blender_render)
                    and (self.blender_effective_workers > 1))
                parallel_render_tasks = []
                parallel_validation_context = {}
                if blender_parallel_enabled:
                    print(
                        "\033[36m[BLENDER-PARALLEL] Enabled for OOD render stage: "
                        f"workers={self.blender_effective_workers}, "
                        f"gpu_plan={self.blender_worker_gpu_plan}\033[0m")
                for i in range(self.num_objects):
                    try:
                        self._update_run_progress(
                            stage="ood_loop",
                            status="running",
                            ood_index=int(i + 1),
                            total_ood=int(self.num_objects),
                        )
                        model_info = self.models[i]
                        if len(model_info) >= 4:
                            file_identifier, lvis, path, scale = model_info[
                                0], model_info[1], model_info[2], float(
                                    model_info[3])
                        else:
                            file_identifier, lvis, path = model_info[
                                0], model_info[1], model_info[2]
                            scale = 1.0
                        model_meta = {}
                        if i < len(self.sampled_model_metadata):
                            candidate_meta = self.sampled_model_metadata[i]
                            if isinstance(candidate_meta, dict):
                                model_meta = candidate_meta
                        base_scale = float(model_meta.get("base_scale", scale))
                        path = os.path.expanduser(os.path.expandvars(path))
                        print(lvis, path)
                        self.ood_details[i + 1] = {
                            'file_identifier': file_identifier,
                            'lvis': lvis,
                            'path': path,
                            'scale': base_scale,
                            'base_scale': base_scale,
                        }
                        if model_meta:
                            if model_meta.get("description"):
                                self.ood_details[
                                    i + 1]['description'] = model_meta.get(
                                        "description")
                            if model_meta.get("size_bucket"):
                                self.ood_details[
                                    i + 1]['size_bucket'] = model_meta.get(
                                        "size_bucket")
                            if model_meta.get("scale_policy"):
                                self.ood_details[
                                    i + 1]['scale_policy'] = model_meta.get(
                                        "scale_policy")
                            if model_meta.get("target_ref_dim_m") is not None:
                                self.ood_details[
                                    i + 1]['target_ref_dim_m'] = float(
                                        model_meta.get("target_ref_dim_m"))
                            if model_meta.get("target_ref_dim_source"):
                                self.ood_details[
                                    i + 1]['target_ref_dim_source'] = str(
                                        model_meta.get(
                                            "target_ref_dim_source"))
                            if model_meta.get(
                                    "target_ref_dim_raw") is not None:
                                try:
                                    self.ood_details[
                                        i + 1]['target_ref_dim_raw'] = float(
                                            model_meta.get(
                                                "target_ref_dim_raw"))
                                except Exception:
                                    pass
                            if model_meta.get("target_max_dim_m") is not None:
                                try:
                                    self.ood_details[
                                        i + 1]['target_max_dim_m'] = float(
                                            model_meta.get("target_max_dim_m"))
                                except Exception:
                                    pass
                            if model_meta.get("candidate_id"):
                                self.ood_details[
                                    i + 1]['candidate_id'] = model_meta.get(
                                        "candidate_id")
                            if model_meta.get("source_uid"):
                                self.ood_details[
                                    i + 1]['source_uid'] = model_meta.get(
                                        "source_uid")
                            if model_meta.get("pool_index") is not None:
                                self.ood_details[i + 1]['pool_index'] = int(
                                    model_meta.get("pool_index"))
                            if model_meta.get(
                                    "placement_slot_index") is not None:
                                try:
                                    self.ood_details[
                                        i + 1]['placement_slot_index'] = int(
                                            model_meta.get(
                                                "placement_slot_index"))
                                except Exception:
                                    pass
                            if model_meta.get(
                                    "placement_candidate_id") is not None:
                                try:
                                    self.ood_details[
                                        i + 1]['placement_candidate_id'] = int(
                                            model_meta.get(
                                                "placement_candidate_id"))
                                except Exception:
                                    pass
                            if model_meta.get(
                                    "placement_point_xyz") is not None:
                                try:
                                    placement_point_xyz = np.asarray(
                                        model_meta.get("placement_point_xyz"),
                                        dtype=np.float64,
                                    ).reshape(-1)
                                    if placement_point_xyz.shape[0] >= 3:
                                        self.ood_details[
                                            i + 1]['placement_point_xyz'] = [
                                                float(placement_point_xyz[0]),
                                                float(placement_point_xyz[1]),
                                                float(placement_point_xyz[2]),
                                            ]
                                except Exception:
                                    pass
                            for numeric_key in (
                                    "placement_distance_m",
                                    "placement_distance_q",
                                    "placement_visibility_score",
                            ):
                                if model_meta.get(numeric_key) is None:
                                    continue
                                try:
                                    self.ood_details[
                                        i + 1][numeric_key] = float(
                                            model_meta.get(numeric_key))
                                except Exception:
                                    pass
                            if model_meta.get("placement_visible_camera_count"
                                              ) is not None:
                                try:
                                    self.ood_details[i + 1][
                                        'placement_visible_camera_count'] = int(
                                            model_meta.get(
                                                "placement_visible_camera_count"
                                            ))
                                except Exception:
                                    pass
                            if model_meta.get("placement_plan"):
                                self.ood_details[
                                    i + 1]['placement_plan'] = str(
                                        model_meta.get("placement_plan"))
                            if model_meta.get("preserve_manual_scale_at_render"
                                              ) is not None:
                                self.ood_details[i + 1][
                                    'preserve_manual_scale_at_render'] = bool(
                                        model_meta.get(
                                            "preserve_manual_scale_at_render"))
                            if model_meta.get("manual_scale_review_source"):
                                self.ood_details[
                                    i + 1]['manual_scale_review_source'] = str(
                                        model_meta.get(
                                            "manual_scale_review_source"))
                            if model_meta.get("original_scale") is not None:
                                try:
                                    self.ood_details[
                                        i + 1]['original_scale'] = float(
                                            model_meta.get("original_scale"))
                                except Exception:
                                    pass
                        loaded_ood_entry = None

                        should_sample_new = True
                        if ood_details and str(i + 1) in ood_details:
                            loaded_ood_entry = ood_details[str(i + 1)]
                            if isinstance(loaded_ood_entry, dict):
                                preserve_saved_render_state = bool(retry)
                                skipped_saved_keys = {
                                    "sampled_position",
                                    "position",
                                }
                                if bool(self.ood_details[i + 1].get(
                                        "preserve_manual_scale_at_render",
                                        False)):
                                    skipped_saved_keys.update({
                                        "target_ref_dim_m",
                                        "target_ref_dim_source",
                                        "target_ref_dim_raw",
                                        "target_max_dim_m",
                                        "effective_scale",
                                        "distance_scale_multiplier",
                                        "scale_policy",
                                    })
                                if not preserve_saved_render_state:
                                    skipped_saved_keys.update({
                                        "pose",
                                        "bbox_3d",
                                        "render_pose_override",
                                        "shadow_plane_point_override",
                                        "shadow_plane_normal_override",
                                        "ground_contact_epsilon_override",
                                        "shadow_catcher_size_override",
                                        "camera_labels",
                                    })

                                for key, value in loaded_ood_entry.items():
                                    if key in skipped_saved_keys:
                                        continue
                                    if key == "pose":
                                        normalized_pose = self._normalize_pose_4x4(
                                            value)
                                        if normalized_pose is None:
                                            continue
                                        self.ood_details[
                                            i +
                                            1][key] = normalized_pose.tolist()
                                        continue
                                    self.ood_details[i + 1][key] = value

                            if isinstance(
                                    loaded_ood_entry, dict
                            ) and 'sampled_position' in loaded_ood_entry:
                                existing_pos = loaded_ood_entry[
                                    'sampled_position']
                                print(
                                    f"[DEBUG] OOD {i+1}: Found in loaded file, sampled_position = {existing_pos}"
                                )
                                if existing_pos is not None:
                                    should_sample_new = False
                                    position = existing_pos
                                else:
                                    print(
                                        f"[DEBUG] OOD {i+1}: sampled_position is null - will sample new location"
                                    )
                            else:
                                print(
                                    f"[DEBUG] OOD {i+1}: sampled_position key missing - will sample new location"
                                )
                        else:
                            print(
                                f"[DEBUG] OOD {i+1}: Not found in loaded file or ood_details is empty"
                            )

                        if self.skip_blender_render and should_sample_new:
                            if self.require_exact_placement:
                                raise RuntimeError(
                                    f"OOD {i+1} has no saved placement for --skip_blender_render. "
                                    "Render it first without --skip_blender_render.")
                            print(
                                f"\033[33m[REUSE] OOD {i+1} has no sampled_position in retry metadata; "
                                "skipping this object instead of rerendering.\033[0m"
                            )
                            self.ood_details[i + 1]['position'] = None
                            self.ood_details[i + 1].pop('pose', None)
                            continue

                        if should_sample_new:
                            if self.enable_distance_buckets:
                                bucket_idx = self._choose_distance_bucket_for_object(
                                    i + 1)
                                if bucket_idx is not None:
                                    bounds = self._bucket_bounds(bucket_idx)
                                    if bounds is not None:
                                        print(
                                            f"\033[36mPlacement target for OOD {i+1}: "
                                            f"distance bucket [{bounds[0]:.1f}, {bounds[1]:.1f})\033[0m"
                                        )
                            print('inside ood location finder')

                            position, frame_id = self.sample_ood_location(
                                scale=scale,
                                ood_id=i + 1,
                                sensor_type='lidar',
                                render_mode=render_mode,
                                eval=eval,
                                dist=dist)

                        self._assert_saved_placement(i + 1, position)

                        self.ood_details[i + 1]['position'] = position

                        self.ood_details[i + 1]['pose'] = position
                        effective_scale = float(self.ood_details[i + 1].get(
                            'effective_scale', base_scale))
                        if position is not None:
                            if (not should_sample_new) and (
                                    'effective_scale'
                                    in self.ood_details[i + 1]):
                                effective_scale = float(
                                    self.ood_details[i + 1].get(
                                        'effective_scale', base_scale))
                            else:
                                distance_to_ego_m = self._estimate_distance_to_vehicle_pose(
                                    position,
                                    preferred_sensor='camera',
                                )
                                preserve_manual_scale = bool(
                                    self.ood_details[i + 1].get(
                                        'preserve_manual_scale_at_render',
                                        False))
                                if preserve_manual_scale:
                                    distance_scale_multiplier = 1.0
                                    effective_scale = float(base_scale)
                                else:
                                    distance_scale_multiplier = self._compute_distance_scale_multiplier(
                                        distance_to_ego_m)
                                    effective_scale = float(
                                        base_scale) * float(
                                            distance_scale_multiplier)
                                effective_scale = float(
                                    np.clip(
                                        effective_scale,
                                        self.object_scale_min_clip,
                                        self.object_scale_max_clip,
                                    ))
                                if distance_to_ego_m is not None and np.isfinite(
                                        distance_to_ego_m):
                                    self.ood_details[
                                        i + 1]['distance_to_ego_m'] = float(
                                            distance_to_ego_m)
                                self.ood_details[
                                    i +
                                    1]['distance_scale_multiplier'] = float(
                                        distance_scale_multiplier)
                                self.ood_details[i +
                                                 1]['effective_scale'] = float(
                                                     effective_scale)
                        else:
                            self.ood_details[i + 1]['effective_scale'] = float(
                                effective_scale)

                        if position is not None:
                            try:
                                if self.stop_after_sampling:
                                    print(
                                        f"\033[36m[SAMPLE-ONLY] OOD {i+1}: placement saved; "
                                        "skipping Blender render.\033[0m")
                                    render_ok = True
                                elif blender_parallel_enabled:
                                    gpu_hint = self._gpu_for_parallel_render_task(
                                        len(parallel_render_tasks))
                                    blender_data = self._build_blender_payload_for_ood(
                                        ood_id=i + 1,
                                        position=position,
                                        fallback_scale=scale,
                                        render_sensor_mode=render_sensor_mode,
                                        eval=eval,
                                        size_multiplier=size_multiplier,
                                        loaded_ood_details_present=bool(
                                            ood_details),
                                        gpu_id_override=gpu_hint,
                                    )
                                    parallel_render_tasks.append({
                                        "ood_id":
                                        int(i + 1),
                                        "blender_data":
                                        blender_data,
                                    })
                                    parallel_validation_context[int(i + 1)] = {
                                        "scale":
                                        float(scale),
                                        "lvis":
                                        lvis,
                                        "loaded_ood_details_present":
                                        bool(ood_details),
                                    }
                                    print(
                                        f"\033[36m[BLENDER-PARALLEL] queued OOD {i+1} "
                                        f"for initial render on GPU hint {gpu_hint}.\033[0m"
                                    )
                                    self._update_run_progress(
                                        stage="blender_parallel_queue",
                                        status="queued",
                                        ood_id=int(i + 1),
                                        queued=int(len(parallel_render_tasks)),
                                    )
                                    render_ok = True
                                else:
                                    render_ok = False
                                    visibility_attempt = 0
                                    max_visibility_attempts = int(
                                        max(1,
                                            self.required_render_max_retries))
                                    while position is not None and not render_ok:
                                        blender_data = None
                                        existing_render_status = None
                                        if self.skip_blender_render or self.retry:
                                            blender_data = self._build_blender_payload_for_ood(
                                                ood_id=i + 1,
                                                position=position,
                                                fallback_scale=scale,
                                                render_sensor_mode=
                                                render_sensor_mode,
                                                eval=eval,
                                                size_multiplier=size_multiplier,
                                                loaded_ood_details_present=bool(
                                                    ood_details),
                                            )
                                            existing_render_status = self._has_complete_existing_renders_for_ood(
                                                ood_id=i + 1,
                                                blender_data=blender_data,
                                                eval=eval,
                                            )
                                        if self.skip_blender_render:
                                            if self.require_exact_placement and not (
                                                    existing_render_status or {}).get(
                                                        "is_complete", False):
                                                raise RuntimeError(
                                                    f"OOD {i+1} has incomplete saved Blender renders; "
                                                    "cannot use --skip_blender_render.")
                                            existing_render_files = int(
                                                (existing_render_status
                                                 or {}).get(
                                                     "existing_render_files",
                                                     0))
                                            if existing_render_files <= 0:
                                                print(
                                                    f"\033[33m[REUSE] No existing renders found for OOD {i+1}; "
                                                    "skipping this object in retry mode.\033[0m"
                                                )
                                                self._update_run_progress(
                                                    stage="blender_render",
                                                    status=
                                                    "skipped_missing_reuse",
                                                    ood_id=int(i + 1),
                                                )
                                                self.ood_details[
                                                    i + 1]['position'] = None
                                                self.ood_details[i + 1].pop(
                                                    'pose', None)
                                                position = None
                                                break
                                            render_root = (
                                                f"{self.base_dir}/{self.file_name}/Evaluation/Renders/"
                                                if eval else
                                                f"{self.base_dir}/{self.file_name}/Renders/"
                                            )
                                            self.master[self.file_name][
                                                'num_ood_objects'] = int(
                                                    self.num_objects)
                                            self.master[self.file_name][
                                                'ood_renders'] = render_root
                                            print(
                                                f"[BLENDER] Reusing existing renders for OOD {i+1} "
                                                f"(files={existing_render_files}); skipping Blender subprocess."
                                            )
                                            self._update_run_progress(
                                                stage="blender_render",
                                                status="skipped_reuse",
                                                ood_id=int(i + 1),
                                                existing_render_files=int(
                                                    existing_render_files),
                                            )
                                        elif self.retry and existing_render_status and existing_render_status[
                                                "is_complete"]:
                                            render_root = (
                                                f"{self.base_dir}/{self.file_name}/Evaluation/Renders/"
                                                if eval else
                                                f"{self.base_dir}/{self.file_name}/Renders/"
                                            )
                                            self.master[self.file_name][
                                                'num_ood_objects'] = int(
                                                    self.num_objects)
                                            self.master[self.file_name][
                                                'ood_renders'] = render_root
                                            print(
                                                f"[BLENDER] Reusing complete existing renders for OOD {i+1} "
                                                f"(files={existing_render_status['existing_render_files']}/"
                                                f"{existing_render_status['expected_render_files']}); "
                                                "skipping Blender subprocess.")
                                            self._update_run_progress(
                                                stage="blender_render",
                                                status=
                                                "skipped_retry_complete_reuse",
                                                ood_id=int(i + 1),
                                                existing_render_files=int(
                                                    existing_render_status[
                                                        "existing_render_files"]
                                                ),
                                                expected_total_render_calls=
                                                int(existing_render_status[
                                                    "expected_render_files"]),
                                            )
                                        else:
                                            if blender_data is None:
                                                blender_data = self._build_blender_payload_for_ood(
                                                    ood_id=i + 1,
                                                    position=position,
                                                    fallback_scale=scale,
                                                    render_sensor_mode=
                                                    render_sensor_mode,
                                                    eval=eval,
                                                    size_multiplier=
                                                    size_multiplier,
                                                    loaded_ood_details_present=
                                                    bool(ood_details),
                                                )
                                            self.run_blender_render_subprocess(
                                                ood_id=i + 1,
                                                blender_data=blender_data)

                                        if ((not self.skip_blender_render)
                                                and render_sensor_mode
                                                in {"camera", "both"} and
                                                self.required_visible_cameras):
                                            visibility_check = self._validate_required_camera_render_visibility(
                                                ood_id=i + 1,
                                                eval=eval,
                                            )
                                            if not visibility_check["ok"]:
                                                visibility_attempt += 1
                                                print(
                                                    f"\033[33mRender visibility miss for OOD {i+1}: "
                                                    f"missing={visibility_check['missing']} "
                                                    f"counts={visibility_check['counts']} "
                                                    f"frames={visibility_check['frames']} "
                                                    f"(attempt {visibility_attempt}/{max_visibility_attempts}).\033[0m"
                                                )
                                                if visibility_attempt >= max_visibility_attempts:
                                                    print(
                                                        f"\033[31mFailed to satisfy required camera render visibility "
                                                        f"for OOD {i+1} after {max_visibility_attempts} attempts.\033[0m"
                                                    )
                                                    position = None
                                                    break

                                                self._delete_ood_render_artifacts(
                                                    ood_id=i + 1, eval=eval)
                                                position, frame_id = self.sample_ood_location(
                                                    scale=scale,
                                                    ood_id=i + 1,
                                                    sensor_type='lidar',
                                                    render_mode=render_mode,
                                                    eval=eval,
                                                    dist=dist,
                                                )
                                                if position is None:
                                                    break
                                                self.ood_details[
                                                    i +
                                                    1]['position'] = position
                                                self.ood_details[
                                                    i + 1]['pose'] = position
                                                continue

                                        if ((not self.skip_blender_render)
                                                and render_sensor_mode
                                                in {"camera", "both"} and
                                                self.enable_shadow_quality_gate
                                            ):
                                            shadow_check = self._validate_required_camera_shadow_quality(
                                                ood_id=i + 1,
                                                eval=eval,
                                            )
                                            if not shadow_check["ok"]:
                                                reason_map = shadow_check.get(
                                                    "fail_reasons", {})
                                                compact_reasons = {
                                                    cam: reasons
                                                    for cam, reasons in
                                                    reason_map.items()
                                                    if reasons
                                                }
                                                visibility_attempt += 1
                                                print(
                                                    f"\033[33mShadow quality miss for OOD {i+1}: "
                                                    f"missing={shadow_check['missing']} "
                                                    f"reasons={compact_reasons} "
                                                    f"(attempt {visibility_attempt}/{max_visibility_attempts}).\033[0m"
                                                )
                                                if visibility_attempt >= max_visibility_attempts:
                                                    print(
                                                        f"\033[31mFailed to satisfy required camera shadow quality "
                                                        f"for OOD {i+1} after {max_visibility_attempts} attempts.\033[0m"
                                                    )
                                                    position = None
                                                    break

                                                self._delete_ood_render_artifacts(
                                                    ood_id=i + 1, eval=eval)
                                                position, frame_id = self.sample_ood_location(
                                                    scale=scale,
                                                    ood_id=i + 1,
                                                    sensor_type='lidar',
                                                    render_mode=render_mode,
                                                    eval=eval,
                                                    dist=dist,
                                                )
                                                if position is None:
                                                    break
                                                self.ood_details[
                                                    i +
                                                    1]['position'] = position
                                                self.ood_details[
                                                    i + 1]['pose'] = position
                                                continue

                                        render_ok = True

                                if render_ok:
                                    if blender_parallel_enabled and (
                                            not self.stop_after_sampling):
                                        self._update_run_progress(
                                            stage="ood_loop",
                                            status="queued_for_parallel_render",
                                            ood_index=int(i + 1),
                                            total_ood=int(self.num_objects),
                                        )
                                    else:
                                        self._update_run_progress(
                                            stage="ood_loop",
                                            status="completed",
                                            ood_index=int(i + 1),
                                            total_ood=int(self.num_objects),
                                        )
                                else:
                                    print(f"\033[33mSkipping {lvis}.\033[0m")
                                    times_failed += 1
                                    print(
                                        f"\033[31mNo valid locations/render visibility found.\033[0m\n"
                                    )
                                    if times_failed == self.num_objects:
                                        print(
                                            "\033[31mSkipping this data segment.\033[0m"
                                        )
                                        skip_rest = True
                                        break
                                    continue
                            except KeyboardInterrupt:
                                skip_rest = True
                                break
                        else:
                            print(f"\033[33mSkipping {lvis}.\033[0m")
                            times_failed += 1
                            print(f"\033[31mNo locations found.\033[0m\n")
                            if times_failed == self.num_objects:
                                print(
                                    "\033[31mSkipping this data segment.\033[0m"
                                )
                                skip_rest = True
                                break
                            else:
                                continue
                    except tf.errors.InvalidArgumentError as e:
                        print("Caught Tensorflow Exception:", e)
                        skip_rest = True
                    except KeyboardInterrupt:
                        skip_rest = True
                        break

                if (not skip_rest and blender_parallel_enabled
                        and len(parallel_render_tasks) > 0):
                    run_stage(
                        "blender_render_parallel_batch",
                        self._run_parallel_blender_renders,
                        parallel_render_tasks,
                    )
                    for ood_id in sorted(parallel_validation_context.keys()):
                        ctx = parallel_validation_context[ood_id]
                        render_ok = self._post_render_validate_with_resample(
                            ood_id=int(ood_id),
                            base_scale=float(ctx.get("scale", 1.0)),
                            render_mode=render_mode,
                            eval=eval,
                            dist=dist,
                            render_sensor_mode=render_sensor_mode,
                            size_multiplier=size_multiplier,
                            loaded_ood_details_present=bool(
                                ctx.get("loaded_ood_details_present", False)),
                        )
                        if render_ok:
                            self._update_run_progress(
                                stage="ood_loop",
                                status="completed",
                                ood_index=int(ood_id),
                                total_ood=int(self.num_objects),
                                mode="parallel_render",
                            )
                        else:
                            lvis = str(ctx.get("lvis", "unknown"))
                            print(f"\033[33mSkipping {lvis}.\033[0m")
                            times_failed += 1
                            print(
                                f"\033[31mNo valid locations/render visibility found.\033[0m\n"
                            )
                            if times_failed == self.num_objects:
                                print(
                                    "\033[31mSkipping this data segment.\033[0m"
                                )
                                skip_rest = True
                                break

                if skip_rest == False:
                    run_stage(
                        "export_sampling_spread_debug",
                        self._export_sampling_selected_spread,
                        eval=eval,
                        finalize=True,
                    )
                    if self.stop_after_sampling:
                        print(
                            "[GO STAGE] SAMPLE-ONLY mode: saving ood_details and exiting "
                            "before Blender/merge stages.",
                            flush=True,
                        )
                        if render_mode == 'multi':
                            run_stage("save_ood_details",
                                      self.save_ood_details,
                                      eval=eval)
                        self._update_run_progress(
                            stage="go",
                            status="completed",
                            mode="sampling_only",
                        )
                        return True
                    print('entering save_ood_details')
                    if render_mode == 'multi':
                        run_stage("save_ood_details",
                                  self.save_ood_details,
                                  eval=eval)

                    if not self.enable_raw_image_export:
                        print(
                            "[GO STAGE] save_raw_images will populate in-memory frames only "
                            "(enable_raw_image_export=False).",
                            flush=True,
                        )
                    run_stage("save_raw_images", self.save_raw_images)

                    if (lidar_pipeline_requested and
                        (sensor == 'lidar' or self.enable_projection_debug
                         or self.enable_intensity_debug)):
                        run_stage("merge_ood_into_image",
                                  self.merge_ood_into_image,
                                  sensor,
                                  eval=eval)
                    else:
                        print(
                            "[GO STAGE] SKIP merge_ood_into_image pre-pass "
                            "(improved merge will generate camera outputs).",
                            flush=True,
                        )
                    if lidar_pipeline_requested:
                        run_stage("get_range_image_top_pose",
                                  self.get_range_image_top_pose)
                        check_pc_path = f'{self.base_dir}/{self.file_name}/Point Cloud'
                        check_render_path = f'{self.base_dir}/{self.file_name}/Renders/ood_{self.num_objects}/lidar'

                        run_stage(
                            "visualize_ood_locations_per_frame_in_global_frame",
                            self.
                            visualize_ood_locations_per_frame_in_global_frame)
                        run_stage("get_range_image_indices_of_point_cloud_post_raycast",
                                  self.get_range_image_indices_of_point_cloud)
                        run_stage("combine_point_clouds",
                                  self.combine_point_clouds)
                        run_stage("ood_point_cloud_to_range_image",
                                  self.ood_point_cloud_to_range_image)

                        run_stage(
                            "get_point_cloud_from_range_image_via_indices",
                            self.get_point_cloud_from_range_image_via_indices,
                            eval=eval,
                            render_mode=render_mode,
                            master_save=master_save)
                        if self.save_point_cloud_comparison:
                            run_stage("compare_point_cloud",
                                      self.compare_point_cloud)
                        else:
                            print(
                                "[GO STAGE] SKIP compare_point_cloud "
                                "(save_point_cloud_comparison=False).",
                                flush=True,
                            )
                        if render_mode == 'multi':
                            run_stage("get_ood_bbox", self.get_ood_bbox)

                            run_stage("save_framewise_ood_details",
                                      self.save_framewise_ood_details)
                    else:
                        print(
                            "[GO STAGE] SKIP lidar raw insertion pipeline "
                            "(enable_lidar_insertion=False).",
                            flush=True,
                        )

                    if camera_pipeline_requested:
                        print(
                            f"\033[32m[DEBUG] ✓ Entering camera processing block "
                            f"(sensor == '{sensor}', lidar_only_resume={self.lidar_only_resume})\033[0m"
                        )
                        if self.enable_projection_debug:
                            run_stage("project_points_to_image_plane",
                                      self.project_points_to_image_plane,
                                      eval=eval)
                        else:
                            print(
                                "[GO STAGE] SKIP project_points_to_image_plane "
                                "(enable_projection_debug=False).",
                                flush=True,
                            )

                        if self.enable_intensity_debug:
                            run_stage("get_intensity", self.get_intensity)
                        else:
                            print(
                                "[GO STAGE] SKIP get_intensity "
                                "(enable_intensity_debug=False).",
                                flush=True,
                            )
                        run_stage("find_occlusion_map",
                                  self.find_occlusion_map,
                                  eval=eval)
                        run_stage("merge_ood_into_image_improved",
                                  self.merge_ood_into_image_improved,
                                  sensor,
                                  eval=eval,
                                  render_mode=render_mode)
                        if render_mode == 'multi' and self.enable_pct:
                            if isinstance(self.raw_images.get('camera'), dict):
                                print(
                                    "\033[33mSkipping global color consistency for multi-camera folder layout.\033[0m"
                                )
                            else:
                                run_stage("get_consistent_colors",
                                          self.get_consistent_colors)
                    else:
                        print(
                            f"\033[33m[DEBUG] ✗ Skipping camera processing "
                            f"(sensor == '{sensor}', lidar_only_resume={self.lidar_only_resume})\033[0m"
                        )
            stage_timing_path = os.path.join(
                self.base_dir,
                self.file_name,
                "stage_timing_summary.json",
            )
            try:
                os.makedirs(os.path.dirname(stage_timing_path), exist_ok=True)
                timing_payload = {
                    "segment": self.file_name,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "stages": stage_timing_summary,
                }
                with open(stage_timing_path, "w") as timing_file:
                    json.dump(timing_payload, timing_file, indent=2)
                print(f"[GO STAGE] Wrote timing summary: {stage_timing_path}",
                      flush=True)
            except Exception as timing_exc:
                print(
                    f"[GO STAGE] WARN could not write timing summary: {timing_exc}",
                    flush=True)

            self._update_run_progress(
                stage="go",
                status="completed",
                stage_timing_path=stage_timing_path,
            )
            return True
        except Exception as e:
            print(f"ERROR in go(): {e}")
            import traceback
            traceback.print_exc()
            self._update_run_progress(stage="go",
                                      status="failed",
                                      error=str(e))
            return False
        except KeyboardInterrupt:
            print("KeyboardInterrupt in go()")
            self._update_run_progress(stage="go", status="cancelled")
            return False
        finally:
            self._clear_parsed_range_image_cache()


def get_git_commit_hash():
    try:
        repo_root = Path(__file__).resolve().parent
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=repo_root,
                                       text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def write_run_manifest(manifest_path, payload):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as manifest_file:
        json.dump(payload, manifest_file, indent=2)


CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _as_pos_int(value, default=0):
    try:
        out = int(value)
        return out if out > 0 else int(default)
    except Exception:
        return int(default)


def _resolve_parallel_jobs(runtime_config):
    cfg_jobs = _as_pos_int(runtime_config.get("cpu_parallel_jobs", 0),
                           default=0)
    env_jobs = _as_pos_int(os.environ.get("WAYPOP_PARALLEL_JOBS", 0),
                           default=0)
    worker_jobs = _as_pos_int(runtime_config.get("_effective_worker_count", 0),
                              default=0)
    return max(1, cfg_jobs, env_jobs, worker_jobs)


def _resolve_thread_cap(runtime_config):
    if not bool(runtime_config.get("cpu_oversubscription_guard", True)):
        return 0
    manual_cap = _as_pos_int(runtime_config.get("cpu_threads_per_worker", 0),
                             default=0)
    if manual_cap > 0:
        return manual_cap
    parallel_jobs = _resolve_parallel_jobs(runtime_config)

    if parallel_jobs <= 1:
        return 0
    cpu_count = max(1, int(os.cpu_count() or 1))
    return max(1, cpu_count // parallel_jobs)


def _apply_cpu_oversubscription_guard(runtime_config):
    cap = _resolve_thread_cap(runtime_config)
    if cap <= 0:
        print(
            "[CPU] Oversubscription guard: no thread cap applied "
            f"(parallel_jobs={_resolve_parallel_jobs(runtime_config)}, guard={runtime_config.get('cpu_oversubscription_guard', True)})."
        )
        return 0

    respect_existing = bool(
        runtime_config.get("cpu_respect_preexisting_env", True))
    env_updates = []
    for var_name in CPU_THREAD_ENV_VARS:
        existing = os.environ.get(var_name, "").strip()
        if respect_existing and existing:
            continue
        os.environ[var_name] = str(cap)
        env_updates.append(var_name)

    cv2_threads = _as_pos_int(runtime_config.get("cpu_cv2_threads", 0),
                              default=0)
    if cv2_threads <= 0:
        cv2_threads = cap
    try:
        cv2.setNumThreads(int(cv2_threads))
    except Exception as e:
        print(f"[CPU] cv2.setNumThreads skipped: {e}")

    tf_intra = _as_pos_int(runtime_config.get("cpu_tf_intra_threads", 0),
                           default=0)
    if tf_intra <= 0:
        tf_intra = cap
    tf_inter = _as_pos_int(runtime_config.get("cpu_tf_inter_threads", 0),
                           default=0)
    if tf_inter <= 0:
        tf_inter = max(1, min(4, cap))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(tf_intra))
        tf.config.threading.set_inter_op_parallelism_threads(int(tf_inter))
    except Exception as e:
        print(f"[CPU] TensorFlow thread config skipped: {e}")

    torch_threads = _as_pos_int(runtime_config.get("cpu_torch_threads", 0),
                                default=0)
    if torch_threads <= 0:
        torch_threads = cap
    lazy_torch = torch
    if lazy_torch is None:
        print(
            "[CPU] torch thread caps skipped: torch not imported in this runtime path."
        )
    else:
        try:
            lazy_torch.set_num_threads(int(torch_threads))
        except Exception as e:
            print(f"[CPU] torch.set_num_threads skipped: {e}")
        try:

            if hasattr(lazy_torch, "set_num_interop_threads"):
                lazy_torch.set_num_interop_threads(
                    max(1, min(4, int(torch_threads))))
        except Exception:
            pass

    print(
        "[CPU] Oversubscription guard applied: "
        f"parallel_jobs={_resolve_parallel_jobs(runtime_config)} cap={cap} "
        f"env_vars={','.join(env_updates) if env_updates else 'none(respected existing)'} "
        f"tf_intra={tf_intra} tf_inter={tf_inter} cv2={cv2_threads} torch={torch_threads}"
    )
    return int(cap)


def _configure_tensorflow_runtime(runtime_config):
    runtime_config = dict(runtime_config or {})
    tf_use_cpu_only = bool(runtime_config.get("tf_use_cpu_only", True))
    tf_disable_jit = bool(runtime_config.get("tf_disable_jit", True))

    try:
        tf.config.set_soft_device_placement(True)
    except Exception as e:
        print(f"[TF] soft device placement config skipped: {e}")

    if tf_disable_jit:
        try:
            tf.config.optimizer.set_jit(False)
            print("[TF] XLA JIT disabled for sampler worker.")
        except Exception as e:
            print(f"[TF] XLA JIT disable skipped: {e}")

    if tf_use_cpu_only:
        try:
            visible_gpus_before = tf.config.list_physical_devices('GPU')
            tf.config.set_visible_devices([], 'GPU')
            visible_gpus_after = tf.config.list_physical_devices('GPU')
            print("[TF] CPU-only mode enabled for sampler worker: "
                  f"visible_gpus_before={len(visible_gpus_before)} "
                  f"visible_gpus_after={len(visible_gpus_after)}")
        except RuntimeError as e:
            print(
                f"[TF] CPU-only mode requested too late to reconfigure devices: {e}"
            )
        except Exception as e:
            print(f"[TF] CPU-only mode setup skipped: {e}")
        return

    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(
            f"[TF] GPU mode enabled for sampler worker with {len(gpus)} visible GPU(s)."
        )
    except Exception as e:
        print(f"[TF] GPU runtime setup skipped: {e}")


def run_sampler(task_info):
    file, is_training, gpu_id, proj, file_dir, sensor_type, eval, render_mode, dist, object_path, size_multiplier, retry, save_path, image_path, wandb_logging, master_save, runtime_config = task_info

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    runtime_config = dict(runtime_config or {})
    _configure_tensorflow_runtime(runtime_config)

    ResourceLogger.set_gpu_id(gpu_id)
    ResourceLogger.wandb_enabled = wandb_logging

    train_valid = "Train" if is_training else "Valid"
    file_name = file.replace(".tfrecord", "")

    if image_path:
        output_dir = image_path
    else:
        output_dir = f"Pipeline_2_{train_valid}"

    check_path = f"{output_dir}/{file_name}/merged_directly/{sensor_type}"
    if sensor_type == "both" or sensor_type == "lidar":
        check_path = f"{output_dir}/{file_name}"
    elif sensor_type == "camera":
        check_path = f"{output_dir}/{file_name}/merged_directly/{sensor_type}"

    print(f"Processing {file_name} on GPU {gpu_id}...", flush=True)

    try:
        effective_cpu_thread_cap = _apply_cpu_oversubscription_guard(
            runtime_config)
        runtime_config["_effective_cpu_thread_cap"] = int(
            effective_cpu_thread_cap)

        if save_path:
            base_dir = save_path
        else:
            base_dir = 'Transfer'

        print(f"[DEBUG] Creating Sampler object...", flush=True)
        sampler = Sampler(
            file_name,
            base_dir=base_dir,
            image_path=image_path,
            runtime_config=runtime_config,
        )
        print(f"[DEBUG] Sampler created successfully", flush=True)

        if render_mode == 'single':
            sampler.num_frames = 1
            sampler.num_objects = 1

            if object_path:
                try:
                    object_info = json.loads(object_path)
                    path = object_info.get('path')
                    lvis = object_info.get('lvis')
                    file_identifier = object_info.get('fileIdentifier')
                    scale = float(object_info.get('scale', 1.0))

                    if path:
                        sampler.models = [(file_identifier, lvis, path, scale)]
                        print(f"Using custom object: {lvis} from {path}")
                    else:
                        print("Warning: 'path' not found in object_path JSON")
                except json.JSONDecodeError as e:
                    print(f"Error parsing object_path JSON: {e}")
        else:
            sampler.num_frames = max(
                1, int(runtime_config.get("num_frames", sampler.num_frames)))

        seed = int(hashlib.sha256(file_name.encode()).hexdigest(), 16) % (2**
                                                                          32)
        sampler.seed = np.random.default_rng(seed)
        sampler.train = is_training
        print(f"[DEBUG] Sampler configured (seed={seed}, train={is_training})",
              flush=True)

        manifest_payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "file_name": file_name,
            "split": train_valid.lower(),
            "seed": seed,
            "git_commit": get_git_commit_hash(),
            "run_args": {
                "sensor_type": sensor_type,
                "eval": eval,
                "render_mode": render_mode,
                "dist": dist,
                "object_path": object_path,
                "size_multiplier": size_multiplier,
                "retry": retry,
                "save_path": save_path,
                "image_path": image_path,
                "wandb_logging": wandb_logging,
                "master_save": master_save,
                "gpu_id": gpu_id,
            },
            "runtime_config": runtime_config,
        }
        manifest_path = f"{base_dir}/{file_name}/run_manifest.json"
        write_run_manifest(manifest_path, manifest_payload)
        print(f"[DEBUG] Wrote run manifest: {manifest_path}", flush=True)

        if wandb_logging:
            if wandb is None:
                raise RuntimeError(
                    "wandb_logging=True requires the wandb package to be installed."
                )
            print(f"[DEBUG] Initializing wandb...", flush=True)
            wandb.init(project=proj,
                       name=f"{file_name}-{train_valid}-gpu{gpu_id}",
                       reinit=True)
            print(f"[DEBUG] wandb initialized", flush=True)

        check_dir = f'{sampler.base_dir}/{sampler.file_name}/Reconstructed_Point_Cloud_via_Indices'
        print(f"[DEBUG] Starting sampler.go()...", flush=True)
        print(
            f"[DEBUG] Parameters: file_dir={file_dir}, sensor_type={sensor_type}, eval={eval}, render_mode={render_mode}, dist={dist}",
            flush=True)
        success = sampler.go(file_dir, sensor_type, eval, render_mode, dist,
                             size_multiplier, retry, save_path, image_path,
                             master_save)
        print(f"[DEBUG] sampler.go() completed with success={success}",
              flush=True)
        if wandb_logging:
            wandb.finish()

        if success:
            return f"Success: {file_name}"
        else:
            return f"Failed: {file_name}"
    except Exception as e:
        print(f"!!! ERROR processing {file_name} on GPU {gpu_id}: {e}",
              file=sys.stderr)
        import traceback
        traceback.print_exc()
        return f"Failed: {file_name}"


def get_free_gpus_in_main_process(num_gpus_to_select):
    gpus_info = ResourceLogger.get_gpu_utilization()
    print('All GPUs info:', gpus_info)

    available_gpus = {}
    for gpu_id, info in gpus_info.items():

        if 'memory_total' in info and 'memory' in info:
            free_memory = info['memory_total'] - info['memory']
            if free_memory >= 5:
                available_gpus[gpu_id] = info

    sorted_gpus = sorted(
        available_gpus.items(),
        key=lambda item: item[1]['memory'] + item[1]['compute'])

    free_gpus = [gpu_id for gpu_id, _ in sorted_gpus[:num_gpus_to_select]]

    return free_gpus


def prune_release_outputs(segment_root):
    root = Path(segment_root).resolve()
    keep_top_level_dirs = {
        "Placement", "Renders", "Raw_Images", "precompute_cache"
    }
    keep_top_level_files = {
        "ood_details.json", "run_manifest.json", "run_progress.json"
    }
    drop_dirs = {
        "Actual_Top_Lidar_Point_Cloud",
        "Comparison",
        "Intensity",
        "Occlusion_Masks",
        "OOD_Intensity",
        "OOD_Merged_Intensity",
        "OOD_Merged_Panoptic_Labels",
        "OOD_Merged_Range_Image",
        "Point_Cloud",
        "Point_Cloud_Metadata",
        "Points_Projected_in_Image",
        "Raw_Images_debug",
        "Reconstructed_Point_Cloud_via_Indices",
        "Sam_Masks",
        "Scene_3D",
        "Segmentation_Masks",
        "Shadow_Debug",
        "Single_Rendered_Images_CNN_pct",
        "Single_Rendered_Images_ViT_pct",
        "Support_Debug",
        "Ultralytics_Segmented_Masks",
        "camera_data",
        "lidar_data",
        "lidar_label_metadata",
    }

    if not root.is_dir():
        return

    for child in root.iterdir():
        if child.is_dir() and child.name == "Placement":
            replay_inputs = child / "replay_inputs"
            if replay_inputs.is_dir():
                for path in replay_inputs.iterdir():
                    if path.is_file(
                    ) and path.name == "sampled_positions_only.json":
                        continue
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                for nested in child.iterdir():
                    if nested.name == "replay_inputs":
                        continue
                    if nested.is_dir():
                        shutil.rmtree(nested)
                    else:
                        nested.unlink()
            else:
                shutil.rmtree(child)
            continue
        if child.is_dir() and child.name in drop_dirs:
            shutil.rmtree(child)
            continue
        if child.is_dir() and child.name not in keep_top_level_dirs:
            shutil.rmtree(child)
            continue
        if child.is_file() and child.name not in keep_top_level_files:
            child.unlink()


def main():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=
        "Render a WayPOP reconstruction example from explicit source inputs."
    )
    parser.add_argument(
        '--segment', required=True,
        help='Waymo segment basename, without the .tfrecord suffix.')
    parser.add_argument(
        '--waymo_dir',
        type=str,
        required=True,
        help='Directory containing the user-provided source or mini TFRecord.')
    parser.add_argument(
        '--objects_json',
        type=str,
        required=True,
        help='Object assignment JSON used for this example.')
    parser.add_argument('--placement_json',
                        type=str,
                        required=True,
                        help='Placement candidate table used for this example.')
    parser.add_argument('--save_path',
                        type=str,
                        default=str(repo_root / "run_outputs" /
                                    "outputs"),
                        help='Directory where run artifacts are written.')
    parser.add_argument(
        '--image_path',
        type=str,
        default=str(repo_root / "run_outputs" / "images"),
        help='Directory where final image products are written.')
    parser.add_argument('--blender_logs_dir',
                        type=str,
        default=str(repo_root / "run_outputs" /
                                    "blender_logs"),
                        help='Directory where Blender logs are written.')
    parser.add_argument('--blender_path',
                        type=str,
                        default=os.environ.get("BLENDER_PATH") or shutil.which("blender") or "",
                        help='Path to Blender executable.')
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='GPU id used by the sampler worker.')
    parser.add_argument('--name',
                        type=str,
                        default='validation_example',
                        help='Run name written into manifests.')
    parser.add_argument(
        '--frame_token_list',
        type=str,
        required=True,
        help='Comma-separated 1-based source frame tokens to render.')
    parser.add_argument(
        '--fixed_ood_objects',
        type=int,
        default=12,
        help=
        'Number of OOD objects rendered from the provided object assignment.')
    parser.add_argument(
        '--retry',
        type=str2bool,
        default=False,
        help='Reuse saved placements and intermediate state when available.')
    parser.add_argument(
        '--skip_blender_render',
        type=str2bool,
        default=False,
        help='Reuse existing Blender renders and skip rerendering.')
    parser.add_argument('--camera_occlusion_depth_fill',
                        type=str,
                        choices=['none', 'multiscale'],
                        default='multiscale',
                        help='Camera occlusion depth densification mode.')
    parser.add_argument(
        '--enable_stuff_occlusion_expansion',
        type=str2bool,
        default=True,
        help=
        'Allow non-ground stuff panoptic regions to expand occlusion masks.')
    parser.add_argument(
        '--enable_stuff_occlusion_fallback',
        type=str2bool,
        default=True,
        help='Allow fallback expansion into non-ground stuff regions.')
    parser.add_argument(
        '--enable_thing_occlusion_fallback',
        type=str2bool,
        default=True,
        help='Allow fallback expansion into thing-instance regions.')
    parser.add_argument(
        '--enable_shadow_quality_gate',
        type=str2bool,
        default=False,
        help='Re-sample objects that fail the shadow-quality heuristic.')
    parser.add_argument('--required_render_max_retries',
                        type=int,
                        default=1,
                        help='Maximum rerender attempts per OOD.')
    parser.add_argument(
        '--prune_intermediates', type=str2bool, default=False,
        help='Remove intermediate render files after success. Default: False.')
    args = parser.parse_args()

    args.waymo_dir = resolve_path(args.waymo_dir)
    args.objects_json = resolve_path(args.objects_json)
    args.placement_json = resolve_path(args.placement_json)
    args.save_path = resolve_path(args.save_path)
    args.image_path = resolve_path(args.image_path)
    args.blender_logs_dir = resolve_path(args.blender_logs_dir)
    args.blender_path = resolve_path(args.blender_path) if args.blender_path else ""

    if not re.fullmatch(r"segment-[A-Za-z0-9_]+", args.segment):
        parser.error('--segment must be a Waymo segment basename')
    tfrecord_path = os.path.join(args.waymo_dir,
                                 f"{args.segment}.tfrecord")
    if not os.path.isdir(args.waymo_dir):
        raise FileNotFoundError(f"Waymo directory not found: {args.waymo_dir}")
    if not os.path.isfile(tfrecord_path):
        raise FileNotFoundError(f"Mini TFRecord not found: {tfrecord_path}")
    if not os.path.isfile(args.objects_json):
        raise FileNotFoundError(
            f"Object assignment file not found: {args.objects_json}")
    if not os.path.isfile(args.placement_json):
        raise FileNotFoundError(
            f"Placement file not found: {args.placement_json}")
    if args.fixed_ood_objects < 1:
        parser.error('--fixed_ood_objects must be >= 1')
    if args.required_render_max_retries < 1:
        parser.error('--required_render_max_retries must be >= 1')
    if args.skip_blender_render and not args.retry:
        parser.error('--skip_blender_render requires --retry')
    if not args.skip_blender_render and not os.access(args.blender_path, os.X_OK):
        parser.error('--blender_path must name an executable Blender binary')

    frame_tokens = [
        tok.strip() for tok in str(args.frame_token_list).split(',')
        if tok.strip()
    ]
    if not frame_tokens:
        parser.error(
            '--frame_token_list must contain at least one frame token')

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.image_path, exist_ok=True)
    os.makedirs(args.blender_logs_dir, exist_ok=True)

    runtime_config = {
        'blender_path': args.blender_path,
        'blender_logs_dir': args.blender_logs_dir,
        'blender_require_gpu': False,
        'tf_use_cpu_only': True,
        'tf_disable_jit': True,
        'objects_train_json': args.objects_json,
        'objects_valid_json': args.objects_json,
        'objects_eval_json': args.objects_json,
        'skip_blender_render': args.skip_blender_render,
        'stop_after_sampling': False,
        'request_only': False,
        'lidar_only_resume': False,
        'num_frames': len(frame_tokens),
        'frame_id_start': 1,
        'frame_id_end': len(frame_tokens),
        'frame_token_list': ','.join(frame_tokens),
        'enable_extended_accumulation_window': False,
        'accumulation_window_offset_frames': 0,
        'accumulation_voxel_size': 0.12,
        'enable_sam': False,
        'enable_pct': False,
        'enable_weather_api': False,
        'required_render_max_retries': args.required_render_max_retries,
        'enable_shadow_quality_gate': args.enable_shadow_quality_gate,
        'camera_occlusion_depth_fill': args.camera_occlusion_depth_fill,
        'camera_occlusion_depth_fill_bbox_pad_px': 48,
        'camera_occlusion_depth_fill_radius_px': 96,
        'camera_occlusion_depth_fill_max_depth': 120.0,
        'enable_stuff_occlusion_expansion':
        args.enable_stuff_occlusion_expansion,
        'enable_stuff_occlusion_fallback':
        args.enable_stuff_occlusion_fallback,
        'enable_thing_occlusion_fallback':
        args.enable_thing_occlusion_fallback,
        'enable_precompute_cache': True,
        'force_recompute_precompute_cache': False,
        'enable_raw_image_export': True,
        'reuse_saved_pose_for_render': False,
        'load_saved_ood_details': False,
        'enable_lidar_insertion': False,
        'placement_policy': 'gpt_oneshot',
        'placement_gpt_enabled': False,
        'placement_gpt_response_json': args.placement_json,
        'placement_guard_mode': 'reuse_external_candidates',
        'require_exact_placement': True,
        'placement_bev_export': False,
        'object_count_policy': 'fixed',
        'fixed_ood_objects': args.fixed_ood_objects,
        '_effective_worker_count': 1,
    }

    task = (
        f"{args.segment}.tfrecord",
        False,
        args.gpu_id,
        args.name,
        args.waymo_dir,
        'both',
        False,
        'multi',
        None,
        None,
        1.0,
        args.retry,
        args.save_path,
        args.image_path,
        False,
        False,
        runtime_config,
    )

    result = run_sampler(task)
    print(result)
    if not result.startswith('Success:'):
        raise RuntimeError(result)

    if args.prune_intermediates:
        prune_release_outputs(Path(args.save_path) / args.segment)


if __name__ == "__main__":
    main()
