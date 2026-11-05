"""Small, dependency-free checks for saved object placements."""

from __future__ import annotations

import math


def assigned_point(rows, ood_id: int, *, required: bool = False):
    """Look up an object's saved point by object order, not sparse slot number."""
    index = int(ood_id) - 1
    row = rows[index] if 0 <= index < len(rows) else {}
    raw = row.get("placement_point_xyz")
    if raw is None:
        if required:
            raise ValueError(f"OOD {ood_id} has no saved placement_point_xyz")
        return None
    try:
        point = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OOD {ood_id} has an invalid placement_point_xyz") from exc
    if len(point) != 3 or not all(math.isfinite(value) for value in point):
        raise ValueError(f"OOD {ood_id} has an invalid placement_point_xyz")
    return point


def checked_saved_object_ids(details, rows):
    """Return saved IDs in assignment order after checking asset identity."""
    expected = {str(ood_id) for ood_id in range(1, len(rows) + 1)}
    actual = {str(key) for key in details if str(key) != "0"}
    if actual != expected:
        raise ValueError(
            f"Saved object IDs differ from the input assignment: "
            f"expected={sorted(expected, key=int)}, actual={sorted(actual)}"
        )
    for ood_id, row in enumerate(rows, 1):
        saved = details.get(str(ood_id))
        if not isinstance(saved, dict) or saved.get("file_identifier") != row.get(
            "fileIdentifier"
        ):
            raise ValueError(f"Saved asset differs from the input assignment for OOD {ood_id}")
    return range(1, len(rows) + 1)


def check_sampled_position(position, expected, ood_id: int, tolerance_m: float = 1e-5):
    """Reject saved retry state or a new sample at a different world point."""
    try:
        if len(position) == 4 and all(len(row) == 4 for row in position):
            actual = tuple(float(position[axis][3]) for axis in range(3))
        else:
            actual = tuple(float(value) for value in position)
    except (TypeError, ValueError, IndexError) as exc:
        raise RuntimeError(f"OOD {ood_id} has no valid sampled position") from exc
    if len(actual) != 3 or not all(math.isfinite(value) for value in actual):
        raise RuntimeError(f"OOD {ood_id} has no valid sampled position")
    error_m = math.dist(actual, expected)
    if error_m > tolerance_m:
        raise RuntimeError(
            f"OOD {ood_id} sampled position differs from its saved assignment "
            f"by {error_m:.6f} m: actual={actual}, expected={expected}. "
            "Start a fresh run without --retry if saved outputs came from an older renderer."
        )
