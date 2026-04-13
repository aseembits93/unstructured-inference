"""Tests for numba_kernels.py.

Covers:
- Output parity: numba path vs numpy fallback for all three kernels
- Fallback path: monkeypatched NUMBA_AVAILABLE=False produces identical outputs
- dtype handling: float32 inputs are handled correctly by wrappers
- Edge cases: empty arrays, single element, all-identical, all-disjoint
"""

import numpy as np
import pytest

import unstructured_inference.numba_kernels as _mod
from unstructured_inference.numba_kernels import (
    _coords_intersections_numpy,
    _intersection_areas_numpy,
    _nms_numpy,
    coords_intersections_numba,
    intersection_areas_numba,
    nms_numba,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_boxes(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 900, n)
    y1 = rng.uniform(0, 900, n)
    x2 = x1 + rng.uniform(10, 100, n)
    y2 = y1 + rng.uniform(10, 100, n)
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    scores = rng.uniform(0, 1, n).astype(np.float32)
    return boxes, scores


def _make_random_coords(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 500, n)
    y1 = rng.uniform(0, 500, n)
    x2 = x1 + rng.uniform(5, 100, n)
    y2 = y1 + rng.uniform(5, 100, n)
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# Kernel 1: NMS — output parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [10, 50, 100, 500, 1000])
def test_nms_parity(n):
    """numba NMS must return the same kept indices as the numpy fallback."""
    boxes, scores = _make_random_boxes(n)
    nms_thr = 0.45

    numpy_keep = sorted(_nms_numpy(boxes, scores, nms_thr))
    numba_keep = sorted(nms_numba(boxes, scores, nms_thr))

    assert numpy_keep == numba_keep, (
        f"NMS mismatch for N={n}: numpy kept {len(numpy_keep)}, numba kept {len(numba_keep)}"
    )


def test_nms_empty():
    boxes = np.empty((0, 4), dtype=np.float32)
    scores = np.empty(0, dtype=np.float32)
    assert nms_numba(boxes, scores, 0.5) == []


def test_nms_single():
    boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    assert nms_numba(boxes, scores, 0.5) == [0]


def test_nms_all_identical():
    """All identical boxes: only one should be kept."""
    boxes = np.tile([0.0, 0.0, 10.0, 10.0], (5, 1)).astype(np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    keep = nms_numba(boxes, scores, 0.5)
    assert len(keep) == 1


def test_nms_all_disjoint():
    """Completely disjoint boxes: all should be kept."""
    boxes = np.array(
        [[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms_numba(boxes, scores, 0.5)
    assert len(keep) == 3


def test_nms_float32_input_no_error():
    """Float32 inputs must not raise a numba type-mismatch error."""
    boxes, scores = _make_random_boxes(20)
    assert boxes.dtype == np.float32
    # Should not raise
    keep = nms_numba(boxes, scores, 0.45)
    assert isinstance(keep, list)


# ---------------------------------------------------------------------------
# Kernel 1: NMS — fallback path
# ---------------------------------------------------------------------------

def test_nms_fallback(monkeypatch):
    """With NUMBA_AVAILABLE=False the numpy fallback produces identical results."""
    monkeypatch.setattr(_mod, "NUMBA_AVAILABLE", False)
    boxes, scores = _make_random_boxes(50)
    numpy_keep = sorted(_nms_numpy(boxes, scores, 0.45))
    fallback_keep = sorted(nms_numba(boxes, scores, 0.45))
    assert numpy_keep == fallback_keep


# ---------------------------------------------------------------------------
# Kernel 2: coords_intersections — output parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 5, 30, 70, 200])
def test_coords_intersections_parity(n):
    coords = _make_random_coords(n)
    expected = _coords_intersections_numpy(coords)
    result = coords_intersections_numba(coords)
    np.testing.assert_array_equal(result, expected, err_msg=f"Mismatch for N={n}")


def test_coords_intersections_empty():
    coords = np.empty((0, 4), dtype=np.float64)
    result = coords_intersections_numba(coords)
    assert result.shape == (0, 0)


def test_coords_intersections_single():
    coords = np.array([[0.0, 0.0, 10.0, 10.0]])
    result = coords_intersections_numba(coords)
    assert result.shape == (1, 1)
    assert result[0, 0]  # a rect always intersects itself


def test_coords_intersections_diagonal():
    """Self-intersection diagonal must always be True."""
    coords = _make_random_coords(20)
    result = coords_intersections_numba(coords)
    assert result.diagonal().all()


def test_coords_intersections_symmetry():
    """Intersection matrix must be symmetric."""
    coords = _make_random_coords(30)
    result = coords_intersections_numba(coords)
    np.testing.assert_array_equal(result, result.T)


def test_coords_intersections_parallel_vs_serial():
    """Both parallel (N>=64) and serial (N<64) variants must agree on a boundary case."""
    if not _mod.NUMBA_AVAILABLE:
        pytest.skip("numba not installed")
    coords = _make_random_coords(64)
    # Force serial path
    n = coords.shape[0]
    out_serial = np.empty((n, n), dtype=np.bool_)
    _mod._coords_intersections_serial(
        coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3], out_serial
    )
    # Force parallel path
    out_parallel = np.empty((n, n), dtype=np.bool_)
    _mod._coords_intersections_parallel(
        coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3], out_parallel
    )
    np.testing.assert_array_equal(out_serial, out_parallel)


def test_coords_intersections_fallback(monkeypatch):
    monkeypatch.setattr(_mod, "NUMBA_AVAILABLE", False)
    coords = _make_random_coords(20)
    expected = _coords_intersections_numpy(coords)
    result = coords_intersections_numba(coords)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Kernel 3: intersection_areas — output parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,m", [(1, 1), (10, 10), (50, 30), (100, 50)])
def test_intersection_areas_parity(n, m):
    c1 = _make_random_coords(n, seed=1)
    c2 = _make_random_coords(m, seed=2)
    expected = _intersection_areas_numpy(c1, c2)
    result = intersection_areas_numba(c1, c2)
    np.testing.assert_allclose(result, expected, rtol=1e-10, err_msg=f"Mismatch for N={n},M={m}")


def test_intersection_areas_empty_n():
    c1 = np.empty((0, 4), dtype=np.float64)
    c2 = _make_random_coords(5)
    result = intersection_areas_numba(c1, c2)
    assert result.shape == (0, 5)


def test_intersection_areas_empty_m():
    c1 = _make_random_coords(5)
    c2 = np.empty((0, 4), dtype=np.float64)
    result = intersection_areas_numba(c1, c2)
    assert result.shape == (5, 0)


def test_intersection_areas_disjoint():
    c1 = np.array([[0.0, 0.0, 10.0, 10.0]])
    c2 = np.array([[100.0, 100.0, 200.0, 200.0]])
    result = intersection_areas_numba(c1, c2)
    assert result[0, 0] == 0.0


def test_intersection_areas_contained():
    """Smaller box fully inside larger: area should equal area of smaller box."""
    outer = np.array([[0.0, 0.0, 100.0, 100.0]])
    inner = np.array([[10.0, 10.0, 50.0, 50.0]])
    result = intersection_areas_numba(outer, inner)
    assert result[0, 0] == pytest.approx(40.0 * 40.0)


def test_intersection_areas_float32_input():
    c1 = _make_random_coords(10).astype(np.float32)
    c2 = _make_random_coords(10).astype(np.float32)
    c1_f64 = c1.astype(np.float64)
    c2_f64 = c2.astype(np.float64)
    result = intersection_areas_numba(c1, c2)
    expected = _intersection_areas_numpy(c1_f64, c2_f64)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_intersection_areas_fallback(monkeypatch):
    monkeypatch.setattr(_mod, "NUMBA_AVAILABLE", False)
    c1 = _make_random_coords(15, seed=3)
    c2 = _make_random_coords(10, seed=4)
    expected = _intersection_areas_numpy(c1, c2)
    result = intersection_areas_numba(c1, c2)
    np.testing.assert_allclose(result, expected, rtol=1e-10)
