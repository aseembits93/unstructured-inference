"""Numba JIT-compiled kernels for performance-critical geometry and NMS operations.

When numba is installed, these kernels compile the entire computation to native
machine code, eliminating Python/numpy dispatch overhead and intermediate array
allocations. Falls back transparently to numpy when numba is not available.

Install the optional numba dependency group to enable acceleration:
    uv sync --group numba
"""

from __future__ import annotations

import numpy as np

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Kernel 1: Non-maxima suppression (NMS)
# Used by yolox.py multiclass_nms_class_agnostic
# ---------------------------------------------------------------------------

if NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def _nms_kernel(
        x1: numba.float64[:],  # type: ignore[valid-type]
        y1: numba.float64[:],  # type: ignore[valid-type]
        x2: numba.float64[:],  # type: ignore[valid-type]
        y2: numba.float64[:],  # type: ignore[valid-type]
        areas: numba.float64[:],  # type: ignore[valid-type]
        order: numba.int64[:],  # type: ignore[valid-type]
        nms_thr: float,
    ):
        """NMS inner loop compiled to native code.

        Replaces the while-loop-over-shrinking-array pattern with an alive[] mask,
        which is numba-compatible and avoids repeated array creation.
        """
        n = order.shape[0]
        keep = numba.typed.List.empty_list(numba.int64)
        alive = numba.typed.List.empty_list(numba.boolean)
        for _ in range(n):
            alive.append(True)

        i_outer = 0
        while i_outer < n:
            if not alive[i_outer]:
                i_outer += 1
                continue
            i = order[i_outer]
            keep.append(i)
            for j in range(i_outer + 1, n):
                if not alive[j]:
                    continue
                jj = order[j]
                xx1 = x1[i] if x1[i] > x1[jj] else x1[jj]
                yy1 = y1[i] if y1[i] > y1[jj] else y1[jj]
                xx2 = x2[i] if x2[i] < x2[jj] else x2[jj]
                yy2 = y2[i] if y2[i] < y2[jj] else y2[jj]
                w = xx2 - xx1 + 1.0
                h = yy2 - yy1 + 1.0
                if w <= 0.0 or h <= 0.0:
                    continue
                inter = w * h
                ovr = inter / (areas[i] + areas[jj] - inter)
                if ovr > nms_thr:
                    alive[j] = False
            i_outer += 1
        return keep


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, nms_thr: float) -> list:
    """Numpy fallback for NMS — exact copy of yolox.py::nms."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def nms_numba(boxes: np.ndarray, scores: np.ndarray, nms_thr: float) -> list:
    """NMS with numba acceleration when available.

    Parameters
    ----------
    boxes:
        (N, 4) float array of bounding boxes in [x1, y1, x2, y2] format.
    scores:
        (N,) float array of confidence scores.
    nms_thr:
        IoU threshold above which lower-confidence boxes are suppressed.

    Returns
    -------
    list of int indices of kept boxes (sorted highest-score first).
    """
    # Size guard: typed.List overhead not worth it for very small inputs
    if not NUMBA_AVAILABLE or len(boxes) < 10:
        return _nms_numpy(boxes, scores, nms_thr)

    # Always cast to float64: cache=True compiles per dtype; ONNX output is float32
    x1 = boxes[:, 0].astype(np.float64)
    y1 = boxes[:, 1].astype(np.float64)
    x2 = boxes[:, 2].astype(np.float64)
    y2 = boxes[:, 3].astype(np.float64)
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1].astype(np.int64)
    return list(_nms_kernel(x1, y1, x2, y2, areas, order, float(nms_thr)))


# ---------------------------------------------------------------------------
# Kernel 2: Pairwise rectangle intersection boolean matrix
# Used by elements.py::coords_intersections
# ---------------------------------------------------------------------------

if NUMBA_AVAILABLE:

    @numba.njit(cache=True, parallel=True)
    def _coords_intersections_parallel(
        x1s: numba.float64[:],  # type: ignore[valid-type]
        y1s: numba.float64[:],  # type: ignore[valid-type]
        x2s: numba.float64[:],  # type: ignore[valid-type]
        y2s: numba.float64[:],  # type: ignore[valid-type]
        out: numba.boolean[:, :],  # type: ignore[valid-type]
    ) -> None:
        """Fill out[i,j] = True iff rect i and rect j intersect. Parallelized outer loop."""
        n = x1s.shape[0]
        for i in numba.prange(n):
            for j in range(n):
                out[i, j] = not (
                    x1s[i] > x2s[j]
                    or y1s[i] > y2s[j]
                    or x1s[j] > x2s[i]
                    or y1s[j] > y2s[i]
                )

    @numba.njit(cache=True)
    def _coords_intersections_serial(
        x1s: numba.float64[:],  # type: ignore[valid-type]
        y1s: numba.float64[:],  # type: ignore[valid-type]
        x2s: numba.float64[:],  # type: ignore[valid-type]
        y2s: numba.float64[:],  # type: ignore[valid-type]
        out: numba.boolean[:, :],  # type: ignore[valid-type]
    ) -> None:
        """Fill out[i,j] = True iff rect i and rect j intersect. Serial version for small N."""
        n = x1s.shape[0]
        for i in range(n):
            for j in range(n):
                out[i, j] = not (
                    x1s[i] > x2s[j]
                    or y1s[i] > y2s[j]
                    or x1s[j] > x2s[i]
                    or y1s[j] > y2s[i]
                )


def _coords_intersections_numpy(coords: np.ndarray) -> np.ndarray:
    """Numpy fallback — original broadcasting implementation."""
    x1s, y1s, x2s, y2s = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    return ~(
        (x1s[None] > x2s[..., None])
        | (y1s[None] > y2s[..., None])
        | (x1s[None] > x2s[..., None]).T
        | (y1s[None] > y2s[..., None]).T
    )


def coords_intersections_numba(coords: np.ndarray) -> np.ndarray:
    """Boolean N×N matrix: entry (i,j) is True iff rect i and rect j intersect.

    Numba version eliminates the 4 intermediate N×N boolean arrays created by
    numpy broadcasting, writing directly to the output array.

    Parameters
    ----------
    coords:
        (N, 4) array of [x1, y1, x2, y2] coordinates.

    Returns
    -------
    (N, N) boolean numpy array.
    """
    if not NUMBA_AVAILABLE:
        return _coords_intersections_numpy(coords)

    c = coords.astype(np.float64)
    n = c.shape[0]
    out = np.empty((n, n), dtype=np.bool_)
    # Use parallel kernel for large N (amortizes thread-pool startup cost)
    if n >= 64:
        _coords_intersections_parallel(c[:, 0], c[:, 1], c[:, 2], c[:, 3], out)
    else:
        _coords_intersections_serial(c[:, 0], c[:, 1], c[:, 2], c[:, 3], out)
    return out


# ---------------------------------------------------------------------------
# Kernel 3: Pairwise intersection area matrix
# Used by layoutelement.py::intersection_areas_between_coords
# ---------------------------------------------------------------------------

if NUMBA_AVAILABLE:

    @numba.njit(cache=True, parallel=True)
    def _intersection_areas_kernel(
        x11: numba.float64[:],  # type: ignore[valid-type]
        y11: numba.float64[:],  # type: ignore[valid-type]
        x12: numba.float64[:],  # type: ignore[valid-type]
        y12: numba.float64[:],  # type: ignore[valid-type]
        x21: numba.float64[:],  # type: ignore[valid-type]
        y21: numba.float64[:],  # type: ignore[valid-type]
        x22: numba.float64[:],  # type: ignore[valid-type]
        y22: numba.float64[:],  # type: ignore[valid-type]
        out: numba.float64[:, :],  # type: ignore[valid-type]
    ) -> None:
        """Fill out[i,j] with the intersection area of rect i from coords1 and rect j from coords2."""
        n = x11.shape[0]
        m = x21.shape[0]
        for i in numba.prange(n):
            for j in range(m):
                xa = x11[i] if x11[i] > x21[j] else x21[j]
                ya = y11[i] if y11[i] > y21[j] else y21[j]
                xb = x12[i] if x12[i] < x22[j] else x22[j]
                yb = y12[i] if y12[i] < y22[j] else y22[j]
                dw = xb - xa
                dh = yb - ya
                out[i, j] = (dw if dw > 0.0 else 0.0) * (dh if dh > 0.0 else 0.0)


def _intersection_areas_numpy(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Numpy fallback — original broadcasting implementation."""
    x11, y11, x12, y12 = np.split(coords1, 4, axis=1)
    x21, y21, x22, y22 = np.split(coords2, 4, axis=1)
    xa = np.maximum(x11, np.transpose(x21))
    ya = np.maximum(y11, np.transpose(y21))
    xb = np.minimum(x12, np.transpose(x22))
    yb = np.minimum(y12, np.transpose(y22))
    return np.maximum((xb - xa), 0) * np.maximum((yb - ya), 0)


def intersection_areas_numba(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """N×M matrix of intersection areas between two sets of bounding boxes.

    Numba version writes directly to output, avoiding the 8 np.split +
    4 np.maximum/minimum + 2 np.transpose intermediate arrays.

    Parameters
    ----------
    coords1:
        (N, 4) array of [x1, y1, x2, y2] bounding boxes.
    coords2:
        (M, 4) array of [x1, y1, x2, y2] bounding boxes.

    Returns
    -------
    (N, M) float64 array of intersection areas.
    """
    if not NUMBA_AVAILABLE:
        return _intersection_areas_numpy(coords1, coords2)

    c1 = coords1.astype(np.float64)
    c2 = coords2.astype(np.float64)
    out = np.empty((c1.shape[0], c2.shape[0]), dtype=np.float64)
    _intersection_areas_kernel(
        c1[:, 0], c1[:, 1], c1[:, 2], c1[:, 3],
        c2[:, 0], c2[:, 1], c2[:, 2], c2[:, 3],
        out,
    )
    return out


# ---------------------------------------------------------------------------
# Warm-up: pre-compile all kernels on import
# With cache=True, JIT happens once and is saved to __pycache__.
# Subsequent imports load the compiled binary directly (milliseconds, not seconds).
# ---------------------------------------------------------------------------

def _warmup() -> None:
    """Trigger JIT compilation of all kernels with representative dummy inputs.

    Only pays the compilation cost the first time in a fresh environment.
    cache=True ensures compiled code persists across process restarts.
    """
    if not NUMBA_AVAILABLE:
        return
    dummy_boxes = np.array(
        [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]], dtype=np.float64
    )
    dummy_scores = np.array([0.9, 0.8], dtype=np.float64)
    nms_numba(dummy_boxes, dummy_scores, 0.5)

    dummy_coords = np.array(
        [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 20.0]], dtype=np.float64
    )
    coords_intersections_numba(dummy_coords)

    dummy_c1 = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float64)
    dummy_c2 = np.array([[5.0, 5.0, 15.0, 15.0]], dtype=np.float64)
    intersection_areas_numba(dummy_c1, dummy_c2)


_warmup()
