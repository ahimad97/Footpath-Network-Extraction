"""Node proposals from predicted class-probability masks."""

import numpy as np
from scipy.spatial import KDTree
from skimage.morphology import binary_dilation, disk, remove_small_objects, skeletonize


def nms_points(points, scores, radius=30, return_indices=False):
    """Non-maximum suppression on an unordered point cloud.

    Keeps the highest-scoring point in each ``radius``-pixel ball and suppresses
    its neighbours.

    Parameters
    ----------
    points : (N, 2) array of (x, y) pixel coordinates.
    scores : (N,) confidence per point (higher is better).
    radius : minimum spacing in pixels between kept points.
    return_indices : also return the indices of kept points in ``points``.
    """
    if len(points) == 0:
        if return_indices:
            return np.empty((0, 2), points.dtype), np.empty((0,), np.int64)
        return np.empty((0, 2), points.dtype)

    order = np.argsort(scores)[::-1]
    pts_sorted = points[order]
    kept = np.ones(len(points), dtype=bool)
    tree = KDTree(pts_sorted)
    for i, p in enumerate(pts_sorted):
        if not kept[i]:
            continue
        for j in tree.query_ball_point(p, r=radius):
            if j != i:
                kept[j] = False

    if return_indices:
        return pts_sorted[kept], order[kept]
    return pts_sorted[kept]


def get_points_and_scores_from_mask(mask, threshold):
    """Every pixel above ``threshold`` as an (x, y) candidate, scored by its value."""
    rcs = np.column_stack(np.where(mask > threshold))
    return rcs[:, ::-1], mask[mask > threshold]


def get_skeleton_points_and_scores(mask, threshold, min_blob_px=0, thickness=0):
    """Candidates on the centreline (skeleton) of the thresholded mask.

    Drop-in alternative to :func:`get_points_and_scores_from_mask`:

    1. binarise ``mask`` at ``threshold``;
    2. optionally drop connected blobs smaller than ``min_blob_px`` pixels;
    3. skeletonise to a 1-px medial axis;
    4. optionally widen the skeleton to a ribbon of half-width ``thickness``
       (clipped to the mask);
    5. return skeleton pixels as (x, y) with the mask value as score.

    Parameters
    ----------
    mask : (H, W) uint8 class probability scaled to 0..255.
    threshold : same convention as ``get_points_and_scores_from_mask``.
    min_blob_px : minimum connected-component size kept (0 keeps all).
    thickness : skeleton half-width in pixels (0 keeps the 1-px medial axis).
    """
    empty = np.empty((0, 2), np.float32), np.empty((0,), np.float32)
    binary = mask > threshold
    if not binary.any():
        return empty

    if min_blob_px and int(min_blob_px) > 0:
        binary = remove_small_objects(binary, min_size=int(min_blob_px))
        if not binary.any():
            return empty

    skel = skeletonize(binary)
    if thickness and int(thickness) > 0:
        skel = binary_dilation(skel, footprint=disk(int(thickness))) & binary
    rcs = np.column_stack(np.where(skel))
    if rcs.size == 0:
        return empty

    xys = rcs[:, ::-1].astype(np.float32)
    scores = mask[rcs[:, 0], rcs[:, 1]].astype(np.float32)
    return xys, scores
