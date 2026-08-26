"""Training data: aerial tiles, class masks, patch grids and graph labels.

A dataset root contains::

    images/            RGB tiles
    grayscale_masks/   single-channel class masks (same file names as images)
    gt_graph*/         pickled Sat2Graph adjacency dicts, one per tile
    tree_locations/    optional pickled tree coordinates per tile

The file-name patterns depend on ``dataset_location`` (see ``SatMapDataset``).
"""

import math
import os
import pickle
import random
import re
from pathlib import Path

import cv2
import numpy as np
import rtree
import scipy.spatial
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torch_geometric.data import Batch, Data

from . import graph_utils
from .graph_extraction import nms_points
from .utils import cfg_get

# Margin of the patch grid used to build training/validation patches. The
# inference grid uses the config's SAMPLE_MARGIN instead.
TRAIN_SAMPLE_MARGIN = 32

# TRAIN_GRAPH_NODE_SOURCE values that sample training nodes from the class mask
# (as at inference) instead of from the reference graph vertices.
MASK_NODE_SOURCES = {"mask", "seg_mask", "inference", "inference_mask"}

_IMAGENET_NORMALIZE = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def read_rgb_img(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Failed to read RGB image with cv2.imread: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_grayscale_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        return mask
    # Fall back to PIL for PNGs that OpenCV cannot decode.
    try:
        with Image.open(path) as img:
            return np.array(img if img.mode == "L" else img.convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Failed to read grayscale mask: {path}") from exc


def extract_number(filename):
    match = re.search(r'(\d+)\.p', filename)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------- graphs
def _canonical_undirected_pairs(edge_pairs, edge_labels=None):
    """Return one (min, max) row per unordered edge, dropping self-loops.

    Duplicate candidates are merged; their labels are combined with logical OR.
    """
    pairs = np.asarray(edge_pairs, dtype=np.int64).reshape(-1, 2)
    keep = pairs[:, 0] != pairs[:, 1]
    pairs = pairs[keep]
    labels = None
    if edge_labels is not None:
        labels = np.asarray(edge_labels, dtype=np.float32).reshape(-1)[keep]
    if pairs.size == 0:
        empty_labels = None if edge_labels is None else np.zeros((0,), dtype=np.float32)
        return np.zeros((0, 2), dtype=np.int64), empty_labels

    pairs = np.stack([np.minimum(pairs[:, 0], pairs[:, 1]),
                      np.maximum(pairs[:, 0], pairs[:, 1])], axis=1)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    if labels is not None:
        labels = labels[order]

    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    if labels is None:
        return unique_pairs.astype(np.int64), None
    merged = np.zeros((unique_pairs.shape[0],), dtype=np.float32)
    np.maximum.at(merged, inverse, labels)
    return unique_pairs.astype(np.int64), merged


def build_undirected_knn_edge_pairs(points, radius, max_neighbors):
    """Unique unordered candidate edges to the ``max_neighbors`` nearest points within ``radius``."""
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]
    if n < 2 or int(max_neighbors) <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    k = min(int(max_neighbors) + 1, n)
    _, idx = scipy.spatial.KDTree(points).query(points, k=k, distance_upper_bound=radius)
    if idx.ndim == 1:
        idx = idx[:, None]
    idx = idx[:, 1:]  # drop self
    if idx.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    valid = idx < n
    src = np.repeat(np.arange(n)[:, None], idx.shape[1], axis=1)[valid]
    dst = idx[valid]
    if src.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs, _ = _canonical_undirected_pairs(np.stack([src, dst], axis=1))
    return pairs


def _edge_attr_from_index(pos, edge_index):
    if edge_index.numel() == 0:
        return pos.new_zeros((0, 2))
    src, dst = edge_index
    return pos[dst] - pos[src]


def refresh_graph_edge_attrs(data):
    """Recompute (dx, dy) attributes after node positions changed."""
    data.edge_attr = _edge_attr_from_index(data.pos, data.edge_index)
    label_index = getattr(data, "edge_label_index", None)
    if label_index is not None:
        data.edge_label_attr = _edge_attr_from_index(data.pos, label_index)
    return data


def build_undirected_pyg_data(points, edge_pairs, embed_dim, edge_labels=None):
    """Build a PyG ``Data`` with undirected topology semantics.

    ``edge_label_index`` holds one canonical unordered edge per candidate.
    ``edge_index`` holds both directions and is used only for message passing.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    undirected_pairs, labels = _canonical_undirected_pairs(edge_pairs, edge_labels)
    if undirected_pairs.shape[0] == 0:
        directed_pairs = np.zeros((0, 2), dtype=np.int64)
    else:
        directed_pairs = np.concatenate(
            [undirected_pairs, undirected_pairs[:, ::-1]], axis=0).astype(np.int64)

    data = Data()
    data.x = torch.empty((points.shape[0], int(embed_dim)), dtype=torch.float)
    data.pos = torch.tensor(points, dtype=torch.float)
    data.edge_index = torch.tensor(directed_pairs.T, dtype=torch.long)
    data.edge_label_index = torch.tensor(undirected_pairs.T, dtype=torch.long)
    data.edge_attr = _edge_attr_from_index(data.pos, data.edge_index)
    data.edge_label_attr = _edge_attr_from_index(data.pos, data.edge_label_index)
    if edge_labels is not None:
        data.edge_label = torch.tensor(labels, dtype=torch.float)
    return data


def build_inference_pyg_graph(points, radius, max_neighbors, embed_dim):
    """Candidate graph over node proposals; None if there are no candidate edges."""
    edge_pairs = build_undirected_knn_edge_pairs(points, radius, max_neighbors)
    if edge_pairs.shape[0] == 0:
        return None
    return build_undirected_pyg_data(points, edge_pairs, embed_dim)


# --------------------------------------------------------------------------- splits
def _normalize_dataset_location(dataset_location):
    loc = str(dataset_location).strip().lower()
    if loc in {"sf", "sanfrancisco", "san_francisco", "san-francisco"}:
        return "sf"
    if loc in {"combined", "combined_dataset"}:
        return "combined"
    if loc in {"amsterdam", "amsterdam_dataset"}:
        return "amsterdam"
    if loc in {"london", "london_dataset"}:
        return "london"
    return "cityscale"


def _default_dataset_root(dataset_location):
    return {
        "sf": "SF_Dataset",
        "combined": "Full_Combined_Dataset",
        "amsterdam": "CVAT_dataset/amsterdam_dataset",
        "london": "CVAT_dataset/london_dataset",
    }.get(dataset_location, "SidewalkFormer_first_training")


def _tile_sort_key(tile_id):
    if isinstance(tile_id, (int, np.integer)):
        return (0, int(tile_id))
    match = re.search(r'(\d+)$', str(tile_id))
    return (1, int(match.group(1)) if match else str(tile_id))


def _list_dataset_tile_ids(base_dir, dataset_location):
    images_dir = Path(base_dir) / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if _normalize_dataset_location(dataset_location) == "sf":
        tile_ids = [p.stem for p in images_dir.glob("*.png")]
    else:
        tile_ids = [n for n in map(extract_number, os.listdir(images_dir)) if n is not None]

    tile_ids = sorted(tile_ids, key=_tile_sort_key)
    if not tile_ids:
        raise RuntimeError(f"No dataset tiles found under {images_dir}")
    return tile_ids


def _partition_tile_ids(tile_ids, seed=42, legacy_split=True):
    indices = list(tile_ids)
    random.Random(seed).shuffle(indices)

    if legacy_split:
        # Split used for the released models: 80% train / 20% test. The train
        # split already contains the (near-empty) validation remainder.
        train_size = int(0.8 * len(indices))
        test_size = int(0.2 * len(indices))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:train_size + test_size]
        val_indices = indices[train_size + test_size:]
        return train_indices + val_indices, val_indices, test_indices

    n = len(indices)
    if n == 1:
        return indices, [], indices
    if n == 2:
        return [indices[0]], [], [indices[1]]

    train_size = max(1, int(0.8 * n))
    val_size = max(1, int(0.1 * n))
    test_size = n - train_size - val_size
    if test_size < 1:
        test_size = 1
        if train_size > val_size and train_size > 1:
            train_size -= 1
        elif val_size > 1:
            val_size -= 1

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:train_size + val_size + test_size]
    if not test_indices:
        test_indices = val_indices[-1:]
        val_indices = val_indices[:-1]
    return train_indices, val_indices, test_indices


def _expand_tile_id_spec(spec):
    """Expand ints, numeric strings and inclusive "lo-hi" ranges into a set of ids.

    Both the int and str form of each id are included so the result matches
    either representation returned by ``_list_dataset_tile_ids``.
    """
    out = set()
    if spec is None:
        return out
    if isinstance(spec, (int, str)):
        spec = [spec]
    for item in spec:
        if isinstance(item, (int, np.integer)):
            vals = [int(item)]
        else:
            text = str(item).strip()
            if not text:
                continue
            match = re.fullmatch(r'(\d+)\s*-\s*(\d+)', text)
            if match:
                lo, hi = int(match.group(1)), int(match.group(2))
                vals = list(range(min(lo, hi), max(lo, hi) + 1))
            elif text.lstrip('-').isdigit():
                vals = [int(text)]
            else:
                out.add(text)
                continue
        for v in vals:
            out.add(v)
            out.add(str(v))
    return out


def cityscale_data_partition(base_dir="SidewalkFormer_first_training", dataset_location="cityscale",
                             split_seed=42, legacy_split=True,
                             tile_exclude=None, tile_include=None,
                             train_fraction=1.0):
    """Split dataset tiles into (train, val, test) tile-id lists.

    Ablation hooks (no-ops at their defaults):
      tile_exclude   : ids or "lo-hi" ranges dropped before splitting
                       (leave-one-city-out runs).
      tile_include   : if given, only these ids are kept before splitting.
      train_fraction : keep this fraction of the train split only, leaving
                       val/test untouched. Subsets are deterministic given
                       ``split_seed`` and nested across fractions.
    """
    tile_ids = _list_dataset_tile_ids(base_dir, dataset_location)

    include = _expand_tile_id_spec(tile_include)
    if include:
        tile_ids = [t for t in tile_ids if t in include]
        if not tile_ids:
            raise RuntimeError(f"tile_include removed every tile under {base_dir}")

    exclude = _expand_tile_id_spec(tile_exclude)
    if exclude:
        kept = [t for t in tile_ids if t not in exclude]
        if not kept:
            raise RuntimeError(f"tile_exclude removed every tile under {base_dir}")
        print(f"[data_partition] tile_exclude dropped {len(tile_ids) - len(kept)} "
              f"of {len(tile_ids)} tiles")
        tile_ids = kept

    train, val, test = _partition_tile_ids(tile_ids, seed=split_seed, legacy_split=legacy_split)

    train_fraction = float(train_fraction)
    if 0.0 < train_fraction < 1.0:
        # With legacy_split the train list also contains val; subsample only
        # the train-only tiles so val stays intact.
        val_set = set(val)
        order = [t for t in train if t not in val_set]
        n_train_only = len(order)
        random.Random(split_seed).shuffle(order)
        keep_n = max(1, int(round(train_fraction * n_train_only)))
        keep = set(order[:keep_n])
        train = [t for t in train if t in keep or t in val_set]
        print(f"[data_partition] train_fraction={train_fraction} -> "
              f"{keep_n}/{n_train_only} train tiles")
    elif train_fraction <= 0.0:
        raise ValueError("train_fraction must be > 0")

    return train, val, test


# --------------------------------------------------------------------------- patch grids
def get_patch_info_one_img(image_index, image_size, sample_margin, patch_size, patches_per_edge):
    """Square patch grid over a square image, touching both image edges.

    Returns ``[(image_index, (x0, y0), (x1, y1)), ...]``. ``patches_per_edge``
    positions per axis are used; ``sample_margin`` only shifts the interior
    positions. With ``patches_per_edge == 1`` the single patch is at (0, 0).
    """
    assert patch_size <= image_size, "patch_size must be <= image_size"
    coords = _edge_inclusive_coords(image_size, patch_size, patches_per_edge, sample_margin)
    return [
        (image_index, (x, y), (x + patch_size, y + patch_size))
        for x in coords
        for y in coords
    ]


def get_patch_info_one_img_rect(image_index, H, W, sample_margin, patch_size, patches_per_edge):
    """Edge-inclusive patch grid over a rectangular H x W image.

    ``patch_size`` is an int (square) or a (patch_h, patch_w) tuple. Output
    format matches :func:`get_patch_info_one_img`.
    """
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    else:
        patch_h = patch_w = int(patch_size)
    assert patch_h <= H and patch_w <= W, "patch size must fit inside the image"

    xs = _edge_inclusive_coords(W, patch_w, patches_per_edge, sample_margin)
    ys = _edge_inclusive_coords(H, patch_h, patches_per_edge, sample_margin)
    return [(image_index, (x, y), (x + patch_w, y + patch_h)) for y in ys for x in xs]


def _edge_inclusive_coords(length, patch, num, margin):
    """``num`` patch start positions along one axis, always including 0 and the far edge."""
    end = length - patch
    if num <= 1:
        coords = [0]
    elif num == 2:
        coords = [0, end]
    else:
        interior_start = min(max(0, margin), end)
        interior_end = max(min(end - margin, end), interior_start)
        interior = []
        if interior_end > interior_start:
            interior = [int(round(x)) for x in np.linspace(interior_start, interior_end, num=num - 2)]
        coords = [0] + interior + [end]
    # Clamp, de-duplicate and sort (avoids rounding duplicates).
    return sorted(set(max(0, min(end, int(c))) for c in coords))


def get_patch_info_strided(image_index, image_size, sample_margin, patch_size, patches_per_edge):
    """Strided patch grid with full coverage (stride = patch_size - 2 * sample_margin).

    ``image_size`` is an int (square) or (H, W). The last patch on each axis
    touches the far edge. If fewer than ``patches_per_edge`` positions result,
    the axis is densified to ``patches_per_edge`` evenly spaced positions.
    """
    if isinstance(image_size, int):
        H = W = int(image_size)
    else:
        H, W = int(image_size[0]), int(image_size[1])
    P = int(patch_size)
    assert 0 < P <= H and P <= W, "patch_size must fit inside image"

    stride = max(1, P - 2 * int(sample_margin))
    xs = list(range(0, max(W - P, 0) + 1, stride))
    ys = list(range(0, max(H - P, 0) + 1, stride))
    if xs[-1] != W - P:
        xs.append(W - P)
    if ys[-1] != H - P:
        ys.append(H - P)

    if patches_per_edge and patches_per_edge > 0:
        if len(xs) < patches_per_edge:
            xs = [int(round(t)) for t in np.linspace(0, W - P, patches_per_edge)]
            xs[-1] = W - P
        if len(ys) < patches_per_edge:
            ys = [int(round(t)) for t in np.linspace(0, H - P, patches_per_edge)]
            ys[-1] = H - P

    return [(image_index, (x0, y0), (min(x0 + P, W), min(y0 + P, H))) for y0 in ys for x0 in xs]


# --------------------------------------------------------------------------- graph labels
class GraphLabelGenerator:
    """Samples per-patch candidate graphs with edge labels from a reference graph.

    ``graph`` is a Sat2Graph adjacency dict; ``coord_transform`` maps its
    [N, 2] vertex array to (x, y) image coordinates.
    """

    def __init__(self, config, graph, coord_transform, tree_nodes=None):
        self.config = config
        self.graph = graph_utils.igraph_from_adj_dict(graph, coord_transform)
        self.crossover_points = graph_utils.find_crossover_points(self.graph)
        self.points = np.array(self.graph.vs['point'])

        self.rtree_index = rtree_from_points(self.points)
        self.kdtree = scipy.spatial.KDTree(self.points)

        # Junctions and dead ends (degree != 2) are always kept by NMS.
        point_num = len(self.graph.vs)
        itsc_indices = {i for i in range(point_num) if self.graph.degree(i) != 2}
        self.nms_score_override = np.zeros((point_num,), dtype=np.float32)
        self.nms_score_override[list(itsc_indices)] = 2.0

        # Vertices near junctions, crossovers and trees are "interesting" and
        # sampled more often when Weighted_Sampling is enabled.
        interesting_indices = set()
        interesting_radius = config.NEIGHBOR_RADIUS
        if tree_nodes is not None:
            near_tree_points, _ = graph_utils.find_near_tree_nodes(
                self.graph, tree_coords=tree_nodes, max_distance=config.NEIGHBOR_RADIUS)
            for p in near_tree_points:
                nearby = self.kdtree.query_ball_point(p, 30)
                interesting_indices.update(nearby)
                itsc_indices.update(nearby)
                self.nms_score_override[nearby] = 2.0
        for i in itsc_indices:
            interesting_indices.update(self.kdtree.query_ball_point(self.points[i], interesting_radius))
        for p in self.crossover_points:
            interesting_indices.update(self.kdtree.query_ball_point(np.array(p), interesting_radius))

        self.sample_weights = np.full((point_num,), 0.1, dtype=np.float32)
        self.sample_weights[list(interesting_indices)] = 0.9

    def _empty_pyg_data(self):
        data = Data()
        data.x = torch.zeros((0, cfg_get(self.config, "EMB_OUT_CH", 512)))
        data.pos = torch.zeros((0, 2))
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.edge_attr = torch.zeros((0, 2))
        data.edge_label_index = torch.zeros((2, 0), dtype=torch.long)
        data.edge_label_attr = torch.zeros((0, 2))
        data.edge_label = torch.zeros((0,))
        return data

    def _transform_patch_points(self, points_xy, patch, rot_index=0,
                                hflip=False, vflip=False, noise_scale=0.0):
        """Map absolute image (x, y) points into the augmented patch frame."""
        points_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        if points_xy.shape[0] == 0:
            return points_xy.copy()

        (x0, y0), _ = patch
        pts = points_xy - np.array([x0, y0], dtype=np.float32)[np.newaxis, :]
        pts = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        trans, rot = _patch_centering_and_rotation(self.config.PATCH_SIZE)
        pts = pts @ trans.T @ np.linalg.matrix_power(rot.T, rot_index) @ np.linalg.inv(trans.T)
        pts = pts[:, :2]

        size = float(self.config.PATCH_SIZE)
        if hflip:
            pts[:, 0] = size - pts[:, 0] - 1
        if vflip:
            pts[:, 1] = size - pts[:, 1] - 1
        if noise_scale > 0:
            pts += np.random.normal(0.0, noise_scale, size=pts.shape)
        return pts.astype(np.float32)

    def sample_patch_from_mask_points(self, patch, candidate_points_patch,
                                      rot_index=0, hflip=False, vflip=False):
        """Label mask-derived node proposals by snapping them to the reference graph.

        Keeps the training node distribution close to inference while edge
        labels still come from the annotated graph. Returns (points, data).
        """
        D = cfg_get(self.config, "EMB_OUT_CH", 512)
        candidate_points_patch = np.asarray(candidate_points_patch, dtype=np.float32).reshape(-1, 2)
        if candidate_points_patch.shape[0] < 2:
            return candidate_points_patch, self._empty_pyg_data()

        snap_radius = float(cfg_get(self.config, "TRAIN_MASK_NODE_SNAP_RADIUS",
                                    max(20.0, 0.5 * float(self.config.NEIGHBOR_RADIUS))))
        drop_unsnapped = bool(cfg_get(self.config, "TRAIN_MASK_DROP_UNSNAPPED_NODES", True))
        edge_depth = int(cfg_get(self.config, "TRAIN_MASK_EDGE_MAX_DEPTH",
                                 cfg_get(self.config, "MAX_DEPTH", 1)))

        (x0, y0), (x1, y1) = patch
        label_box = (min(x0, x1) - snap_radius, min(y0, y1) - snap_radius,
                     max(x0, x1) + snap_radius, max(y0, y1) + snap_radius)
        gt_indices = np.array(list(self.rtree_index.intersection(label_box)), dtype=np.int64)
        if gt_indices.shape[0] == 0:
            return candidate_points_patch, self._empty_pyg_data()

        gt_points_aug = self._transform_patch_points(
            self.points[gt_indices], patch, rot_index=rot_index, hflip=hflip, vflip=vflip)
        _, nearest = scipy.spatial.KDTree(gt_points_aug).query(
            candidate_points_patch, distance_upper_bound=snap_radius)
        snapped_ok = nearest < gt_indices.shape[0]

        points = candidate_points_patch
        nearest_gt = np.full((points.shape[0],), -1, dtype=np.int64)
        nearest_gt[snapped_ok] = gt_indices[nearest[snapped_ok]]
        if drop_unsnapped:
            if snapped_ok.sum() < 2:
                return candidate_points_patch, self._empty_pyg_data()
            points = points[snapped_ok]
            nearest_gt = nearest_gt[snapped_ok]

        edge_pairs = build_undirected_knn_edge_pairs(
            points, self.config.NEIGHBOR_RADIUS, self.config.MAX_NEIGHBOR_QUERIES)
        if edge_pairs.shape[0] == 0:
            return points, build_undirected_pyg_data(
                points, edge_pairs, D, edge_labels=np.zeros((0,), dtype=np.float32))

        # An edge is positive when the snapped reference vertices are within
        # ``edge_depth`` hops of each other.
        label_cache = {}
        edge_labels = []
        for u, v in edge_pairs:
            gu, gv = int(nearest_gt[u]), int(nearest_gt[v])
            if gu < 0 or gv < 0 or gu == gv:
                edge_labels.append(0.0)
                continue
            key = (min(gu, gv), max(gu, gv), edge_depth)
            if key not in label_cache:
                reached = graph_utils.bfs_with_conditions(self.graph, gu, {gv}, edge_depth)
                label_cache[key] = gv in reached
            edge_labels.append(float(label_cache[key]))

        return points, build_undirected_pyg_data(
            points, edge_pairs, D, edge_labels=np.asarray(edge_labels, dtype=np.float32))

    def sample_patch(self, patch, rot_index=0):
        """Candidate graph over reference vertices inside ``patch`` (training default).

        Vertices are thinned with NMS (junctions always kept), connected to their
        nearest neighbours within NEIGHBOR_RADIUS, and an edge is labelled
        positive when the two vertices are adjacent in the reference graph.
        Node positions are rotated by ``rot_index`` * 90 degrees and jittered by
        NODE_JITTER_SIGMA pixels. Returns (points, data).
        """
        (x0, y0), (x1, y1) = patch
        query_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        patch_indices = np.array(list(set(self.rtree_index.intersection(query_box))))
        D = cfg_get(self.config, "EMB_OUT_CH", 512)
        if len(patch_indices) == 0:
            return np.zeros((0, 2), dtype=np.float32), self._empty_pyg_data()

        # Random scores give varied but similarly spaced NMS selections.
        nms_scores = np.random.uniform(low=0.9, high=1.0, size=patch_indices.shape[0])
        nms_scores = np.maximum(nms_scores, self.nms_score_override[patch_indices])
        nmsed_points, kept_indices = graph_utils.nms_points_training(
            self.points[patch_indices, :], nms_scores,
            radius=self.config.ROAD_NMS_RADIUS, return_indices=True)
        nmsed_indices = patch_indices[kept_indices]
        nmsed_point_num = nmsed_points.shape[0]

        if cfg_get(self.config, 'Weighted_Sampling', False):
            sample_num = min(self.config.TOPO_SAMPLE_NUM, len(patch_indices))
            sample_weights = self.sample_weights[nmsed_indices]
            sample_indices_in_nmsed = np.random.choice(
                np.arange(start=0, stop=nmsed_point_num, dtype=np.int32),
                size=sample_num, replace=True, p=sample_weights / np.sum(sample_weights))
            sample_indices = nmsed_indices[sample_indices_in_nmsed]
        else:
            sample_indices_in_nmsed = np.arange(start=0, stop=nmsed_point_num, dtype=np.int32)
            sample_indices = nmsed_indices

        # k + 1 neighbours because the nearest one is the point itself.
        _, knn_idx = scipy.spatial.KDTree(nmsed_points).query(
            self.points[sample_indices, :], k=self.config.MAX_NEIGHBOR_QUERIES + 1,
            distance_upper_bound=self.config.NEIGHBOR_RADIUS)

        edge_pairs, edge_labels = [], []
        for i, source_node in enumerate(sample_indices):
            valid_nbr_indices = knn_idx[i, knn_idx[i, :] < nmsed_point_num][1:]
            target_nodes = [nmsed_indices[ni] for ni in valid_nbr_indices]
            reached = graph_utils.bfs_with_conditions(self.graph, source_node, set(target_nodes), 1)
            source_nmsed_idx = sample_indices_in_nmsed[i]
            for target_nmsed_idx, target in zip(valid_nbr_indices, target_nodes):
                edge_pairs.append((source_nmsed_idx, target_nmsed_idx))
                edge_labels.append(float(target in reached))

        # Move to patch coordinates, then rotate about the patch centre.
        nmsed_points -= np.array([x0, y0])[np.newaxis, :]
        nmsed_points = np.concatenate(
            [nmsed_points, np.ones((nmsed_point_num, 1), dtype=nmsed_points.dtype)], axis=1)
        trans, rot = _patch_centering_and_rotation(self.config.PATCH_SIZE)
        nmsed_points = nmsed_points @ trans.T @ np.linalg.matrix_power(rot.T, rot_index) @ np.linalg.inv(trans.T)
        nmsed_points = nmsed_points[:, :2]

        noise_scale = float(cfg_get(self.config, 'NODE_JITTER_SIGMA', 5.0))
        if noise_scale > 0:
            nmsed_points += np.random.normal(0.0, noise_scale, size=nmsed_points.shape)

        return nmsed_points, build_undirected_pyg_data(nmsed_points, edge_pairs, D, edge_labels=edge_labels)


def rtree_from_points(points):
    """R-tree over (x, y) points (each inserted as a degenerate box)."""
    index = rtree.index.Index()
    for i, (x, y) in enumerate(points):
        index.insert(i, (x, y, x, y))
    return index


def _patch_centering_and_rotation(patch_size):
    """Homogeneous matrices: shift the patch centre to the origin, and rotate 90 degrees CCW."""
    trans = np.array([
        [1, 0, -0.5 * patch_size],
        [0, 1, -0.5 * patch_size],
        [0, 0, 1],
    ], dtype=np.float32)
    rot = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ], dtype=np.float32)
    return trans, rot


# --------------------------------------------------------------------------- dataset
class SatMapDataset(Dataset):
    """Patches of aerial tiles with class masks and candidate-graph labels.

    All tiles of the split are loaded into memory. Training patches are
    shuffled and only those containing reference-graph vertices are kept;
    validation keeps every patch.
    """

    def __init__(self, config, is_train=False, dev_run=False):
        self.config = config
        self.is_train = is_train
        self.return_image_pil = bool(cfg_get(config, "RETURN_IMAGE_PIL", False))
        self.train_graph_node_source = str(
            cfg_get(config, "TRAIN_GRAPH_NODE_SOURCE", "gt_graph")).strip().lower()
        self.train_mask_empty_fallback = bool(cfg_get(config, "TRAIN_MASK_EMPTY_FALLBACK", True))

        self.dataset_location = _normalize_dataset_location(
            cfg_get(config, "dataset_location", cfg_get(config, "DATASET", "cityscale")))
        self.base_dir = cfg_get(config, "DATASET_ROOT", "") or _default_dataset_root(self.dataset_location)
        self.PATCH = config.PATCH_SIZE

        rgb_pattern, mask_pattern, graph_pattern = self._file_patterns()
        print(f"[dataset] dataset_location={self.dataset_location} base_dir={self.base_dir}")
        print(f"[dataset] Using graph labels from: {os.path.dirname(graph_pattern)}")
        tree_pattern = os.path.join(self.base_dir, "tree_locations", "tree_coordinates_{}.p")

        legacy_split = bool(cfg_get(config, "LEGACY_SPLIT", True))
        train, val, test = cityscale_data_partition(
            base_dir=self.base_dir,
            dataset_location=self.dataset_location,
            split_seed=int(cfg_get(config, "SPLIT_SEED", 42)),
            legacy_split=legacy_split,
            tile_exclude=cfg_get(config, "TILE_ID_EXCLUDE", None),
            tile_include=cfg_get(config, "TILE_ID_INCLUDE", None),
            train_fraction=cfg_get(config, "TRAIN_TILE_FRACTION", 1.0),
        )
        print(f"[dataset] split sizes train={len(train)} val={len(val)} test={len(test)} "
              f"legacy_split={legacy_split}")
        tile_indices = train + val if is_train else test
        if dev_run:
            tile_indices = tile_indices[:4]

        coord_transform = lambda v: v[:, ::-1]  # (row, col) -> (x, y)  # noqa: E731
        self.rgbs, self.road_masks, self.image_sizes = [], [], []
        self.graph_label_generators = []
        self.loaded_tile_indices = []
        for tile_idx in tile_indices:
            graph_path = graph_pattern.format(tile_idx)
            rgb_path = rgb_pattern.format(tile_idx)
            mask_path = mask_pattern.format(tile_idx)
            if not os.path.exists(graph_path):
                print(f"[dataset] skipping tile {tile_idx}: no pedestrian graph")
                continue
            if not os.path.exists(rgb_path):
                raise FileNotFoundError(f"Missing image tile: {rgb_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing grayscale mask: {mask_path}")

            with open(graph_path, 'rb') as f:
                gt_graph_adj = pickle.load(f)
            if len(gt_graph_adj) == 0:
                print(f"[dataset] skipping tile {tile_idx}: empty graph")
                continue
            tree_nodes = None
            tree_path = tree_pattern.format(tile_idx)
            if os.path.exists(tree_path):
                with open(tree_path, 'rb') as f:
                    tree_nodes = coord_transform(pickle.load(f))

            rgb_img = read_rgb_img(rgb_path)
            road_mask = read_grayscale_mask(mask_path)
            if rgb_img.shape[:2] != road_mask.shape[:2]:
                raise AssertionError(
                    f"masks and rgb shape mismatch for tile {tile_idx}: "
                    f"rgb={rgb_img.shape[:2]} mask={road_mask.shape[:2]} "
                    f"(rgb_path={rgb_path}, mask_path={mask_path})")

            self.loaded_tile_indices.append(tile_idx)
            self.rgbs.append(rgb_img)
            self.road_masks.append(road_mask)
            self.image_sizes.append(rgb_img.shape[:2])
            self.graph_label_generators.append(
                GraphLabelGenerator(config, gt_graph_adj, coord_transform, tree_nodes=tree_nodes))

        if not self.rgbs:
            raise RuntimeError("No tiles were loaded for this split. Check the graph label files.")
        print(f"[dataset] loaded {len(self.rgbs)} tiles")

        self.patches = self._build_patches()
        if is_train:
            random.shuffle(self.patches)
        if not self.patches:
            raise RuntimeError("Dataset produced zero patches. Check graph labels and PATCH_SIZE.")

    def _file_patterns(self):
        """(rgb, mask, graph) path patterns with ``{}`` for the tile id."""
        images_dir = os.path.join(self.base_dir, "images")
        masks_dir = os.path.join(self.base_dir, "grayscale_masks")
        gt_graph_dir = os.path.join(self.base_dir, "gt_graph")
        resampled_dir = os.path.join(self.base_dir, "gt_graph_resampled60")

        if self.dataset_location == "sf":
            rgb_pattern = os.path.join(images_dir, "{}.png")
            mask_pattern = os.path.join(masks_dir, "{}.png")
            graph_candidates = [os.path.join(resampled_dir, "{}.p")]
        else:
            rgb_pattern = os.path.join(images_dir, "image{}.png")
            mask_pattern = os.path.join(masks_dir, "image{}.png")
            if self.dataset_location == "combined":
                graph_candidates = [os.path.join(resampled_dir, "adj_sidewalk_{}_updated.p")]
            elif self.dataset_location in {"amsterdam", "london"}:
                # Prefer graphs resampled to ~60 px spacing, else the raw graphs.
                graph_candidates = [
                    os.path.join(resampled_dir, "graph_GT_adj_dict_pedestrian_{}.p"),
                    os.path.join(gt_graph_dir, "graph_GT_adj_dict_pedestrian_{}.p"),
                ]
            else:
                sparse_3x = os.path.join(self.base_dir, 'gt_graph_sparse_3x', 'adj_sidewalk_{}_updated.p')
                sparse_2x = os.path.join(self.base_dir, 'gt_graph_sparse_2x', 'adj_sidewalk_{}_updated.p')
                dense = os.path.join(gt_graph_dir, 'adj_sidewalk_{}_updated.p')
                graph_candidates = {
                    2: [sparse_2x, sparse_3x, dense],
                    3: [sparse_3x, sparse_2x, dense],
                }.get(cfg_get(self.config, "gt_graph_pattern", 0), [dense, sparse_3x, sparse_2x])

        for candidate in graph_candidates:
            if os.path.isdir(os.path.dirname(candidate)):
                return rgb_pattern, mask_pattern, candidate
        raise FileNotFoundError(
            "Could not find any graph directory. Checked: "
            + ", ".join(sorted({os.path.dirname(p) for p in graph_candidates})))

    def _build_patches(self):
        patches = []
        legacy_patch_grid = bool(cfg_get(self.config, "LEGACY_PATCH_GRID", True))
        for img_idx, (img_h, img_w) in enumerate(self.image_sizes):
            if legacy_patch_grid:
                per_edge = max(1, math.ceil((img_h - 2 * TRAIN_SAMPLE_MARGIN) / self.PATCH))
                if img_h == img_w:
                    coords = get_patch_info_one_img(
                        img_idx, img_h, TRAIN_SAMPLE_MARGIN, self.PATCH, per_edge)
                else:
                    coords = get_patch_info_one_img_rect(
                        img_idx, img_h, img_w, TRAIN_SAMPLE_MARGIN, self.PATCH, per_edge)
            else:
                per_edge = max(1, math.ceil((max(img_h, img_w) - 2 * TRAIN_SAMPLE_MARGIN) / self.PATCH))
                coords = get_patch_info_strided(
                    img_idx, (img_h, img_w), TRAIN_SAMPLE_MARGIN, self.PATCH, per_edge)

            if self.is_train:
                # Keep only patches with at least two reference vertices.
                index = self.graph_label_generators[img_idx].rtree_index
                for _, (bx, by), (ex, ey) in coords:
                    if len(list(index.intersection((bx, by, ex, ey)))) > 1:
                        patches.append((img_idx, (bx, by), (ex, ey)))
            else:
                patches += coords
        return patches

    def __len__(self):
        return len(self.patches)

    def _extract_train_mask_graph_points(self, mask_np):
        """Inference-style node proposals from a reference class mask."""
        side_labels = cfg_get(self.config, "TRAIN_MASK_SIDEWALK_LABELS", [1])
        cross_labels = cfg_get(self.config, "TRAIN_MASK_CROSSING_LABELS", [3, 4])
        side_radius = int(cfg_get(self.config, "TRAIN_MASK_SIDEWALK_NMS_RADIUS",
                                  cfg_get(self.config, "SIDEWALK_NMS_RADIUS", 60)))
        cross_radius = int(cfg_get(self.config, "TRAIN_MASK_CROSSING_NMS_RADIUS",
                                   cfg_get(self.config, "CROSSING_NMS_RADIUS", 60)))

        points = []
        for labels, radius in ((side_labels, side_radius), (cross_labels, cross_radius)):
            binary = np.isin(mask_np, np.asarray(labels, dtype=mask_np.dtype))
            if not binary.any():
                continue
            xys = np.column_stack(np.where(binary))[:, ::-1].astype(np.float32)
            scores = np.ones((xys.shape[0],), dtype=np.float32)
            points.append(nms_points(xys, scores, radius=max(1, radius)))
        if not points:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate(points, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        img_idx, (x0, y0), (x1, y1) = self.patches[idx]
        image_pil = Image.fromarray(self.rgbs[img_idx][y0:y1, x0:x1, :].astype(np.uint8)).convert('RGB')
        mask_pil = Image.fromarray(self.road_masks[img_idx][y0:y1, x0:x1].astype(np.uint8))

        rotation, hflip, vflip = 0, False, False
        if self.is_train:
            rotation = random.randint(0, 3)
            hflip = random.random() > 0.5
            vflip = random.random() > 0.5
            blur = random.random() > 0.6
            blur_radius = random.uniform(0, 1.5)

            if rotation > 0:
                image_pil = image_pil.rotate(90 * rotation)
                mask_pil = mask_pil.rotate(90 * rotation)
            if hflip:
                image_pil = TF.hflip(image_pil)
                mask_pil = TF.hflip(mask_pil)
            if vflip:
                image_pil = TF.vflip(image_pil)
                mask_pil = TF.vflip(mask_pil)
            image_pil = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)(image_pil)
            if blur:
                image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        mask_np = np.array(mask_pil)
        pixel_values = _IMAGENET_NORMALIZE(image_pil)   # native PATCH_SIZE, as at inference
        labels = torch.from_numpy(np.array(mask_np, dtype=np.int64))

        patch = ((x0, y0), (x1, y1))
        generator = self.graph_label_generators[img_idx]
        flips_applied = False
        if self.is_train and self.train_graph_node_source in MASK_NODE_SOURCES:
            mask_points = self._extract_train_mask_graph_points(mask_np)
            _, graph = generator.sample_patch_from_mask_points(
                patch, mask_points, rot_index=rotation, hflip=hflip, vflip=vflip)
            flips_applied = True
            if ((graph.num_nodes < 2 or graph.edge_label_index.numel() == 0)
                    and self.train_mask_empty_fallback):
                _, graph = generator.sample_patch(patch, rot_index=rotation)
                flips_applied = False
        else:
            _, graph = generator.sample_patch(patch, rot_index=rotation)

        if self.is_train and not flips_applied:
            size = self.config.PATCH_SIZE
            if hflip:
                graph.pos[:, 0] = size - graph.pos[:, 0] - 1
            if vflip:
                graph.pos[:, 1] = size - graph.pos[:, 1] - 1
        graph = refresh_graph_edge_attrs(graph)

        item = {
            'pixel_values': pixel_values,
            'labels': labels,
            'graph_points': graph.pos.detach().cpu().clone(),
            'graph_data': graph,
        }
        if self.return_image_pil:
            item["image_pil"] = image_pil
        return item


def custom_graph_collate_fn(batch):
    """Collate dataset items: tensors are stacked, graphs are merged into a PyG ``Batch``."""
    graphs = [item['graph_data'] for item in batch]
    image_pils = [item['image_pil'] for item in batch] if 'image_pil' in batch[0] else None

    # Pad graph_points to the same length.
    max_pts = max(item['graph_points'].shape[0] for item in batch)
    for item in batch:
        p = item['graph_points']
        pad_n = max_pts - p.shape[0]
        if pad_n > 0:
            item['graph_points'] = torch.cat([p, torch.zeros(pad_n, 2, dtype=p.dtype)], dim=0)

    collated = default_collate([
        {k: v for k, v in item.items() if k not in ('graph_data', 'image_pil')}
        for item in batch
    ])
    collated['graph_data'] = Batch.from_data_list(graphs)
    if image_pils is not None:
        collated['image_pil'] = image_pils
    return collated
