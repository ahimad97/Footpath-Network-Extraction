"""City-scale SidewalkFormer inference: per-tile graphs stitched into one network.

Pipeline:
  1. Each tile is covered by overlapping patches; class probabilities are
     averaged, node proposals are extracted from the sidewalk/crossing masks,
     and the topology head scores candidate edges. Every edge keeps its
     averaged probability and its sidewalk/crossing type.
  2. Per tile, components with fewer than 3 edges are pruned.
  3. Tiles are merged by snapping nearby nodes (``--snap_tol_m``); duplicate
     edges keep their highest score, and ``TOPO_MERGE_THRESHOLD`` is applied
     once after the merge.
  4. Synthetic seam bridges join unsnapped cross-tile neighbours
     (``--bridge_tol_m``); they are flagged and carry no model confidence.
  5. High-degree node clusters within ``--junction_snap_tol_m`` are collapsed.

Outputs per-tile results under ``tiles/`` and the merged graph (NPZ + GeoJSON)
under ``global/``. Re-running with the same output directory resumes.
"""

import argparse
import collections
import contextlib
import csv
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rtree
import torch
import torch.nn.functional as F
import torchvision.transforms as tvT
from PIL import Image
from torch_geometric.data import Batch, Data

from sidewalkformer import graph_utils
from sidewalkformer.dataset import (
    build_inference_pyg_graph,
    get_patch_info_one_img,
    get_patch_info_one_img_rect,
)
from sidewalkformer.graph_extraction import (
    get_points_and_scores_from_mask,
    get_skeleton_points_and_scores,
    nms_points,
)
from sidewalkformer.model import SidewalkFormer
from sidewalkformer.utils import load_config

_IMAGENET_TRANSFORM = tvT.Compose(
    [
        tvT.ToTensor(),
        tvT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


_EARTH_RADIUS_M = 6_371_008.8


def wgs84_to_local_meters(nodes_xy: np.ndarray) -> np.ndarray:
    """Project WGS84 ``(lon, lat)`` coordinates to a local metric plane.

    All city-scale manifests use WGS84 bboxes.  A local equirectangular
    projection keeps merge radii in physical metres without changing the
    WGS84 coordinates written to graph outputs.  Its distortion is negligible
    at city scale and avoids the latitude-dependent scale of raw degrees.
    """
    if nodes_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    origin = nodes_xy.mean(axis=0)
    meters_per_degree = _EARTH_RADIUS_M * math.pi / 180.0
    x = (nodes_xy[:, 0] - origin[0]) * meters_per_degree * math.cos(math.radians(origin[1]))
    y = (nodes_xy[:, 1] - origin[1]) * meters_per_degree
    return np.column_stack((x, y))


def preprocess_patches(pil_imgs: Sequence[Image.Image]) -> torch.Tensor:
    return torch.stack([_IMAGENET_TRANSFORM(img) for img in pil_imgs], dim=0)


def read_manifest(path: str) -> List[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "tiles" in data:
            return list(data["tiles"])
        if isinstance(data, list):
            return data
        raise ValueError("JSON manifest must be a list or {'tiles': [...]} structure.")
    if ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        parsed: List[dict] = []
        for row in rows:
            parsed.append({
                "tile_id": row.get("tile_id") or row.get("id"),
                "image_path": row.get("image_path"),
                "bbox": [
                    float(row["minx"]),
                    float(row["miny"]),
                    float(row["maxx"]),
                    float(row["maxy"]),
                ],
            })
        return parsed
    raise ValueError("Manifest must be .json or .csv")


def extract_graph_points_separate(cross_mask_u8: np.ndarray, side_mask_u8: np.ndarray, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """Extract candidate node coords for the sidewalk and crossing classes.

    With ``SAMPLE_FROM_CENTERLINE`` (config or --sample_from_centerline) the
    candidates are drawn from the mask skeleton/medial axis, so nodes land on
    sidewalk/crossing centerlines. Otherwise every foreground pixel above the
    class threshold is a candidate (default). Both paths feed the same
    confidence-ordered NMS, so node spacing is unchanged.
    """
    use_centerline = bool(getattr(cfg, "SAMPLE_FROM_CENTERLINE", False))
    min_blob = int(getattr(cfg, "CENTERLINE_MIN_BLOB_PX", 0))
    thickness = int(getattr(cfg, "CENTERLINE_THICKNESS_PX", 0))
    sample = (
        (lambda m, t: get_skeleton_points_and_scores(m, t, min_blob, thickness))
        if use_centerline
        else get_points_and_scores_from_mask
    )

    cand, sc = sample(cross_mask_u8, cfg.CROSSING_THRESHOLD * 255)
    kps_cross = nms_points(cand, sc, radius=cfg.CROSSING_NMS_RADIUS)
    cand, sc = sample(side_mask_u8, cfg.SIDEWALK_THRESHOLD * 255)
    kps_side = nms_points(cand, sc, radius=cfg.SIDEWALK_NMS_RADIUS)
    return kps_side.astype(np.float32), kps_cross.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-edge metadata (scores, types, synthetic flags) stays aligned with edges
# through every pruning / remapping step below.
# ---------------------------------------------------------------------------


def _empty_edge_metadata() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 2), np.int32),
        np.zeros((0,), np.float32),
        np.zeros((0,), np.uint8),
        np.zeros((0,), bool),
    )


def _validate_edge_metadata(
    edges: np.ndarray,
    scores: np.ndarray,
    edge_types: np.ndarray,
    edge_is_synthetic: np.ndarray,
) -> None:
    edge_count = len(edges)
    if len(scores) != edge_count:
        raise ValueError("edge_scores must have one value per edge")
    if len(edge_types) != edge_count:
        raise ValueError("edge_types must have one value per edge")
    if len(edge_is_synthetic) != edge_count:
        raise ValueError("edge_is_synthetic must have one value per edge")
    scores_arr = np.asarray(scores, dtype=np.float32)
    types_arr = np.asarray(edge_types, dtype=np.uint8)
    synthetic_arr = np.asarray(edge_is_synthetic, dtype=bool)
    if np.any((types_arr != 0) & (types_arr != 1)):
        raise ValueError("edge_types values must be 0=sidewalk or 1=crossing")
    if np.any(~synthetic_arr & ~np.isfinite(scores_arr)):
        raise ValueError("non-synthetic edges must have finite edge_scores")
    if np.any(synthetic_arr & ~np.isnan(scores_arr)):
        raise ValueError("synthetic edges must use NaN edge_scores")


def prune_isolated_nodes_with_metadata(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    scores: np.ndarray,
    edge_types: np.ndarray,
    edge_is_synthetic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop nodes with zero edges; keep boundary fragments for cross-tile merge."""
    _validate_edge_metadata(edges, scores, edge_types, edge_is_synthetic)
    if nodes_rc.size == 0:
        return nodes_rc.astype(np.int32), *_empty_edge_metadata()

    e = np.asarray(edges, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float32)
    t = np.asarray(edge_types, dtype=np.uint8)
    y = np.asarray(edge_is_synthetic, dtype=bool)
    n = int(nodes_rc.shape[0])
    if e.size == 0:
        return np.zeros((0, 2), np.int32), *_empty_edge_metadata()

    valid = (e[:, 0] >= 0) & (e[:, 0] < n) & (e[:, 1] >= 0) & (e[:, 1] < n)
    e = e[valid]
    s = s[valid]
    t = t[valid]
    y = y[valid]
    if e.size == 0:
        return np.zeros((0, 2), np.int32), *_empty_edge_metadata()

    incident = np.zeros(n, dtype=bool)
    incident[e[:, 0]] = True
    incident[e[:, 1]] = True
    kept_idx = np.nonzero(incident)[0]
    if kept_idx.size == 0:
        return np.zeros((0, 2), np.int32), *_empty_edge_metadata()

    old2new = -np.ones(n, dtype=np.int64)
    old2new[kept_idx] = np.arange(kept_idx.size, dtype=np.int64)
    keep_e = incident[e[:, 0]] & incident[e[:, 1]]
    return (
        nodes_rc[kept_idx].astype(np.int32),
        old2new[e[keep_e]].astype(np.int32),
        s[keep_e].astype(np.float32),
        t[keep_e].astype(np.uint8),
        y[keep_e].astype(bool),
    )


def prune_keep_components_ge3_edges_with_metadata(
    nodes_rc: np.ndarray,
    edges: np.ndarray,
    scores: np.ndarray,
    edge_types: np.ndarray,
    edge_is_synthetic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep only connected components with at least 3 edges."""
    _validate_edge_metadata(edges, scores, edge_types, edge_is_synthetic)
    if nodes_rc.size == 0:
        return nodes_rc.astype(np.int32), *_empty_edge_metadata()

    e = np.asarray(edges, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float32)
    t = np.asarray(edge_types, dtype=np.uint8)
    y = np.asarray(edge_is_synthetic, dtype=bool)
    n = int(nodes_rc.shape[0])
    if e.size == 0:
        return np.zeros((0, 2), np.int32), *_empty_edge_metadata()

    valid = (e[:, 0] >= 0) & (e[:, 0] < n) & (e[:, 1] >= 0) & (e[:, 1] < n)
    e = e[valid]
    s = s[valid]
    t = t[valid]
    y = y[valid]
    if e.size == 0:
        return np.zeros((0, 2), np.int32), *_empty_edge_metadata()

    incident = np.zeros(n, dtype=bool)
    incident[e[:, 0]] = True
    incident[e[:, 1]] = True

    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in e:
        adj[u].append(int(v))
        if u != v:
            adj[v].append(int(u))

    comp_id = -np.ones(n, dtype=np.int64)
    comp_nodes: List[List[int]] = []
    cid = 0
    for st in range(n):
        if (not incident[st]) or comp_id[st] != -1:
            continue
        stack = [st]
        comp: List[int] = []
        comp_id[st] = cid
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if comp_id[w] == -1:
                    comp_id[w] = cid
                    stack.append(w)
        comp_nodes.append(comp)
        cid += 1

    if cid == 0:
        return np.zeros((0, 2), np.int32), *_empty_edge_metadata()

    comp_edge_count = np.zeros(cid, dtype=np.int64)
    for u, v in e:
        cu, cv = comp_id[u], comp_id[v]
        if cu >= 0 and cu == cv:
            comp_edge_count[cu] += 1

    keep_comp = comp_edge_count >= 3
    keep_nodes = np.zeros(n, dtype=bool)
    for c, nodes in enumerate(comp_nodes):
        if keep_comp[c]:
            keep_nodes[nodes] = True

    kept_idx = np.nonzero(keep_nodes)[0]
    if kept_idx.size == 0:
        return np.zeros((0, 2), np.int32), *_empty_edge_metadata()

    old2new = -np.ones(n, dtype=np.int64)
    old2new[kept_idx] = np.arange(kept_idx.size, dtype=np.int64)
    keep_e = keep_nodes[e[:, 0]] & keep_nodes[e[:, 1]]
    return (
        nodes_rc[kept_idx].astype(np.int32),
        old2new[e[keep_e]].astype(np.int32),
        s[keep_e].astype(np.float32),
        t[keep_e].astype(np.uint8),
        y[keep_e].astype(bool),
    )


# ---------------------------------------------------------------------------
# Junction snapping.
# ---------------------------------------------------------------------------


def snap_high_degree_clusters(
    nodes_xy: np.ndarray,
    edges: np.ndarray,
    radius_m: float,
    edge_scores: np.ndarray,
    edge_types: np.ndarray,
    edge_is_synthetic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse degree-≥3 node clusters within ``radius_m`` metres into a centroid.

    Reroutes incident edges; drops self-loops and duplicate (u,v) pairs.
    Lower-degree nodes are passed through unchanged. ``nodes_xy`` remains in
    WGS84 throughout; metric coordinates are used only for distance checks.
    All aligned edge metadata is preserved. If snapping creates duplicate links,
    a real edge is preferred over a synthetic edge, then the highest-scoring real
    edge is retained (ties prefer the crossing type).
    """
    _validate_edge_metadata(edges, edge_scores, edge_types, edge_is_synthetic)
    scores = np.asarray(edge_scores, dtype=np.float32)
    types = np.asarray(edge_types, dtype=np.uint8)
    synthetic = np.asarray(edge_is_synthetic, dtype=bool)

    n = int(nodes_xy.shape[0])
    if n == 0 or edges.shape[0] == 0 or radius_m <= 0:
        return nodes_xy, edges, scores, types, synthetic

    degree = np.zeros(n, dtype=np.int64)
    for u, v in edges:
        degree[u] += 1
        if u != v:
            degree[v] += 1
    junction = degree >= 3
    if not junction.any():
        return nodes_xy, edges, scores, types, synthetic

    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    metric_nodes = wgs84_to_local_meters(nodes_xy)
    idx = rtree.index.Index()
    junction_ids = np.nonzero(junction)[0]
    for i in junction_ids:
        x, y = float(metric_nodes[i, 0]), float(metric_nodes[i, 1])
        idx.insert(int(i), (x, y, x, y))

    r2 = radius_m * radius_m
    for i in junction_ids:
        x, y = float(metric_nodes[i, 0]), float(metric_nodes[i, 1])
        for j in idx.intersection((x - radius_m, y - radius_m, x + radius_m, y + radius_m)):
            if j <= int(i):
                continue
            dx = metric_nodes[j, 0] - x
            dy = metric_nodes[j, 1] - y
            if dx * dx + dy * dy <= r2:
                union(int(i), int(j))

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    rep_ids = sorted(clusters.keys())
    new_nodes = np.zeros((len(rep_ids), 2), dtype=np.float64)
    root2new: Dict[int, int] = {}
    for ni, r in enumerate(rep_ids):
        members = np.array(clusters[r], dtype=np.int64)
        new_nodes[ni] = nodes_xy[members].mean(axis=0)
        root2new[r] = ni

    mapped: Dict[Tuple[int, int], Tuple[float, int, bool]] = {}
    for (u, v), score, edge_type, is_synthetic in zip(edges, scores, types, synthetic):
        nu = root2new[find(int(u))]
        nv = root2new[find(int(v))]
        if nu == nv:
            continue
        if nu > nv:
            nu, nv = nv, nu
        key = (int(nu), int(nv))
        candidate = (float(score), int(edge_type), bool(is_synthetic))
        previous = mapped.get(key)
        if previous is None:
            mapped[key] = candidate
        elif previous[2] and not candidate[2]:
            mapped[key] = candidate
        elif previous[2] == candidate[2]:
            if candidate[2]:
                if candidate[1] > previous[1]:
                    mapped[key] = candidate
            elif candidate[0] > previous[0] or (
                candidate[0] == previous[0] and candidate[1] > previous[1]
            ):
                mapped[key] = candidate

    if not mapped:
        return new_nodes, *_empty_edge_metadata()

    sorted_pairs = sorted(mapped.items())
    snapped_edges = np.asarray([key for key, _ in sorted_pairs], dtype=np.int32)
    snapped_scores = np.asarray([value[0] for _, value in sorted_pairs], dtype=np.float32)
    snapped_types = np.asarray([value[1] for _, value in sorted_pairs], dtype=np.uint8)
    snapped_synthetic = np.asarray([value[2] for _, value in sorted_pairs], dtype=bool)
    return new_nodes, snapped_edges, snapped_scores, snapped_types, snapped_synthetic


# ---------------------------------------------------------------------------
# Per-tile inference.
# ---------------------------------------------------------------------------


@dataclass
class TileResult:
    tile_id: str
    core_mask_u8: np.ndarray
    core_pred_mask: np.ndarray
    nodes_rc_core: np.ndarray
    edges: np.ndarray
    edge_scores: np.ndarray
    edge_types: np.ndarray
    edge_is_synthetic: np.ndarray
    nodes_global_xy: np.ndarray
    bbox: Tuple[float, float, float, float]
    core_shape_hw: Tuple[int, int]


class CityScaleInferencer:
    def __init__(
        self,
        config,
        checkpoint_path: str,
        device: str,
        use_bf16: bool = True,
        use_compile: bool = True,
        gnn_chunk: int = 16,
    ):
        self.cfg = config
        self.device = torch.device(device)
        self.net = SidewalkFormer(self.cfg)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.net.load_state_dict(state["state_dict"], strict=False)
        self.net.eval().to(self.device)

        is_cuda = self.device.type == "cuda"
        self.use_bf16 = bool(use_bf16) and is_cuda
        self.gnn_chunk = max(1, int(gnn_chunk))

        if is_cuda:
            torch.backends.cudnn.benchmark = True
            try:
                self.net.to(memory_format=torch.channels_last)
                self._channels_last = True
            except (RuntimeError, ValueError):
                self._channels_last = False
        else:
            self._channels_last = False

        # torch.compile the segmentation/feature forward only. The GNN head has
        # variable graph shapes and is not a good compile target.
        model_type = getattr(self.cfg, "MODEL_TYPE", "segformer")
        if is_cuda and use_compile and model_type != "tile2net":
            try:
                self.net.infer_masks_and_img_features = torch.compile(
                    self.net.infer_masks_and_img_features,
                    mode="reduce-overhead",
                    dynamic=False,
                )
            except Exception as exc:  # pragma: no cover - torch.compile is best-effort
                print(f"[warn] torch.compile disabled: {exc}")

        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and self.device.type == "cuda":
            self.net = torch.nn.DataParallel(self.net)

        defaults = {
            "SIDEWALK_THRESHOLD": 0.3,
            "CROSSING_THRESHOLD": 0.3,
            "SIDEWALK_NMS_RADIUS": 60,
            "CROSSING_NMS_RADIUS": 60,
            # Per-tile retention floor — keep low so the merge sees the full
            # confidence distribution before the global threshold is applied.
            "TOPO_THRESHOLD": 0.05,
            # Single confidence threshold applied once after merging.
            "TOPO_MERGE_THRESHOLD": 0.6,
            "NEIGHBOR_RADIUS": 120,
            "MAX_NEIGHBOR_QUERIES": 8,
            "INFER_BATCH_SIZE": 8,
            "INFER_PATCHES_PER_EDGE": 8,
            # Node-proposal source: False = every foreground pixel above the
            # class threshold; True = mask skeleton / centerline only.
            "SAMPLE_FROM_CENTERLINE": False,
            # When sampling from centerline, drop connected blobs smaller than
            # this (pixels) before skeletonizing (0 = keep all).
            "CENTERLINE_MIN_BLOB_PX": 0,
            # Half-width (px) the 1-px skeleton is widened to (0 = strict 1-px
            # medial axis; N>0 = centred ribbon ~(2N+1) px wide).
            "CENTERLINE_THICKNESS_PX": 0,
        }
        # addict.Dict answers hasattr() with True for missing keys, so test
        # membership instead; otherwise these defaults are never applied.
        for key, value in defaults.items():
            if key not in self.cfg:
                setattr(self.cfg, key, value)

        self.legacy_patch_grid = bool(getattr(self.cfg, "LEGACY_PATCH_GRID", True))
        if "SAMPLE_MARGIN" not in self.cfg:
            self.cfg.SAMPLE_MARGIN = int(self.cfg.NEIGHBOR_RADIUS)
        elif self.legacy_patch_grid and int(self.cfg.SAMPLE_MARGIN) == 0:
            self.cfg.SAMPLE_MARGIN = int(self.cfg.NEIGHBOR_RADIUS)

    def _amp_ctx(self):
        """Combined inference_mode + (optional) bf16 autocast on CUDA."""
        stack = contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        if self.use_bf16:
            stack.enter_context(torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        return stack

    def infer_tile(self, tile_id: str, image_rgb: np.ndarray, bbox: Tuple[float, float, float, float], halo_px: int) -> TileResult:
        h, w = image_rgb.shape[:2]
        if min(h, w) < int(self.cfg.PATCH_SIZE):
            raise ValueError(f"Tile {tile_id}: tile size smaller than PATCH_SIZE.")

        net = self.net.module if isinstance(self.net, torch.nn.DataParallel) else self.net
        if self.legacy_patch_grid and h == w:
            all_patches = get_patch_info_one_img(
                0, h, int(self.cfg.SAMPLE_MARGIN),
                int(self.cfg.PATCH_SIZE), int(self.cfg.INFER_PATCHES_PER_EDGE),
            )
        else:
            all_patches = get_patch_info_one_img_rect(
                0, h, w, int(self.cfg.SAMPLE_MARGIN),
                int(self.cfg.PATCH_SIZE), int(self.cfg.INFER_PATCHES_PER_EDGE),
            )

        fused_bg = torch.zeros((h, w), device=self.device)
        fused_side = torch.zeros((h, w), device=self.device)
        fused_road = torch.zeros((h, w), device=self.device)
        fused_cross = torch.zeros((h, w), device=self.device)
        fused_walk = torch.zeros((h, w), device=self.device)
        pixel_cnt = torch.zeros((h, w), device=self.device)

        feats_per_patch: List[torch.Tensor] = []
        meta_per_patch: List[Tuple[int, int, int, int]] = []
        bs = int(self.cfg.INFER_BATCH_SIZE)

        for i in range(0, len(all_patches), bs):
            batch_info = all_patches[i : i + bs]
            pil_imgs = [
                Image.fromarray(image_rgb[y0:y1, x0:x1]) for _, (x0, y0), (x1, y1) in batch_info
            ]
            px = preprocess_patches(pil_imgs).to(self.device)
            if self._channels_last:
                px = px.contiguous(memory_format=torch.channels_last)

            with self._amp_ctx():
                mask_scores, feats = net.infer_masks_and_img_features(px)

            # Promote outputs back to fp32 (no-op if already) so accumulation
            # buffers stay in fp32. Clone so torch.compile's reduce-overhead
            # mode (CUDA graphs) cannot overwrite them on the next call.
            mask_scores = mask_scores.float().clone()
            feats = feats.float().clone()

            for j, (_, (x0, y0), (x1, y1)) in enumerate(batch_info):
                ph, pw = (y1 - y0), (x1 - x0)
                bg = F.interpolate(mask_scores[j, 0][None, None], size=(ph, pw), mode="bilinear", align_corners=False).squeeze()
                side = F.interpolate(mask_scores[j, 1][None, None], size=(ph, pw), mode="bilinear", align_corners=False).squeeze()
                road = F.interpolate(mask_scores[j, 2][None, None], size=(ph, pw), mode="bilinear", align_corners=False).squeeze()
                # 5-class (CityScale) merges crossing+zebra; 4-class (SF) is crossing only.
                if mask_scores.shape[1] >= 5:
                    cross = torch.maximum(mask_scores[j, 3], mask_scores[j, 4])
                else:
                    cross = mask_scores[j, 3]
                cross = F.interpolate(cross[None, None], size=(ph, pw), mode="bilinear", align_corners=False).squeeze()

                fused_bg[y0:y1, x0:x1] += bg
                fused_side[y0:y1, x0:x1] += side
                fused_road[y0:y1, x0:x1] += road
                fused_cross[y0:y1, x0:x1] += cross
                fused_walk[y0:y1, x0:x1] += side + cross
                pixel_cnt[y0:y1, x0:x1] += 1.0

                feats_per_patch.append(feats[j])
                meta_per_patch.append((x0, y0, x1, y1))

        pixel_cnt = torch.clamp(pixel_cnt, min=1.0)
        fused_walk = fused_walk / pixel_cnt
        fused_bg = fused_bg / pixel_cnt
        fused_side = fused_side / pixel_cnt
        fused_road = fused_road / pixel_cnt
        fused_cross = fused_cross / pixel_cnt

        fused_probs = torch.stack([fused_bg, fused_side, fused_road, fused_cross], dim=0)
        pred_mask = fused_probs.argmax(dim=0).byte().cpu().numpy()
        fused_u8 = (fused_walk.cpu() * 255).byte().cpu().numpy()
        side_u8 = (fused_side.cpu() * 255).byte().cpu().numpy()
        cross_u8 = (fused_cross.cpu() * 255).byte().cpu().numpy()

        nodes_rc, edges, edge_scores, edge_types, edge_is_synthetic = self._infer_topology(
            feats_per_patch, meta_per_patch, side_u8, cross_u8
        )
        core_x0, core_y0, core_x1, core_y1 = self._core_window(w, h, halo_px)
        nodes_rc_core, edges_core, scores_core, types_core, synthetic_core = self._keep_graph_in_core(
            nodes_rc,
            edges,
            edge_scores,
            edge_types,
            edge_is_synthetic,
            core_x0,
            core_y0,
            core_x1,
            core_y1,
        )
        # Drop small fragments before the cross-tile merge.
        nodes_rc_core, edges_core, scores_core, types_core, synthetic_core = (
            prune_keep_components_ge3_edges_with_metadata(
                nodes_rc_core, edges_core, scores_core, types_core, synthetic_core
            )
        )

        core_mask_u8 = fused_u8[core_y0:core_y1, core_x0:core_x1]
        core_pred_mask = pred_mask[core_y0:core_y1, core_x0:core_x1]

        nodes_global = self._nodes_to_global(
            nodes_rc_core, bbox, full_w=w, full_h=h, core_x0=core_x0, core_y0=core_y0,
        )
        return TileResult(
            tile_id=str(tile_id),
            core_mask_u8=core_mask_u8,
            core_pred_mask=core_pred_mask,
            nodes_rc_core=nodes_rc_core,
            edges=edges_core,
            edge_scores=scores_core,
            edge_types=types_core,
            edge_is_synthetic=synthetic_core,
            nodes_global_xy=nodes_global,
            bbox=bbox,
            core_shape_hw=core_mask_u8.shape[:2],
        )

    def _infer_topology(
        self,
        feats_per_patch: List[torch.Tensor],
        meta_per_patch: List[Tuple[int, int, int, int]],
        side_u8: np.ndarray,
        cross_u8: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the topology head per patch and return averaged per-edge scores.

        Two passes are made: one over sidewalk nodes only, and one over all
        nodes that keeps edges touching a crossing node (type 1). Edges below
        TOPO_THRESHOLD are dropped only to bound memory; the final threshold
        (TOPO_MERGE_THRESHOLD) is applied once after the cross-tile merge.
        """
        net = self.net.module if isinstance(self.net, torch.nn.DataParallel) else self.net
        g_side, g_cross = extract_graph_points_separate(cross_u8, side_u8, self.cfg)
        if g_side.shape[0] + g_cross.shape[0] == 0:
            return np.zeros((0, 2), np.int32), *_empty_edge_metadata()

        g_all = np.concatenate([g_side, g_cross], axis=0)
        n_side = g_side.shape[0]
        node_type = np.zeros((g_all.shape[0],), dtype=np.uint8)
        node_type[n_side:] = 1

        r_side = rtree.index.Index()
        for i, (x, y) in enumerate(g_side):
            r_side.insert(i, (x, y, x, y))
        r_all = rtree.index.Index()
        for i, (x, y) in enumerate(g_all):
            r_all.insert(i, (x, y, x, y))

        edge_score_side: Dict[Tuple[int, int], float] = collections.defaultdict(float)
        edge_count_side: Dict[Tuple[int, int], float] = collections.defaultdict(float)
        edge_score_mix: Dict[Tuple[int, int], float] = collections.defaultdict(float)
        edge_count_mix: Dict[Tuple[int, int], float] = collections.defaultdict(float)

        embed_dim = net.encoder_output_dim

        # Candidate graphs per patch for both passes.
        side_jobs: List[Tuple[int, List[int], Data]] = []  # (patch_idx, global_ids, data)
        mix_jobs: List[Tuple[int, List[int], Data]] = []
        for p_idx, (x0, y0, x1, y1) in enumerate(meta_per_patch):
            ids = list(r_side.intersection((x0, y0, x1, y1)))
            if len(ids) >= 2:
                pts_patch = g_side[ids] - np.array([[x0, y0]], dtype=np.float32)
                data = build_inference_pyg_graph(pts_patch, self.cfg.NEIGHBOR_RADIUS, self.cfg.MAX_NEIGHBOR_QUERIES, embed_dim)
                if data is not None:
                    side_jobs.append((p_idx, ids, data))

            ids = list(r_all.intersection((x0, y0, x1, y1)))
            if len(ids) >= 2:
                pts_patch = g_all[ids] - np.array([[x0, y0]], dtype=np.float32)
                data = build_inference_pyg_graph(pts_patch, self.cfg.NEIGHBOR_RADIUS, self.cfg.MAX_NEIGHBOR_QUERIES, embed_dim)
                if data is not None:
                    mix_jobs.append((p_idx, ids, data))

        def _run_chunked(jobs, score_dict, count_dict, mix_pass: bool) -> None:
            chunk = self.gnn_chunk
            for start in range(0, len(jobs), chunk):
                group = jobs[start:start + chunk]
                feats_stack = torch.stack(
                    [feats_per_patch[p_idx] for p_idx, _, _ in group], dim=0
                ).to(self.device, non_blocking=True)
                batch = Batch.from_data_list([d for _, _, d in group]).to(self.device)
                net._fill_node_features(batch, feats_stack)
                if hasattr(net, 'edge_visual_enc'):
                    net._fill_edge_features(batch, feats_stack)
                _, s = net.predict_patch_topo(batch)
                s_np = s.detach().float().cpu().numpy()
                eli = batch.edge_label_index.detach().cpu().numpy()  # [2, E]
                ptr = batch.ptr.detach().cpu().numpy()               # [B+1]
                node_batch = batch.batch.detach().cpu().numpy()      # [N]
                ids_per_graph = [ids for _, ids, _ in group]
                src = eli[0]
                dst = eli[1]
                for e in range(s_np.shape[0]):
                    u_g = int(src[e]); v_g = int(dst[e])
                    if u_g == v_g:
                        continue
                    gi = int(node_batch[u_g])
                    off = int(ptr[gi])
                    ids_local = ids_per_graph[gi]
                    gu = ids_local[u_g - off]
                    gv = ids_local[v_g - off]
                    if mix_pass and node_type[gu] == 0 and node_type[gv] == 0:
                        continue
                    if gu > gv:
                        gu, gv = gv, gu
                    score_dict[(gu, gv)] += float(s_np[e])
                    count_dict[(gu, gv)] += 1.0

        with self._amp_ctx():
            _run_chunked(side_jobs, edge_score_side, edge_count_side, mix_pass=False)
            _run_chunked(mix_jobs, edge_score_mix, edge_count_mix, mix_pass=True)

        # Average each edge over the patches that scored it; where the two
        # passes overlap, keep the higher probability.
        retain_min = float(getattr(self.cfg, 'TOPO_THRESHOLD', 0.05))
        edge_prob: Dict[Tuple[int, int], float] = {}
        edge_type: Dict[Tuple[int, int], int] = {}
        for (u, v), s_sum in edge_score_side.items():
            edge_prob[(u, v)] = float(s_sum / edge_count_side[(u, v)])
            edge_type[(u, v)] = 0  # sidewalk-only topology pass
        for (u, v), s_sum in edge_score_mix.items():
            p = float(s_sum / edge_count_mix[(u, v)])
            old_p = edge_prob.get((u, v), -1.0)
            if p > old_p:
                edge_prob[(u, v)] = p
                edge_type[(u, v)] = 1  # contains at least one crossing node
            elif p == old_p:
                edge_type[(u, v)] = 1

        kept = [
            (u, v, p, edge_type[(u, v)])
            for (u, v), p in edge_prob.items()
            if p >= retain_min
        ]
        kept.sort()
        if not kept:
            edges = np.zeros((0, 2), np.int32)
            scores = np.zeros((0,), np.float32)
            types = np.zeros((0,), np.uint8)
        else:
            edges = np.asarray([[u, v] for (u, v, _, _) in kept], dtype=np.int32)
            scores = np.asarray([p for (_, _, p, _) in kept], dtype=np.float32)
            types = np.asarray([link_type for (_, _, _, link_type) in kept], dtype=np.uint8)

        nodes_rc = g_all[:, ::-1].astype(np.int32)
        synthetic = np.zeros((len(edges),), dtype=bool)
        # Only isolated nodes are dropped here; the >=3-edge component filter
        # runs after core cropping in infer_tile so boundary fragments survive.
        return prune_isolated_nodes_with_metadata(nodes_rc, edges, scores, types, synthetic)

    @staticmethod
    def _core_window(w: int, h: int, halo_px: int) -> Tuple[int, int, int, int]:
        if halo_px <= 0:
            return 0, 0, w, h
        x0 = min(max(halo_px, 0), w - 1)
        y0 = min(max(halo_px, 0), h - 1)
        x1 = max(x0 + 1, w - halo_px)
        y1 = max(y0 + 1, h - halo_px)
        return int(x0), int(y0), int(x1), int(y1)

    @staticmethod
    def _keep_graph_in_core(
        nodes_rc: np.ndarray,
        edges: np.ndarray,
        scores: np.ndarray,
        edge_types: np.ndarray,
        edge_is_synthetic: np.ndarray,
        core_x0: int,
        core_y0: int,
        core_x1: int,
        core_y1: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _validate_edge_metadata(edges, scores, edge_types, edge_is_synthetic)
        if nodes_rc.shape[0] == 0:
            return nodes_rc.astype(np.int32), *_empty_edge_metadata()

        in_core = (
            (nodes_rc[:, 1] >= core_x0)
            & (nodes_rc[:, 1] < core_x1)
            & (nodes_rc[:, 0] >= core_y0)
            & (nodes_rc[:, 0] < core_y1)
        )

        # Boundary keep: nodes just outside the core that share an edge with an
        # in-core node would otherwise lose all their edges before the cross-tile
        # merge and never snap to neighbours in adjacent tiles.
        keep = in_core.copy()
        valid_edges = np.zeros((len(edges),), dtype=bool)
        edges_valid = np.zeros((0, 2), dtype=np.int32)
        if edges.size > 0:
            e = np.asarray(edges, dtype=np.int64)
            n = nodes_rc.shape[0]
            valid_edges = (
                (e[:, 0] >= 0)
                & (e[:, 0] < n)
                & (e[:, 1] >= 0)
                & (e[:, 1] < n)
            )
            edges_valid = e[valid_edges]
            if edges_valid.size > 0:
                has_core_neighbour = np.zeros(n, dtype=bool)
                in_core_u = in_core[edges_valid[:, 0]]
                in_core_v = in_core[edges_valid[:, 1]]
                has_core_neighbour[edges_valid[in_core_v, 0]] = True
                has_core_neighbour[edges_valid[in_core_u, 1]] = True
                keep |= has_core_neighbour

        kept_idx = np.where(keep)[0]
        if kept_idx.size == 0:
            return np.zeros((0, 2), np.int32), *_empty_edge_metadata()

        old2new = -np.ones(nodes_rc.shape[0], dtype=np.int64)
        old2new[kept_idx] = np.arange(kept_idx.shape[0], dtype=np.int64)
        nodes_new = nodes_rc[kept_idx].copy()
        nodes_new[:, 1] -= core_x0
        nodes_new[:, 0] -= core_y0
        if edges.size == 0:
            return nodes_new.astype(np.int32), *_empty_edge_metadata()

        keep_valid = keep[edges_valid[:, 0]] & keep[edges_valid[:, 1]]
        e_kept = edges_valid[keep_valid]
        return (
            nodes_new.astype(np.int32),
            old2new[e_kept].astype(np.int32),
            np.asarray(scores)[valid_edges][keep_valid].astype(np.float32),
            np.asarray(edge_types)[valid_edges][keep_valid].astype(np.uint8),
            np.asarray(edge_is_synthetic)[valid_edges][keep_valid].astype(bool),
        )

    @staticmethod
    def _nodes_to_global(
        nodes_rc: np.ndarray,
        bbox: Tuple[float, float, float, float],
        full_w: int,
        full_h: int,
        core_x0: int,
        core_y0: int,
    ) -> np.ndarray:
        if nodes_rc.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)
        minx, miny, maxx, maxy = map(float, bbox)
        abs_cols = (nodes_rc[:, 1] + core_x0).astype(np.float64)
        abs_rows = (nodes_rc[:, 0] + core_y0).astype(np.float64)
        xs = minx + ((abs_cols + 0.5) / max(1.0, float(full_w))) * (maxx - minx)
        ys = maxy - ((abs_rows + 0.5) / max(1.0, float(full_h))) * (maxy - miny)
        return np.stack([xs, ys], axis=1)


# ---------------------------------------------------------------------------
# Cross-tile merge.
# ---------------------------------------------------------------------------


def merge_global_graph(
    tile_results: Sequence[TileResult],
    snap_tol_m: float,
    bridge_tol_m: float = 0.0,
    topo_merge_threshold: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stitch per-tile graphs into one.

    Edges are deduplicated across overlapping patches/tiles by max score.
    A single `topo_merge_threshold` is applied once after node snapping.
    Synthetic bridge edges are appended after thresholding with score=NaN and an
    explicit synthetic flag because they were not evaluated by the model.
    """
    global_nodes = []
    global_edges = []
    global_scores = []
    global_types = []
    global_synthetic = []
    tile_id_per_pre: List[str] = []
    running = 0
    for tr in tile_results:
        global_nodes.append(tr.nodes_global_xy)
        tile_id_per_pre.extend([tr.tile_id] * tr.nodes_global_xy.shape[0])
        if tr.edges.size > 0:
            _validate_edge_metadata(
                tr.edges, tr.edge_scores, tr.edge_types, tr.edge_is_synthetic
            )
            global_edges.append(tr.edges + running)
            global_scores.append(tr.edge_scores)
            global_types.append(tr.edge_types)
            global_synthetic.append(tr.edge_is_synthetic)
        running += tr.nodes_global_xy.shape[0]

    if running == 0:
        return (
            np.zeros((0, 2), np.float64),
            *_empty_edge_metadata(),
        )

    nodes = np.vstack(global_nodes).astype(np.float64)
    if global_edges:
        edges = np.vstack(global_edges).astype(np.int64)
        scores = np.concatenate(global_scores).astype(np.float32)
        types = np.concatenate(global_types).astype(np.uint8)
        synthetic = np.concatenate(global_synthetic).astype(bool)
    else:
        edges, scores, types, synthetic = _empty_edge_metadata()

    metric_nodes = wgs84_to_local_meters(nodes)
    idx = rtree.index.Index()
    for i, (x, y) in enumerate(metric_nodes):
        idx.insert(i, (x, y, x, y))

    parent = np.arange(nodes.shape[0], dtype=np.int64)
    rank = np.zeros(nodes.shape[0], dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for i, (x, y) in enumerate(metric_nodes):
        for j in idx.intersection((x - snap_tol_m, y - snap_tol_m, x + snap_tol_m, y + snap_tol_m)):
            if j <= i:
                continue
            dx = metric_nodes[j, 0] - x
            dy = metric_nodes[j, 1] - y
            if (dx * dx + dy * dy) <= (snap_tol_m * snap_tol_m):
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(nodes.shape[0]):
        clusters.setdefault(find(i), []).append(i)

    rep_ids = sorted(clusters.keys())
    new_nodes = np.zeros((len(rep_ids), 2), dtype=np.float64)
    root2new: Dict[int, int] = {}
    for ni, r in enumerate(rep_ids):
        pts = nodes[np.array(clusters[r], dtype=np.int64)]
        new_nodes[ni] = pts.mean(axis=0)
        root2new[r] = ni

    # Confidence-aware edge dedup: max real score across all overlapping tiles.
    edge_max: Dict[Tuple[int, int], Tuple[float, int, bool]] = {}
    for (u, v), score, edge_type, is_synthetic in zip(
        edges, scores, types, synthetic
    ):
        nu = root2new[find(int(u))]
        nv = root2new[find(int(v))]
        if nu == nv:
            continue
        if nu > nv:
            nu, nv = nv, nu
        key = (int(nu), int(nv))
        candidate = (float(score), int(edge_type), bool(is_synthetic))
        previous = edge_max.get(key)
        if previous is None:
            edge_max[key] = candidate
        elif previous[2] and not candidate[2]:
            edge_max[key] = candidate
        elif previous[2] == candidate[2]:
            if candidate[2]:
                if candidate[1] > previous[1]:
                    edge_max[key] = candidate
            elif candidate[0] > previous[0] or (
                candidate[0] == previous[0] and candidate[1] > previous[1]
            ):
                edge_max[key] = candidate

    # The confidence threshold is applied once, after the merge.
    kept_pairs = [
        (key, value)
        for key, value in edge_max.items()
        if not value[2] and value[0] >= topo_merge_threshold
    ]
    if kept_pairs:
        kept_pairs.sort()
        merged_edges = np.asarray([k for k, _ in kept_pairs], dtype=np.int32)
        merged_scores = np.asarray([v[0] for _, v in kept_pairs], dtype=np.float32)
        merged_types = np.asarray([v[1] for _, v in kept_pairs], dtype=np.uint8)
        merged_synthetic = np.zeros((len(kept_pairs),), dtype=bool)
    else:
        merged_edges, merged_scores, merged_types, merged_synthetic = _empty_edge_metadata()

    # Cross-tile seam bridges are added after model-score thresholding.
    if bridge_tol_m > snap_tol_m and merged_edges.size > 0:
        origins: List[set] = [set() for _ in range(new_nodes.shape[0])]
        for pre_i in range(nodes.shape[0]):
            origins[root2new[find(pre_i)]].add(tile_id_per_pre[pre_i])

        has_edge = np.zeros(new_nodes.shape[0], dtype=bool)
        existing = set()
        for u, v in merged_edges:
            a, b = (int(u), int(v))
            if a > b:
                a, b = b, a
            existing.add((a, b))
            has_edge[a] = True
            has_edge[b] = True

        metric_new_nodes = wgs84_to_local_meters(new_nodes)
        idx2 = rtree.index.Index()
        for i, (x, y) in enumerate(metric_new_nodes):
            idx2.insert(i, (x, y, x, y))

        bridge_pairs: List[Tuple[int, int]] = []
        bt2 = bridge_tol_m * bridge_tol_m
        st2 = snap_tol_m * snap_tol_m
        for i in range(new_nodes.shape[0]):
            if not has_edge[i]:
                continue
            x, y = float(metric_new_nodes[i, 0]), float(metric_new_nodes[i, 1])
            for j in idx2.intersection((x - bridge_tol_m, y - bridge_tol_m, x + bridge_tol_m, y + bridge_tol_m)):
                if j <= i or not has_edge[j]:
                    continue
                key = (i, j)
                if key in existing or not origins[i].isdisjoint(origins[j]):
                    continue
                dx = float(metric_new_nodes[j, 0]) - x
                dy = float(metric_new_nodes[j, 1]) - y
                d2 = dx * dx + dy * dy
                if st2 < d2 <= bt2:
                    bridge_pairs.append(key)
                    existing.add(key)

        if bridge_pairs:
            extra = np.asarray(bridge_pairs, dtype=np.int32)
            extra_scores = np.full((len(bridge_pairs),), np.nan, dtype=np.float32)
            # Infer each seam bridge type from its incident links. Confidence-
            # weighted voting makes crossing fragments join as crossing while
            # keeping ordinary sidewalk seams labelled sidewalk.
            incident_votes = np.zeros((new_nodes.shape[0], 2), dtype=np.float64)
            for (u, v), score, edge_type in zip(merged_edges, merged_scores, merged_types):
                incident_votes[int(u), int(edge_type)] += float(score)
                incident_votes[int(v), int(edge_type)] += float(score)
            extra_types = []
            for u, v in bridge_pairs:
                votes = incident_votes[int(u)] + incident_votes[int(v)]
                extra_types.append(1 if votes[1] > votes[0] else 0)
            merged_edges = np.vstack([merged_edges, extra]).astype(np.int32)
            merged_scores = np.concatenate([merged_scores, extra_scores]).astype(np.float32)
            merged_types = np.concatenate(
                [merged_types, np.asarray(extra_types, dtype=np.uint8)]
            )
            merged_synthetic = np.concatenate(
                [merged_synthetic, np.ones((len(bridge_pairs),), dtype=bool)]
            )
            print(f"[merge] added {len(bridge_pairs)} cross-tile seam bridges (bridge_tol_m={bridge_tol_m})")

    return new_nodes, merged_edges, merged_scores, merged_types, merged_synthetic


# ---------------------------------------------------------------------------
# I/O.
# ---------------------------------------------------------------------------


def save_tile_result(out_dir: str, tr: TileResult, image_rgb: Optional[np.ndarray] = None, halo_px: int = 0) -> None:
    tile_dir = os.path.join(out_dir, "tiles", str(tr.tile_id))
    os.makedirs(tile_dir, exist_ok=True)
    cv2.imwrite(os.path.join(tile_dir, "walkmask_core.png"), tr.core_mask_u8)
    np.save(os.path.join(tile_dir, "pred_mask_core.npy"), tr.core_pred_mask)
    _validate_edge_metadata(
        tr.edges, tr.edge_scores, tr.edge_types, tr.edge_is_synthetic
    )
    graph_arrays = dict(
        nodes_rc=tr.nodes_rc_core.astype(np.int32),
        edges=tr.edges.astype(np.int32),
        edge_scores=tr.edge_scores.astype(np.float32),
        edge_probs=tr.edge_scores.astype(np.float32),
        edge_types=tr.edge_types.astype(np.uint8),
        edge_is_synthetic=tr.edge_is_synthetic.astype(bool),
        nodes_global_xy=tr.nodes_global_xy.astype(np.float64),
    )
    np.savez_compressed(os.path.join(tile_dir, "graph_core.npz"), **graph_arrays)
    meta = {
        "tile_id": tr.tile_id,
        "bbox": list(map(float, tr.bbox)),
        "core_shape_hw": [int(tr.core_shape_hw[0]), int(tr.core_shape_hw[1])],
        "edge_type_labels": {"0": "sidewalk", "1": "crossing"},
    }
    with open(os.path.join(tile_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    sat2graph = (
        graph_utils.convert_to_sat2graph_format(tr.nodes_rc_core.astype(np.int32), tr.edges.astype(np.int32))
        if tr.nodes_rc_core.size
        else {}
    )
    with open(os.path.join(tile_dir, "graph_core.p"), "wb") as f:
        pickle.dump(sat2graph, f)

    if image_rgb is not None:
        h_img, w_img = image_rgb.shape[:2]
        core_x0, core_y0, core_x1, core_y1 = CityScaleInferencer._core_window(w_img, h_img, halo_px)
        viz = image_rgb[core_y0:core_y1, core_x0:core_x1].copy()
        for (u, v), score in zip(tr.edges, tr.edge_scores):
            confidence = float(np.clip(score, 0.0, 1.0))
            color = (int(255 * (1.0 - confidence)), int(255 * confidence), 0)
            cv2.line(
                viz,
                (int(tr.nodes_rc_core[u, 1]), int(tr.nodes_rc_core[u, 0])),
                (int(tr.nodes_rc_core[v, 1]), int(tr.nodes_rc_core[v, 0])),
                color, 2,
            )
        for r, c in tr.nodes_rc_core:
            cv2.circle(viz, (int(c), int(r)), 5, (255, 230, 0), -1)
        cv2.imwrite(
            os.path.join(tile_dir, "graph_viz.png"),
            cv2.cvtColor(viz, cv2.COLOR_RGB2BGR),
        )


def _confidence_bin(score: float) -> str:
    if score < 0.25:
        return "very_low"
    if score < 0.50:
        return "low"
    if score < 0.70:
        return "med"
    if score < 0.85:
        return "high"
    return "very_high"


def save_global_graph(
    out_dir: str,
    nodes_xy: np.ndarray,
    edges: np.ndarray,
    edge_scores: np.ndarray,
    edge_types: np.ndarray,
    edge_is_synthetic: np.ndarray,
) -> None:
    gdir = os.path.join(out_dir, "global")
    os.makedirs(gdir, exist_ok=True)
    _validate_edge_metadata(edges, edge_scores, edge_types, edge_is_synthetic)
    edge_scores = np.asarray(edge_scores, dtype=np.float32)
    edge_types = np.asarray(edge_types, dtype=np.uint8)
    edge_is_synthetic = np.asarray(edge_is_synthetic, dtype=bool)
    graph_arrays = dict(
        nodes_xy=nodes_xy.astype(np.float64),
        edges=edges.astype(np.int32),
        edge_scores=edge_scores,
        edge_probs=edge_scores,
        edge_types=edge_types,
        edge_type_names=np.asarray(["sidewalk", "crossing"]),
        edge_is_synthetic=edge_is_synthetic,
    )
    np.savez_compressed(os.path.join(gdir, "graph_merged.npz"), **graph_arrays)

    node_features: List[dict] = []
    for i, (x, y) in enumerate(nodes_xy):
        node_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
            "properties": {"type": "node", "id": int(i)},
        })

    edge_features: List[dict] = []
    for (u, v), score, edge_type, is_synthetic in zip(
        edges, edge_scores, edge_types, edge_is_synthetic
    ):
        synthetic = bool(is_synthetic)
        confidence = None if synthetic or not np.isfinite(score) else float(score)
        properties = {
            "type": "edge",
            "src": int(u),
            "dst": int(v),
            "link_type": "crossing" if int(edge_type) == 1 else "sidewalk",
            "confidence": confidence,
            "confidence_bin": "synthetic" if synthetic else _confidence_bin(float(score)),
            "synthetic": synthetic,
        }
        edge_features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(nodes_xy[u, 0]), float(nodes_xy[u, 1])],
                    [float(nodes_xy[v, 0]), float(nodes_xy[v, 1])],
                ],
            },
            "properties": properties,
        })

    edges_geojson = {"type": "FeatureCollection", "features": edge_features}
    with open(os.path.join(gdir, "graph_merged_edges.geojson"), "w", encoding="utf-8") as f:
        json.dump(edges_geojson, f)

    nodes_geojson = {"type": "FeatureCollection", "features": node_features}
    with open(os.path.join(gdir, "graph_merged_nodes.geojson"), "w", encoding="utf-8") as f:
        json.dump(nodes_geojson, f)

    geojson = {
        "type": "FeatureCollection",
        "features": node_features + edge_features,
    }
    with open(os.path.join(gdir, "graph_merged.geojson"), "w", encoding="utf-8") as f:
        json.dump(geojson, f)


def load_resume_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"done": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_resume_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("City-scale SidewalkFormer inference")
    p.add_argument("--config", required=True, help="Path to model config yaml")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path (.ckpt)")
    p.add_argument("--manifest", required=True, help="Tile manifest (.json or .csv)")
    p.add_argument("--output_dir", required=True, help="Output directory")
    p.add_argument("--halo_px", type=int, default=0,
                   help="Pixels cropped from each tile side (use 0 for non-overlapping tiles)")
    p.add_argument("--snap_tol_m", type=float, default=2.0,
                   help="Node snap tolerance in metres (default: 2.0). Nodes closer than this are collapsed.")
    p.add_argument("--bridge_tol_m", type=float, default=-1.0,
                   help="Add a synthetic edge between cross-tile node pairs within this many metres "
                        "that did not snap. Default -1 auto-sets to 3x snap_tol_m; set 0 to disable.")
    p.add_argument("--topo_merge_threshold", type=float, default=None,
                   help="Single confidence threshold applied after merging (default from config "
                        "TOPO_MERGE_THRESHOLD or 0.6). Sweep on a held-out subset.")
    p.add_argument("--prob_min", type=float, default=None,
                   help="Per-tile edge-probability retention floor (overrides config "
                        "TOPO_THRESHOLD). Lower it, together with --topo_merge_threshold, "
                        "to keep low-confidence edges in the outputs.")
    p.add_argument("--junction_snap_tol_m", type=float, default=None,
                   help="Snap radius in metres for high-degree node clusters after merge. "
                        "Default: 3.0; set to 0 to disable.")
    p.add_argument("--device", default="auto", help="auto|cuda|cpu")
    p.add_argument("--sample_from_centerline", dest="sample_from_centerline",
                   action="store_true", default=None,
                   help="Sample candidate nodes from the mask skeleton/centerline "
                        "(medial axis) instead of every foreground pixel. Overrides "
                        "config SAMPLE_FROM_CENTERLINE when passed.")
    p.add_argument("--centerline_min_blob_px", type=int, default=None,
                   help="When sampling from centerline, drop connected mask blobs "
                        "smaller than this many pixels before skeletonizing "
                        "(0 = keep all). Overrides config CENTERLINE_MIN_BLOB_PX.")
    p.add_argument("--centerline_thickness_px", type=int, default=None,
                   help="Half-width (px) to widen the 1-px skeleton to: 0 = strict "
                        "1-px centerline, N>0 = centred ribbon ~(2N+1) px wide. "
                        "Overrides config CENTERLINE_THICKNESS_PX.")
    p.add_argument("--skip_merged_graph", action="store_true", help="Only run per-tile outputs, skip global merge")
    p.add_argument("--no-bf16", dest="no_bf16", action="store_true",
                   help="Disable bf16 autocast on CUDA (debugging numerics).")
    p.add_argument("--no-compile", dest="no_compile", action="store_true",
                   help="Disable torch.compile on the segmentation forward (debugging).")
    p.add_argument("--gnn-chunk", type=int, default=16,
                   help="Number of patches batched into a single GNN forward (default 16). "
                        "Set 1 to run patches one at a time.")
    return p.parse_args()


def _validate_record(rec: dict) -> None:
    for key in ("tile_id", "image_path", "bbox"):
        if key not in rec:
            raise ValueError(f"Manifest row missing key: {key}")
    if not os.path.exists(rec["image_path"]):
        raise FileNotFoundError(f"Missing image path: {rec['image_path']}")
    if len(rec["bbox"]) != 4:
        raise ValueError(f"bbox must have 4 values for tile {rec['tile_id']}")


def _load_resumed_tile(tile_dir: str, rec: dict) -> TileResult:
    meta_path = os.path.join(tile_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        core_hw = tuple(meta.get("core_shape_hw", [0, 0]))
        bbox = tuple(meta.get("bbox", rec["bbox"]))
    else:
        core_hw = (0, 0)
        bbox = tuple(rec["bbox"])
    with np.load(os.path.join(tile_dir, "graph_core.npz")) as g:
        edges_arr = g["edges"].astype(np.int32)
        if "edge_scores" in g.files:
            scores_arr = g["edge_scores"].astype(np.float32)
        elif "edge_probs" in g.files:
            scores_arr = g["edge_probs"].astype(np.float32)
        elif len(edges_arr) == 0:
            scores_arr = np.zeros((0,), dtype=np.float32)
        else:
            raise ValueError("legacy graph_core.npz has no edge_scores or edge_probs")

        if "edge_types" in g.files:
            types_arr = g["edge_types"].astype(np.uint8)
        elif len(edges_arr) == 0:
            types_arr = np.zeros((0,), dtype=np.uint8)
        else:
            raise ValueError("legacy graph_core.npz has no edge_types")

        synthetic_arr = (
            g["edge_is_synthetic"].astype(bool)
            if "edge_is_synthetic" in g.files
            else np.zeros((len(edges_arr),), dtype=bool)
        )
        nodes_rc = g["nodes_rc"].astype(np.int32)
        nodes_global_xy = g["nodes_global_xy"].astype(np.float64)

    walkmask_path = os.path.join(tile_dir, "walkmask_core.png")
    pred_mask_path = os.path.join(tile_dir, "pred_mask_core.npy")
    core_mask = cv2.imread(walkmask_path, cv2.IMREAD_GRAYSCALE)
    if core_mask is None:
        core_mask = np.zeros(tuple(map(int, core_hw)), dtype=np.uint8)
    core_pred_mask = (
        np.load(pred_mask_path)
        if os.path.exists(pred_mask_path)
        else np.zeros(tuple(map(int, core_hw)), dtype=np.uint8)
    )
    _validate_edge_metadata(edges_arr, scores_arr, types_arr, synthetic_arr)
    return TileResult(
        tile_id=str(rec["tile_id"]),
        core_mask_u8=core_mask,
        core_pred_mask=core_pred_mask,
        nodes_rc_core=nodes_rc,
        edges=edges_arr,
        edge_scores=scores_arr,
        edge_types=types_arr,
        edge_is_synthetic=synthetic_arr,
        nodes_global_xy=nodes_global_xy,
        bbox=tuple(map(float, bbox)),
        core_shape_hw=tuple(map(int, core_hw)),
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    state_path = os.path.join(args.output_dir, "resume_state.json")
    resume_state = load_resume_state(state_path)
    done = set(resume_state.get("done", []))

    cfg = load_config(args.config)
    if args.topo_merge_threshold is not None:
        cfg.TOPO_MERGE_THRESHOLD = float(args.topo_merge_threshold)
    if args.prob_min is not None:
        cfg.TOPO_THRESHOLD = float(args.prob_min)
    if args.junction_snap_tol_m is not None:
        cfg.JUNCTION_SNAP_TOL_M = float(args.junction_snap_tol_m)
    if args.sample_from_centerline is not None:
        cfg.SAMPLE_FROM_CENTERLINE = bool(args.sample_from_centerline)
    if args.centerline_min_blob_px is not None:
        cfg.CENTERLINE_MIN_BLOB_PX = int(args.centerline_min_blob_px)
    if args.centerline_thickness_px is not None:
        cfg.CENTERLINE_THICKNESS_PX = int(args.centerline_thickness_px)
    if bool(getattr(cfg, "SAMPLE_FROM_CENTERLINE", False)):
        print(f"[nodes] sampling from centerline (skeleton)  "
              f"min_blob_px={int(getattr(cfg, 'CENTERLINE_MIN_BLOB_PX', 0))}  "
              f"thickness_px={int(getattr(cfg, 'CENTERLINE_THICKNESS_PX', 0))}")

    device = choose_device(args.device)
    inferencer = CityScaleInferencer(
        cfg, args.checkpoint, device=device,
        use_bf16=not args.no_bf16,
        use_compile=not args.no_compile,
        gnn_chunk=int(args.gnn_chunk),
    )
    manifest = read_manifest(args.manifest)

    tile_results: List[TileResult] = []
    for rec in manifest:
        _validate_record(rec)
        tile_id = str(rec["tile_id"])
        tile_dir = os.path.join(args.output_dir, "tiles", tile_id)

        if tile_id in done and os.path.exists(os.path.join(tile_dir, "graph_core.npz")):
            try:
                tile_results.append(_load_resumed_tile(tile_dir, rec))
                print(f"[skip] tile {tile_id} already completed.")
                continue
            except ValueError as exc:
                print(f"[resume] tile {tile_id} requires one-time reprocessing: {exc}")

        img_bgr = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {rec['image_path']}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        bbox = tuple(map(float, rec["bbox"]))
        print(f"[run] tile {tile_id}  image={rec['image_path']}")
        tr = inferencer.infer_tile(tile_id=tile_id, image_rgb=img_rgb, bbox=bbox, halo_px=args.halo_px)
        save_tile_result(args.output_dir, tr, image_rgb=img_rgb, halo_px=args.halo_px)
        tile_results.append(tr)

        done.add(tile_id)
        resume_state["done"] = sorted(done)
        save_resume_state(state_path, resume_state)

    if args.skip_merged_graph:
        print("[done] per-tile inference completed. Global merge skipped.")
        return

    bridge_tol_m = args.bridge_tol_m if args.bridge_tol_m >= 0.0 else 3.0 * args.snap_tol_m
    topo_thresh = float(getattr(cfg, "TOPO_MERGE_THRESHOLD", 0.6))
    print(
        f"[merge] building global stitched graph  snap_tol_m={args.snap_tol_m:.2f}  "
        f"bridge_tol_m={bridge_tol_m:.2f}  topo_merge_threshold={topo_thresh:.2f} ..."
    )
    nodes_xy, edges, edge_scores, edge_types, edge_is_synthetic = merge_global_graph(
        tile_results,
        snap_tol_m=args.snap_tol_m,
        bridge_tol_m=bridge_tol_m,
        topo_merge_threshold=topo_thresh,
    )

    # Collapse high-degree node clusters within JUNCTION_SNAP_TOL_M.
    junction_tol_m = float(cfg.get("JUNCTION_SNAP_TOL_M", 3.0))
    if junction_tol_m > 0 and edges.shape[0] > 0:
        before_n, before_e = nodes_xy.shape[0], edges.shape[0]
        nodes_xy, edges, edge_scores, edge_types, edge_is_synthetic = snap_high_degree_clusters(
            nodes_xy,
            edges,
            radius_m=junction_tol_m,
            edge_scores=edge_scores,
            edge_types=edge_types,
            edge_is_synthetic=edge_is_synthetic,
        )
        print(
            f"[snap] junction snap radius_m={junction_tol_m:.2f}: "
            f"nodes {before_n} → {nodes_xy.shape[0]}, edges {before_e} → {edges.shape[0]}"
        )

    save_global_graph(
        args.output_dir,
        nodes_xy,
        edges,
        edge_scores,
        edge_types,
        edge_is_synthetic,
    )
    real_scores = edge_scores[~edge_is_synthetic]
    score_summary = ""
    if real_scores.size:
        score_summary = (
            f" prob[min/median/max]={real_scores.min():.3f}/"
            f"{np.median(real_scores):.3f}/{real_scores.max():.3f}"
        )
    print(
        f"[done] merged graph nodes={nodes_xy.shape[0]} edges={edges.shape[0]}"
        f"{score_summary}"
    )


if __name__ == "__main__":
    main()
