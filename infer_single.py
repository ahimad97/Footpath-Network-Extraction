"""Single-tile inference for inspection and debugging.

Runs each selected manifest tile independently (no cross-tile merge) and
writes per-tile masks, a pixel-space graph pickle and an overlay image. Edges
are thresholded directly at TOPO_THRESHOLD. Also supports the ``tile2net``
baseline, whose graph comes from Tile2Net post-processing of the mask.
"""

import argparse
import collections
import json
import os
import pickle
import time

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rtree
import torch
import torch.nn.functional as F
import torchvision.transforms as tvT
import yaml
from PIL import Image
from rasterio.features import shapes as rio_shapes
from rasterio.transform import from_bounds as rio_from_bounds
from shapely.geometry import shape as shp_shape
from torch_geometric.data import Batch

from sidewalkformer import graph_utils
from sidewalkformer.dataset import build_inference_pyg_graph, get_patch_info_one_img
from sidewalkformer.graph_extraction import get_points_and_scores_from_mask, nms_points
from sidewalkformer.model import SidewalkFormer
from sidewalkformer.tile2net_postprocess import (
    metric_coords_to_pixel,
    seg_mask_to_graph as t2n_seg_mask_to_graph,
)
from sidewalkformer.utils import cfg_get, load_config


_IMAGENET_TRANSFORM = tvT.Compose([
    tvT.ToTensor(),
    tvT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _preprocess_patches(pil_imgs):
    """List of PIL images -> ImageNet-normalised tensor [B,3,H,W].

    All variants take ImageNet-normalised input; SAM-based models re-normalise
    internally.
    """
    return torch.stack([_IMAGENET_TRANSFORM(img) for img in pil_imgs], dim=0)


def _polygons_gdf_from_mask(pred_mask, bbox, class_map=None):
    """Convert a class-labeled mask to polygons in EPSG:4326 using `bbox`."""
    if class_map is None:
        class_map = {1: "sidewalk", 2: "road", 3: "crosswalk"}

    H, W = pred_mask.shape
    minx, miny, maxx, maxy = map(float, bbox)
    transform = rio_from_bounds(minx, miny, maxx, maxy, W, H)

    rows_ftype = []
    rows_geom = []
    for cid, fname in class_map.items():
        bmask = (pred_mask == cid).astype("uint8")
        if bmask.max() == 0:
            continue
        for geom, _ in rio_shapes(bmask, mask=bmask.astype(bool), transform=transform):
            g = shp_shape(geom)
            if g.is_empty or not g.is_valid:
                continue
            rows_ftype.append(fname)
            rows_geom.append(g)

    return gpd.GeoDataFrame({"f_type": rows_ftype, "geometry": rows_geom}, crs="EPSG:4326")


def prune_keep_components_ge3_edges(nodes_rc, edges):
    """Keep only connected components with >= 3 edges. Drops isolated nodes too.

    Returns (pruned_nodes_rc, pruned_edges) with reindexed node IDs.
    """
    if nodes_rc.size == 0:
        return nodes_rc.astype(np.int32), np.zeros((0, 2), np.int32)

    E = np.asarray(edges, dtype=np.int64)
    N = int(nodes_rc.shape[0])
    if E.size == 0:
        return np.zeros((0, 2), np.int32), np.zeros((0, 2), np.int32)

    valid = (E[:, 0] >= 0) & (E[:, 0] < N) & (E[:, 1] >= 0) & (E[:, 1] < N)
    E = E[valid]
    if E.size == 0:
        return np.zeros((0, 2), np.int32), np.zeros((0, 2), np.int32)

    incident = np.zeros(N, dtype=bool)
    incident[E[:, 0]] = True
    incident[E[:, 1]] = True

    adj = [[] for _ in range(N)]
    for u, v in E:
        adj[u].append(int(v))
        if u != v:
            adj[v].append(int(u))

    comp_id = -np.ones(N, dtype=np.int64)
    comp_nodes = []
    cid = 0
    for s in range(N):
        if not incident[s] or comp_id[s] != -1:
            continue
        stack = [s]
        comp = []
        comp_id[s] = cid
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
        return np.zeros((0, 2), np.int32), np.zeros((0, 2), np.int32)

    comp_edge_count = np.zeros(cid, dtype=np.int64)
    for u, v in E:
        cu, cv = comp_id[u], comp_id[v]
        if cu >= 0 and cu == cv:
            comp_edge_count[cu] += 1

    keep_comp = comp_edge_count >= 3
    keep_nodes_mask = np.zeros(N, dtype=bool)
    for c, nodes in enumerate(comp_nodes):
        if keep_comp[c]:
            keep_nodes_mask[nodes] = True

    kept_idx = np.nonzero(keep_nodes_mask)[0]
    if kept_idx.size == 0:
        return np.zeros((0, 2), np.int32), np.zeros((0, 2), np.int32)

    old2new = -np.ones(N, dtype=np.int64)
    old2new[kept_idx] = np.arange(kept_idx.size, dtype=np.int64)
    pruned_nodes = nodes_rc[kept_idx].astype(np.int32)
    keep_edge_mask = keep_nodes_mask[E[:, 0]] & keep_nodes_mask[E[:, 1]]
    pruned_edges = old2new[E[keep_edge_mask]].astype(np.int32)
    return pruned_nodes, pruned_edges


def extract_graph_points_separate(cross_mask_u8, side_mask_u8, cfg):
    cand, sc = get_points_and_scores_from_mask(cross_mask_u8, cfg.CROSSING_THRESHOLD * 255)
    kps_cross = nms_points(cand, sc, radius=cfg.CROSSING_NMS_RADIUS)

    cand, sc = get_points_and_scores_from_mask(side_mask_u8, cfg.SIDEWALK_THRESHOLD * 255)
    kps_side = nms_points(cand, sc, radius=cfg.SIDEWALK_NMS_RADIUS)

    return kps_side.astype(np.float32), kps_cross.astype(np.float32)


def infer_one_img(net, img_rgb, cfg, device):
    """Run segmentation + topology on one image.

    Returns: (pred_nodes_rc, pred_edges, fused_walk_mask_uint8, pred_mask, pred_gdf)
    """
    net = net.module if isinstance(net, torch.nn.DataParallel) else net
    model_type = getattr(cfg, 'MODEL_TYPE', 'segformer')
    force_t2n_post = getattr(cfg, 'T2N_POSTPROCESS', False)

    H, W = img_rgb.shape[:2]
    all_patches = get_patch_info_one_img(
        0, H, cfg.SAMPLE_MARGIN, cfg.PATCH_SIZE, cfg.INFER_PATCHES_PER_EDGE,
    )

    fused_bg = torch.zeros((H, W), device=device)
    fused_road = torch.zeros((H, W), device=device)
    fused_sidewalk = torch.zeros((H, W), device=device)
    fused_crossing = torch.zeros((H, W), device=device)
    fused_walk = torch.zeros((H, W), device=device)
    pixel_cnt = torch.zeros_like(fused_sidewalk)

    # Per-patch feature maps are kept for the topology stage; set
    # OFFLOAD_PATCH_FEATURES_TO_CPU to hold them in host memory on small GPUs.
    offload_patch_feats = cfg_get(cfg, "OFFLOAD_PATCH_FEATURES_TO_CPU", False)
    feats_per_patch, meta_per_patch = [], []

    bs = cfg.INFER_BATCH_SIZE
    for i in range(0, len(all_patches), bs):
        batch_info = all_patches[i:i + bs]
        pil_imgs = [
            Image.fromarray(img_rgb[y0:y1, x0:x1])
            for _, (x0, y0), (x1, y1) in batch_info
        ]
        px = _preprocess_patches(pil_imgs).to(device)

        with torch.no_grad():
            mask_scores, feats = net.infer_masks_and_img_features(px)

        for j, (_, (x0, y0), (x1, y1)) in enumerate(batch_info):
            meta_per_patch.append((x0, y0, x1, y1))
            patch_feat = feats[j].detach()
            if offload_patch_feats:
                patch_feat = patch_feat.cpu()
            feats_per_patch.append(patch_feat)

            bg_logits = mask_scores[j, 0]
            cls_sidewalk = mask_scores[j, 1]
            road_logits = mask_scores[j, 2]
            if mask_scores.shape[1] >= 5:
                cls_crossing = torch.maximum(mask_scores[j, 3], mask_scores[j, 4])
            else:
                cls_crossing = mask_scores[j, 3]

            def upsample(cls_map):
                return F.interpolate(
                    cls_map[None, None],
                    size=(y1 - y0, x1 - x0),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            fused_bg[y0:y1, x0:x1] += upsample(bg_logits)
            fused_road[y0:y1, x0:x1] += upsample(road_logits)
            fused_sidewalk[y0:y1, x0:x1] += upsample(cls_sidewalk)
            fused_crossing[y0:y1, x0:x1] += upsample(cls_crossing)
            fused_walk[y0:y1, x0:x1] += upsample(cls_sidewalk)
            fused_walk[y0:y1, x0:x1] += upsample(cls_crossing)
            pixel_cnt[y0:y1, x0:x1] += 1

        del px, mask_scores, feats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pixel_cnt[pixel_cnt == 0] = 1.0
    fused_walk /= pixel_cnt
    fused_bg /= pixel_cnt
    fused_road /= pixel_cnt
    fused_sidewalk /= pixel_cnt
    fused_crossing /= pixel_cnt

    fused_u8 = (fused_walk.cpu() * 255).byte().cpu().numpy()
    fused_probs = torch.stack(
        [fused_bg, fused_sidewalk, fused_road, fused_crossing], dim=0
    ).float().cpu().numpy()
    pred_mask = fused_probs.argmax(axis=0).astype(np.uint8)
    sidewalk_u8 = (fused_sidewalk.cpu() * 255).byte().cpu().numpy()
    crossing_u8 = (fused_crossing.cpu() * 255).byte().cpu().numpy()

    # Polygon export is opt-in (--write_polygons).
    pred_gdf = gpd.GeoDataFrame({"f_type": [], "geometry": []}, crs="EPSG:4326")
    if getattr(cfg, "EXPORT_POLYGONS", False):
        bbox = getattr(cfg, "TILE_BBOX", None)
        if bbox is not None:
            try:
                pred_mask_poly = pred_mask.copy()
                pred_mask_poly[pred_mask_poly == 4] = 3
                pred_gdf = _polygons_gdf_from_mask(pred_mask_poly, bbox)
            except Exception as e:
                print(f"[poly-export] Polygonization failed: {e}")

    # tile2net baseline (or T2N_POSTPROCESS): graph from Tile2Net mask ->
    # polygon -> centreline post-processing instead of the topology head.
    if model_type == 'tile2net' or force_t2n_post:
        bbox = getattr(cfg, 'TILE_BBOX', None)
        snap_tol = cfg_get(cfg, 'T2N_SNAP_TOL', 5.0)
        if bbox is not None:
            try:
                pred_mask_t2n = pred_mask.copy()
                pred_mask_t2n[pred_mask_t2n == 4] = 3
                nodes_metric, edges, _ = t2n_seg_mask_to_graph(
                    pred_mask_t2n, bbox, snap_tol=snap_tol,
                )
                nodes_rc = metric_coords_to_pixel(nodes_metric, bbox, img_rgb.shape[:2])
                edges = edges.astype(np.int32) if len(edges) > 0 else np.zeros((0, 2), np.int32)
            except Exception as e:
                print(f"[tile2net-postprocess] Failed: {e}")
                nodes_rc = np.zeros((0, 2), np.int32)
                edges = np.zeros((0, 2), np.int32)
        else:
            print("[tile2net-postprocess] No TILE_BBOX — returning empty graph.")
            nodes_rc = np.zeros((0, 2), np.int32)
            edges = np.zeros((0, 2), np.int32)
        return nodes_rc, edges, fused_u8, pred_mask, pred_gdf

    # Topology head: sidewalk-only nodes first, then all nodes (crossing edges).
    graph_pts_side, graph_pts_cross = extract_graph_points_separate(
        crossing_u8, sidewalk_u8, cfg,
    )
    if graph_pts_side.shape[0] + graph_pts_cross.shape[0] == 0:
        return (
            np.zeros((0, 2), np.int32),
            np.zeros((0, 2), np.int32),
            fused_u8,
            pred_mask,
            pred_gdf,
        )

    graph_pts_all = np.concatenate([graph_pts_side, graph_pts_cross], axis=0)
    n_side = graph_pts_side.shape[0]
    node_type = np.zeros((graph_pts_all.shape[0],), dtype=np.uint8)
    node_type[n_side:] = 1

    r_side = rtree.index.Index()
    for i, (x, y) in enumerate(graph_pts_side):
        r_side.insert(i, (x, y, x, y))
    r_all = rtree.index.Index()
    for i, (x, y) in enumerate(graph_pts_all):
        r_all.insert(i, (x, y, x, y))

    edge_score_side = collections.defaultdict(float)
    edge_count_side = collections.defaultdict(float)
    edge_score_mix = collections.defaultdict(float)
    edge_count_mix = collections.defaultdict(float)
    embed_dim = net.encoder_output_dim

    # Stage 1: sidewalk-only edges.
    with torch.no_grad():
        for feat, (x0, y0, x1, y1) in zip(feats_per_patch, meta_per_patch):
            ids = list(r_side.intersection((x0, y0, x1, y1)))
            if len(ids) < 2:
                continue
            pts_patch = graph_pts_side[ids] - np.array([[x0, y0]], dtype=np.float32)
            data = build_inference_pyg_graph(pts_patch, cfg.NEIGHBOR_RADIUS, cfg.MAX_NEIGHBOR_QUERIES, embed_dim)
            if data is None:
                continue
            batch = Batch.from_data_list([data]).to(device)
            feat = feat.unsqueeze(0).to(device, non_blocking=True)
            net._fill_node_features(batch, feat)
            if hasattr(net, 'edge_visual_enc'):
                net._fill_edge_features(batch, feat)
            _, s = net.predict_patch_topo(batch)
            s = s.cpu().numpy()
            eidx = batch.edge_label_index.cpu().numpy().T
            for (u, v), score in zip(eidx, s):
                if u == v:
                    continue
                gu, gv = ids[u], ids[v]
                if gu > gv:
                    gu, gv = gv, gu
                edge_score_side[(gu, gv)] += float(score)
                edge_count_side[(gu, gv)] += 1
            del batch, feat, s, eidx
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Stage 2: edges touching a crossing node (sidewalk-sidewalk handled in stage 1).
    with torch.no_grad():
        for feat, (x0, y0, x1, y1) in zip(feats_per_patch, meta_per_patch):
            ids = list(r_all.intersection((x0, y0, x1, y1)))
            if len(ids) < 2:
                continue
            pts_patch = graph_pts_all[ids] - np.array([[x0, y0]], dtype=np.float32)
            data = build_inference_pyg_graph(pts_patch, cfg.NEIGHBOR_RADIUS, cfg.MAX_NEIGHBOR_QUERIES, embed_dim)
            if data is None:
                continue
            batch = Batch.from_data_list([data]).to(device)
            feat = feat.unsqueeze(0).to(device, non_blocking=True)
            net._fill_node_features(batch, feat)
            if hasattr(net, 'edge_visual_enc'):
                net._fill_edge_features(batch, feat)
            _, s = net.predict_patch_topo(batch)
            s = s.cpu().numpy()
            eidx = batch.edge_label_index.cpu().numpy().T
            for (u, v), score in zip(eidx, s):
                if u == v:
                    continue
                gu, gv = ids[u], ids[v]
                if node_type[gu] == 0 and node_type[gv] == 0:
                    continue
                if gu > gv:
                    gu, gv = gv, gu
                edge_score_mix[(gu, gv)] += float(score)
                edge_count_mix[(gu, gv)] += 1
            del batch, feat, s, eidx
            if device.type == "cuda":
                torch.cuda.empty_cache()

    edges_side = [
        (u, v) for (u, v), sc_sum in edge_score_side.items()
        if (sc_sum / edge_count_side[(u, v)]) >= cfg.TOPO_THRESHOLD
    ]
    edges_mix = [
        (u, v) for (u, v), sc_sum in edge_score_mix.items()
        if (sc_sum / edge_count_mix[(u, v)]) >= cfg.TOPO_THRESHOLD
    ]
    pred_edges = np.asarray(sorted(set(edges_side) | set(edges_mix)), dtype=np.int32)
    pred_nodes_rc = graph_pts_all[:, ::-1].astype(np.int32)  # xy -> rc
    pred_nodes_rc, pred_edges = prune_keep_components_ge3_edges(pred_nodes_rc, pred_edges)
    return pred_nodes_rc, pred_edges, fused_u8, pred_mask, pred_gdf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser("Single-tile SidewalkFormer inference")
    parser.add_argument("--config", default="config/sidewalkformer.yaml")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path (.ckpt)")
    parser.add_argument("--manifest", required=True,
                        help="JSON manifest with tile_id, image_path, and bbox.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--image_ids", nargs="*", default=None,
                        help="Optional subset of tile_ids to run.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of manifest rows.")
    parser.add_argument("--write_polygons", action="store_true",
                        help="Write polygon shapefiles when bbox is available.")
    return parser.parse_args()


def read_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "tiles" in data:
        data = data["tiles"]
    if not isinstance(data, list):
        raise ValueError("Manifest must be a list or {'tiles': [...]} structure.")
    return data


def select_records(records, image_ids=None, limit=None):
    selected = list(records)
    if image_ids:
        wanted = {str(tile_id) for tile_id in image_ids}
        selected = [rec for rec in selected if str(rec.get("tile_id")) in wanted]
    if limit is not None:
        selected = selected[:limit]
    return selected


def ensure_runtime_defaults(config):
    legacy_patch_grid = bool(getattr(config, "LEGACY_PATCH_GRID", True))
    defaults = {
        "NEIGHBOR_RADIUS": 120,
        "SIDEWALK_NMS_RADIUS": 60,
        "CROSSING_NMS_RADIUS": 60,
        "SIDEWALK_THRESHOLD": 0.3,
        "CROSSING_THRESHOLD": 0.3,
        "INFER_BATCH_SIZE": 8,
    }
    # addict.Dict answers hasattr() with True for missing keys, so test membership.
    for key, value in defaults.items():
        if key not in config:
            setattr(config, key, value)
    if "SAMPLE_MARGIN" not in config:
        config.SAMPLE_MARGIN = config.NEIGHBOR_RADIUS
    elif legacy_patch_grid and int(config.SAMPLE_MARGIN) == 0:
        config.SAMPLE_MARGIN = config.NEIGHBOR_RADIUS
    return config


def main():
    args = parse_args()
    device = choose_device(args.device)
    print(f"Using device: {device}")

    config = ensure_runtime_defaults(load_config(args.config))
    out_root = args.output_dir
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "config.yaml"), "w") as f:
        yaml.dump(config.to_dict(), f)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    net = SidewalkFormer(config)
    state = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(state["state_dict"], strict=False)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and device == "cuda":
        print("Using", torch.cuda.device_count(), "GPUs")
        net = torch.nn.DataParallel(net)
    net.eval().to(device)

    records = select_records(read_manifest(args.manifest), args.image_ids, args.limit)
    if not records:
        raise ValueError("No manifest rows selected for inference.")

    for sub in ("mask", "pred_mask", "viz", "graph"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    if args.write_polygons:
        os.makedirs(os.path.join(out_root, "polygons"), exist_ok=True)

    rows = []
    for rec in records:
        tile_id = str(rec.get("tile_id"))
        image_path = rec.get("image_path")
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image path for {tile_id}: {image_path}")

        print(f"[run] tile {tile_id}  image={image_path}")
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        bbox = rec.get("bbox")
        config.TILE_BBOX = tuple(map(float, bbox)) if bbox is not None else None
        config.EXPORT_POLYGONS = bool(args.write_polygons and config.TILE_BBOX is not None)

        t0 = time.time()
        nodes_rc, edges, mask_u8, pred_mask, pred_gdf = infer_one_img(
            net, img_rgb, config, torch.device(device),
        )
        runtime_sec = time.time() - t0
        print("   done in %.1fs" % runtime_sec)

        cv2.imwrite(os.path.join(out_root, "mask", f"{tile_id}_walkmask.png"), mask_u8)
        np.save(os.path.join(out_root, "pred_mask", f"{tile_id}_pred_mask.npy"), pred_mask)

        viz = img_rgb.copy()
        for u, v in edges:
            cv2.line(
                viz,
                (int(nodes_rc[u, 1]), int(nodes_rc[u, 0])),
                (int(nodes_rc[v, 1]), int(nodes_rc[v, 0])),
                (255, 0, 0), 2,
            )
        for r, c in nodes_rc:
            cv2.circle(viz, (int(c), int(r)), 5, (255, 230, 0), -1)
        cv2.imwrite(
            os.path.join(out_root, "viz", f"{tile_id}_graph.png"),
            cv2.cvtColor(viz, cv2.COLOR_RGB2BGR),
        )

        sat2graph = graph_utils.convert_to_sat2graph_format(nodes_rc, edges)
        with open(os.path.join(out_root, "graph", f"{tile_id}.p"), "wb") as f:
            pickle.dump(sat2graph, f)

        if args.write_polygons and config.TILE_BBOX is not None and not pred_gdf.empty:
            try:
                pred_gdf.to_file(os.path.join(out_root, "polygons", f"{tile_id}.shp"))
            except Exception as exc:
                print(f"[warn] Failed to write polygons for {tile_id}: {exc}")

        rows.append({
            "tile_id": tile_id,
            "image_path": image_path,
            "nodes": int(nodes_rc.shape[0]),
            "edges": int(edges.shape[0]),
            "runtime_sec": round(runtime_sec, 3),
            "bbox": bbox,
        })

    pd.DataFrame(rows).to_csv(os.path.join(out_root, "summary.csv"), index=False)
    print("Saved summary table -> summary.csv")
    print("All outputs saved to", out_root)


if __name__ == "__main__":
    main()
