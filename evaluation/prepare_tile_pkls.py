#!/usr/bin/env python3
"""Prepare per-tile pickle adjacency files for pedestrian-network evaluation.

The output pickle format is:
  {(row, col): [(neighbor_row, neighbor_col), ...], ...}

Rows/cols are pixel coordinates within each evaluation tile. Bbox JSON files are
expected as a list of [south, west, north, east] entries.

Sources: Tile2Net network shapefiles, the SidewalkFormer merged graph
(``global/graph_merged.npz`` or ``.geojson``) and OpenStreetMap ground truth
fetched with OSMnx (the full walk network or sidewalk-tag filters).
"""

import argparse
import glob
import json
import math
import os
import pickle
from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString, box

TILE_SIZE = 2048
DEFAULT_OUTPUT_ROOT = "evaluation_results"
SIDEWALK_FILTERS = {
    "footway_sidewalk": '["highway"~"footway"]["footway"~"sidewalk"]',
    "foot_designated": '["foot"~"designated"]',
    "footway_not_crossing": '["highway"="footway"]["footway"!~"crossing"]',
}


def load_bboxes(path: str) -> List[List[float]]:
    with open(path) as f:
        bboxes = json.load(f)
    if not isinstance(bboxes, list) or not bboxes:
        raise ValueError(f"Expected a non-empty bbox list in {path}")
    return bboxes


def union_bbox_swne(bboxes: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    south = min(b[0] for b in bboxes)
    west = min(b[1] for b in bboxes)
    north = max(b[2] for b in bboxes)
    east = max(b[3] for b in bboxes)
    return float(south), float(west), float(north), float(east)


def parse_bbox_swne(value: str) -> Tuple[float, float, float, float]:
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be south,west,north,east")
    south, west, north, east = parts
    if not (south < north and west < east):
        raise argparse.ArgumentTypeError("bbox values must satisfy south<north and west<east")
    return south, west, north, east


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def lonlat_to_pixel(lon: float, lat: float, bbox_wsen: Sequence[float],
                    tile_size: int = TILE_SIZE) -> Tuple[int, int]:
    """Convert lon/lat to integer (row, col) for bbox [west, south, east, north]."""
    west, south, east, north = bbox_wsen
    col = (lon - west) / (east - west) * (tile_size - 1)
    row = (north - lat) / (north - south) * (tile_size - 1)
    row = int(round(min(max(row, 0), tile_size - 1)))
    col = int(round(min(max(col, 0), tile_size - 1)))
    return row, col


def iter_lines(geometry) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def linestrings_to_adjacency(lines: Iterable[LineString], bbox_wsen: Sequence[float],
                             tile_size: int = TILE_SIZE) -> dict:
    adj = {}
    for line in lines:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        pixels = [lonlat_to_pixel(lon, lat, bbox_wsen, tile_size) for lon, lat in coords]
        for a, b in zip(pixels[:-1], pixels[1:]):
            if a == b:
                continue
            adj.setdefault(a, [])
            adj.setdefault(b, [])
            if b not in adj[a]:
                adj[a].append(b)
            if a not in adj[b]:
                adj[b].append(a)
    return adj


def _as_int_tuple(pt):
    return (int(round(float(pt[0]))), int(round(float(pt[1]))))


def sanitize_undirected_adj(adj):
    out = defaultdict(set)
    for u_raw, nbrs in adj.items():
        u = _as_int_tuple(u_raw)
        out.setdefault(u, set())
        for v_raw in nbrs:
            v = _as_int_tuple(v_raw)
            if u == v:
                continue
            out[u].add(v)
            out[v].add(u)
    for u_raw in adj.keys():
        out.setdefault(_as_int_tuple(u_raw), set())
    return {u: sorted(vs) for u, vs in out.items()}


def adj_to_nx(adj):
    graph = nx.Graph()
    for u, nbrs in adj.items():
        graph.add_node(u)
        for v in nbrs:
            if u != v:
                graph.add_edge(u, v)
    return graph


def _edge_key(u, v):
    return (u, v) if u <= v else (v, u)


def decompose_into_anchor_chains_and_deg2_loops(graph):
    deg = dict(graph.degree())
    anchors = {n for n, d in deg.items() if d != 2}
    visited = set()
    chains = []

    for anchor in anchors:
        for neighbor in graph.neighbors(anchor):
            edge = _edge_key(anchor, neighbor)
            if edge in visited:
                continue
            visited.add(edge)
            path = [anchor, neighbor]
            prev, curr = anchor, neighbor

            while deg[curr] == 2:
                nbrs = list(graph.neighbors(curr))
                nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
                edge = _edge_key(curr, nxt)
                if edge in visited:
                    break
                visited.add(edge)
                path.append(nxt)
                prev, curr = curr, nxt

            chains.append(path)

    loops = []
    for u, v in graph.edges:
        edge = _edge_key(u, v)
        if edge in visited:
            continue

        visited.add(edge)
        loop = [u, v]
        start = u
        prev, curr = u, v
        closed = False

        while True:
            nbrs = list(graph.neighbors(curr))
            if len(nbrs) < 2:
                break
            nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
            edge = _edge_key(curr, nxt)

            if nxt == start:
                visited.add(edge)
                closed = True
                break
            if edge in visited:
                break

            visited.add(edge)
            loop.append(nxt)
            prev, curr = curr, nxt

        if closed and len(loop) >= 3:
            loops.append(loop)
        else:
            chains.append(loop)

    return chains, loops, anchors


def _sample_along_polyline_open(points_xy, target_dist):
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] <= 2:
        return pts

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total <= 1e-6:
        return np.vstack([pts[0], pts[-1]])

    n_segments = max(1, int(round(total / float(target_dist))))
    d_samples = np.linspace(0.0, total, n_segments + 1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    sampled = []
    j = 0
    for dist in d_samples:
        while j < len(seg_len) - 1 and cum[j + 1] < dist:
            j += 1
        if seg_len[j] <= 1e-9:
            point = pts[j + 1]
        else:
            t = (dist - cum[j]) / seg_len[j]
            point = pts[j] + t * (pts[j + 1] - pts[j])
        sampled.append(point)

    sampled = np.asarray(sampled, dtype=np.float64)
    sampled[0] = pts[0]
    sampled[-1] = pts[-1]
    return sampled


def _sample_along_polyline_closed(points_xy, target_dist):
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] < 3:
        return pts

    pts_closed = np.vstack([pts, pts[0]])
    seg = pts_closed[1:] - pts_closed[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total <= 1e-6:
        return pts

    n_segments = max(3, int(round(total / float(target_dist))))
    d_samples = np.linspace(0.0, total, n_segments, endpoint=False)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    sampled = []
    j = 0
    for dist in d_samples:
        while j < len(seg_len) - 1 and cum[j + 1] < dist:
            j += 1
        if seg_len[j] <= 1e-9:
            point = pts_closed[j + 1]
        else:
            t = (dist - cum[j]) / seg_len[j]
            point = pts_closed[j] + t * (pts_closed[j + 1] - pts_closed[j])
        sampled.append(point)

    return np.asarray(sampled, dtype=np.float64)


def _round_open_path(sampled_points, start_anchor, end_anchor):
    out = [_as_int_tuple(start_anchor)]
    for point in sampled_points[1:-1]:
        q = _as_int_tuple(point)
        if q != out[-1]:
            out.append(q)
    q_end = _as_int_tuple(end_anchor)
    if q_end != out[-1]:
        out.append(q_end)
    if len(out) < 2:
        out = [_as_int_tuple(start_anchor), _as_int_tuple(end_anchor)]
    return out


def _round_closed_cycle(sampled_points, fallback_cycle):
    pts = [_as_int_tuple(point) for point in sampled_points]
    dedup = []
    for q in pts:
        if not dedup or q != dedup[-1]:
            dedup.append(q)
    if len(dedup) > 1 and dedup[0] == dedup[-1]:
        dedup.pop()

    if len(list(dict.fromkeys(dedup))) >= 3:
        return dedup

    fallback = []
    for point in fallback_cycle:
        q = _as_int_tuple(point)
        if not fallback or q != fallback[-1]:
            fallback.append(q)
    if len(fallback) > 1 and fallback[0] == fallback[-1]:
        fallback.pop()
    return list(dict.fromkeys(fallback))


def _add_path_edges(adj_sets, path_points, closed=False):
    for u, v in zip(path_points[:-1], path_points[1:]):
        if u == v:
            continue
        adj_sets[u].add(v)
        adj_sets[v].add(u)

    if closed and len(path_points) >= 3:
        u, v = path_points[-1], path_points[0]
        if u != v:
            adj_sets[u].add(v)
            adj_sets[v].add(u)


def resample_degree2_nodes_keep_anchors(adj_in, target_dist=60.0):
    """Resample degree-2 chains and loops to ~``target_dist`` px spacing; junctions and ends are kept."""
    adj = sanitize_undirected_adj(adj_in)
    graph = adj_to_nx(adj)
    if graph.number_of_nodes() == 0:
        return {}

    chains, loops, anchors = decompose_into_anchor_chains_and_deg2_loops(graph)
    out_sets = defaultdict(set)

    for anchor in anchors:
        out_sets.setdefault(anchor, set())

    for path in chains:
        if len(path) < 2:
            continue
        if len(path) == 2:
            _add_path_edges(out_sets, path, closed=False)
            continue
        sampled = _sample_along_polyline_open(path, target_dist=target_dist)
        rounded = _round_open_path(sampled, start_anchor=path[0], end_anchor=path[-1])
        _add_path_edges(out_sets, rounded, closed=False)

    for loop in loops:
        sampled = _sample_along_polyline_closed(loop, target_dist=target_dist)
        rounded = _round_closed_cycle(sampled, fallback_cycle=loop)
        if len(rounded) >= 3:
            _add_path_edges(out_sets, rounded, closed=True)

    for u, nbrs in adj.items():
        if len(nbrs) == 0:
            out_sets.setdefault(u, set())

    return sanitize_undirected_adj({u: sorted(vs) for u, vs in out_sets.items()})


def densify_graph_with_long_edges(adj_xy, max_edge_length=60.0):
    """Split remaining direct edges so each segment is at most max_edge_length pixels."""
    new_adj = defaultdict(list)
    processed_edges = set()

    for node, neighbors in adj_xy.items():
        new_adj.setdefault(node, [])
        for neighbor in neighbors:
            edge = _edge_key(node, neighbor)
            if edge in processed_edges:
                continue
            processed_edges.add(edge)

            node_arr = np.array(node, dtype=float)
            neighbor_arr = np.array(neighbor, dtype=float)
            edge_length = np.linalg.norm(neighbor_arr - node_arr)
            if edge_length <= max_edge_length:
                chain = [node, neighbor]
            else:
                num_segments = int(np.ceil(edge_length / max_edge_length))
                chain = []
                for i in range(num_segments + 1):
                    t = i / num_segments
                    point = tuple(np.round(node_arr + t * (neighbor_arr - node_arr)).astype(int))
                    if not chain or point != chain[-1]:
                        chain.append(point)

            for u, v in zip(chain[:-1], chain[1:]):
                if u == v:
                    continue
                new_adj[u].append(v)
                new_adj[v].append(u)

    return {node: list(dict.fromkeys(neighbors)) for node, neighbors in new_adj.items()}


def edge_length_stats(adj):
    lengths = []
    seen = set()
    for u, nbrs in adj.items():
        for v in nbrs:
            edge = _edge_key(u, v)
            if edge in seen:
                continue
            seen.add(edge)
            lengths.append(math.dist(u, v))
    return {
        "nodes": len(adj),
        "edges": len(lengths),
        "mean": float(np.mean(lengths)) if lengths else 0.0,
        "max": float(np.max(lengths)) if lengths else 0.0,
    }


def write_pickle(path: str, adj: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(adj, f)


def clip_lines_to_tiles(lines_gdf: gpd.GeoDataFrame, bboxes: Sequence[Sequence[float]],
                        out_dir: str, tile_size: int = TILE_SIZE,
                        resample_px: float | None = None,
                        max_edge_px: float | None = None) -> None:
    ensure_dir(out_dir)
    if lines_gdf.crs is None:
        lines_gdf = lines_gdf.set_crs("EPSG:4326")
    elif str(lines_gdf.crs).upper() not in {"EPSG:4326", "WGS84"}:
        lines_gdf = lines_gdf.to_crs("EPSG:4326")

    for tile_idx, (south, west, north, east) in enumerate(bboxes):
        tile_box = box(west, south, east, north)
        bbox_wsen = [west, south, east, north]
        if lines_gdf.empty:
            write_pickle(os.path.join(out_dir, f"tile_{tile_idx}.p"), {})
            continue
        clipped = gpd.clip(lines_gdf, tile_box)
        lines = []
        for geom in clipped.geometry:
            lines.extend(iter_lines(geom))
        adj = linestrings_to_adjacency(lines, bbox_wsen, tile_size)
        if resample_px is not None or max_edge_px is not None:
            before = edge_length_stats(adj)
        if resample_px is not None:
            adj = resample_degree2_nodes_keep_anchors(adj, target_dist=resample_px)
        if max_edge_px is not None:
            adj = densify_graph_with_long_edges(adj, max_edge_length=max_edge_px)
        if resample_px is not None or max_edge_px is not None:
            after = edge_length_stats(adj)
            print(
                f"  tile_{tile_idx}: GT resampled {before['nodes']}n/{before['edges']}e "
                f"-> {after['nodes']}n/{after['edges']}e, "
                f"mean edge {before['mean']:.1f}px -> {after['mean']:.1f}px"
            )
        write_pickle(os.path.join(out_dir, f"tile_{tile_idx}.p"), adj)


def latest_network_shapefile(tile_dir: str) -> str | None:
    shp_files = glob.glob(os.path.join(tile_dir, "**", "network", "**", "*.shp"), recursive=True)
    if not shp_files:
        return None
    return max(shp_files, key=os.path.getmtime)


def process_tile2net(tile2net_dir: str, bboxes: Sequence[Sequence[float]],
                     out_dir: str, tile_size: int = TILE_SIZE) -> None:
    tile_dirs = sorted(glob.glob(os.path.join(tile2net_dir, "tile_*")))
    print(f"Processing Tile2Net shapefiles from {tile2net_dir}")

    # Tile2Net source tile indices are not guaranteed to match the metric
    # comparison-grid indices. Merge all georeferenced network lines first,
    # then clip them spatially to each requested evaluation bbox.
    geometries = []
    shapefile_count = 0
    for tile_dir in tile_dirs:
        shp_path = latest_network_shapefile(tile_dir)
        if shp_path is None:
            continue
        gdf = gpd.read_file(shp_path)
        if gdf.empty:
            continue
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")
        geometries.extend(
            geometry
            for geometry in gdf.geometry
            if geometry is not None and not geometry.is_empty
        )
        shapefile_count += 1

    lines_gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    clip_lines_to_tiles(lines_gdf, bboxes, out_dir, tile_size)
    print(
        f"  Done. Spatially clipped {len(geometries)} features from "
        f"{shapefile_count} Tile2Net shapefiles into {len(bboxes)} metric tiles. "
        f"Saved to {out_dir}"
    )


def process_sidewalkformer(sf_inference_dir: str, bboxes: Sequence[Sequence[float]],
                           out_dir: str, tile_size: int = TILE_SIZE) -> None:
    graph_path = os.path.join(sf_inference_dir, "global", "graph_merged.npz")
    npz = np.load(graph_path)
    nodes_xy = npz["nodes_xy"]  # [lon, lat]
    edges = npz["edges"]

    print(f"SidewalkFormer global graph: {len(nodes_xy)} nodes, {len(edges)} edges")
    lines = []
    for u, v in edges:
        u, v = int(u), int(v)
        if u == v:
            continue
        p1 = nodes_xy[u]
        p2 = nodes_xy[v]
        if np.any(~np.isfinite(p1)) or np.any(~np.isfinite(p2)):
            continue
        lines.append(LineString([(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))]))

    gdf = gpd.GeoDataFrame({"geometry": lines}, crs="EPSG:4326")
    print(f"Cropping SidewalkFormer graph into {len(bboxes)} evaluation tiles")
    clip_lines_to_tiles(gdf, bboxes, out_dir, tile_size)
    print(f"  Done. Saved to {out_dir}")


def process_sidewalkformer_geojson(sf_geojson_path: str, bboxes: Sequence[Sequence[float]],
                                   out_dir: str, tile_size: int = TILE_SIZE) -> None:
    gdf = gpd.read_file(sf_geojson_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])][["geometry"]]
    print(f"SidewalkFormer global GeoJSON lines: {len(gdf)} from {sf_geojson_path}")
    print(f"Cropping SidewalkFormer GeoJSON graph into {len(bboxes)} evaluation tiles")
    clip_lines_to_tiles(gdf, bboxes, out_dir, tile_size)
    print(f"  Done. Saved to {out_dir}")


def graph_from_bbox_compat(ox, bbox_swne: Sequence[float], **kwargs):
    south, west, north, east = bbox_swne
    try:
        return ox.graph_from_bbox((west, south, east, north), **kwargs)
    except TypeError:
        return ox.graph_from_bbox(north, south, east, west, **kwargs)


def fetch_osmnx_walk_lines(bbox_swne: Sequence[float], graphml_path: str | None = None,
                           geojson_path: str | None = None) -> gpd.GeoDataFrame:
    """Fetch the OSMnx walk network only: graph_from_bbox(..., network_type='walk')."""
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError(
            "OSMnx is required for ground truth. Install osmnx (see requirements.txt)."
        ) from exc

    south, west, north, east = bbox_swne
    print("Fetching OSMnx walk network ground truth")
    print(f"  graph_from_bbox(({west}, {south}, {east}, {north}), network_type='walk')")
    graph = graph_from_bbox_compat(
        ox,
        bbox_swne,
        network_type="walk",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )

    if graphml_path:
        ensure_dir(os.path.dirname(graphml_path) or ".")
        ox.save_graphml(graph, graphml_path)
        print(f"  Saved OSMnx graphml to {graphml_path}")

    edges_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
    edges_gdf = edges_gdf.to_crs("EPSG:4326")
    edges_gdf = edges_gdf[["geometry"]].reset_index(drop=True)
    if geojson_path:
        ensure_dir(os.path.dirname(geojson_path) or ".")
        edges_gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"  Saved OSMnx walk edges to {geojson_path}")
    return edges_gdf


def fetch_osmnx_sidewalk_filter_lines(bbox_swne: Sequence[float],
                                      geojson_path: str | None = None) -> gpd.GeoDataFrame:
    """Fetch OSM edges matching ``SIDEWALK_FILTERS`` instead of the full walk graph."""
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError(
            "OSMnx is required for ground truth. Install osmnx (see requirements.txt)."
        ) from exc

    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.timeout = 180

    south, west, north, east = bbox_swne
    print("Fetching OSMnx sidewalk-filter ground truth")
    print(f"  bbox=({south}, {west}, {north}, {east})")

    geometries = []
    seen = set()
    for name, filter_str in SIDEWALK_FILTERS.items():
        print(f"  custom_filter[{name}] = {filter_str}")
        try:
            graph = graph_from_bbox_compat(
                ox,
                bbox_swne,
                custom_filter=filter_str,
                network_type="walk",
                retain_all=True,
                simplify=False,
                truncate_by_edge=True,
            )
        except Exception as exc:
            print(f"    skipped {name}: {exc}")
            continue
        if graph.is_directed():
            graph = graph.to_undirected()
        edges_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
        edges_gdf = edges_gdf.to_crs("EPSG:4326")
        for geom in edges_gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            key = geom.wkb
            if key in seen:
                continue
            seen.add(key)
            geometries.append(geom)

    out = gpd.GeoDataFrame({"geometry": geometries}, crs="EPSG:4326")
    if geojson_path:
        ensure_dir(os.path.dirname(geojson_path) or ".")
        out.to_file(geojson_path, driver="GeoJSON")
        print(f"  Saved OSMnx sidewalk-filter edges to {geojson_path}")
    return out


def process_ground_truth(bboxes: Sequence[Sequence[float]], out_dir: str,
                         osm_bbox_swne: Sequence[float] | None = None,
                         graphml_path: str | None = None,
                         geojson_path: str | None = None,
                         tile_size: int = TILE_SIZE,
                         resample_px: float | None = None,
                         max_edge_px: float | None = None,
                         gt_source: str = "walk") -> None:
    if osm_bbox_swne is None:
        osm_bbox_swne = union_bbox_swne(bboxes)
    if gt_source == "walk":
        osm_edges = fetch_osmnx_walk_lines(osm_bbox_swne, graphml_path, geojson_path)
        source_label = "OSMnx walk graph"
    elif gt_source == "sidewalk_filter":
        osm_edges = fetch_osmnx_sidewalk_filter_lines(osm_bbox_swne, geojson_path)
        source_label = "OSMnx sidewalk-filter graph"
    else:
        raise ValueError(f"Unknown gt_source: {gt_source}")
    print(f"{source_label} edges: {len(osm_edges)}")
    print(f"Cropping {source_label} into {len(bboxes)} evaluation tiles")
    clip_lines_to_tiles(
        osm_edges,
        bboxes,
        out_dir,
        tile_size,
        resample_px=resample_px,
        max_edge_px=max_edge_px,
    )
    print(f"  Done. Saved to {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Prepare Tile2Net, SidewalkFormer, and OSMnx-walk GT tile pickles."
    )
    ap.add_argument("--bbox_file", required=True,
                    help="JSON list of [south, west, north, east] evaluation tile bboxes")
    ap.add_argument("--tile2net_dir", default=None)
    ap.add_argument("--sf_inference_dir", default=None,
                    help="Directory containing global/graph_merged.npz")
    ap.add_argument("--sf_geojson", default=None,
                    help="Optional SidewalkFormer global graph_merged.geojson to use instead of graph_merged.npz.")
    ap.add_argument(
        "--out_tile2net",
        default=os.path.join(DEFAULT_OUTPUT_ROOT, "tile2net_pkls"),
    )
    ap.add_argument(
        "--out_sidewalkformer",
        default=os.path.join(DEFAULT_OUTPUT_ROOT, "sidewalkformer_pkls"),
    )
    ap.add_argument("--out_gt", default=os.path.join(DEFAULT_OUTPUT_ROOT, "gt_pkls"))
    ap.add_argument("--osm_bbox", type=parse_bbox_swne, default=None,
                    help="Optional OSM download bbox as south,west,north,east. Defaults to union of bbox_file.")
    ap.add_argument(
        "--osm_graphml",
        default=os.path.join(DEFAULT_OUTPUT_ROOT, "osm_walk.graphml"),
    )
    ap.add_argument(
        "--osm_geojson",
        default=os.path.join(DEFAULT_OUTPUT_ROOT, "osm_walk_edges.geojson"),
    )
    ap.add_argument("--gt_source", choices=["walk", "sidewalk_filter"], default="walk",
                    help="Ground truth from the full OSMnx walk graph or from the sidewalk tag filters.")
    ap.add_argument("--tile_size", type=int, default=TILE_SIZE)
    ap.add_argument("--gt_resample_px", type=float, default=None,
                    help="Resample OSM GT degree-2 chains/loops to this pixel spacing, e.g. 60.")
    ap.add_argument("--gt_max_edge_px", type=float, default=None,
                    help="Split any remaining OSM GT edges longer than this many pixels, e.g. 60.")
    ap.add_argument("--skip_tile2net", action="store_true")
    ap.add_argument("--skip_sidewalkformer", action="store_true")
    ap.add_argument("--skip_gt", action="store_true")
    return ap


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.skip_tile2net and not args.tile2net_dir:
        parser.error("--tile2net_dir is required unless --skip_tile2net is set")
    if (
        not args.skip_sidewalkformer
        and not args.sf_inference_dir
        and not args.sf_geojson
    ):
        parser.error(
            "--sf_inference_dir or --sf_geojson is required unless "
            "--skip_sidewalkformer is set"
        )
    bboxes = load_bboxes(args.bbox_file)
    print(f"Loaded {len(bboxes)} evaluation bboxes from {args.bbox_file}")

    if not args.skip_tile2net:
        print("\n" + "=" * 60)
        print("1. Tile2Net shapefiles -> pickle adjacency files")
        print("=" * 60)
        process_tile2net(args.tile2net_dir, bboxes, args.out_tile2net, args.tile_size)

    if not args.skip_sidewalkformer:
        print("\n" + "=" * 60)
        print("2. SidewalkFormer global graph -> per-tile pickles")
        print("=" * 60)
        if args.sf_geojson:
            process_sidewalkformer_geojson(args.sf_geojson, bboxes, args.out_sidewalkformer, args.tile_size)
        else:
            process_sidewalkformer(args.sf_inference_dir, bboxes, args.out_sidewalkformer, args.tile_size)

    if not args.skip_gt:
        print("\n" + "=" * 60)
        print("3. OSMnx walk network -> ground-truth per-tile pickles")
        print("=" * 60)
        process_ground_truth(
            bboxes,
            args.out_gt,
            osm_bbox_swne=args.osm_bbox,
            graphml_path=args.osm_graphml,
            geojson_path=args.osm_geojson,
            tile_size=args.tile_size,
            resample_px=args.gt_resample_px,
            max_edge_px=args.gt_max_edge_px,
            gt_source=args.gt_source,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
