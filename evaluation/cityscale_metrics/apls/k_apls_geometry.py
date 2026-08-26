#!/usr/bin/env python3
"""K-APLS + Geometry for pedestrian-network comparison.

The CLI accepts either:

* the legacy CityScale APLS JSON format: ``[nodes, edges]`` where nodes are
  ``[latitude, longitude]``; or
* GeoJSON LineString/MultiLineString networks. Mixed SidewalkFormer
  node/edge FeatureCollections are supported because only line features are
  needed to reconstruct the graph.

All scoring is performed in a projected CRS measured in metres.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import networkx as nx
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from shapely import frechet_distance
from shapely.geometry import LineString, box, shape
from shapely.ops import transform as transform_geometry


@dataclass(frozen=True)
class MetricConfig:
    k: int = 3
    theta: float = 0.6
    w_length: float = 0.5
    w_geometry: float = 0.5
    tau: float = 20.0
    geometry_metric: str = "hausdorff"
    max_candidates: int = 30
    penalize_missing_routes: bool = True
    return_details: bool = False
    control_point_spacing: float = 30.0
    max_control_points: int = 160
    max_od_pairs: int = 500
    min_path_length: float = 60.0
    max_path_length: float = 0.0
    snap_tolerance: float = 4.0
    snap_exclusion_hops: int = 4
    densify_spacing: float = 5.0
    random_seed: int = 0

    def validate(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must be in [0, 1]")
        if self.w_length < 0.0 or self.w_geometry < 0.0:
            raise ValueError("metric weights cannot be negative")
        if not math.isclose(self.w_length + self.w_geometry, 1.0, abs_tol=1e-9):
            raise ValueError("w_length + w_geometry must equal 1")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.geometry_metric not in {"hausdorff", "frechet"}:
            raise ValueError("geometry_metric must be 'hausdorff' or 'frechet'")
        if self.max_candidates < self.k:
            raise ValueError("max_candidates must be at least k")
        if self.control_point_spacing <= 0.0:
            raise ValueError("control_point_spacing must be positive")
        if self.max_control_points < 0:
            raise ValueError("max_control_points cannot be negative")
        if self.max_od_pairs < 0:
            raise ValueError("max_od_pairs cannot be negative")
        if self.min_path_length < 0.0:
            raise ValueError("min_path_length cannot be negative")
        if self.max_path_length < 0.0:
            raise ValueError("max_path_length cannot be negative")
        if (
            self.max_path_length > 0.0
            and self.max_path_length < self.min_path_length
        ):
            raise ValueError("max_path_length must be zero or at least min_path_length")
        if self.snap_tolerance < 0.0:
            raise ValueError("snap_tolerance cannot be negative")
        if self.snap_exclusion_hops < 0:
            raise ValueError("snap_exclusion_hops cannot be negative")
        if self.densify_spacing <= 0.0:
            raise ValueError("densify_spacing must be positive")


@dataclass
class RawNetwork:
    polylines: list[list[tuple[float, float]]]
    crs: CRS
    source_format: str


@dataclass
class PathRecord:
    nodes: tuple[int, ...]
    length: float
    geometry: LineString
    edge_lengths: dict[tuple[int, int], float]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _geojson_crs(data: dict[str, Any]) -> CRS | None:
    crs_data = data.get("crs")
    if not isinstance(crs_data, dict):
        return None
    properties = crs_data.get("properties", {})
    name = properties.get("name") if isinstance(properties, dict) else None
    if not name:
        return None
    if name.startswith("urn:ogc:def:crs:EPSG::"):
        name = f"EPSG:{name.rsplit(':', 1)[-1]}"
    return CRS.from_user_input(name)


def _iter_geojson_lines(geometry: dict[str, Any] | None) -> Iterator[list[tuple[float, float]]]:
    if not geometry:
        return
    geom = shape(geometry)
    if geom.geom_type == "LineString":
        coords = [(float(x), float(y)) for x, y, *_ in geom.coords]
        if len(coords) >= 2:
            yield coords
    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            coords = [(float(x), float(y)) for x, y, *_ in part.coords]
            if len(coords) >= 2:
                yield coords
    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from _iter_geojson_lines(part.__geo_interface__)


def _infer_coordinate_crs(polylines: Sequence[Sequence[tuple[float, float]]]) -> CRS:
    coordinates = list(itertools.chain.from_iterable(polylines))
    if not coordinates:
        raise ValueError("network contains no line coordinates")
    xs = [point[0] for point in coordinates]
    ys = [point[1] for point in coordinates]
    if min(xs) >= -180.0 and max(xs) <= 180.0 and min(ys) >= -90.0 and max(ys) <= 90.0:
        return CRS.from_epsg(4326)
    raise ValueError(
        "input has projected-looking coordinates but no CRS; pass an explicit source CRS"
    )


def read_network(path: str | Path, source_crs: str | None = None) -> RawNetwork:
    """Read legacy APLS JSON, GeoJSON, or a vector file into raw polylines."""
    path = Path(path)
    override_crs = CRS.from_user_input(source_crs) if source_crs else None

    if path.suffix.lower() not in {".json", ".geojson"}:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise RuntimeError(
                f"geopandas is required to read vector network file {path}"
            ) from exc
        frame = gpd.read_file(path)
        polylines = []
        for geometry in frame.geometry:
            if geometry is not None and not geometry.is_empty:
                polylines.extend(_iter_geojson_lines(geometry.__geo_interface__))
        if not polylines:
            raise ValueError(f"vector file has no LineString features: {path}")
        crs = override_crs or (CRS.from_user_input(frame.crs) if frame.crs else None)
        if crs is None:
            crs = _infer_coordinate_crs(polylines)
        return RawNetwork(polylines, crs, path.suffix.lower().lstrip("."))

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list) and len(data) == 2:
        nodes, edges = data
        polylines = []
        for edge in edges:
            if not isinstance(edge, list) or len(edge) < 2:
                continue
            u, v = int(edge[0]), int(edge[1])
            if u == v or u < 0 or v < 0 or u >= len(nodes) or v >= len(nodes):
                continue
            # The bundled Go APLS converter writes [latitude, longitude].
            lat_u, lon_u = float(nodes[u][0]), float(nodes[u][1])
            lat_v, lon_v = float(nodes[v][0]), float(nodes[v][1])
            polylines.append([(lon_u, lat_u), (lon_v, lat_v)])
        if not polylines:
            raise ValueError(f"legacy APLS graph has no usable edges: {path}")
        return RawNetwork(polylines, override_crs or CRS.from_epsg(4326), "legacy_apls_json")

    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        polylines = []
        for feature in data.get("features", []):
            polylines.extend(_iter_geojson_lines(feature.get("geometry")))
        if not polylines:
            raise ValueError(f"GeoJSON has no LineString features: {path}")
        crs = override_crs or _geojson_crs(data) or _infer_coordinate_crs(polylines)
        return RawNetwork(polylines, crs, "geojson")

    raise ValueError(
        f"unsupported graph format in {path}; expected legacy [nodes, edges] JSON or GeoJSON"
    )


def _crs_uses_metres(crs: CRS) -> bool:
    if not crs.is_projected:
        return False
    return all(
        axis.unit_name and axis.unit_name.lower() in {"metre", "meter"}
        for axis in crs.axis_info[:2]
    )


def choose_metric_crs(raw_gt: RawNetwork, requested_crs: str | None = None) -> CRS:
    """Choose a metre-based CRS, preferring the GT CRS when already suitable."""
    if requested_crs:
        metric_crs = CRS.from_user_input(requested_crs)
        if not _crs_uses_metres(metric_crs):
            raise ValueError("--metric-crs must be a projected CRS measured in metres")
        return metric_crs
    if _crs_uses_metres(raw_gt.crs):
        return raw_gt.crs

    all_points = list(itertools.chain.from_iterable(raw_gt.polylines))
    mean_x = float(np.mean([point[0] for point in all_points]))
    mean_y = float(np.mean([point[1] for point in all_points]))
    to_wgs84 = Transformer.from_crs(raw_gt.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(mean_x, mean_y)
    zone = max(1, min(60, int((lon + 180.0) // 6.0) + 1))
    epsg = (32600 if lat >= 0.0 else 32700) + zone
    return CRS.from_epsg(epsg)


def project_network(raw: RawNetwork, metric_crs: CRS) -> RawNetwork:
    if raw.crs == metric_crs:
        return raw
    transformer = Transformer.from_crs(raw.crs, metric_crs, always_xy=True)
    polylines = []
    for line in raw.polylines:
        projected = [transformer.transform(x, y) for x, y in line]
        polylines.append([(float(x), float(y)) for x, y in projected])
    return RawNetwork(polylines, metric_crs, raw.source_format)


def clip_network_wgs84(
    raw: RawNetwork,
    bbox_swne: Sequence[float],
) -> RawNetwork:
    """Clip a raw network to a WGS84 [south, west, north, east] bbox."""
    south, west, north, east = [float(value) for value in bbox_swne]
    clip_geometry = box(west, south, east, north)
    if raw.crs != CRS.from_epsg(4326):
        transformer = Transformer.from_crs("EPSG:4326", raw.crs, always_xy=True)
        clip_geometry = transform_geometry(transformer.transform, clip_geometry)

    clipped_polylines = []
    for coordinates in raw.polylines:
        geometry = LineString(coordinates)
        if not geometry.intersects(clip_geometry):
            continue
        intersection = geometry.intersection(clip_geometry)
        if not intersection.is_empty:
            clipped_polylines.extend(_iter_geojson_lines(intersection.__geo_interface__))
    return RawNetwork(clipped_polylines, raw.crs, raw.source_format)


def graph_from_polylines(
    polylines: Sequence[Sequence[tuple[float, float]]],
    node_merge_tolerance: float = 0.01,
    allow_empty: bool = False,
) -> nx.Graph:
    """Build an undirected weighted graph from projected polylines."""
    if node_merge_tolerance <= 0.0:
        raise ValueError("node_merge_tolerance must be positive")
    graph = nx.Graph()
    node_ids: dict[tuple[int, int], int] = {}

    def get_node(point: tuple[float, float]) -> int:
        key = (
            int(round(point[0] / node_merge_tolerance)),
            int(round(point[1] / node_merge_tolerance)),
        )
        node_id = node_ids.get(key)
        if node_id is None:
            node_id = len(node_ids)
            node_ids[key] = node_id
            graph.add_node(node_id, x=float(point[0]), y=float(point[1]))
        return node_id

    for polyline in polylines:
        for start, end in itertools.pairwise(polyline):
            u = get_node(start)
            v = get_node(end)
            if u == v:
                continue
            length = math.dist(start, end)
            if not math.isfinite(length) or length <= 0.0:
                continue
            if graph.has_edge(u, v):
                if length < graph[u][v]["length"]:
                    graph[u][v]["length"] = length
                continue
            graph.add_edge(u, v, length=length)

    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_edges() == 0 and not allow_empty:
        raise ValueError("network has no usable edges after graph construction")
    return graph


def densify_graph(graph: nx.Graph, max_segment_length: float) -> nx.Graph:
    """Split straight graph edges so snapping behaves like the bundled APLS."""
    dense = nx.Graph()
    coordinate_to_node: dict[tuple[float, float], int] = {}

    def get_node(x: float, y: float) -> int:
        key = (round(float(x), 6), round(float(y), 6))
        node_id = coordinate_to_node.get(key)
        if node_id is None:
            node_id = len(coordinate_to_node)
            coordinate_to_node[key] = node_id
            dense.add_node(node_id, x=float(x), y=float(y))
        return node_id

    for u, v in graph.edges:
        x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
        x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]
        length = math.hypot(x2 - x1, y2 - y1)
        segment_count = max(1, int(math.ceil(length / max_segment_length)))
        previous = get_node(x1, y1)
        for index in range(1, segment_count + 1):
            alpha = index / segment_count
            x = x1 + alpha * (x2 - x1)
            y = y1 + alpha * (y2 - y1)
            current = get_node(x, y)
            if previous != current:
                segment_length = math.hypot(
                    dense.nodes[current]["x"] - dense.nodes[previous]["x"],
                    dense.nodes[current]["y"] - dense.nodes[previous]["y"],
                )
                dense.add_edge(previous, current, length=segment_length)
            previous = current
    return dense


def load_metric_graphs(
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    ground_truth_crs: str | None = None,
    prediction_crs: str | None = None,
    metric_crs: str | None = None,
    node_merge_tolerance: float = 0.01,
    densify_spacing: float = 5.0,
) -> tuple[nx.Graph, nx.Graph, CRS, dict[str, Any]]:
    raw_gt = read_network(ground_truth_path, ground_truth_crs)
    raw_pred = read_network(prediction_path, prediction_crs)
    target_crs = choose_metric_crs(raw_gt, metric_crs)
    gt_projected = project_network(raw_gt, target_crs)
    pred_projected = project_network(raw_pred, target_crs)
    graph_gt = densify_graph(
        graph_from_polylines(gt_projected.polylines, node_merge_tolerance),
        densify_spacing,
    )
    graph_pred = densify_graph(
        graph_from_polylines(pred_projected.polylines, node_merge_tolerance),
        densify_spacing,
    )
    metadata = {
        "ground_truth_format": raw_gt.source_format,
        "prediction_format": raw_pred.source_format,
        "ground_truth_source_crs": raw_gt.crs.to_string(),
        "prediction_source_crs": raw_pred.crs.to_string(),
        "metric_crs": target_crs.to_string(),
    }
    return graph_gt, graph_pred, target_crs, metadata


def metric_graph_from_raw(
    raw: RawNetwork,
    metric_crs: CRS,
    node_merge_tolerance: float = 0.01,
    densify_spacing: float = 5.0,
    allow_empty: bool = False,
) -> nx.Graph:
    projected = project_network(raw, metric_crs)
    graph = graph_from_polylines(
        projected.polylines,
        node_merge_tolerance,
        allow_empty=allow_empty,
    )
    return densify_graph(graph, densify_spacing)


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _path_record(graph: nx.Graph, nodes: Sequence[int]) -> PathRecord:
    if len(nodes) < 2:
        raise ValueError("a path must contain at least two nodes")
    coordinates = [(graph.nodes[node]["x"], graph.nodes[node]["y"]) for node in nodes]
    edge_lengths = {}
    total_length = 0.0
    for u, v in itertools.pairwise(nodes):
        length = float(graph[u][v]["length"])
        edge_lengths[_edge_key(u, v)] = length
        total_length += length
    return PathRecord(
        nodes=tuple(nodes),
        length=total_length,
        geometry=LineString(coordinates),
        edge_lengths=edge_lengths,
    )


def path_overlap(path_a: PathRecord, path_b: PathRecord) -> float:
    """Shared undirected edge length divided by the shorter path length."""
    denominator = min(path_a.length, path_b.length)
    if denominator <= 0.0:
        return 1.0
    shared_edges = set(path_a.edge_lengths).intersection(path_b.edge_lengths)
    shared_length = sum(
        min(path_a.edge_lengths[edge], path_b.edge_lengths[edge])
        for edge in shared_edges
    )
    return _clamp01(shared_length / denominator)


def diverse_paths(
    graph: nx.Graph,
    source: int,
    target: int,
    k: int,
    theta: float,
    max_candidates: int,
) -> list[PathRecord]:
    """Return up to k length-ranked simple paths satisfying the overlap limit."""
    if source not in graph or target not in graph:
        return []
    selected: list[PathRecord] = []
    try:
        candidates = nx.shortest_simple_paths(graph, source, target, weight="length")
        for node_path in itertools.islice(candidates, max_candidates):
            record = _path_record(graph, node_path)
            if all(path_overlap(record, accepted) < theta for accepted in selected):
                selected.append(record)
                if len(selected) >= k:
                    break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return selected


def geometry_distance(
    ground_truth: LineString,
    prediction: LineString,
    metric: str,
) -> float:
    if metric == "hausdorff":
        return float(ground_truth.hausdorff_distance(prediction))
    if metric == "frechet":
        return float(frechet_distance(ground_truth, prediction))
    raise ValueError(f"unsupported geometry metric: {metric}")


def score_path_pair(
    ground_truth: PathRecord,
    prediction: PathRecord,
    config: MetricConfig,
) -> dict[str, float]:
    if ground_truth.length <= 0.0:
        return {
            "length_gt": ground_truth.length,
            "length_pred": prediction.length,
            "length_score": 0.0,
            "geometry_distance": math.inf,
            "geometry_score": 0.0,
            "combined_score": 0.0,
        }
    length_score = _clamp01(
        1.0 - abs(ground_truth.length - prediction.length) / ground_truth.length
    )
    distance = geometry_distance(
        ground_truth.geometry,
        prediction.geometry,
        config.geometry_metric,
    )
    geometry_score = _clamp01(1.0 - distance / config.tau)
    combined_score = _clamp01(
        config.w_length * length_score + config.w_geometry * geometry_score
    )
    return {
        "length_gt": float(ground_truth.length),
        "length_pred": float(prediction.length),
        "length_score": length_score,
        "geometry_distance": distance,
        "geometry_score": geometry_score,
        "combined_score": combined_score,
    }


def score_aligned_paths(
    ground_truth_paths: Sequence[PathRecord],
    prediction_paths: Sequence[PathRecord],
    config: MetricConfig,
) -> tuple[float, float, float, float, list[dict[str, float]]]:
    """Return combined, length, geometry, vanilla, and per-path scores."""
    path_scores = [
        score_path_pair(gt_path, pred_path, config)
        for gt_path, pred_path in zip(ground_truth_paths, prediction_paths)
    ]
    combined_scores = [score["combined_score"] for score in path_scores]
    length_scores = [score["length_score"] for score in path_scores]
    geometry_scores = [score["geometry_score"] for score in path_scores]
    if not ground_truth_paths or not prediction_paths:
        od_score = 0.0
        length_component = 0.0
        geometry_component = 0.0
    elif config.penalize_missing_routes:
        # Penalize only alternatives that exist in GT. Dividing by k would
        # incorrectly cap an identical tree at 1/k.
        denominator = len(ground_truth_paths)
        od_score = sum(combined_scores) / denominator
        length_component = sum(length_scores) / denominator
        geometry_component = sum(geometry_scores) / denominator
    else:
        od_score = _mean_or_zero(combined_scores)
        length_component = _mean_or_zero(length_scores)
        geometry_component = _mean_or_zero(geometry_scores)

    if ground_truth_paths and prediction_paths:
        vanilla_score = _clamp01(
            1.0
            - abs(ground_truth_paths[0].length - prediction_paths[0].length)
            / ground_truth_paths[0].length
        )
    else:
        vanilla_score = 0.0
    return (
        _clamp01(od_score),
        _clamp01(length_component),
        _clamp01(geometry_component),
        vanilla_score,
        path_scores,
    )


def _trace_chains(graph: nx.Graph) -> list[list[int]]:
    """Trace degree-2 chains and pure cycles for control-point sampling."""
    visited_edges: set[tuple[int, int]] = set()
    chains: list[list[int]] = []
    anchors = [node for node in graph if graph.degree[node] != 2]

    def trace(start: int, neighbor: int) -> list[int]:
        chain = [start, neighbor]
        visited_edges.add(_edge_key(start, neighbor))
        previous, current = start, neighbor
        while graph.degree[current] == 2:
            next_nodes = [node for node in graph.neighbors(current) if node != previous]
            if not next_nodes:
                break
            next_node = next_nodes[0]
            edge = _edge_key(current, next_node)
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            chain.append(next_node)
            previous, current = current, next_node
        return chain

    for anchor in anchors:
        for neighbor in graph.neighbors(anchor):
            if _edge_key(anchor, neighbor) not in visited_edges:
                chains.append(trace(anchor, neighbor))

    for u, v in graph.edges:
        if _edge_key(u, v) in visited_edges:
            continue
        cycle = trace(u, v)
        chains.append(cycle)
    return chains


def _nodes_along_chain(graph: nx.Graph, chain: Sequence[int], spacing: float) -> list[int]:
    if not chain:
        return []
    selected = [chain[0]]
    distance_since_selection = 0.0
    for u, v in itertools.pairwise(chain):
        distance_since_selection += float(graph[u][v]["length"])
        if distance_since_selection >= spacing:
            selected.append(v)
            distance_since_selection = 0.0
    if selected[-1] != chain[-1] and graph.degree[chain[-1]] != 2:
        selected.append(chain[-1])
    return selected


def _farthest_point_limit(
    graph: nx.Graph,
    candidates: Sequence[int],
    limit: int,
) -> list[int]:
    unique = list(dict.fromkeys(candidates))
    if limit <= 0 or len(unique) <= limit:
        return unique
    if limit == 1:
        return _farthest_point_limit(graph, unique, 1)
    coordinates = np.asarray(
        [(graph.nodes[node]["x"], graph.nodes[node]["y"]) for node in unique],
        dtype=float,
    )
    center = coordinates.mean(axis=0)
    first = int(np.argmin(np.sum((coordinates - center) ** 2, axis=1)))
    selected_indices = [first]
    min_distances = np.sum((coordinates - coordinates[first]) ** 2, axis=1)
    min_distances[first] = -1.0
    while len(selected_indices) < limit:
        next_index = int(np.argmax(min_distances))
        selected_indices.append(next_index)
        distances = np.sum((coordinates - coordinates[next_index]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distances)
        min_distances[selected_indices] = -1.0
    return [unique[index] for index in selected_indices]


def _component_balanced_limit(
    graph: nx.Graph,
    candidates: Sequence[int],
    limit: int,
) -> list[int]:
    """Keep multiple controls in the largest components under a global cap."""
    unique = list(dict.fromkeys(candidates))
    if limit <= 0 or len(unique) <= limit:
        return unique

    candidate_set = set(unique)
    groups = []
    for component in nx.connected_components(graph):
        component_candidates = [node for node in component if node in candidate_set]
        if len(component_candidates) < 2:
            continue
        edge_length = graph.subgraph(component).size(weight="length")
        groups.append((float(edge_length), component_candidates))
    groups.sort(key=lambda item: item[0], reverse=True)
    if not groups:
        return _farthest_point_limit(graph, unique, limit)

    # Four controls per retained component is a useful initial budget: enough
    # to form OD pairs without allowing hundreds of tiny components to consume
    # one control each.
    group_limit = max(1, limit // 4)
    groups = groups[:group_limit]
    allocations = [min(2, len(nodes)) for _, nodes in groups]
    remaining = limit - sum(allocations)
    while remaining > 0:
        choices = [
            (weight / (allocations[index] + 1), index)
            for index, (weight, nodes) in enumerate(groups)
            if allocations[index] < len(nodes)
        ]
        if not choices:
            break
        _, selected_group = max(choices)
        allocations[selected_group] += 1
        remaining -= 1

    selected = []
    for allocation, (_, nodes) in zip(allocations, groups):
        selected.extend(_farthest_point_limit(graph, nodes, allocation))
    return selected


def sample_control_points(
    graph: nx.Graph,
    spacing: float,
    max_control_points: int,
) -> list[int]:
    candidates = []
    for chain in _trace_chains(graph):
        candidates.extend(_nodes_along_chain(graph, chain, spacing))
    if not candidates:
        candidates = list(graph.nodes)
    return _component_balanced_limit(graph, candidates, max_control_points)


def _nodes_within_hops(graph: nx.Graph, source: int, hops: int) -> set[int]:
    if hops <= 0:
        return {source}
    return set(nx.single_source_shortest_path_length(graph, source, cutoff=hops))


def snap_control_points(
    graph_gt: nx.Graph,
    graph_pred: nx.Graph,
    control_points: Sequence[int],
    tolerance: float,
    exclusion_hops: int,
) -> dict[int, int | None]:
    """Greedily snap GT controls to nearby prediction nodes one-to-one."""
    pred_nodes = list(graph_pred.nodes)
    if not pred_nodes:
        return {gt_node: None for gt_node in control_points}
    pred_coordinates = np.asarray(
        [(graph_pred.nodes[node]["x"], graph_pred.nodes[node]["y"]) for node in pred_nodes],
        dtype=float,
    )
    tree = cKDTree(pred_coordinates)
    blocked: set[int] = set()
    used: set[int] = set()
    result: dict[int, int | None] = {}
    query_count = min(20, len(pred_nodes))

    for gt_node in control_points:
        gt_coordinate = (graph_gt.nodes[gt_node]["x"], graph_gt.nodes[gt_node]["y"])
        distances, indices = tree.query(gt_coordinate, k=query_count)
        distance_values = np.atleast_1d(distances)
        index_values = np.atleast_1d(indices)
        matched = None
        for distance, index in zip(distance_values, index_values):
            if not math.isfinite(float(distance)) or float(distance) > tolerance:
                break
            pred_node = pred_nodes[int(index)]
            # Preserve exact identity matches even when another control point
            # blocked nearby nodes. This makes identical graphs score exactly
            # one while retaining the original APLS-style exclusion for
            # non-exact nearest-neighbour snaps.
            exact_match = float(distance) <= 1e-6 and pred_node not in used
            if not exact_match and pred_node in blocked:
                continue
            if pred_node in used:
                continue
            matched = pred_node
            used.add(pred_node)
            blocked.update(_nodes_within_hops(graph_pred, pred_node, exclusion_hops))
            break
        result[gt_node] = matched
    return result


def select_od_pairs(
    graph: nx.Graph,
    control_points: Sequence[int],
    min_path_length: float,
    max_path_length: float,
    max_od_pairs: int,
    random_seed: int,
) -> list[tuple[int, int, float]]:
    valid_pairs = []
    control_set = set(control_points)
    for index, source in enumerate(control_points):
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="length")
        for target in control_points[index + 1 :]:
            if target not in control_set:
                continue
            length = lengths.get(target)
            if (
                length is not None
                and length >= min_path_length
                and (max_path_length <= 0.0 or length <= max_path_length)
            ):
                valid_pairs.append((source, target, float(length)))
    if max_od_pairs > 0 and len(valid_pairs) > max_od_pairs:
        rng = random.Random(random_seed)
        valid_pairs = rng.sample(valid_pairs, max_od_pairs)
        valid_pairs.sort(key=lambda pair: (pair[0], pair[1]))
    return valid_pairs


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def evaluate_direction(
    graph_gt: nx.Graph,
    graph_pred: nx.Graph,
    config: MetricConfig,
) -> dict[str, Any]:
    """Evaluate one ground-truth-to-prediction direction."""
    config.validate()
    control_points = sample_control_points(
        graph_gt,
        config.control_point_spacing,
        config.max_control_points,
    )
    snapped = snap_control_points(
        graph_gt,
        graph_pred,
        control_points,
        config.snap_tolerance,
        config.snap_exclusion_hops,
    )
    od_pairs = select_od_pairs(
        graph_gt,
        control_points,
        config.min_path_length,
        config.max_path_length,
        config.max_od_pairs,
        config.random_seed,
    )

    od_scores = []
    length_component_scores = []
    geometry_component_scores = []
    vanilla_scores = []
    matched_od_scores = []
    matched_length_component_scores = []
    matched_geometry_component_scores = []
    matched_vanilla_scores = []
    routable_od_scores = []
    routable_length_component_scores = []
    routable_geometry_component_scores = []
    routable_vanilla_scores = []
    aligned_geometry_distances = []
    aligned_length_scores = []
    aligned_geometry_scores = []
    aligned_combined_scores = []
    diagnostics = []
    ground_truth_route_counts = []
    prediction_route_counts = []
    matched_od_pairs = 0
    routable_od_pairs = 0
    for source, target, shortest_gt_length in od_pairs:
        pred_source = snapped[source]
        pred_target = snapped[target]
        endpoints_matched = pred_source is not None and pred_target is not None
        gt_paths = diverse_paths(
            graph_gt,
            source,
            target,
            config.k,
            config.theta,
            config.max_candidates,
        )
        pred_paths = (
            diverse_paths(
                graph_pred,
                pred_source,
                pred_target,
                config.k,
                config.theta,
                config.max_candidates,
            )
            if pred_source is not None and pred_target is not None
            else []
        )
        ground_truth_route_counts.append(len(gt_paths))
        prediction_route_counts.append(len(pred_paths))

        (
            od_score,
            length_component,
            geometry_component,
            vanilla_score,
            path_scores,
        ) = score_aligned_paths(
            gt_paths,
            pred_paths,
            config,
        )
        od_scores.append(od_score)
        length_component_scores.append(length_component)
        geometry_component_scores.append(geometry_component)
        vanilla_scores.append(vanilla_score)
        aligned_geometry_distances.extend(
            score["geometry_distance"]
            for score in path_scores
            if math.isfinite(score["geometry_distance"])
        )
        aligned_length_scores.extend(score["length_score"] for score in path_scores)
        aligned_geometry_scores.extend(score["geometry_score"] for score in path_scores)
        aligned_combined_scores.extend(score["combined_score"] for score in path_scores)
        if endpoints_matched:
            matched_od_pairs += 1
            matched_od_scores.append(od_score)
            matched_length_component_scores.append(length_component)
            matched_geometry_component_scores.append(geometry_component)
            matched_vanilla_scores.append(vanilla_score)
        if pred_paths:
            routable_od_pairs += 1
            routable_od_scores.append(od_score)
            routable_length_component_scores.append(length_component)
            routable_geometry_component_scores.append(geometry_component)
            routable_vanilla_scores.append(vanilla_score)

        if config.return_details:
            diagnostics.append(
                {
                    "source_gt": source,
                    "target_gt": target,
                    "source_pred": pred_source,
                    "target_pred": pred_target,
                    "shortest_gt_length": shortest_gt_length,
                    "ground_truth_path_count": len(gt_paths),
                    "prediction_path_count": len(pred_paths),
                    "aligned_path_count": len(path_scores),
                    "missing_prediction_alternatives": max(0, len(gt_paths) - len(pred_paths)),
                    "path_scores": path_scores,
                    "length_component": length_component,
                    "geometry_component": geometry_component,
                    "vanilla_apls_score": vanilla_score,
                    "od_score": od_score,
                    "endpoints_matched": endpoints_matched,
                    "disconnected": not pred_paths,
                }
            )

    matched_controls = sum(node is not None for node in snapped.values())
    result = {
        "k_apls_geometry": _mean_or_zero(od_scores),
        "k_apls_geometry_strict": _mean_or_zero(od_scores),
        "k_apls_geometry_matched": _mean_or_zero(matched_od_scores),
        "k_apls_geometry_routable": _mean_or_zero(routable_od_scores),
        "length_component_strict": _mean_or_zero(length_component_scores),
        "length_component_matched": _mean_or_zero(matched_length_component_scores),
        "length_component_routable": _mean_or_zero(routable_length_component_scores),
        "geometry_component_strict": _mean_or_zero(geometry_component_scores),
        "geometry_component_matched": _mean_or_zero(matched_geometry_component_scores),
        "geometry_component_routable": _mean_or_zero(routable_geometry_component_scores),
        "vanilla_apls": _mean_or_zero(vanilla_scores),
        "vanilla_apls_strict": _mean_or_zero(vanilla_scores),
        "vanilla_apls_matched": _mean_or_zero(matched_vanilla_scores),
        "vanilla_apls_routable": _mean_or_zero(routable_vanilla_scores),
        "control_points": len(control_points),
        "matched_control_points": matched_controls,
        "control_point_match_rate": matched_controls / len(control_points)
        if control_points
        else 0.0,
        "od_pairs": len(od_pairs),
        "matched_od_pairs": matched_od_pairs,
        "routable_od_pairs": routable_od_pairs,
        "endpoint_pair_coverage": matched_od_pairs / len(od_pairs) if od_pairs else 0.0,
        "connectivity_rate": routable_od_pairs / matched_od_pairs
        if matched_od_pairs
        else 0.0,
        "route_availability": routable_od_pairs / len(od_pairs) if od_pairs else 0.0,
        "aligned_path_pairs": len(aligned_geometry_distances),
        "mean_aligned_hausdorff_m": _mean_or_zero(aligned_geometry_distances),
        "mean_aligned_length_score": _mean_or_zero(aligned_length_scores),
        "mean_aligned_geometry_score": _mean_or_zero(aligned_geometry_scores),
        "mean_aligned_combined_score": _mean_or_zero(aligned_combined_scores),
        "mean_ground_truth_routes": _mean_or_zero(ground_truth_route_counts),
        "mean_prediction_routes": _mean_or_zero(prediction_route_counts),
    }
    if config.return_details:
        result["details"] = diagnostics
    return result


def evaluate_graphs(
    graph_gt: nx.Graph,
    graph_pred: nx.Graph,
    config: MetricConfig,
    symmetric: bool = False,
) -> dict[str, Any]:
    forward = evaluate_direction(graph_gt, graph_pred, config)
    directions = {"ground_truth_to_prediction": forward}
    score_keys = (
        "k_apls_geometry_strict",
        "k_apls_geometry_matched",
        "k_apls_geometry_routable",
        "length_component_strict",
        "length_component_matched",
        "length_component_routable",
        "geometry_component_strict",
        "geometry_component_matched",
        "geometry_component_routable",
        "vanilla_apls_strict",
        "vanilla_apls_matched",
        "vanilla_apls_routable",
    )
    if symmetric:
        reverse = evaluate_direction(graph_pred, graph_gt, config)
        directions["prediction_to_ground_truth"] = reverse
        scores = {
            key: (forward[key] + reverse[key]) / 2.0
            for key in score_keys
        }
    else:
        scores = {key: forward[key] for key in score_keys}
    return {
        "k_apls_geometry": float(scores["k_apls_geometry_strict"]),
        "vanilla_apls": float(scores["vanilla_apls_strict"]),
        **{key: float(value) for key, value in scores.items()},
        "symmetric": symmetric,
        "directions": directions,
    }


def k_apls_geometry(
    graph_gt: nx.Graph,
    graph_pred: nx.Graph,
    *,
    return_details: bool = False,
    symmetric: bool = False,
    **kwargs: Any,
) -> float | tuple[float, dict[str, Any]]:
    """Programmatic API for already projected metre-based NetworkX graphs."""
    config = MetricConfig(return_details=return_details, **kwargs)
    result = evaluate_graphs(graph_gt, graph_pred, config, symmetric=symmetric)
    if return_details:
        return result["k_apls_geometry"], result
    return result["k_apls_geometry"]


def _graph_summary(graph: nx.Graph) -> dict[str, Any]:
    components = list(nx.connected_components(graph))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "connected_components": len(components),
        "largest_component_nodes": max((len(component) for component in components), default=0),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", help="GT legacy APLS JSON or GeoJSON")
    parser.add_argument("prediction", help="prediction legacy APLS JSON or GeoJSON")
    parser.add_argument("--output", help="write summary JSON to this path")
    parser.add_argument("--details-output", help="write full per-OD diagnostics JSON")
    parser.add_argument("--ground-truth-crs", help="override GT source CRS, e.g. EPSG:4326")
    parser.add_argument("--prediction-crs", help="override prediction source CRS")
    parser.add_argument("--metric-crs", help="projected metre CRS used for scoring")
    parser.add_argument("--node-merge-tolerance", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--theta", type=float, default=0.6)
    parser.add_argument("--w-length", type=float, default=0.5)
    parser.add_argument("--w-geometry", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument(
        "--geometry-metric",
        choices=["hausdorff", "frechet"],
        default="hausdorff",
    )
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument(
        "--no-penalize-missing-routes",
        action="store_true",
        help="average only successfully aligned routes",
    )
    parser.add_argument("--control-point-spacing", type=float, default=30.0)
    parser.add_argument(
        "--max-control-points",
        type=int,
        default=160,
        help="0 keeps all sampled controls",
    )
    parser.add_argument(
        "--max-od-pairs",
        type=int,
        default=500,
        help="0 evaluates all valid control-point pairs",
    )
    parser.add_argument("--min-path-length", type=float, default=60.0)
    parser.add_argument(
        "--max-path-length",
        type=float,
        default=0.0,
        help="0 disables the upper OD-path-length limit",
    )
    parser.add_argument("--snap-tolerance", type=float, default=4.0)
    parser.add_argument("--snap-exclusion-hops", type=int, default=4)
    parser.add_argument("--densify-spacing", type=float, default=5.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="also evaluate prediction-to-GT and average both directions",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    return_details = bool(args.details_output)
    config = MetricConfig(
        k=args.k,
        theta=args.theta,
        w_length=args.w_length,
        w_geometry=args.w_geometry,
        tau=args.tau,
        geometry_metric=args.geometry_metric,
        max_candidates=args.max_candidates,
        penalize_missing_routes=not args.no_penalize_missing_routes,
        return_details=return_details,
        control_point_spacing=args.control_point_spacing,
        max_control_points=args.max_control_points,
        max_od_pairs=args.max_od_pairs,
        min_path_length=args.min_path_length,
        max_path_length=args.max_path_length,
        snap_tolerance=args.snap_tolerance,
        snap_exclusion_hops=args.snap_exclusion_hops,
        densify_spacing=args.densify_spacing,
        random_seed=args.random_seed,
    )
    config.validate()
    graph_gt, graph_pred, metric_crs, metadata = load_metric_graphs(
        args.ground_truth,
        args.prediction,
        args.ground_truth_crs,
        args.prediction_crs,
        args.metric_crs,
        args.node_merge_tolerance,
        args.densify_spacing,
    )
    result = evaluate_graphs(graph_gt, graph_pred, config, symmetric=args.symmetric)
    result.update(
        {
            "ground_truth": str(Path(args.ground_truth).resolve()),
            "prediction": str(Path(args.prediction).resolve()),
            "input": metadata,
            "metric_crs": metric_crs.to_string(),
            "parameters": asdict(config),
            "ground_truth_graph": _graph_summary(graph_gt),
            "prediction_graph": _graph_summary(graph_pred),
        }
    )
    safe_result = _json_safe(result)

    summary_result = json.loads(json.dumps(safe_result))
    for direction in summary_result["directions"].values():
        direction.pop("details", None)
    print(json.dumps(summary_result, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary_result, indent=2) + "\n", encoding="utf-8")
    if args.details_output:
        details_path = Path(args.details_output)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text(json.dumps(safe_result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
