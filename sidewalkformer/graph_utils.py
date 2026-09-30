"""Graph utilities for reference graphs stored in the Sat2Graph adjacency format.

The adjacency format is a dict ``{(row, col): [(row, col), ...]}`` mapping each
vertex to its neighbours in image pixel coordinates.
"""

from collections import deque

import igraph as ig
import numpy as np
import rtree
import scipy.spatial
from shapely.geometry import LineString, Point


def edge_list_to_adj_table(nodes, edges):
    """Directed edge list -> list of neighbour sets, one per node."""
    adj_table = [set() for _ in range(len(nodes))]
    for start_idx, end_idx in edges:
        adj_table[start_idx].add(end_idx)
    return adj_table


def convert_to_sat2graph_format(nodes, edges):
    """Nodes [N, 2] (row, col) and undirected edges [E, 2] -> adjacency dict.

    Coordinates are rounded to integers; each edge is stored in both
    directions. Nodes are expected to be unique after rounding.
    """
    all_edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    adj_table = edge_list_to_adj_table(nodes, all_edges)
    int_nodes = [(round(x), round(y)) for x, y in nodes]
    return {
        int_nodes[node_idx]: [int_nodes[n] for n in neighbor_indices]
        for node_idx, neighbor_indices in enumerate(adj_table)
    }


def convert_from_sat2graph_format(graph):
    """Adjacency dict -> (nodes [N, 2] array, list of directed (src, dst) edges).

    Edges are not de-duplicated, so each undirected edge appears twice.
    """
    node_to_idx = {}
    for node, neighbors in graph.items():
        node_to_idx.setdefault(node, len(node_to_idx))
        for neighbor in neighbors:
            node_to_idx.setdefault(neighbor, len(node_to_idx))

    edges = [
        (node_to_idx[node], node_to_idx[neighbor])
        for node, neighbors in graph.items()
        for neighbor in neighbors
    ]
    nodes = [None] * len(node_to_idx)
    for node, idx in node_to_idx.items():
        nodes[idx] = node
    return np.array(nodes), edges


def igraph_from_adj_dict(graph, coord_transform):
    """Adjacency dict -> undirected igraph with a vertex attribute ``point``.

    ``coord_transform`` maps the [N, 2] node array to the stored coordinates,
    e.g. ``lambda v: v[:, ::-1]`` for (row, col) -> (x, y).
    """
    nodes, edges = convert_from_sat2graph_format(graph)
    if nodes.shape[0] == 0:
        nodes = np.zeros((0, 2), dtype=nodes.dtype)
    edges = {(min(src, tgt), max(src, tgt)) for src, tgt in edges}
    g = ig.Graph(nodes.shape[0], list(edges))
    g.vs['point'] = coord_transform(nodes)
    return g


def _line_bbox(line):
    (x0, y0), (x1, y1) = line
    return (min(x0, x1) - 1, min(y0, y1) - 1, max(x0, x1) + 1, max(y0, y1) + 1)


def find_intersection(segment1, segment2):
    """Interior intersection point of two segments ``((x1, y1), (x2, y2))``, or None.

    Intersections at segment endpoints and collinear overlaps are ignored.
    """
    line1 = LineString([segment1[0], segment1[1]])
    line2 = LineString([segment2[0], segment2[1]])
    intersection = line1.intersection(line2)
    if intersection.is_empty or intersection.geom_type != 'Point':
        return None
    endpoints = [segment1[0], segment1[1], segment2[0], segment2[1]]
    if any(intersection.equals(Point(x, y)) for x, y in endpoints):
        return None
    return (intersection.x, intersection.y)


def find_crossover_points(graph):
    """Points where two edges of an igraph cross without sharing a vertex.

    Uses an R-tree over edge bounding boxes to limit the pairwise tests.
    """
    points = graph.vs['point']
    lines = [(points[edge.source], points[edge.target]) for edge in graph.es]
    line_bboxes = [_line_bbox(line) for line in lines]
    line_index = rtree.index.Index()
    for idx, bbox in enumerate(line_bboxes):
        line_index.insert(idx, bbox)

    crossover_points = []
    tested_pairs = set()
    for i, line_0 in enumerate(lines):
        for ni in line_index.intersection(line_bboxes[i]):
            pair = (min(i, ni), max(i, ni))
            if pair in tested_pairs:
                continue
            itsc = find_intersection(line_0, lines[ni])
            if itsc is not None:
                crossover_points.append(itsc)
            tested_pairs.add(pair)
    return crossover_points


def find_near_tree_nodes(graph_sidewalk, tree_coords, max_distance=30):
    """Graph vertices within ``max_distance`` of any tree location.

    Returns (list of vertex coordinates, list of vertex indices).
    """
    sidewalk_coords = np.array(graph_sidewalk.vs['point']).astype(int)
    tree_kd_tree = scipy.spatial.KDTree(tree_coords)
    close_nodes, close_indexes = [], []
    for idx, coord in enumerate(sidewalk_coords):
        if len(tree_kd_tree.query_ball_point(coord, max_distance)) > 0:
            close_nodes.append(coord.tolist())
            close_indexes.append(idx)
    return close_nodes, close_indexes


def nms_points_training(points, scores, radius, return_indices=False):
    """NMS used when sampling training nodes; points with score > 1 are never suppressed."""
    sorted_indices = np.argsort(scores)[::-1]
    sorted_points = points[sorted_indices, :]
    sorted_scores = scores[sorted_indices]
    kept = np.ones(sorted_indices.shape[0], dtype=bool)
    tree = scipy.spatial.KDTree(sorted_points)
    for idx, p in enumerate(sorted_points):
        if not kept[idx]:
            continue
        neighbor_indices = tree.query_ball_point(p, r=radius)
        kept[neighbor_indices] = np.greater(sorted_scores[neighbor_indices], 1.0)
        kept[idx] = True
    if return_indices:
        return sorted_points[kept], sorted_indices[kept]
    return sorted_points[kept]


def bfs_with_conditions(graph, start_node, stop_nodes, max_depth):
    """Breadth-first search on an igraph from ``start_node``.

    Expansion stops at nodes in ``stop_nodes`` and at depth ``max_depth``.
    Returns the set of visited node indices (stop nodes included).
    """
    visited = set()
    queue = deque([(start_node, 0)])
    while queue:
        current_node, current_depth = queue.popleft()
        visited.add(current_node)
        if current_node in stop_nodes or current_depth >= max_depth:
            continue
        for neighbor in graph.neighbors(current_node, mode="all"):
            if neighbor not in visited:
                queue.append((neighbor, current_depth + 1))
    return visited
