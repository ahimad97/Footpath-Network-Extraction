#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx
from shapely.geometry import LineString

from k_apls_geometry import (
    MetricConfig,
    PathRecord,
    diverse_paths,
    evaluate_direction,
    load_metric_graphs,
    path_overlap,
    read_network,
    score_aligned_paths,
    score_path_pair,
)


def add_node(graph, node, x, y):
    graph.add_node(node, x=float(x), y=float(y))


def add_edge(graph, u, v):
    x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
    x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]
    graph.add_edge(u, v, length=((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)


def path_record(points):
    line = LineString(points)
    return PathRecord(tuple(range(len(points))), line.length, line, {})


class KAPLSGeometryTests(unittest.TestCase):
    def config(self, **overrides):
        values = {
            "k": 3,
            "theta": 0.6,
            "tau": 20.0,
            "control_point_spacing": 10.0,
            "max_control_points": 20,
            "max_od_pairs": 20,
            "min_path_length": 1.0,
            "snap_tolerance": 1.0,
            "snap_exclusion_hops": 0,
            "densify_spacing": 5.0,
            "return_details": True,
        }
        values.update(overrides)
        return MetricConfig(**values)

    def test_identical_graphs_score_one(self):
        graph = nx.Graph()
        for node, point in enumerate([(0, 0), (10, 0), (20, 0), (30, 0)]):
            add_node(graph, node, *point)
        for u, v in [(0, 1), (1, 2), (2, 3)]:
            add_edge(graph, u, v)
        result = evaluate_direction(
            graph,
            graph.copy(),
            self.config(k=1, snap_exclusion_hops=4),
        )
        self.assertAlmostEqual(result["k_apls_geometry"], 1.0)
        self.assertAlmostEqual(result["k_apls_geometry_matched"], 1.0)
        self.assertAlmostEqual(result["k_apls_geometry_routable"], 1.0)
        self.assertAlmostEqual(result["vanilla_apls"], 1.0)

    def test_same_length_shifted_geometry_is_penalized(self):
        gt = path_record([(0, 0), (100, 0)])
        pred = path_record([(0, 10), (100, 10)])
        score = score_path_pair(gt, pred, self.config(k=1, tau=20.0))
        self.assertAlmostEqual(score["length_score"], 1.0)
        self.assertAlmostEqual(score["geometry_score"], 0.5)
        self.assertAlmostEqual(score["combined_score"], 0.75)

    def test_disconnected_prediction_scores_zero(self):
        gt = nx.Graph()
        pred = nx.Graph()
        for graph in (gt, pred):
            for node, point in enumerate([(0, 0), (10, 0), (20, 0)]):
                add_node(graph, node, *point)
        add_edge(gt, 0, 1)
        add_edge(gt, 1, 2)
        result = evaluate_direction(gt, pred, self.config(k=1))
        self.assertEqual(result["k_apls_geometry"], 0.0)
        self.assertEqual(result["k_apls_geometry_matched"], 0.0)
        self.assertEqual(result["connectivity_rate"], 0.0)
        self.assertEqual(result["route_availability"], 0.0)

    def test_missing_alternative_route_is_penalized(self):
        gt = nx.Graph()
        pred = nx.Graph()
        points = {
            0: (0, 0),
            1: (10, 10),
            2: (10, -10),
            3: (20, 0),
        }
        for graph in (gt, pred):
            for node, point in points.items():
                add_node(graph, node, *point)
        for edge in [(0, 1), (1, 3), (0, 2), (2, 3)]:
            add_edge(gt, *edge)
        for edge in [(0, 1), (1, 3)]:
            add_edge(pred, *edge)
        config = self.config(k=2, theta=0.5, min_path_length=20.0)
        gt_paths = diverse_paths(gt, 0, 3, 2, config.theta, config.max_candidates)
        pred_paths = diverse_paths(pred, 0, 3, 2, config.theta, config.max_candidates)
        self.assertEqual(len(gt_paths), 2)
        self.assertEqual(len(pred_paths), 1)
        od_score, length_score, geometry_score, _, path_scores = score_aligned_paths(
            gt_paths,
            pred_paths,
            config,
        )
        self.assertAlmostEqual(path_scores[0]["combined_score"], 1.0)
        self.assertAlmostEqual(od_score, 0.5)
        self.assertAlmostEqual(length_score, 0.5)
        self.assertAlmostEqual(geometry_score, 0.5)

    def test_high_overlap_candidate_is_rejected(self):
        path_a = PathRecord((0, 1, 2), 20.0, LineString([(0, 0), (10, 0), (20, 0)]), {
            (0, 1): 10.0,
            (1, 2): 10.0,
        })
        path_b = PathRecord((0, 1, 3, 2), 22.0, LineString([(0, 0), (10, 0), (15, 2), (20, 0)]), {
            (0, 1): 10.0,
            (1, 3): 6.0,
            (2, 3): 6.0,
        })
        self.assertAlmostEqual(path_overlap(path_a, path_b), 0.5)
        self.assertFalse(path_overlap(path_a, path_b) < 0.5)

    def test_geojson_latlon_is_detected(self):
        data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-0.15, 51.51], [-0.149, 51.511]],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.geojson"
            path.write_text(json.dumps(data), encoding="utf-8")
            raw = read_network(path)
            _, _, metric_crs, _ = load_metric_graphs(path, path)
        self.assertEqual(raw.crs.to_epsg(), 4326)
        self.assertTrue(metric_crs.is_projected)


if __name__ == "__main__":
    unittest.main()
