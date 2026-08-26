#!/usr/bin/env python3
"""Evaluate K-APLS + Geometry tile-by-tile from region network files."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from k_apls_geometry import (
    MetricConfig,
    _graph_summary,
    _json_safe,
    choose_metric_crs,
    clip_network_wgs84,
    evaluate_graphs,
    metric_graph_from_raw,
    read_network,
)


def _weighted_mean(rows: Sequence[dict[str, Any]], value_key: str, weight_key: str) -> float:
    denominator = sum(float(row[weight_key]) for row in rows)
    if denominator <= 0.0:
        return 0.0
    return float(
        sum(float(row[value_key]) * float(row[weight_key]) for row in rows)
        / denominator
    )


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else 0.0


def evaluate_tiles(
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    bboxes: Sequence[Sequence[float]],
    config: MetricConfig,
    ground_truth_crs: str | None = None,
    prediction_crs: str | None = None,
    metric_crs: str | None = None,
    node_merge_tolerance: float = 0.01,
    symmetric: bool = False,
) -> dict[str, Any]:
    raw_gt = read_network(ground_truth_path, ground_truth_crs)
    raw_pred = read_network(prediction_path, prediction_crs)
    target_crs = choose_metric_crs(raw_gt, metric_crs)

    rows = []
    for tile_index, bbox in enumerate(bboxes):
        clipped_gt = clip_network_wgs84(raw_gt, bbox)
        if not clipped_gt.polylines:
            continue
        clipped_pred = clip_network_wgs84(raw_pred, bbox)
        graph_gt = metric_graph_from_raw(
            clipped_gt,
            target_crs,
            node_merge_tolerance,
            config.densify_spacing,
        )
        graph_pred = metric_graph_from_raw(
            clipped_pred,
            target_crs,
            node_merge_tolerance,
            config.densify_spacing,
            allow_empty=True,
        )
        result = evaluate_graphs(graph_gt, graph_pred, config, symmetric=symmetric)
        direction = result["directions"]["ground_truth_to_prediction"]
        row = {
            "tile": tile_index,
            "south": float(bbox[0]),
            "west": float(bbox[1]),
            "north": float(bbox[2]),
            "east": float(bbox[3]),
            "gt_nodes": graph_gt.number_of_nodes(),
            "gt_edges": graph_gt.number_of_edges(),
            "pred_nodes": graph_pred.number_of_nodes(),
            "pred_edges": graph_pred.number_of_edges(),
            "k_apls_geometry_strict": result["k_apls_geometry_strict"],
            "k_apls_geometry_matched": result["k_apls_geometry_matched"],
            "k_apls_geometry_routable": result["k_apls_geometry_routable"],
            "length_component_strict": result["length_component_strict"],
            "length_component_matched": result["length_component_matched"],
            "length_component_routable": result["length_component_routable"],
            "geometry_component_strict": result["geometry_component_strict"],
            "geometry_component_matched": result["geometry_component_matched"],
            "geometry_component_routable": result["geometry_component_routable"],
            "vanilla_apls_strict": result["vanilla_apls_strict"],
            "vanilla_apls_matched": result["vanilla_apls_matched"],
            "vanilla_apls_routable": result["vanilla_apls_routable"],
            "control_points": direction["control_points"],
            "matched_control_points": direction["matched_control_points"],
            "control_point_match_rate": direction["control_point_match_rate"],
            "od_pairs": direction["od_pairs"],
            "matched_od_pairs": direction["matched_od_pairs"],
            "routable_od_pairs": direction["routable_od_pairs"],
            "endpoint_pair_coverage": direction["endpoint_pair_coverage"],
            "connectivity_rate": direction["connectivity_rate"],
            "route_availability": direction["route_availability"],
            "aligned_path_pairs": direction["aligned_path_pairs"],
            "mean_aligned_hausdorff_m": direction["mean_aligned_hausdorff_m"],
            "mean_aligned_length_score": direction["mean_aligned_length_score"],
            "mean_aligned_geometry_score": direction["mean_aligned_geometry_score"],
            "mean_aligned_combined_score": direction["mean_aligned_combined_score"],
            "mean_ground_truth_routes": direction["mean_ground_truth_routes"],
            "mean_prediction_routes": direction["mean_prediction_routes"],
        }
        rows.append(row)

    scored_rows = [row for row in rows if row["od_pairs"] > 0]
    available_rows = [row for row in scored_rows if row["pred_edges"] > 0]
    matched_rows = [row for row in scored_rows if row["matched_od_pairs"] > 0]
    routable_rows = [row for row in scored_rows if row["routable_od_pairs"] > 0]
    summary = {
        "mode": "tile_average",
        "ground_truth": str(Path(ground_truth_path).resolve()),
        "prediction": str(Path(prediction_path).resolve()),
        "metric_crs": target_crs.to_string(),
        "bbox_count": len(bboxes),
        "tiles_with_ground_truth": len(rows),
        "tiles_scored": len(scored_rows),
        "tiles_with_prediction": len(available_rows),
        "prediction_tile_coverage": len(available_rows) / len(scored_rows)
        if scored_rows
        else 0.0,
        "tiles_with_matched_od": len(matched_rows),
        "tiles_with_routable_od": len(routable_rows),
        "k_apls_geometry_strict_macro_all_tiles": _mean(
            scored_rows, "k_apls_geometry_strict"
        ),
        "k_apls_geometry_strict_macro_available_tiles": _mean(
            available_rows, "k_apls_geometry_strict"
        ),
        "k_apls_geometry_matched_macro": _mean(
            matched_rows, "k_apls_geometry_matched"
        ),
        "k_apls_geometry_routable_macro": _mean(
            routable_rows, "k_apls_geometry_routable"
        ),
        "k_apls_geometry_strict_od_weighted": _weighted_mean(
            scored_rows, "k_apls_geometry_strict", "od_pairs"
        ),
        "k_apls_geometry_matched_od_weighted": _weighted_mean(
            matched_rows, "k_apls_geometry_matched", "matched_od_pairs"
        ),
        "k_apls_geometry_routable_od_weighted": _weighted_mean(
            routable_rows, "k_apls_geometry_routable", "routable_od_pairs"
        ),
        "length_component_strict_od_weighted": _weighted_mean(
            scored_rows, "length_component_strict", "od_pairs"
        ),
        "length_component_matched_od_weighted": _weighted_mean(
            matched_rows, "length_component_matched", "matched_od_pairs"
        ),
        "length_component_routable_od_weighted": _weighted_mean(
            routable_rows, "length_component_routable", "routable_od_pairs"
        ),
        "geometry_component_strict_od_weighted": _weighted_mean(
            scored_rows, "geometry_component_strict", "od_pairs"
        ),
        "geometry_component_matched_od_weighted": _weighted_mean(
            matched_rows, "geometry_component_matched", "matched_od_pairs"
        ),
        "geometry_component_routable_od_weighted": _weighted_mean(
            routable_rows, "geometry_component_routable", "routable_od_pairs"
        ),
        "vanilla_apls_strict_od_weighted": _weighted_mean(
            scored_rows, "vanilla_apls_strict", "od_pairs"
        ),
        "vanilla_apls_matched_od_weighted": _weighted_mean(
            matched_rows, "vanilla_apls_matched", "matched_od_pairs"
        ),
        "vanilla_apls_routable_od_weighted": _weighted_mean(
            routable_rows, "vanilla_apls_routable", "routable_od_pairs"
        ),
        "od_pairs": sum(row["od_pairs"] for row in scored_rows),
        "matched_od_pairs": sum(row["matched_od_pairs"] for row in scored_rows),
        "routable_od_pairs": sum(row["routable_od_pairs"] for row in scored_rows),
        "endpoint_pair_coverage": (
            sum(row["matched_od_pairs"] for row in scored_rows)
            / sum(row["od_pairs"] for row in scored_rows)
        )
        if sum(row["od_pairs"] for row in scored_rows)
        else 0.0,
        "connectivity_rate": (
            sum(row["routable_od_pairs"] for row in scored_rows)
            / sum(row["matched_od_pairs"] for row in scored_rows)
        )
        if sum(row["matched_od_pairs"] for row in scored_rows)
        else 0.0,
        "route_availability": (
            sum(row["routable_od_pairs"] for row in scored_rows)
            / sum(row["od_pairs"] for row in scored_rows)
        )
        if sum(row["od_pairs"] for row in scored_rows)
        else 0.0,
        "aligned_path_pairs": sum(row["aligned_path_pairs"] for row in scored_rows),
        "mean_aligned_hausdorff_m": (
            sum(
                row["mean_aligned_hausdorff_m"] * row["aligned_path_pairs"]
                for row in scored_rows
            )
            / sum(row["aligned_path_pairs"] for row in scored_rows)
        )
        if sum(row["aligned_path_pairs"] for row in scored_rows)
        else 0.0,
        "mean_aligned_length_score": (
            sum(
                row["mean_aligned_length_score"] * row["aligned_path_pairs"]
                for row in scored_rows
            )
            / sum(row["aligned_path_pairs"] for row in scored_rows)
        )
        if sum(row["aligned_path_pairs"] for row in scored_rows)
        else 0.0,
        "mean_aligned_geometry_score": (
            sum(
                row["mean_aligned_geometry_score"] * row["aligned_path_pairs"]
                for row in scored_rows
            )
            / sum(row["aligned_path_pairs"] for row in scored_rows)
        )
        if sum(row["aligned_path_pairs"] for row in scored_rows)
        else 0.0,
        "mean_aligned_combined_score": (
            sum(
                row["mean_aligned_combined_score"] * row["aligned_path_pairs"]
                for row in scored_rows
            )
            / sum(row["aligned_path_pairs"] for row in scored_rows)
        )
        if sum(row["aligned_path_pairs"] for row in scored_rows)
        else 0.0,
        "parameters": asdict(config),
        "ground_truth_region_graph": _graph_summary(
            metric_graph_from_raw(
                raw_gt,
                target_crs,
                node_merge_tolerance,
                config.densify_spacing,
            )
        ),
        "prediction_region_graph": _graph_summary(
            metric_graph_from_raw(
                raw_pred,
                target_crs,
                node_merge_tolerance,
                config.densify_spacing,
                allow_empty=True,
            )
        ),
        "tiles": rows,
    }
    return _json_safe(summary)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth")
    parser.add_argument("prediction")
    parser.add_argument("bbox_file", help="JSON list of [south, west, north, east] bboxes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    parser.add_argument("--ground-truth-crs")
    parser.add_argument("--prediction-crs")
    parser.add_argument("--metric-crs")
    parser.add_argument("--node-merge-tolerance", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--theta", type=float, default=0.6)
    parser.add_argument("--w-length", type=float, default=0.5)
    parser.add_argument("--w-geometry", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=20.0)
    parser.add_argument("--geometry-metric", choices=["hausdorff", "frechet"], default="hausdorff")
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--no-penalize-missing-routes", action="store_true")
    parser.add_argument("--control-point-spacing", type=float, default=30.0)
    parser.add_argument("--max-control-points", type=int, default=40)
    parser.add_argument("--max-od-pairs", type=int, default=50)
    parser.add_argument("--min-path-length", type=float, default=30.0)
    parser.add_argument(
        "--max-path-length",
        type=float,
        default=300.0,
        help="local OD limit; 0 disables the upper bound",
    )
    parser.add_argument("--snap-tolerance", type=float, default=4.0)
    parser.add_argument("--snap-exclusion-hops", type=int, default=4)
    parser.add_argument("--densify-spacing", type=float, default=5.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--symmetric", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    with Path(args.bbox_file).open("r", encoding="utf-8") as file:
        bboxes = json.load(file)
    config = MetricConfig(
        k=args.k,
        theta=args.theta,
        w_length=args.w_length,
        w_geometry=args.w_geometry,
        tau=args.tau,
        geometry_metric=args.geometry_metric,
        max_candidates=args.max_candidates,
        penalize_missing_routes=not args.no_penalize_missing_routes,
        return_details=False,
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
    result = evaluate_tiles(
        args.ground_truth,
        args.prediction,
        bboxes,
        config,
        args.ground_truth_crs,
        args.prediction_crs,
        args.metric_crs,
        args.node_merge_tolerance,
        args.symmetric,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.csv_output:
        write_csv(Path(args.csv_output), result["tiles"])

    printed = dict(result)
    printed.pop("tiles", None)
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    main()
