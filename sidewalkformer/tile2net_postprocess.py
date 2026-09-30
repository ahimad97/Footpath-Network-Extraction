"""Segmentation mask -> pedestrian network, using Tile2Net's post-processing.

Converts a class-labelled mask into polygons, then into sidewalk/crosswalk
centrelines and a node/edge graph with the helper functions of Tile2Net's
``PedNet`` (``third_party/tile2net``), so the result matches Tile2Net.

Usage::

    from sidewalkformer.tile2net_postprocess import seg_mask_to_graph

    nodes_rc, edges, lines_gdf = seg_mask_to_graph(
        pred_mask,                       # H x W uint8 {0: bg, 1: sidewalk, 2: road, 3: crossing}
        bbox=(minx, miny, maxx, maxy),   # EPSG:4326 bounds
        class_map={1: "sidewalk", 2: "road", 3: "crosswalk"},
    )
"""
from __future__ import annotations

import logging
import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.affinity
from pyproj import Transformer
from rasterio.features import shapes as rio_shapes
from rasterio.transform import Affine
from rasterio.transform import from_bounds as rio_from_bounds
from scipy.spatial import KDTree
from shapely.geometry import MultiLineString, Point
from shapely.geometry import shape as shp_shape
from shapely.validation import make_valid
from tile2net.raster.tile_utils.geodata_utils import buff_dfs, buffer_union_erode, geo2geodf, set_gdf_crs
from tile2net.raster.tile_utils.topology import (
    clean_deadend_dangles,
    extend_lines,
    fill_holes,
    get_crosswalk_cnl,
    get_extrapolated_line,
    get_line_sepoints,
    get_shortest,
    morpho_atts,
    remove_false_nodes,
    replace_convexhull,
    to_cline,
    trim_checkempty,
    wrinkle_remover,
)

METRIC_CRS = "EPSG:3857"

# Silence shapely warnings from degenerate minimum_rotated_rectangle inputs.
warnings.filterwarnings('ignore', message='.*divide by zero.*oriented_envelope.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*invalid value.*oriented_envelope.*', category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Mask -> polygons. Mirrors Tile2Net's tile.mask_to_poly_geojson (pixel space)
#    -> convert_poly_coords (georeference) -> mask2poly (per-class fill_holes and
#    replace_convexhull in a metric CRS), without needing Tile2Net Tile objects.
# ═══════════════════════════════════════════════════════════════════════════════

# Per-class hole sizes from Tile2Net's map_features() in tile.py.
_CLASS_HOLE_SIZES = {"sidewalk": 25, "crosswalk": 15, "road": 30}


def _empty_result():
    empty2 = np.zeros((0, 2), np.int32)
    return empty2, empty2, gpd.GeoDataFrame({"f_type": [], "geometry": []}, crs="EPSG:4326")


def mask_to_polygons(
    pred_mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    class_map: dict[int, str] | None = None,
    min_area: float = 20,
    simplify_tol: float = 0.8,
) -> "gpd.GeoDataFrame":
    """
    Convert a class-labelled mask to geo-referenced polygons in EPSG:4326.

    Replicates Tile2Net's mask -> polygon pipeline:
      1. Polygonize with an identity transform (pixel coordinates)
      2. Drop polygons smaller than ``min_area`` pixels²
      3. ``simplify(simplify_tol)`` in pixel space (sub-pixel smoothing)
      4. Apply the affine transform to EPSG:4326
      5. Convert to metric CRS (EPSG:3857)
      6. ``fill_holes`` with per-class hole sizes (sw=25, cw=15, rd=30)
      7. ``replace_convexhull``
      8. Convert back to EPSG:4326

    Parameters
    ----------
    pred_mask : H×W uint8 with class labels.
    bbox      : (minx, miny, maxx, maxy) in EPSG:4326.
    class_map : mapping {class_id → "f_type"}.
    min_area  : minimum polygon area in pixels² (Tile2Net's default).
    simplify_tol : Douglas-Peucker tolerance in *pixels* (default 0.8).
    """
    if class_map is None:
        class_map = {1: "sidewalk", 2: "road", 3: "crosswalk"}

    H, W = pred_mask.shape
    minx, miny, maxx, maxy = map(float, bbox)

    # Pixel (col, row) -> EPSG:4326 (lon, lat), as shapely affine parameters.
    geo_tfm = rio_from_bounds(minx, miny, maxx, maxy, W, H)
    geo_affine_params = [geo_tfm.a, geo_tfm.b, geo_tfm.d, geo_tfm.e, geo_tfm.c, geo_tfm.f]
    identity_tfm = Affine(1, 0, 0, 0, 1, 0)

    all_class_gdfs = []
    for cid, fname in class_map.items():
        bmask = (pred_mask == cid).astype("uint8")
        if bmask.max() == 0:
            continue

        # Steps 1-2: polygonize in pixel coordinates, drop small polygons.
        pixel_polys = []
        for geom, _ in rio_shapes(bmask, mask=bmask.astype(bool), transform=identity_tfm):
            g = make_valid(shp_shape(geom).buffer(0.0))
            if not g.is_empty and g.area >= min_area:
                pixel_polys.append(g)
        if not pixel_polys:
            continue

        # Steps 3-4: simplify in pixel space, then georeference.
        if simplify_tol > 0:
            pixel_polys = [p.simplify(simplify_tol) for p in pixel_polys]
        geo_polys = [shapely.affinity.affine_transform(p, geo_affine_params) for p in pixel_polys]

        class_gdf = gpd.GeoDataFrame(
            {"f_type": [fname] * len(geo_polys), "geometry": geo_polys},
            crs="EPSG:4326",
        )
        class_gdf = class_gdf[class_gdf.geometry.notna()]
        class_gdf = class_gdf[class_gdf.geometry.is_valid]
        if class_gdf.empty:
            continue

        # Steps 5-8: per-class fill_holes + replace_convexhull in a metric CRS.
        class_metric = (
            class_gdf.to_crs(METRIC_CRS)
            .explode(index_parts=False)
            .reset_index(drop=True)
        )
        class_metric = class_metric[~class_metric.geometry.isna()]
        class_metric["geometry"] = class_metric.apply(
            fill_holes, args=(_CLASS_HOLE_SIZES.get(fname, 25),), axis=1
        )
        class_metric = replace_convexhull(class_metric)
        class_metric = class_metric[class_metric.geometry.notna()]
        all_class_gdfs.append(class_metric.to_crs("EPSG:4326"))

    if not all_class_gdfs:
        return gpd.GeoDataFrame({"f_type": [], "geometry": []}, crs="EPSG:4326")
    return pd.concat(all_class_gdfs, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Polygons -> centrelines. Each helper mirrors the PedNet method it names
#    (third_party/tile2net/src/tile2net/raster/pednet.py).
# ═══════════════════════════════════════════════════════════════════════════════
def _validate_linemerge(merged_line):
    """PedNet.validate_linemerge()."""
    if not isinstance(merged_line, shapely.geometry.LineString):
        merged_line = [ls for ls in merged_line.geoms]
    else:
        merged_line = [merged_line]
    return merged_line


def _make_longer(line, thr):
    """PedNet.make_longer()."""
    lcoord = shapely.get_coordinates(line)
    lcoord = lcoord[-4:] if len(lcoord.shape) == 1 else lcoord[-2:].flatten()
    extended = get_extrapolated_line(lcoord, int(line.length * thr))
    line_checked = _validate_linemerge(line)
    extended_checked = _validate_linemerge(extended)
    new_l = MultiLineString(line_checked + extended_checked)
    merged = shapely.ops.linemerge(new_l)
    return merged


def _pednet_prepare_class_gdf(polygons_gdf, class_name):
    """PedNet.prepare_class_gdf()."""
    nt = polygons_gdf[polygons_gdf.f_type == class_name].copy()
    nt.geometry = nt.geometry.to_crs(3857)
    return nt


def _pednet_create_lines(gdf):
    """PedNet.create_lines() from pednet.py."""
    lin_geom = []
    gdf_atts = morpho_atts(gdf)
    for c, geom in enumerate(gdf.geometry):
        corners = gdf_atts.iloc[c, gdf_atts.columns.get_loc(gdf_atts.corners.name)]
        minpr = 2 * math.sqrt(math.pi * abs(geom.area))
        trim1 = 20
        trim2 = 6
        if geom.area <= 20:
            continue
        elif minpr / geom.length > 0.8:
            cl_arg = math.sqrt(geom.area / math.pi) / 4
        else:
            cl_arg = 0.2

        line = to_cline(geom, cl_arg, 1)
        if not line.is_empty:
            tr_line_ = trim_checkempty(line, trim1, trim2)
            if corners > 100:
                tr_line = trim_checkempty(tr_line_, trim1, trim2)
            else:
                tr_line = tr_line_
        else:
            line_clh = to_cline(geom, cl_arg / 2, 1)
            if not line_clh.is_empty:
                tr_line_ = trim_checkempty(line_clh, trim1, trim2)
                if corners > 100:
                    tr_line = trim_checkempty(tr_line_, trim1, trim2)
                else:
                    tr_line = tr_line_
            else:
                new_line = to_cline(geom, 0.1, 0.5)
                tr_line_ = trim_checkempty(new_line, trim1, trim2)
                if corners > 100:
                    tr_line = trim_checkempty(tr_line_, trim1, trim2)
                else:
                    tr_line = tr_line_

        if tr_line.is_empty:
            logging.debug('empty line')
            continue
        else:
            line_tr = tr_line.simplify(1)
            extended_line = extend_lines(
                geo2geodf([line_tr]),
                target=geo2geodf([geom.boundary]),
                tolerance=6,
                extension=0,
            )
            lin_geom.append(extended_line)

    if len(lin_geom) > 0:
        ntw = pd.concat(lin_geom)
        smoothed = wrinkle_remover(ntw, 1.5)
        return smoothed
    return None


def _pednet_create_sidewalks(merged_polygons):
    """PedNet.create_sidewalks() from pednet.py.

    Parameters
    ----------
    merged_polygons : GeoDataFrame from buff_dfs (metric CRS).

    Returns
    -------
    GeoDataFrame of sidewalk centerlines in EPSG:3857.
    """
    sw_all = _pednet_prepare_class_gdf(merged_polygons, 'sidewalk')

    if len(sw_all) > 0:
        swntw = _pednet_create_lines(sw_all)
        if swntw is None or swntw.empty:
            warnings.warn('No sidewalk lines created')
            sw_result = gpd.GeoDataFrame({'geometry': []}, crs=3857)
            sw_result['f_type'] = 'sidewalk'
            return sw_result

        logging.info('..... creating the processed sidewalk network')

        swntw.geometry = swntw.simplify(0.6)
        sw_modif_uni = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([geom for geom in swntw.unary_union.geoms])
        )
        sw_modif_uni_met = set_gdf_crs(sw_modif_uni, 3857)
        sw_uni_lines = sw_modif_uni_met.explode()
        sw_uni_lines.reset_index(drop=True, inplace=True)
        sw_uni_lines.geometry = sw_uni_lines.simplify(2)
        sw_uni_lines.dropna(inplace=True)
        sw_uni_line2 = sw_uni_lines.copy()

        try:
            sw_cl1 = clean_deadend_dangles(sw_uni_line2)
            sw_extended = extend_lines(sw_cl1, 10, extension=0)
            sw_cleaned = remove_false_nodes(sw_extended)
            sw_cleaned.reset_index(drop=True, inplace=True)
            sw_cleaned.geometry = sw_cleaned.geometry.set_crs(3857)
            sw_cleaned['f_type'] = 'sidewalk'
            return sw_cleaned
        except Exception:
            sw_uni_lines['f_type'] = 'sidewalk'
            return sw_uni_lines
    else:
        warnings.warn('No sidewalk polygons found')
        sw_result = gpd.GeoDataFrame({'geometry': []}, crs=3857)
        sw_result['f_type'] = 'sidewalk'
        return sw_result


def _pednet_create_crosswalk(merged_polygons):
    """PedNet.create_crosswalk() from pednet.py.

    Parameters
    ----------
    merged_polygons : GeoDataFrame from buff_dfs (metric CRS).

    Returns
    -------
    GeoDataFrame of crosswalk centerlines in EPSG:3857.
    """
    nt_cw = _pednet_prepare_class_gdf(merged_polygons, 'crosswalk')

    if len(nt_cw) > 0:
        cw_lin_geom = []
        nt_cw.geometry = nt_cw.simplify(0.6)

        # PedNet's crosswalk merge uses different buffer/erode parameters
        # from buff_dfs (0.3, -0.25, 0.2, 0.3, 0.3).
        cw_union = buffer_union_erode(nt_cw, 1, -0.95, 0.6, 0.8, 0.8)

        cw_explode = cw_union.explode().reset_index(drop=True)
        cw_explode = cw_explode[cw_explode.geometry.notna()].reset_index(drop=True)
        cw_explode_ = morpho_atts(cw_explode)

        for c, geom in enumerate(cw_explode_.geometry):
            if geom.area < 5:
                continue
            else:
                # If crosswalks are T or U shaped (low convexity)
                if cw_explode_.iloc[c, cw_explode_.columns.get_loc(cw_explode_.convexity.name)] < 0.8:
                    av_width = 4 * geom.area / geom.length
                    geom_er = geom.buffer(-av_width / 4)

                    if geom_er.geom_type == "MultiPolygon":
                        for g in list(geom_er.geoms):
                            if g.area > 2:
                                cnl = to_cline(g, 0.3, 1)
                                tr_line_ = trim_checkempty(cnl, 4.5, 2)
                                if tr_line_.length < 8:
                                    extended = _make_longer(tr_line_, 0.8)
                                    extended_line = extend_lines(
                                        geo2geodf([extended]),
                                        tolerance=8,
                                        target=geo2geodf([geom.boundary]),
                                        extension=0,
                                    )
                                    for gi in extended_line.geometry:
                                        cw_lin_geom.append(gi)
                                else:
                                    cw_lin_geom.append(tr_line_)
                            else:
                                continue

                    elif geom_er.geom_type == "Polygon":
                        if geom_er.area > 2:
                            cnl = to_cline(geom_er, 0.2, 1)
                            tr_line_ = trim_checkempty(cnl, 4.5, 2)
                            if tr_line_.length < 8:
                                extended = _make_longer(tr_line_, 0.8)
                                extended_line = extend_lines(
                                    geo2geodf([extended]),
                                    target=geo2geodf([geom.boundary]),
                                    tolerance=8,
                                    extension=0,
                                )
                                for g in extended_line.geometry:
                                    cw_lin_geom.append(g)
                            else:
                                cw_lin_geom.append(tr_line_)
                        else:
                            continue
                    else:
                        continue
                else:
                    # Rectangular / convex crosswalk → centerline via min rotated rect
                    line = get_crosswalk_cnl(geom)
                    if line.length < 8:
                        extended = _make_longer(line, 0.8)
                        extended_line = extend_lines(
                            geo2geodf([extended]),
                            target=geo2geodf([geom.boundary]),
                            tolerance=8,
                            extension=0,
                        )
                        for g in extended_line.geometry:
                            cw_lin_geom.append(g)
                    else:
                        cw_lin_geom.append(line)

        if cw_lin_geom:
            cw_ntw = geo2geodf(cw_lin_geom)
            cw_ntw['f_type'] = 'crosswalk'
            cw_ntw.geometry = cw_ntw.geometry.set_crs(3857)
            smoothed = wrinkle_remover(cw_ntw, 1.3)
            return smoothed
        else:
            cw_result = gpd.GeoDataFrame({'geometry': []}, crs=3857)
            cw_result['f_type'] = 'crosswalk'
            return cw_result
    else:
        warnings.warn('No crosswalks found')
        cw_result = gpd.GeoDataFrame({'geometry': []}, crs=3857)
        cw_result['f_type'] = 'crosswalk'
        return cw_result


def _pednet_connect_crosswalks_to_sidewalks(sidewalk_gdf, crosswalk_gdf, island=None):
    """Replicates the crosswalk-to-sidewalk connection logic from
    PedNet.convert_whole_poly2line().

    Parameters
    ----------
    sidewalk_gdf  : sidewalk centerlines GeoDataFrame (EPSG:3857)
    crosswalk_gdf : crosswalk centerlines GeoDataFrame (EPSG:3857)
    island        : island points GeoDataFrame or None

    Returns
    -------
    GeoDataFrame of combined network in EPSG:3857
    """
    if crosswalk_gdf.empty or sidewalk_gdf.empty:
        combined = pd.concat(
            [c for c in [crosswalk_gdf, sidewalk_gdf] if not c.empty],
            ignore_index=True,
        )
        if combined.empty:
            return gpd.GeoDataFrame({'geometry': [], 'f_type': []}, crs=3857)
        combined.geometry = combined.geometry.set_crs(3857, allow_override=True)
        return combined

    # Get start/end points of crosswalk centerlines
    points = get_line_sepoints(crosswalk_gdf)

    # Find crosswalk endpoints that connect to only one crosswalk segment
    inp, res = crosswalk_gdf.sindex.query(
        geo2geodf(points).geometry, predicate="intersects"
    )
    unique, counts = np.unique(inp, return_counts=True)
    ends = np.unique(res[np.isin(inp, unique[counts == 1])])

    new_geoms_s = []
    new_geoms_e = []
    new_geoms_both = []
    all_connections = []

    pgeom = crosswalk_gdf.geometry.values
    for line in ends:
        l_coords = shapely.get_coordinates(pgeom[line])
        start = Point(l_coords[0])
        end = Point(l_coords[-1])

        first = list(pgeom.sindex.query(start, predicate="intersects"))
        second = list(pgeom.sindex.query(end, predicate="intersects"))
        first.remove(line)
        second.remove(line)

        if first and not second:
            new_geoms_s.append((line, end))
        elif not first and second:
            new_geoms_e.append((line, start))
        if not first and not second:
            new_geoms_both.append((line, start))
            new_geoms_both.append((line, end))

    # Connect unconnected crosswalk endpoints to nearest sidewalks
    if len(new_geoms_s) > 0:
        ps = [g[1] for g in new_geoms_s]
        pdfs = gpd.GeoDataFrame(geometry=ps)
        pdfs.set_crs(3857, inplace=True)
        connect_s = get_shortest(sidewalk_gdf, pdfs, f_type='sidewalk_connection')
        all_connections.append(connect_s)

    if len(new_geoms_e) > 0:
        pe = [g[1] for g in new_geoms_e]
        pdfe = gpd.GeoDataFrame(geometry=pe)
        pdfe.set_crs(3857, inplace=True)
        connect_e = get_shortest(sidewalk_gdf, pdfe, f_type='sidewalk_connection')
        all_connections.append(connect_e)

    pdfb = None
    if len(new_geoms_both) > 0:
        pb = [g[1] for g in new_geoms_both]
        pdfb = gpd.GeoDataFrame(geometry=pb)
        pdfb.set_crs(3857, inplace=True)
        connect_b = get_shortest(sidewalk_gdf, pdfb, f_type='sidewalk_connection')
        all_connections.append(connect_b)

    if len(all_connections) > 1:
        connect = pd.concat(all_connections)
    elif len(all_connections) == 1:
        connect = all_connections[0]
    else:
        connect = []

    # Combine everything
    if len(all_connections) > 0:
        # Manage median islands
        if island is not None and pdfb is not None and len(pdfb) > 0:
            try:
                nearest_cw = island.sindex.nearest(pdfb.geometry, max_distance=7)
                indcwnear = list(zip(nearest_cw[0], nearest_cw[1]))
                island_lines = []
                for k, v in indcwnear:
                    island_lines.append(
                        shapely.shortest_line(
                            island.geometry.values[v],
                            pdfb.geometry.values[k],
                        )
                    )
                island_gdf = gpd.GeoDataFrame(geometry=island_lines)
                island_gdf.geometry = island_gdf.geometry.set_crs(3857)
                island_gdf['f_type'] = 'medians'
                combined = pd.concat([crosswalk_gdf, connect, sidewalk_gdf, island_gdf])
            except Exception:
                combined = pd.concat([crosswalk_gdf, connect, sidewalk_gdf])
        else:
            combined = pd.concat([crosswalk_gdf, connect, sidewalk_gdf])
    else:
        combined = pd.concat([crosswalk_gdf, sidewalk_gdf])

    combined.dropna(inplace=True)
    combined.geometry = combined.geometry.set_crs(3857, allow_override=True)
    combined = combined[~combined.geometry.isna()]
    combined.drop_duplicates(subset='geometry', inplace=True)
    combined.reset_index(drop=True, inplace=True)

    return combined


def _polygons_to_network(poly_gdf, snap_tol):
    """Tile2Net's polygon -> network pipeline.

    Mirrors ``grid.save_ntw_polygons`` (simplify -> buff_dfs -> simplify ->
    fill_holes -> replace_convexhull) followed by ``PedNet.convert_whole_poly2line``
    (sidewalk and crosswalk centrelines, then crosswalk endpoints connected to
    the nearest sidewalks). PedNet never calls ``find_medianisland`` there, so
    median islands are not used.
    """
    poly_metric = poly_gdf.to_crs(METRIC_CRS)

    # grid.save_ntw_polygons
    poly_metric.geometry = poly_metric.geometry.simplify(0.6)
    merged = buff_dfs(poly_metric)
    if merged.empty:
        return _empty_result()
    merged.geometry = merged.geometry.simplify(0.9)
    merged = merged[merged.geometry.notna()]
    merged['geometry'] = merged.apply(fill_holes, args=(25,), axis=1)
    merged = replace_convexhull(merged)
    merged = merged[merged.geometry.notna()]
    merged = merged[['geometry', 'f_type']]
    if not merged.crs or merged.crs.to_epsg() != 3857:
        merged = merged.set_crs(METRIC_CRS, allow_override=True)

    # PedNet.convert_whole_poly2line
    sidewalk_gdf = _pednet_create_sidewalks(merged)
    crosswalk_gdf = _pednet_create_crosswalk(merged)
    combined = _pednet_connect_crosswalks_to_sidewalks(sidewalk_gdf, crosswalk_gdf, island=None)
    if combined.empty:
        return _empty_result()

    combined_metric = combined.copy()
    if combined_metric.crs and combined_metric.crs.to_epsg() != 3857:
        combined_metric = combined_metric.to_crs(METRIC_CRS)
    nodes_rc, edges = _lines_to_node_edge(combined_metric, snap_tol=snap_tol)
    return nodes_rc, edges, combined_metric.to_crs("EPSG:4326")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Centrelines -> node / edge arrays
# ═══════════════════════════════════════════════════════════════════════════════
def _lines_to_node_edge(
    lines_gdf: "gpd.GeoDataFrame",
    snap_tol: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a GeoDataFrame of LineStrings to a topological (nodes_rc, edges)
    graph.

    Uses ``shapely.node()`` to split lines at every mutual intersection
    (including T-junctions where a connection line meets a sidewalk mid-span),
    then builds the graph from **segment endpoints only**.

    nodes_rc : (N,2) int32  (row, col)  – metric coords rounded to int
    edges    : (M,2) int32  (indices into nodes_rc)

    snap_tol : float  – tolerance for matching shared endpoints
              (float-precision only; default 0.001 m = 1 mm).
              This is NOT for spatial snapping of distant features.
    """
    if lines_gdf.empty:
        return np.zeros((0, 2), np.int32), np.zeros((0, 2), np.int32)

    # Collect valid geometries, exploding MultiLineStrings
    geoms = []
    for geom in lines_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiLineString":
            geoms.extend(g for g in geom.geoms if not g.is_empty)
        elif geom.geom_type == "LineString":
            geoms.append(geom)
    if not geoms:
        return np.zeros((0, 2), np.int32), np.zeros((0, 2), np.int32)

    # --- Node all lines at their mutual intersections ----------------------
    #     shapely.node() splits every line at every crossing / T-junction so
    #     the resulting segments share endpoints only at true junctions.
    try:
        noded = shapely.node(shapely.geometry.GeometryCollection(geoms))
        if noded.geom_type == "MultiLineString":
            segments = [g for g in noded.geoms if not g.is_empty]
        elif noded.geom_type == "LineString":
            segments = [noded] if not noded.is_empty else []
        elif noded.geom_type == "GeometryCollection":
            segments = [
                g for g in noded.geoms
                if g.geom_type == "LineString" and not g.is_empty
            ]
        else:
            segments = geoms
    except Exception:
        # Fallback if shapely.node() is unavailable or fails
        segments = geoms

    if not segments:
        return np.zeros((0, 2), np.int32), np.zeros((0, 2), np.int32)

    # --- Collect start / end points of every segment -----------------------
    n_seg = len(segments)
    endpoints_xy = np.empty((2 * n_seg, 2), dtype=np.float64)
    for i, seg in enumerate(segments):
        c = seg.coords
        endpoints_xy[2 * i]     = c[0]
        endpoints_xy[2 * i + 1] = c[-1]

    # --- Deduplicate endpoints (tiny tolerance for float precision) ---------
    tree = KDTree(endpoints_xy)
    labels = np.full(len(endpoints_xy), -1, dtype=int)
    uid = 0
    unique_pts = []
    for i in range(len(endpoints_xy)):
        if labels[i] >= 0:
            continue
        nbrs = tree.query_ball_point(endpoints_xy[i], snap_tol)
        centroid = endpoints_xy[nbrs].mean(axis=0)
        for n in nbrs:
            labels[n] = uid
        unique_pts.append(centroid)
        uid += 1

    nodes_xy = np.array(unique_pts)                    # (N, 2) metric (x, y)
    nodes_rc = nodes_xy[:, ::-1].astype(np.int32)      # (N, 2) as (y, x) = (row, col)

    # --- Build edge list: each segment = one edge --------------------------
    edges_set = set()
    for seg_i in range(n_seg):
        u = labels[2 * seg_i]
        v = labels[2 * seg_i + 1]
        if u != v:
            edges_set.add(tuple(sorted((u, v))))

    edges = (
        np.array(list(edges_set), dtype=np.int32)
        if edges_set
        else np.zeros((0, 2), np.int32)
    )
    return nodes_rc, edges


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Public API
# ═══════════════════════════════════════════════════════════════════════════════
def seg_mask_to_graph(
    pred_mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    class_map: dict[int, str] | None = None,
    snap_tol: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, "gpd.GeoDataFrame"]:
    """
    Segmentation mask -> pedestrian graph with the Tile2Net PedNet pipeline.

    Parameters
    ----------
    pred_mask : H×W uint8  class-label mask.
    bbox      : (minx, miny, maxx, maxy) in EPSG:4326.
    class_map : {class_id → "f_type"}.  Defaults {1:sw, 2:rd, 3:cw}.
    snap_tol  : tolerance (metres) for matching shared endpoints when
                building the node/edge graph.  Default 0.001 m (1 mm) –
                only for float-precision matching, NOT spatial snapping.

    Returns
    -------
    nodes_rc  : (N,2) int32 in metric CRS (y,x → row, col).
    edges     : (M,2) int32.
    lines_gdf : GeoDataFrame of centerlines in EPSG:4326.
    """
    poly_gdf = mask_to_polygons(pred_mask, bbox, class_map=class_map)
    if poly_gdf.empty:
        return _empty_result()
    return _polygons_to_network(poly_gdf, snap_tol)


def metric_coords_to_pixel(
    nodes_metric: np.ndarray,
    bbox_4326: tuple[float, float, float, float],
    img_shape: tuple[int, int],
) -> np.ndarray:
    """
    Convert nodes from metric CRS (EPSG:3857) to image pixel coordinates.

    Parameters
    ----------
    nodes_metric : (N,2) array in EPSG:3857 metric coordinates (y,x) or (row,col) in meters.
    bbox_4326    : (minx, miny, maxx, maxy) in EPSG:4326.
    img_shape    : (height, width) of the image.

    Returns
    -------
    nodes_pixel : (N,2) int32 array in pixel coordinates (row, col).
    """
    if len(nodes_metric) == 0:
        return np.zeros((0, 2), np.int32)

    # nodes_metric are in format (y, x) in EPSG:3857
    y_metric = nodes_metric[:, 0]
    x_metric = nodes_metric[:, 1]

    # Convert bbox from EPSG:4326 to EPSG:3857
    transformer_to_metric = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    minx_4326, miny_4326, maxx_4326, maxy_4326 = bbox_4326
    minx_metric, miny_metric = transformer_to_metric.transform(minx_4326, miny_4326)
    maxx_metric, maxy_metric = transformer_to_metric.transform(maxx_4326, maxy_4326)

    # Convert metric coords to normalized [0,1] within bbox
    H, W = img_shape
    norm_x = (x_metric - minx_metric) / (maxx_metric - minx_metric)
    norm_y = (y_metric - miny_metric) / (maxy_metric - miny_metric)

    # Convert to pixel coordinates (note: y is flipped in images)
    col = (norm_x * W).astype(np.int32)
    row = ((1.0 - norm_y) * H).astype(np.int32)

    # Clip to image bounds
    col = np.clip(col, 0, W - 1)
    row = np.clip(row, 0, H - 1)

    return np.column_stack([row, col])