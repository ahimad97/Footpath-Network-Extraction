"""Convert a pixel adjacency pickle to the APLS JSON format ``[nodes, edges]``.

Usage: python convert.py IN_PICKLE OUT_JSON

The pickle maps ``(row, col)`` pixels to neighbour lists. Pixels are placed in
a synthetic lat/lon frame anchored at (41, -71) with the ground resolution of
zoom-20 Web Mercator tiles, which is the frame ``main.go`` expects.
"""
import json
import math
import pickle
import sys

LAT_TOP_LEFT = 41.0
LON_TOP_LEFT = -71.0
ZOOM = 20


def mpp_at_lat(lat_deg, zoom):
    """Web Mercator metres per pixel at a latitude."""
    return 156543.03392804062 * math.cos(math.radians(lat_deg)) / (2 ** zoom)


MPP = mpp_at_lat(LAT_TOP_LEFT, ZOOM)  # ~0.1125 m/px


def xy2latlon(x_px, y_px):
    """(row, col) pixel -> (lat, lon)."""
    dlat = x_px * MPP / 111_111.0
    dlon = y_px * MPP / (111_111.0 * math.cos(math.radians(LAT_TOP_LEFT)))
    return float(LAT_TOP_LEFT - dlat), float(LON_TOP_LEFT + dlon)


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert.py IN_PICKLE OUT_JSON")
        sys.exit(1)
    f_in, f_out = sys.argv[1], sys.argv[2]
    with open(f_in, "rb") as f:
        neighbors = pickle.load(f)

    nodes, nodemap = [], {}
    for k in neighbors:
        nodemap[k] = len(nodes)
        nodes.append(list(xy2latlon(k[0], k[1])))

    edges, seen = [], set()
    for n1, nbrs in neighbors.items():
        for n2 in nbrs:
            if (n1, n2) in seen or (n2, n1) in seen:
                continue
            seen.add((n1, n2))
            if n2 in nodemap:
                edges.append([nodemap[n1], nodemap[n2]])

    with open(f_out, "w") as f:
        json.dump([nodes, edges], f, indent=2)


if __name__ == "__main__":
    main()
