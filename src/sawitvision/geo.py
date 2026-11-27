"""Geospatial utilities and GeoJSON export for palm tree detections."""

import json
from typing import Dict, List, Optional, Tuple, Any


def detection_to_geojson_feature(
    det: Dict[str, Any],
    transform: Optional[Tuple[float, float, float, float, float, float]] = None,
) -> Dict[str, Any]:
    """Convert a single detection record into a GeoJSON Feature."""
    cx = det["center_x"]
    cy = det["center_y"]

    if transform is not None:
        # Affine transform: x = a*col + b*row + c, y = d*col + e*row + f
        a, b, c, d, e, f = transform
        map_x = a * cx + b * cy + c
        map_y = d * cx + e * cy + f
        coords = [round(map_x, 6), round(map_y, 6)]
    else:
        coords = [round(cx, 2), round(cy, 2)]

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": coords,
        },
        "properties": {
            "tree_id": det.get("tree_id"),
            "confidence": round(float(det.get("confidence", 1.0)), 4),
            "class_name": det.get("class_name", "oil_palm"),
            "box_xywh": [
                round(det["x"], 1),
                round(det["y"], 1),
                round(det["width"], 1),
                round(det["height"], 1),
            ],
            "is_georeferenced": transform is not None,
        },
    }


def export_geojson(
    detections: List[Dict[str, Any]],
    output_path: str,
    transform: Optional[Tuple[float, float, float, float, float, float]] = None,
    crs_name: str = "EPSG:4326",
) -> None:
    """Save a list of detections to a valid GeoJSON FeatureCollection."""
    features = [detection_to_geojson_feature(d, transform) for d in detections]
    collection: Dict[str, Any] = {
        "type": "FeatureCollection",
        "name": "sawit_vision_palms",
        "features": features,
    }
    if transform is not None:
        collection["crs"] = {
            "type": "name",
            "properties": {"name": crs_name},
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)
