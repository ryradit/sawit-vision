"""Unit tests for GeoJSON and deterministic sorting using standard library unittest."""

import json
import os
import sys
import tempfile
import unittest

# Add src to path for direct test execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sawitvision.geo import export_geojson
from sawitvision.viz import sort_detections_deterministically


class TestGeoAndViz(unittest.TestCase):
    def test_sort_detections_deterministically(self):
        dets = [
            {"center_x": 100, "center_y": 150},
            {"center_x": 50, "center_y": 20},
            {"center_x": 120, "center_y": 25},
            {"center_x": 40, "center_y": 160},
        ]
        sorted_dets = sort_detections_deterministically(dets, band_height_px=50.0)
        self.assertEqual(len(sorted_dets), 4)
        # First band (0..50 px) should come first, sorted by x: 50 then 120
        self.assertEqual(sorted_dets[0]["center_x"], 50)
        self.assertEqual(sorted_dets[0]["tree_id"], 1)
        self.assertEqual(sorted_dets[1]["center_x"], 120)
        self.assertEqual(sorted_dets[1]["tree_id"], 2)

    def test_export_geojson(self):
        dets = [
            {
                "tree_id": 1,
                "center_x": 100.5,
                "center_y": 200.5,
                "x": 85.0,
                "y": 185.0,
                "width": 31.0,
                "height": 31.0,
                "confidence": 0.95,
                "class_name": "oil_palm",
            }
        ]
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
            tmp_path = f.name

        try:
            export_geojson(dets, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["type"], "FeatureCollection")
            self.assertEqual(len(data["features"]), 1)
            self.assertEqual(
                data["features"][0]["geometry"]["coordinates"], [100.5, 200.5]
            )
            self.assertEqual(data["features"][0]["properties"]["tree_id"], 1)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
