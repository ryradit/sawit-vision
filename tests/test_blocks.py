"""Unit tests for ManagementBlock, planting regularity, and stand vacancy detection."""

import unittest
import numpy as np
from sawitvision.blocks import (
    ManagementBlock,
    assign_trees_to_blocks,
    calculate_planting_regularity,
    detect_stand_vacancies,
)


class TestManagementBlocks(unittest.TestCase):
    def setUp(self):
        self.poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.block = ManagementBlock("BLK-01", "Test Block", self.poly, target_density_per_ha=140.0)

    def test_point_containment(self):
        self.assertTrue(self.block.contains_point(50, 50))
        self.assertTrue(self.block.contains_point(10, 10))
        self.assertFalse(self.block.contains_point(150, 50))
        self.assertFalse(self.block.contains_point(-10, 50))

    def test_area_hectares(self):
        # 100x100 px = 10,000 px^2. At GSD = 0.1m/px, 10000 * 0.01 = 100 m^2 = 0.01 hectares.
        area_ha = self.block.compute_area_hectares(gsd_m=0.1)
        self.assertAlmostEqual(area_ha, 0.01, places=3)

    def test_assign_trees_to_blocks(self):
        trees = [
            {"center_x": 20, "center_y": 20},
            {"center_x": 80, "center_y": 80},
            {"center_x": 120, "center_y": 50},  # Outside
        ]
        block_map, unassigned = assign_trees_to_blocks(trees, [self.block])
        self.assertEqual(len(block_map["BLK-01"]), 2)
        self.assertEqual(len(unassigned), 1)
        self.assertEqual(unassigned[0]["center_x"], 120)

    def test_planting_regularity_regular_grid(self):
        # Create an equilateral-like regular lattice
        pts = []
        for r in range(5):
            for c in range(5):
                pts.append([c * 20.0 + (10.0 if r % 2 == 1 else 0.0), r * 17.32])
        pts = np.array(pts)
        res = calculate_planting_regularity(pts, gsd_m=0.1)
        self.assertGreater(res["regularity_score"], 60.0)
        self.assertLess(res["cv_spacing"], 0.25)

    def test_detect_stand_vacancies(self):
        # 5x5 regular grid with center tree removed
        pts = []
        for r in range(5):
            for c in range(5):
                if r == 2 and c == 2:
                    continue  # Missing tree vacancy at (40, 40)
                pts.append([c * 20.0, r * 20.0])
        pts = np.array(pts, dtype=np.float32)
        poly = np.array([(-10, -10), (100, -10), (100, 100), (-10, 100)], dtype=np.float32)
        vacancies = detect_stand_vacancies(pts, poly, nominal_spacing_px=20.0, gsd_m=0.1)
        self.assertGreaterEqual(len(vacancies), 1)
        # Verify detected vacancy is near missing (40, 40) coordinate
        found_target = False
        for v in vacancies:
            if abs(v["center_x"] - 40.0) < 10.0 and abs(v["center_y"] - 40.0) < 10.0:
                found_target = True
                break
        self.assertTrue(found_target)


if __name__ == "__main__":
    unittest.main()
