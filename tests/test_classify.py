"""Unit tests for canopy classification and crop extraction."""

import os
import sys
import unittest
import numpy as np

# Add src and root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classify import classify_canopy_foliage


class TestCanopyClassification(unittest.TestCase):
    def test_classify_healthy_green(self):
        # Create green synthetic patch in BGR
        green_patch = np.zeros((32, 32, 3), dtype=np.uint8)
        green_patch[:, :, 1] = 180  # high green
        green_patch[:, :, 0] = 30
        green_patch[:, :, 2] = 30
        result = classify_canopy_foliage(green_patch)
        self.assertEqual(result, "healthy_green")

    def test_classify_empty_crop(self):
        empty_patch = np.zeros((0, 0, 3), dtype=np.uint8)
        result = classify_canopy_foliage(empty_patch)
        self.assertEqual(result, "empty")

    def test_classify_shadowed(self):
        dark_patch = np.ones((32, 32, 3), dtype=np.uint8) * 10
        result = classify_canopy_foliage(dark_patch)
        self.assertEqual(result, "shadowed")


if __name__ == "__main__":
    unittest.main()
