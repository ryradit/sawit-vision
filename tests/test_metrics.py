"""Unit tests for counting metrics evaluation using standard library unittest."""

import os
import sys
import unittest

# Add src to path for direct test execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sawitvision.metrics import compute_count_metrics


class TestCountMetrics(unittest.TestCase):
    def test_perfect_count_prediction(self):
        preds = [100, 150, 200, 80]
        truth = [100, 150, 200, 80]
        res = compute_count_metrics(preds, truth)
        self.assertEqual(res["mae"], 0.0)
        self.assertEqual(res["mape"], 0.0)
        self.assertEqual(res["bias"], 0.0)
        self.assertEqual(res["r2"], 1.0)

    def test_count_prediction_with_bias(self):
        preds = [110, 160, 210, 90]
        truth = [100, 150, 200, 80]
        res = compute_count_metrics(preds, truth)
        self.assertEqual(res["mae"], 10.0)
        self.assertEqual(res["bias"], 10.0)
        self.assertEqual(res["r2"], 1.0)
        self.assertGreater(res["mape"], 0.0)

    def test_empty_predictions(self):
        res = compute_count_metrics([], [])
        self.assertEqual(res["mae"], 0.0)
        self.assertEqual(res["r2"], 0.0)


if __name__ == "__main__":
    unittest.main()
