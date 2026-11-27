"""Evaluation metrics for tree count validation."""

from typing import Dict, Sequence, Union
import numpy as np


def compute_count_metrics(
    predictions: Sequence[Union[int, float]],
    ground_truth: Sequence[Union[int, float]],
) -> Dict[str, float]:
    """Calculate MAE, MAPE, systematic bias, and R-squared for tree count estimates.
    
    Args:
        predictions: Predicted tree counts across plots/images.
        ground_truth: Reference ground truth tree counts.
        
    Returns:
        Dictionary containing MAE, MAPE (percentage), Bias, and R2.
    """
    preds = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(ground_truth, dtype=np.float64)

    if preds.shape != truth.shape:
        raise ValueError(
            f"Shape mismatch: predictions {preds.shape} vs ground_truth {truth.shape}"
        )
    if len(preds) == 0:
        return {"mae": 0.0, "mape": 0.0, "bias": 0.0, "r2": 0.0}

    errors = preds - truth
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors) / np.maximum(truth, 1.0)) * 100.0)
    bias = float(np.mean(errors))

    if len(preds) > 1 and np.std(preds) > 1e-9 and np.std(truth) > 1e-9:
        corr_matrix = np.corrcoef(preds, truth)
        r2 = float(corr_matrix[0, 1] ** 2)
    else:
        r2 = 1.0 if np.allclose(preds, truth) else 0.0

    return {
        "mae": round(mae, 3),
        "mape": round(mape, 3),
        "bias": round(bias, 3),
        "r2": round(r2, 4),
    }
