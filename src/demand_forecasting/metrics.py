from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    error = pred - actual

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    mape = float(np.mean(np.abs(error) / np.clip(np.abs(actual), 1e-9, None)) * 100.0)
    bias = float(np.mean(error))

    denom = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1.0 - np.sum(error**2) / denom) if denom else float("nan")

    return {
        "MAE_MW": mae,
        "RMSE_MW": rmse,
        "MAPE_pct": mape,
        "Bias_MW": bias,
        "R2": r2,
    }

