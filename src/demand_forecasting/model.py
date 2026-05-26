from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import (
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
    feature_columns,
    make_forecast_row,
    make_supervised_frame,
)


@dataclass
class RidgeForecaster:
    alpha: float = 10.0
    lags: tuple[int, ...] = DEFAULT_LAGS
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS
    feature_names: list[str] = field(default_factory=list)
    coef_: np.ndarray | None = None
    x_mean_: np.ndarray | None = None
    x_scale_: np.ndarray | None = None
    y_mean_: float | None = None
    fitted_at_: str | None = None
    training_start_: str | None = None
    training_end_: str | None = None
    training_rows_: int | None = None

    def fit(self, series: pd.Series) -> "RidgeForecaster":
        X, y = make_supervised_frame(series, self.lags, self.rolling_windows)
        if X.empty:
            raise ValueError("No hay suficientes filas para entrenar el modelo.")

        self.feature_names = feature_columns(self.lags, self.rolling_windows)
        X_np = X[self.feature_names].to_numpy(dtype=float)
        y_np = y.to_numpy(dtype=float)

        self.x_mean_ = X_np.mean(axis=0)
        self.x_scale_ = X_np.std(axis=0)
        self.x_scale_[self.x_scale_ == 0.0] = 1.0
        self.y_mean_ = float(y_np.mean())

        X_scaled = (X_np - self.x_mean_) / self.x_scale_
        y_centered = y_np - self.y_mean_

        penalty = np.eye(X_scaled.shape[1], dtype=float) * float(self.alpha)
        lhs = X_scaled.T @ X_scaled + penalty
        rhs = X_scaled.T @ y_centered
        self.coef_ = np.linalg.solve(lhs, rhs)

        self.fitted_at_ = str(pd.Timestamp.utcnow())
        self.training_start_ = str(series.index.min())
        self.training_end_ = str(series.index.max())
        self.training_rows_ = int(len(series))
        return self

    @property
    def is_fitted(self) -> bool:
        return self.coef_ is not None

    def predict_frame(self, X: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        missing = set(self.feature_names).difference(X.columns)
        if missing:
            raise ValueError(f"Faltan features para predecir: {sorted(missing)}")

        X_np = X[self.feature_names].to_numpy(dtype=float)
        X_scaled = (X_np - self.x_mean_) / self.x_scale_
        pred = self.y_mean_ + X_scaled @ self.coef_
        return pd.Series(pred, index=X.index, name="forecast")

    def forecast(self, history: pd.Series, start: pd.Timestamp, horizon: int = 24) -> pd.Series:
        self._check_fitted()
        if horizon <= 0:
            raise ValueError("El horizonte debe ser mayor a cero.")

        future_index = pd.date_range(pd.Timestamp(start), periods=horizon, freq="h")
        working_history = history.sort_index().astype(float).copy()
        predictions: list[float] = []

        for ts in future_index:
            X_row = make_forecast_row(ts, working_history, self.lags, self.rolling_windows)
            y_hat = float(self.predict_frame(X_row).iloc[0])
            predictions.append(y_hat)
            working_history.loc[ts] = y_hat

        return pd.Series(predictions, index=future_index, name="forecast")

    def metadata(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "model": "RidgeForecaster",
            "alpha": self.alpha,
            "lags": list(self.lags),
            "rolling_windows": list(self.rolling_windows),
            "features": self.feature_names,
            "fitted_at": self.fitted_at_,
            "training_start": self.training_start_,
            "training_end": self.training_end_,
            "training_rows": self.training_rows_,
        }

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("El modelo aun no fue entrenado.")


def save_model(model: RidgeForecaster, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as fh:
        pickle.dump(model, fh)


def load_model(path: str | Path) -> RidgeForecaster:
    with Path(path).open("rb") as fh:
        model = pickle.load(fh)
    if not isinstance(model, RidgeForecaster):
        raise TypeError("El archivo no contiene un RidgeForecaster valido.")
    return model

