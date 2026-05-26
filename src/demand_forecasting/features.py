from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_LAGS = (1, 2, 3, 24, 48, 168, 336)
DEFAULT_ROLLING_WINDOWS = (24, 168)

CALENDAR_COLUMNS = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour_angle = 2.0 * math.pi * index.hour.to_numpy() / 24.0
    dow_angle = 2.0 * math.pi * index.dayofweek.to_numpy() / 7.0
    month_angle = 2.0 * math.pi * (index.month.to_numpy() - 1) / 12.0
    doy_angle = 2.0 * math.pi * (index.dayofyear.to_numpy() - 1) / 365.25

    return pd.DataFrame(
        {
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "dow_sin": np.sin(dow_angle),
            "dow_cos": np.cos(dow_angle),
            "month_sin": np.sin(month_angle),
            "month_cos": np.cos(month_angle),
            "doy_sin": np.sin(doy_angle),
            "doy_cos": np.cos(doy_angle),
            "is_weekend": (index.dayofweek >= 5).astype(float),
        },
        index=index,
    )


def feature_columns(
    lags: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
) -> list[str]:
    columns = list(CALENDAR_COLUMNS)
    columns.extend(f"lag_{lag}" for lag in lags)
    for window in rolling_windows:
        columns.append(f"rolling_mean_{window}")
        columns.append(f"rolling_std_{window}")
    return columns


def make_supervised_frame(
    series: pd.Series,
    lags: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("La serie debe tener un DatetimeIndex.")

    X = calendar_features(series.index)
    for lag in lags:
        X[f"lag_{lag}"] = series.shift(lag)

    shifted = series.shift(1)
    for window in rolling_windows:
        rolling = shifted.rolling(window=window, min_periods=window)
        X[f"rolling_mean_{window}"] = rolling.mean()
        X[f"rolling_std_{window}"] = rolling.std(ddof=0)

    X = X[feature_columns(lags, rolling_windows)]
    frame = X.join(series.rename("y")).dropna()
    y = frame.pop("y")
    return frame, y


def make_forecast_row(
    timestamp: pd.Timestamp,
    history: pd.Series,
    lags: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("El historico debe tener un DatetimeIndex.")

    ts = pd.Timestamp(timestamp)
    row = calendar_features(pd.DatetimeIndex([ts]))
    last_allowed = ts - pd.Timedelta(hours=1)
    past = history.loc[:last_allowed]

    for lag in lags:
        lag_ts = ts - pd.Timedelta(hours=lag)
        if lag_ts not in history.index:
            raise ValueError(f"No hay suficiente historico para calcular lag_{lag} en {ts}.")
        row[f"lag_{lag}"] = float(history.loc[lag_ts])

    for window in rolling_windows:
        tail = past.tail(window)
        if len(tail) < window:
            raise ValueError(
                f"No hay suficiente historico para rolling window {window} en {ts}."
            )
        row[f"rolling_mean_{window}"] = float(tail.mean())
        row[f"rolling_std_{window}"] = float(tail.std(ddof=0))

    return row[feature_columns(lags, rolling_windows)]

