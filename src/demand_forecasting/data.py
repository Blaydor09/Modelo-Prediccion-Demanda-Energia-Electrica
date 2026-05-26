from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DATETIME_COL = "Datetime"
TARGET_COL = "PJME_MW"
FREQ = "h"


def load_raw(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {csv_path}")

    df = pd.read_csv(csv_path)
    expected = {DATETIME_COL, TARGET_COL}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    df = df[[DATETIME_COL, TARGET_COL]].copy()
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return df


def load_series(path: str | Path) -> pd.Series:
    """Load the hourly demand series and normalize it to a regular hourly index."""
    df = load_raw(path).dropna(subset=[DATETIME_COL, TARGET_COL])
    grouped = df.groupby(DATETIME_COL, as_index=True)[TARGET_COL].mean()
    series = grouped.sort_index().asfreq(FREQ)

    if series.isna().any():
        series = series.interpolate(method="time").ffill().bfill()

    series.name = TARGET_COL
    return series.astype(float)


def quality_report(path: str | Path) -> dict[str, Any]:
    raw = load_raw(path)
    parsed = raw.dropna(subset=[DATETIME_COL, TARGET_COL]).copy()
    parsed = parsed.sort_values(DATETIME_COL)
    grouped = parsed.groupby(DATETIME_COL, as_index=True)[TARGET_COL].mean()

    full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq=FREQ)
    missing_index = full_index.difference(grouped.index)

    return {
        "rows_raw": int(len(raw)),
        "rows_valid": int(len(parsed)),
        "start": str(grouped.index.min()),
        "end": str(grouped.index.max()),
        "duplicates": int(parsed[DATETIME_COL].duplicated().sum()),
        "missing_hours": int(len(missing_index)),
        "first_missing_hours": [str(ts) for ts in missing_index[:10]],
        "target_min": float(grouped.min()),
        "target_mean": float(grouped.mean()),
        "target_median": float(grouped.median()),
        "target_max": float(grouped.max()),
    }

