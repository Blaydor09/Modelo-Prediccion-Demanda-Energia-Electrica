from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data import quality_report
from .metrics import regression_metrics
from .model import RidgeForecaster


def seasonal_naive_forecast(
    history: pd.Series,
    start: pd.Timestamp,
    horizon: int = 24,
    lag_hours: int = 24,
) -> pd.Series:
    index = pd.date_range(pd.Timestamp(start), periods=horizon, freq="h")
    values: list[float] = []
    for ts in index:
        ref = ts - pd.Timedelta(hours=lag_hours)
        values.append(float(history.loc[ref] if ref in history.index else history.iloc[-1]))
    return pd.Series(values, index=index, name="seasonal_naive")


def rolling_backtest(
    forecaster: RidgeForecaster,
    series: pd.Series,
    test_start: pd.Timestamp,
    horizon: int = 24,
    step_hours: int = 24,
) -> pd.DataFrame:
    last_origin = series.index.max() - pd.Timedelta(hours=horizon - 1)
    origins = pd.date_range(pd.Timestamp(test_start), last_origin, freq=f"{step_hours}h")
    if len(origins) == 0:
        raise ValueError("El periodo de prueba es demasiado corto para el horizonte solicitado.")

    rows = []
    for origin in origins:
        history_end = origin - pd.Timedelta(hours=1)
        history = series.loc[:history_end]
        forecast = forecaster.forecast(history, origin, horizon)
        naive = seasonal_naive_forecast(history, origin, horizon)
        actual = series.reindex(forecast.index)

        frame = pd.DataFrame(
            {
                "origin": origin,
                "Datetime": forecast.index,
                "actual": actual.to_numpy(dtype=float),
                "forecast": forecast.to_numpy(dtype=float),
                "seasonal_naive": naive.to_numpy(dtype=float),
            }
        )
        rows.append(frame)

    return pd.concat(rows, ignore_index=True)


def evaluate_alphas(
    series: pd.Series,
    alphas: Iterable[float],
    test_days: int = 90,
    horizon: int = 24,
    step_hours: int = 24,
) -> tuple[RidgeForecaster, pd.DataFrame, pd.DataFrame]:
    test_hours = test_days * 24
    if len(series) <= test_hours + max(RidgeForecaster().lags):
        raise ValueError("La serie no tiene suficiente historico para la evaluacion solicitada.")

    train = series.iloc[:-test_hours]
    test_start = series.index[-test_hours]

    summary_rows = []
    backtests: dict[float, pd.DataFrame] = {}
    for alpha in alphas:
        model = RidgeForecaster(alpha=float(alpha)).fit(train)
        backtest = rolling_backtest(model, series, test_start, horizon, step_hours)
        model_metrics = regression_metrics(backtest["actual"], backtest["forecast"])
        naive_metrics = regression_metrics(backtest["actual"], backtest["seasonal_naive"])

        row = {"alpha": float(alpha)}
        row.update({f"model_{k}": v for k, v in model_metrics.items()})
        row.update({f"naive_{k}": v for k, v in naive_metrics.items()})
        summary_rows.append(row)
        backtests[float(alpha)] = backtest

    summary = pd.DataFrame(summary_rows).sort_values("model_MAE_MW").reset_index(drop=True)
    best_alpha = float(summary.iloc[0]["alpha"])
    best_model = RidgeForecaster(alpha=best_alpha).fit(train)
    return best_model, summary, backtests[best_alpha]


def write_model_selection_report(
    path: str | Path,
    data_path: str | Path,
    summary: pd.DataFrame,
    best_backtest: pd.DataFrame,
    model: RidgeForecaster,
    forecast: pd.Series,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    data_info = quality_report(data_path)
    model_metrics = regression_metrics(best_backtest["actual"], best_backtest["forecast"])
    naive_metrics = regression_metrics(best_backtest["actual"], best_backtest["seasonal_naive"])

    top_rows = summary.head(10).copy()
    metric_table = _to_markdown_table(top_rows)

    report = f"""# Analisis de seleccion de modelo

## Calidad de datos

```json
{json.dumps(data_info, indent=2)}
```

El dataset queda normalizado a frecuencia horaria. Los duplicados se agregan por promedio y las horas faltantes se interpolan por tiempo para mantener una serie regular.

## LSTM vs Prophet

**LSTM** no es la primera opcion para esta fase. Puede capturar patrones no lineales, pero exige mas infraestructura, escalado, ventanas secuenciales, tuning y validacion. Sin variables externas como clima o feriados, el beneficio esperado frente a un modelo autoregresivo fuerte no justifica la complejidad inicial.

**Prophet** es util para tendencia y estacionalidades, pero el problema operativo pide 24 horas de anticipacion. En este caso los rezagos recientes de demanda son muy informativos, y Prophet puro no los usa salvo que se agreguen como regresores externos. Es una alternativa razonable para comparar despues contra el benchmark.

**Decision:** usar primero un modelo autoregresivo Ridge con calendario + rezagos. Es rapido, interpretable, reproducible y deja una metrica base para decidir si Prophet o LSTM realmente agregan valor.

## Evaluacion 24h

Backtest rolling: cada origen pronostica las siguientes 24 horas, usando solo historico disponible antes del origen. Se compara contra un baseline estacional naive que repite el valor de la misma hora del dia anterior.

### Mejor modelo

```json
{json.dumps(model.metadata(), indent=2)}
```

Metricas del modelo:

```json
{json.dumps(model_metrics, indent=2)}
```

Metricas baseline estacional naive:

```json
{json.dumps(naive_metrics, indent=2)}
```

### Comparacion de alphas

{metric_table}

## Forecast generado

Inicio: `{forecast.index.min()}`

Fin: `{forecast.index.max()}`

Filas: `{len(forecast)}`
"""
    target.write_text(report, encoding="utf-8")


def _to_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(fmt(row[col]) for col in columns) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, separator, *rows])
