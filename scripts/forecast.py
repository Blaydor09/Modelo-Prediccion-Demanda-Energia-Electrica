from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from demand_forecasting.data import load_series
from demand_forecasting.model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera forecast con modelo entrenado.")
    parser.add_argument("--data", default="PJME_hourly.csv/PJME_hourly.csv", help="Ruta al CSV historico.")
    parser.add_argument("--model", default="models/demand_ridge.pkl", help="Ruta al modelo pickle.")
    parser.add_argument("--output", default="outputs/forecast_24h.csv", help="Ruta de salida CSV.")
    parser.add_argument("--horizon", type=int, default=24, help="Horas a pronosticar.")
    parser.add_argument("--start", default=None, help="Fecha-hora inicial opcional del forecast.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series = load_series(ROOT / args.data)
    model = load_model(ROOT / args.model)

    start = pd.Timestamp(args.start) if args.start else series.index.max() + pd.Timedelta(hours=1)
    forecast = model.forecast(series, start, args.horizon)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    forecast.rename("PJME_MW_forecast").to_csv(output, index_label="Datetime")
    print(f"Forecast guardado en: {output}")
    print(forecast.to_string())


if __name__ == "__main__":
    main()
