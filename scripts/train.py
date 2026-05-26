from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_forecasting.data import load_series, quality_report
from demand_forecasting.model import RidgeForecaster, save_model
from demand_forecasting.pipeline import evaluate_alphas, write_model_selection_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena modelo de prediccion de demanda.")
    parser.add_argument("--data", default="PJME_hourly.csv/PJME_hourly.csv", help="Ruta al CSV historico.")
    parser.add_argument("--model-out", default="models/demand_ridge.pkl", help="Salida del modelo pickle.")
    parser.add_argument("--forecast-out", default="outputs/forecast_24h.csv", help="Salida CSV del forecast.")
    parser.add_argument("--report-out", default="reports/model_selection.md", help="Reporte markdown.")
    parser.add_argument("--horizon", type=int, default=24, help="Horas a pronosticar.")
    parser.add_argument("--test-days", type=int, default=30, help="Dias para backtest rolling.")
    parser.add_argument("--step-hours", type=int, default=24, help="Separacion entre origenes de backtest.")
    parser.add_argument(
        "--alphas",
        default="1,10,100",
        help="Valores alpha separados por coma para Ridge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = ROOT / args.data
    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]

    print("Cargando datos...")
    series = load_series(data_path)
    print(json.dumps(quality_report(data_path), indent=2))

    print("Evaluando modelos con backtest rolling 24h...")
    _, summary, best_backtest = evaluate_alphas(
        series,
        alphas=alphas,
        test_days=args.test_days,
        horizon=args.horizon,
        step_hours=args.step_hours,
    )
    best_alpha = float(summary.iloc[0]["alpha"])
    print(summary.to_string(index=False))

    print(f"Entrenando modelo final con alpha={best_alpha} sobre todo el historico...")
    final_model = RidgeForecaster(alpha=best_alpha).fit(series)
    model_path = ROOT / args.model_out
    save_model(final_model, model_path)

    next_start = series.index.max() + pd.Timedelta(hours=1)
    forecast = final_model.forecast(series, next_start, args.horizon)
    forecast_path = ROOT / args.forecast_out
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.rename("PJME_MW_forecast").to_csv(forecast_path, index_label="Datetime")

    report_path = ROOT / args.report_out
    write_model_selection_report(
        report_path,
        data_path,
        summary,
        best_backtest,
        final_model,
        forecast,
    )

    print(f"Modelo guardado en: {model_path}")
    print(f"Forecast guardado en: {forecast_path}")
    print(f"Reporte guardado en: {report_path}")

if __name__ == "__main__":
    main()
