# Analisis de seleccion de modelo

## Calidad de datos

```json
{
  "rows_raw": 145366,
  "rows_valid": 145366,
  "start": "2002-01-01 01:00:00",
  "end": "2018-08-03 00:00:00",
  "duplicates": 4,
  "missing_hours": 30,
  "first_missing_hours": [
    "2002-04-07 03:00:00",
    "2002-10-27 02:00:00",
    "2003-04-06 03:00:00",
    "2003-10-26 02:00:00",
    "2004-04-04 03:00:00",
    "2004-10-31 02:00:00",
    "2005-04-03 03:00:00",
    "2005-10-30 02:00:00",
    "2006-04-02 03:00:00",
    "2006-10-29 02:00:00"
  ],
  "target_min": 14544.0,
  "target_mean": 32080.507722100687,
  "target_median": 31421.0,
  "target_max": 62009.0
}
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
{
  "model": "RidgeForecaster",
  "alpha": 100.0,
  "lags": [
    1,
    2,
    3,
    24,
    48,
    168,
    336
  ],
  "rolling_windows": [
    24,
    168
  ],
  "features": [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",
    "lag_168",
    "lag_336",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "rolling_std_168"
  ],
  "fitted_at": "2026-05-26 20:57:23.779399+00:00",
  "training_start": "2002-01-01 01:00:00",
  "training_end": "2018-08-03 00:00:00",
  "training_rows": 145392
}
```

Metricas del modelo:

```json
{
  "MAE_MW": 2656.4377020631828,
  "RMSE_MW": 3495.776614607471,
  "MAPE_pct": 7.268532756290867,
  "Bias_MW": -30.95009531403537,
  "R2": 0.7678183227480824
}
```

Metricas baseline estacional naive:

```json
{
  "MAE_MW": 3575.929166666667,
  "RMSE_MW": 4664.308898581749,
  "MAPE_pct": 9.969311468796747,
  "Bias_MW": 186.18472222222223,
  "R2": 0.5866526028706484
}
```

### Comparacion de alphas

| alpha | model_MAE_MW | model_RMSE_MW | model_MAPE_pct | model_Bias_MW | model_R2 | naive_MAE_MW | naive_RMSE_MW | naive_MAPE_pct | naive_Bias_MW | naive_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100.000 | 2656.438 | 3495.777 | 7.269 | -30.950 | 0.768 | 3575.929 | 4664.309 | 9.969 | 186.185 | 0.587 |
| 10.000 | 2684.357 | 3519.941 | 7.371 | -18.720 | 0.765 | 3575.929 | 4664.309 | 9.969 | 186.185 | 0.587 |
| 1.000 | 2689.550 | 3524.433 | 7.390 | -16.700 | 0.764 | 3575.929 | 4664.309 | 9.969 | 186.185 | 0.587 |

## Forecast generado

Inicio: `2018-08-03 01:00:00`

Fin: `2018-08-04 00:00:00`

Filas: `24`
