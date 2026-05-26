# Prediccion de demanda electrica horaria

Sistema de prediccion para estimar demanda electrica con horizonte minimo de 24 horas. El dataset incluido es `PJME_hourly.csv/PJME_hourly.csv`, con demanda horaria historica en MW.

## Decision tecnica inicial

Antes de usar LSTM o Prophet conviene establecer una linea base fuerte y reproducible. Para este dataset:

- Hay muchos años de datos horarios, con estacionalidad diaria, semanal y anual clara.
- Solo hay una variable historica de demanda; no hay clima, feriados locales u otras variables externas.
- LSTM requiere mas datos preparados, escalado, validacion cuidadosa y dependencias pesadas. Es util si luego agregamos variables externas o patrones no lineales complejos.
- Prophet es una buena opcion para tendencia y estacionalidad, pero para prediccion operativa de 24 horas suele necesitar regresores/rezagos adicionales para competir con modelos autoregresivos.

Por eso el sistema implementa primero un modelo autoregresivo Ridge con:

- Rezagos de demanda: 1, 2, 3, 24, 48, 168 y 336 horas.
- Promedios y desviaciones moviles de 24 y 168 horas.
- Variables ciclicas de hora del dia, dia de semana, mes y dia del ano.
- Evaluacion rolling con pronosticos recursivos de 24 horas.

Esta base deja una referencia medible. Prophet o LSTM deben incorporarse solo si superan este benchmark en el mismo backtest de 24 horas.

## Estructura

```text
src/demand_forecasting/
  data.py        # carga, limpieza y reporte de calidad
  features.py    # variables de calendario, rezagos y rolling windows
  model.py       # modelo Ridge autoregresivo serializable
  metrics.py     # metricas de error
  pipeline.py    # backtesting, comparacion y reportes
scripts/
  train.py       # entrena, evalua, guarda modelo y forecast 24h
  forecast.py    # carga modelo y genera nuevo forecast
```

## Uso rapido

Desde la raiz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python scripts\train.py
```

Si estas usando el runtime empaquetado de Codex, tambien puedes ejecutar directamente:

```powershell
& "C:\Users\josef\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" scripts\train.py
```

El entrenamiento genera:

- `models/demand_ridge.pkl`: modelo entrenado.
- `outputs/forecast_24h.csv`: prediccion de las siguientes 24 horas despues del ultimo dato historico.
- `reports/model_selection.md`: analisis de datos, decision LSTM/Prophet y metricas de backtest.

Por defecto se usa un backtest rolling de 30 dias para que el flujo sea agil. Si quieres una evaluacion mas larga:

```powershell
.\.venv\Scripts\python scripts\train.py --test-days 90 --alphas 0.1,1,10,100,1000
```

Para volver a generar un forecast con un modelo ya entrenado:

```powershell
.\.venv\Scripts\python scripts\forecast.py --horizon 24
```
