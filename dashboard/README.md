# California Gasoline Forecast Dashboard

Streamlit dashboard for the project (proposal Option B).

## Run locally

From the `Project` folder:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/Home.py
```

The dashboard reads files produced by the rest of the project, so run these first:

1. `python data_pull.py` → `data/weekly_data.csv`
2. `forecasting_models.ipynb` → `forecasts/*.csv`

## Pages

| Page | Reads | Notes |
|---|---|---|
| Home | `data/weekly_data.csv`, `forecasts/*.csv` | Latest prices, best model |
| Historical Prices | `data/weekly_data.csv` | Prices, CA–U.S. premium, Brent, seasonality, event markers |
| Model Comparison | `forecasts/*.csv` | Metrics and Diebold-Mariano tests are computed from the forecast files |
| Forecast Viewer | `forecasts/*.csv` | Forecasts at a fixed horizon, or from one origin week |
| Crude Scenario | `data/weekly_data.csv` | Runs the ARDL scenario model (`models.py`) live |

## Adding a model (e.g. Chronos, TimesFM, an LLM)

Save its forecasts as `forecasts/<model>.csv` with the shared columns
`model, origin_date, horizon, target_date, forecast, lower80, upper80`.
It appears on the Model Comparison and Forecast Viewer pages automatically. To give it a display
name and the "AI" color, add it to `MODEL_LABELS` and `AI_MODELS` in `common.py`
(models whose name starts with `llm` are treated as AI automatically).

## Deploy (Streamlit Community Cloud)

Push the `Project` folder to GitHub, including `data/`, `forecasts/` and `.streamlit/config.toml`.
At share.streamlit.io, create an app with main file `dashboard/Home.py` and point it to
`dashboard/requirements.txt`.
