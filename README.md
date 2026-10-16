# Forecasting California Gasoline Prices

Weekly forecasts of California regular gasoline prices, comparing conventional time-series models
(naive, ETS, ARIMA, ARIMAX) with AI forecasters (Chronos, TimesFM, Claude, ChatGPT), plus a
Streamlit dashboard.

## Pipeline

| Step | Run | Produces |
|---|---|---|
| 1. Data | `python data_pull.py` | `data/weekly_data.csv` (+ `.xlsx`, raw series in `data/raw/`) |
| 2. Conventional models, evaluation, VAR, asymmetry test, scenarios | `forecasting_models.ipynb` | `forecasts/*.csv`, `results/*.csv` |
| 3. Time-series foundation models | `.venv-ai/bin/python ai_models.py` and `ai_models.py --validation` | `forecasts/` and `forecasts_validation/` (`chronos_bolt`, `chronos2`, `timesfm`), `results/ai_model_log.csv` |
| 4. LLM experiments | `.venv-ai/bin/python llm/llm_experiments.py` | `forecasts/llm_*.csv`, `results/llm_log.jsonl` |
| 5. Dashboard | `streamlit run dashboard/Home.py` | interactive app (see `dashboard/README.md`) |

Re-run the notebook after steps 3–4 so the AI forecasts are included in the evaluation.

## Setup

```bash
# main environment (data, notebook, dashboard)
pip install pandas numpy scipy statsmodels matplotlib requests openpyxl jupyter streamlit altair

# AI environment (needs PyTorch; kept separate so it doesn't disturb the main one)
uv venv .venv-ai --python 3.11
uv pip install --python .venv-ai/bin/python "chronos-forecasting>=2.2" "timesfm[torch]" pandas anthropic openai
```

**Keys** (never commit them; `.gitignore` covers `.eia_key` and `.env`):
- EIA: get a free key at https://www.eia.gov/opendata/register.php, then set `EIA_API_KEY` or save it in `.eia_key`.
- LLM experiments: `ANTHROPIC_API_KEY` for Claude, `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`) for ChatGPT.
  Check cost first with `--dry-run`; the full run is 17 dates × 3 repeats × 2 experiments = 102 calls per provider.

## Shared forecast format

Every model writes `forecasts/<model>.csv` with the columns
`model, origin_date, horizon, target_date, forecast, lower80, upper80` (prices in USD/gal).
The notebook's evaluation and the dashboard pick up any file in this format automatically.

## Test design

| Period | Dates | Used for |
|---|---|---|
| Training | May 2000 – June 2025 | Estimating models and their basic settings |
| Validation | Forecasts every 4th week, Jan 2022 – Mar 2025 (targets end by June 2025) | Choosing between models: ARIMAX crude version, best conventional model, the combination |
| Test | Weekly forecasts, July 2025 – latest week | Final scoring only; no choice is made by looking at it |

At each forecast date every model uses only data available that week and forecasts 1–12 weeks ahead;
4- and 12-week forecasts are scored. Models that use crude oil assume it stays at its last price.
AI models' release dates are logged for the look-ahead check.

Key results are summarized at the top of `forecasting_models.ipynb` and in `results/summary.json`.
