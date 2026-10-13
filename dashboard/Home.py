"""
California Gasoline Price Forecasting - Streamlit dashboard.

Run from the Project folder:
    streamlit run dashboard/Home.py
"""
import pandas as pd
import streamlit as st

from common import TRAIN_END, compute_metrics, label, load_data, load_forecasts

st.set_page_config(page_title="CA Gasoline Forecasts", page_icon="⛽", layout="wide")

df = load_data()
fc = load_forecasts()
last, prev = df.iloc[-1], df.iloc[-2]

st.title("⛽ Forecasting California Gasoline Prices")
st.caption(f"Weekly data through {df.index[-1]:%B %d, %Y} · Sources: EIA, FRED · "
           f"Test period: {TRAIN_END + pd.Timedelta(weeks=1):%b %Y} onward")

c1, c2, c3, c4 = st.columns(4)
c1.metric("California regular", f"${last.ca_gas:.2f}/gal", f"{last.ca_gas - prev.ca_gas:+.2f} vs last week",
          delta_color="inverse", border=True)
c2.metric("U.S. regular", f"${last.us_gas:.2f}/gal", f"{last.us_gas - prev.us_gas:+.2f}", delta_color="inverse", border=True)
c3.metric("California premium", f"${last.spread:.2f}/gal", f"{last.spread - prev.spread:+.2f}", delta_color="inverse",
          border=True, help="California price minus U.S. average")
c4.metric("Brent crude", f"${last.brent:.0f}/bbl", f"{last.brent - prev.brent:+.1f}", delta_color="inverse", border=True)

if not fc.empty:
    m4 = compute_metrics(4).drop(index=["naive", "snaive"], errors="ignore")
    best = m4.index[0]
    st.success(f"**Most accurate model (4 weeks ahead): {label(best)}** — average error "
               f"${m4.loc[best, 'MAE']:.2f}/gal, {1 - m4.loc[best, 'Skill_vs_naive']:.0%} smaller than the naive forecast.")
    ai = m4[m4["Family"] == "AI"]
    if not ai.empty:
        best_ai = ai.index[0]
        verdict = "beats" if ai.loc[best_ai, "MAE"] < m4.loc[best, "MAE"] else "does not beat"
        st.info(f"**Do AI models win?** The best AI model, {label(best_ai)}, averages "
                f"${ai.loc[best_ai, 'MAE']:.2f}/gal error 4 weeks ahead and {verdict} {label(best)}.")

st.markdown("""
### What's inside

| Page | Question it answers |
|---|---|
| **Historical Prices** | How have California prices, the California–U.S. premium and crude oil moved since 2000? |
| **Model Comparison** | Which forecasting model is most accurate 4 and 12 weeks ahead? Do AI models beat conventional ones? |
| **Forecast Viewer** | How did each model's forecasts compare with what actually happened? |
| **Crude Scenario** | If Brent crude moves to a price you choose, where will California gasoline go? |

Use the sidebar to switch pages.
""")

with st.expander("About the project"):
    st.markdown("""
California drivers pay far more than the U.S. average because of a special fuel blend, seasonal formula
switches, state taxes and recent refinery closures. This project forecasts weekly California regular
gasoline prices and compares conventional time-series models (naive, ETS, ARIMA, ARIMAX) with AI
forecasting models.

**Method.** Every model is tested the same way: at each test week we fit the model only on data available
up to that week and forecast 1–12 weeks ahead (rolling origin). Models that use crude oil assume crude
stays at its last observed price, so no model sees the future.
""")
