"""
California Gasoline Price Forecasting - Streamlit dashboard.

Run from the Project folder:
    streamlit run dashboard/Home.py
"""
import pandas as pd
import streamlit as st

from common import GAL_PER_BBL, TRAIN_END, compute_metrics, label, load_data, load_forecasts, load_summary

st.set_page_config(page_title="CA Gasoline Forecasts", page_icon="⛽", layout="wide")

df = load_data()
fc = load_forecasts()
summary = load_summary()
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
c4.metric("Brent crude", f"${last.brent / GAL_PER_BBL:.2f}/gal", f"{(last.brent - prev.brent) / GAL_PER_BBL:+.2f}",
          delta_color="inverse", border=True, help="Crude oil price per gallon (USD per barrel ÷ 42)")

if summary:
    s = summary
    mae = lambda m: s["mae"][m]
    best, ai = s["best_model_test"], s.get("best_ai_test")
    st.subheader("What we found")
    q1, q2, q3 = st.columns(3)
    with q1.container(border=True):
        st.markdown("**1. How accurately can we forecast?**")
        st.markdown(f"The best model, **{label(best)}**, misses by **\\${mae(best)['h4']:.2f}/gal** on average "
                    f"4 weeks ahead (\\${mae(best)['h12']:.2f} at 12 weeks). The simple rule *same as this week* "
                    f"misses by \\${mae('naive')['h4']:.2f} (\\${mae('naive')['h12']:.2f}).")
        picked = s["best_conventional_validation"]
        st.caption(f"Better than the simple rule, but not statistically significant over a short test period that "
                   f"includes a war (p = {s['best_vs_naive_dm_p']['h4']:.2f})."
                   + ("" if picked == best else
                      f" The model picked in advance on the validation period, {label(picked)}, scored "
                      f"\\${mae(picked)['h4']:.2f} / \\${mae(picked)['h12']:.2f}."))
    with q2.container(border=True):
        st.markdown("**2. Does crude oil lead pump prices?**")
        st.markdown(f"**Yes.** A 10% rise in crude lifts California prices about **{s['crude10_effect_4w']:.1%}** "
                    f"within 4 weeks. Increases reach the pump faster in the first week "
                    f"({s['asym_first_week']['up']:.2f} vs {s['asym_first_week']['down']:.2f}), but decreases catch up.")
        st.caption(f"Crude predicts gasoline (p < 0.001), not the other way round. Total pass-through is the same "
                   f"for increases and decreases (p = {s['asym_total']['p']:.2f}).")
    with q3.container(border=True):
        st.markdown("**3. Do AI models beat conventional ones?**")
        if ai:
            wins = mae(ai)["h4"] < mae(best)["h4"]
            st.markdown(f"**{'Yes' if wins else 'No'}.** The best AI model, {label(ai)}, misses by "
                        f"\\${mae(ai)['h4']:.2f}/gal vs \\${mae(best)['h4']:.2f} for {label(best)}.")
            st.caption("The AI models were used zero-shot (not trained on this data); they mostly match the simple rule.")
        else:
            st.markdown("AI forecasts have not been run yet.")
    st.caption(f"All model choices were made on a separate validation period (2022 – mid 2025) and scored once on "
               f"{s['test_dates']} weekly test forecasts from July 2025. MAE = average size of the miss.")
elif not fc.empty:
    m4 = compute_metrics(4)
    m4 = m4[m4["Family"] != "Benchmark"]
    best = m4.index[0]
    st.success(f"**Most accurate model (4 weeks ahead): {label(best)}** — average error ${m4.loc[best, 'MAE']:.2f}/gal.")

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
up to that week and forecast 1–12 weeks ahead (rolling origin). Choices between models are made on a
separate validation period, so the test results are never used to pick a winner. Models that use crude
oil assume it stays at its last known price, so no model sees the future.
""")
