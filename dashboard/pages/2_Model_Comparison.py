import altair as alt
import pandas as pd
import streamlit as st

from common import FAMILY_COLORS, compute_dm, compute_metrics, label, load_forecasts, load_oracle

st.set_page_config(page_title="Model Comparison", page_icon="⛽", layout="wide")
st.title("Model Comparison")

fc = load_forecasts()
if fc.empty:
    st.error("No forecast files found in forecasts/. Run forecasting_models.ipynb first.")
    st.stop()

METRICS = {
    "MAE": ("Mean absolute error (USD/gal)", "$.3f", "Average size of the forecast error. Lower is better."),
    "RMSE": ("Root mean squared error (USD/gal)", "$.3f", "Like MAE, but punishes large misses more. Lower is better."),
    "MAPE": ("Mean absolute % error", ".2f", "Average error as % of the price. Lower is better."),
    "Skill_vs_naive": ("Skill vs naive (MAE ÷ naive MAE)", ".2f", "Below 1 means the model beats the naive forecast."),
    "Coverage80": ("80% interval coverage", ".0%", "Share of actual prices inside the 80% interval. Ideal: 0.80."),
    "DirAcc": ("Direction correct (%)", ".0f", "Share of forecasts that called the direction of the price change "
               "(up or down) correctly. Higher is better; a coin flip scores about 50%. Naive always forecasts "
               "no change, so it has no score."),
}
HIGHER_IS_BETTER = {"DirAcc"}

c1, c2 = st.columns([1, 2])
horizon = c1.segmented_control("Forecast horizon", [4, 12], default=4, format_func=lambda h: f"{h} weeks ahead",
                               required=True)
metric = c2.selectbox("Metric", list(METRICS), format_func=lambda k: METRICS[k][0])
title, fmt, help_text = METRICS[metric]
st.caption(help_text)

m = compute_metrics(horizon)
plot = m.reset_index().dropna(subset=[metric])
bars = alt.Chart(plot).mark_bar().encode(
    y=alt.Y("Model:N", sort=list(plot.sort_values(metric, ascending=metric not in HIGHER_IS_BETTER)["Model"]),
            title=None, axis=alt.Axis(labelLimit=220)),
    x=alt.X(f"{metric}:Q", title=title),
    color=alt.Color("Family:N", scale=FAMILY_COLORS, title="Model type"),
    tooltip=["Model", "Family", alt.Tooltip(f"{metric}:Q", format=fmt), alt.Tooltip("n:Q", title="weeks scored")])
text = bars.mark_text(align="left", dx=4).encode(text=alt.Text(f"{metric}:Q", format=fmt), color=alt.value("black"))
chart = bars + text
ref = {"Skill_vs_naive": 1.0, "Coverage80": 0.8, "DirAcc": 50.0}.get(metric)
if ref is None and "naive" in m.index:
    ref = m.loc["naive", metric]
if ref is not None:
    chart += alt.Chart(pd.DataFrame({"ref": [ref]})).mark_rule(color="red", strokeDash=[5, 3], size=2).encode(x="ref:Q")
st.altair_chart(chart.properties(height=40 * len(plot) + 40), width="stretch")
st.caption("Red dashed line: naive benchmark (or the ideal value for skill and coverage, and 50% for direction). "
           "Grey bars are benchmarks: naive, seasonal naive, and the 10-year seasonal mean. LLM models forecast "
           "only every 4th week, so their scores cover fewer weeks than the other models.")

st.subheader("All metrics")
table = m[["Model", "Family", "n", "MAE", "RMSE", "MAPE", "Skill_vs_naive", "Coverage80", "DirAcc"]].copy()
table["DirAcc"] = table["DirAcc"].map(lambda v: "—" if pd.isna(v) else f"{v:.0f}%")   # naive forecasts no change
st.dataframe(table, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3, column_config={
    "n": st.column_config.NumberColumn("Weeks scored"),
    "MAE": st.column_config.NumberColumn(format="$%.3f"),
    "RMSE": st.column_config.NumberColumn(format="$%.3f"),
    "MAPE": st.column_config.NumberColumn("MAPE (%)", format="%.2f"),
    "Skill_vs_naive": st.column_config.NumberColumn("Skill vs naive", format="%.2f"),
    "Coverage80": st.column_config.ProgressColumn("80% coverage", min_value=0, max_value=1, format="percent"),
    "DirAcc": st.column_config.TextColumn("Direction correct"),
})

oracle = load_oracle()
if not oracle.empty and horizon in oracle.index:
    st.subheader("How much error comes from not knowing future crude?")
    r = oracle.loc[horizon]
    c1, c2, c3 = st.columns(3)
    c1.metric("ARIMAX, crude held flat", f"${r['arimax']:.3f}", help="What a forecaster could actually do")
    c2.metric("ARIMAX, real crude path", f"${r['arimax_oracle']:.3f}",
              f"{-r['share_of_ARIMAX_error_from_crude']:.0%} error", delta_color="inverse",
              help="Diagnostic only: uses the actual future Brent prices, which no forecaster knows")
    c3.metric("Best conventional model", f"${r['best_conventional']:.3f}",
              help="Most accurate conventional model, for reference")
    st.caption(f"Mean absolute error {horizon} weeks ahead. Giving ARIMAX the real future Brent prices cuts its "
               f"error by {r['share_of_ARIMAX_error_from_crude']:.0%}: that share of its error is crude-price "
               "uncertainty. The rest comes from everything else (seasonality, refinery news, local shocks). This is "
               "why the dashboard treats crude as a scenario input rather than trying to forecast it.")

st.subheader("Is the difference statistically significant?")
candidates = [x for x in m.index if x not in ("snaive",)]
bench = st.selectbox("Compare every model against", candidates,
                     index=candidates.index("naive") if "naive" in candidates else 0, format_func=label)
dm = compute_dm(horizon, bench)
st.dataframe(dm.style.format({"DM statistic": "{:.2f}", "p-value": "{:.3f}"})
             .map(lambda p: "background-color: #d4edda" if p < 0.05 else "", subset=["p-value"]),
             hide_index=True, width="stretch")
st.caption("Diebold-Mariano test (squared errors). Negative DM = the model is more accurate than the benchmark; "
           "green = significant at 5%. The test period is short (about 60 weeks), so large differences in "
           "MAE can still be insignificant.")
