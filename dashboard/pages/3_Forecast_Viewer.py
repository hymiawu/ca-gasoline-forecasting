import altair as alt
import pandas as pd
import streamlit as st

from common import TRAIN_END, actual_line, band_and_line, label, load_data, load_forecasts

st.set_page_config(page_title="Forecast Viewer", page_icon="⛽", layout="wide")
st.title("Forecast Viewer")

df = load_data()
fc = load_forecasts()
if fc.empty:
    st.error("No forecast files found in forecasts/. Run forecasting_models.ipynb first.")
    st.stop()

models = sorted(fc["model"].unique(), key=label)
default = [m for m in ("arima_fourier", "naive") if m in models]
chosen = st.multiselect("Models", models, default=default, format_func=label)
if not chosen:
    st.info("Pick at least one model.")
    st.stop()
sel = fc[fc["model"].isin(chosen)]

tab1, tab2 = st.tabs(["Forecasts at a fixed horizon", "Forecast from one week"])

# ---- tab 1: all origins, one horizon ----
with tab1:
    h = st.slider("Weeks ahead", 1, 12, 4)
    d = sel[sel["horizon"] == h]
    chart = actual_line(df, "2025-01-01") + band_and_line(d, "target_date", "label")
    split = alt.Chart(pd.DataFrame({"x": [TRAIN_END]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:T")
    st.altair_chart((chart + split).properties(height=380).interactive(bind_y=False), width="stretch")
    st.caption(f"Each point is the forecast made {h} week(s) earlier for that date. Black: actual price. "
               "Shaded: 80% interval. Dashed line: end of the training period.")

    st.markdown("**Forecast error over time** (actual − forecast)")
    err = alt.Chart(d.dropna(subset=["err"])).mark_line(point=True).encode(
        x=alt.X("target_date:T", title=None), y=alt.Y("err:Q", title="USD/gal", axis=alt.Axis(format="+.1f")),
        color=alt.Color("label:N", title=None, scale=alt.Scale(scheme="tableau10"), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("target_date:T", format="%Y-%m-%d"), "label", alt.Tooltip("err:Q", format="+.3f")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black").encode(y="y:Q")
    st.altair_chart((err + zero).properties(height=220), width="stretch")
    st.caption("Positive = the model under-predicted (price rose more than expected). Big misses cluster around "
               "sudden shocks such as the start of the 2026 Iran war.")

# ---- tab 2: one origin, all horizons ----
with tab2:
    origins = sorted(sel["origin_date"].unique())
    origin = st.select_slider("Forecast made in the week of", options=origins, value=origins[len(origins) // 2],
                              format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
    origin = pd.Timestamp(origin)
    d = sel[sel["origin_date"] == origin]
    start = origin - pd.Timedelta(weeks=26)
    end = d["target_date"].max()
    known = alt.Chart(pd.DataFrame({"x": [origin]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(x="x:T")
    chart = actual_line(df, start, end) + band_and_line(d, "target_date", "label") + known
    st.altair_chart(chart.properties(height=380), width="stretch")
    st.caption("What each model predicted for the next 12 weeks, using only data up to the dashed line, "
               "compared with what actually happened (black).")

    tbl = d.pivot(index="target_date", columns="label", values="forecast")
    tbl.insert(0, "Actual", tbl.index.map(df["ca_gas"]))
    tbl.index = [f"+{i} wk ({t:%b %d})" for i, t in enumerate(tbl.index, 1)]
    st.dataframe(tbl.style.format("${:.2f}", na_rep="—"), width="stretch")
