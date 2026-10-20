import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.seasonal import STL

from common import GAL_PER_BBL, data_version, event_layer, load_data

st.set_page_config(page_title="Historical Prices", page_icon="⛽", layout="wide")
st.title("Historical Prices")

df = load_data()

with st.sidebar:
    st.header("Options")
    start, end = st.slider("Date range", min_value=df.index.min().date(), max_value=df.index.max().date(),
                           value=(pd.Timestamp("2015-01-01").date(), df.index.max().date()), format="YYYY")
    show_events = st.toggle("Show events", value=True, help="Grey bands: wars and COVID. Dashed lines: refinery events.")

d = df.loc[str(start):str(end)]
ev = event_layer(start, end) if show_events else None
with_events = lambda chart: (ev + chart) if ev is not None else chart

# ---- KPIs for the selected range ----
c1, c2, c3 = st.columns(3)
c1.metric("Average California price", f"${d.ca_gas.mean():.2f}/gal", border=True)
c2.metric("Average California premium", f"${d.spread.mean():.2f}/gal", border=True,
          help="California price minus the U.S. average")
peak = d.ca_gas.idxmax()
c3.metric("Highest California price", f"${d.ca_gas.max():.2f}/gal", f"week of {peak:%b %d, %Y}",
          delta_color="off", border=True)

# ---- prices ----
st.subheader("California vs U.S. regular gasoline")
long = d[["ca_gas", "us_gas"]].rename(columns={"ca_gas": "California", "us_gas": "U.S. average"}) \
    .reset_index().melt("date", var_name="series", value_name="price")
hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
lines = alt.Chart(long).mark_line().encode(
    x=alt.X("date:T", title=None), y=alt.Y("price:Q", title="USD/gal", scale=alt.Scale(zero=False)),
    color=alt.Color("series:N", title=None, scale=alt.Scale(range=["#1f77b4", "#ff7f0e"]), legend=alt.Legend(orient="top")))
points = lines.mark_point(size=40, filled=True).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
rule = alt.Chart(d.reset_index()).mark_rule(color="gray").encode(
    x="date:T", opacity=alt.condition(hover, alt.value(0.4), alt.value(0)),
    tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), alt.Tooltip("ca_gas:Q", title="California", format="$.3f"),
             alt.Tooltip("us_gas:Q", title="U.S.", format="$.3f"), alt.Tooltip("spread:Q", title="Premium", format="$.3f")]
).add_params(hover)
st.altair_chart(with_events(lines + points + rule).properties(height=340), width="stretch")

# ---- spread ----
st.subheader("California premium over the U.S. average")
spread = alt.Chart(d.reset_index()).mark_area(color="#d62728", opacity=0.35, line={"color": "#d62728"}).encode(
    x=alt.X("date:T", title=None), y=alt.Y("spread:Q", title="USD/gal"),
    tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), alt.Tooltip("spread:Q", format="$.2f", title="Premium")])
st.altair_chart(with_events(spread).properties(height=240), width="stretch")
st.caption("The premium has widened from about \\$0.30 before 2015 to over \\$1.50 recently, as California's "
           "refining capacity shrank and its isolated fuel market became tighter.")

# ---- crude ----
st.subheader("California gasoline vs Brent crude")
gc = d.assign(brent_gal=d["brent"] / GAL_PER_BBL)[["ca_gas", "brent_gal"]] \
    .rename(columns={"ca_gas": "California gasoline", "brent_gal": "Brent crude"})
gc_long = gc.reset_index().melt("date", var_name="series", value_name="price")
crude_chart = alt.Chart(gc_long).mark_line().encode(
    x=alt.X("date:T", title=None), y=alt.Y("price:Q", title="USD/gal"),
    color=alt.Color("series:N", title=None, scale=alt.Scale(range=["#ff7f0e", "#1f77b4"]), legend=alt.Legend(orient="top")),
    tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), "series", alt.Tooltip("price:Q", format="$.2f", title="USD/gal")])
st.altair_chart(with_events(crude_chart).properties(height=300), width="stretch")
st.caption("Both prices in USD per gallon (crude = USD per barrel ÷ 42). The gap between the lines is everything "
           "besides crude: refining, taxes, distribution and retail margins. Crude oil leads pump prices: most of a "
           "change in Brent shows up in California prices within 0–5 weeks.")


# ---- seasonality ----
@st.cache_data
def seasonal_by_month(version):
    train = load_data().loc[:"2025-06-30", "log_ca_gas"]
    seas = STL(train, period=52, robust=True).fit().seasonal
    return ((np.exp(seas.groupby(seas.index.month).mean()) - 1) * 100).rename("effect")


st.subheader("Seasonal pattern")
seas = seasonal_by_month(data_version()).reset_index().rename(columns={"date": "month"})
seas["Month"] = pd.to_datetime(seas["month"], format="%m").dt.strftime("%b")
bars = alt.Chart(seas).mark_bar().encode(
    x=alt.X("Month:N", sort=list(seas["Month"]), title=None),
    y=alt.Y("effect:Q", title="% vs average"),
    color=alt.condition("datum.effect > 0", alt.value("#d62728"), alt.value("#1f77b4")),
    tooltip=["Month", alt.Tooltip("effect:Q", format="+.1f", title="% vs average")])
st.altair_chart(bars.properties(height=240), width="stretch")
st.caption("Prices are typically higher from spring to early fall, when California requires the more "
           "expensive summer-blend gasoline (STL decomposition, 2000 – June 2025).")
