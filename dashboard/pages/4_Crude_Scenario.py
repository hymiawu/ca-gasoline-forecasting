import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import actual_line, band_and_line, load_data
from models import event_scenario, fit_ardl, future_dates, scenario

st.set_page_config(page_title="Crude Scenario", page_icon="⛽", layout="wide")
st.title("Crude Oil Scenario Tool")
st.caption("Choose a path for Brent crude and see where California gasoline is likely to go. "
           "The baseline assumes Brent stays at today's price.")

df = load_data()
last_brent = float(df["brent"].iloc[-1])


@st.cache_resource
def model():
    return fit_ardl(load_data())


res = model()
PRESETS = {  # label: (target USD/bbl, weeks to reach it)
    "Ceasefire ($70)": (70, 4),
    "Today's price": (round(last_brent), 1),
    "War escalation ($150)": (150, 4),
    "Oil shock ($200)": (200, 2),
}
if "target" not in st.session_state:
    st.session_state.target, st.session_state.weeks = round(last_brent), 4


def apply_preset():
    choice = st.session_state.preset
    if choice:
        st.session_state.target, st.session_state.weeks = PRESETS[choice]


left, right = st.columns([1, 2.4], gap="large")

with left:
    st.subheader("Brent path")
    st.pills("Quick scenarios", list(PRESETS), key="preset", on_change=apply_preset)
    mode = st.radio("Path type", ["Move to a target price", "Custom weekly path"], horizontal=True)
    H = st.slider("Weeks to forecast", 4, 26, 12)

    if mode == "Move to a target price":
        target = st.slider("Target Brent price (USD/bbl)", 30, 250, key="target", step=5)
        st.session_state.weeks = min(st.session_state.weeks, H)
        weeks = st.slider("Weeks to reach the target", 1, H, key="weeks")
        path = np.r_[np.linspace(last_brent, target, weeks + 1)[1:], np.repeat(target, H - weeks)]
    else:
        st.caption("Edit the weekly Brent prices below.")
        dates = future_dates(df, H)
        start = pd.DataFrame({"week": dates.strftime("%Y-%m-%d"), "Brent (USD/bbl)": round(last_brent, 1)})
        edited = st.data_editor(start, hide_index=True, disabled=["week"], width="stretch", height=min(35 * H + 40, 420),
                                column_config={"Brent (USD/bbl)": st.column_config.NumberColumn(min_value=1.0, step=5.0)})
        path = edited["Brent (USD/bbl)"].astype(float).values

    st.subheader("Refinery outage")
    outage = st.toggle("Add an unplanned refinery outage",
                       help="Based on 4 past California outages (Richmond 2012, Torrance 2012 and 2015, "
                            "El Segundo 2025): prices jump for ~3 weeks, then partly fall back.")
    outage_at = st.slider("Outage starts in week", 1, H, 2) if outage else None

scen = scenario(df, res, path, outage_at=outage_at)
base = event_scenario(df, res, h=H)

with right:
    end = scen["forecast"].iloc[-1]
    chg = round(end - base["forecast"].iloc[-1], 2) + 0.0  # avoid "-0.00"
    c1, c2, c3 = st.columns(3)
    c1.metric(f"California gas in {H} weeks", f"${end:.2f}/gal",
              f"{end - df['ca_gas'].iloc[-1]:+.2f} vs today", delta_color="inverse", border=True)
    c2.metric("vs baseline (Brent flat)", f"{chg:+.2f}/gal", f"{chg / base['forecast'].iloc[-1]:+.1%}",
              delta_color="inverse", border=True)
    c3.metric("80% range (USD/gal)", f"{scen['lower80'].iloc[-1]:.2f}–{scen['upper80'].iloc[-1]:.2f}", border=True,
              help="80% of simulated paths end inside this range")

    both = pd.concat([scen.assign(series="Your scenario"), base.assign(series="Baseline (Brent flat)")]) \
        .rename_axis("date").reset_index()
    color = alt.Scale(domain=["Your scenario", "Baseline (Brent flat)"], range=["#d62728", "#1f77b4"])
    chart = actual_line(df, df.index[-1] - pd.Timedelta(weeks=40)) + band_and_line(both, "date", "series", scale=color)
    st.altair_chart(chart.properties(height=360, title="California regular gasoline (USD/gal)"), width="stretch")

    brent = pd.concat([df["brent"].iloc[-40:], pd.Series(path, index=scen.index)]).rename("brent").rename_axis("date") \
        .reset_index().assign(kind=lambda d: np.where(d["date"] > df.index[-1], "Scenario", "Actual"))
    bchart = alt.Chart(brent).mark_line(point=True).encode(
        x=alt.X("date:T", title=None), y=alt.Y("brent:Q", title="USD/bbl", scale=alt.Scale(zero=False)),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Actual", "Scenario"], range=["#ff7f0e", "#d62728"]), title=None),
        tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), alt.Tooltip("brent:Q", format="$.1f")])
    st.altair_chart(bchart.properties(height=180, title="Brent crude path"), width="stretch")

    with st.expander("Week-by-week numbers"):
        tbl = pd.DataFrame({"Brent (USD/bbl)": path, "Scenario": scen["forecast"], "Baseline": base["forecast"],
                            "Difference": scen["forecast"] - base["forecast"],
                            "80% low": scen["lower80"], "80% high": scen["upper80"]}, index=scen.index.strftime("%Y-%m-%d"))
        st.dataframe(tbl.style.format({"Brent (USD/bbl)": "${:.1f}", "Difference": "{:+.2f}"} |
                                      {c: "${:.2f}" for c in ["Scenario", "Baseline", "80% low", "80% high"]}),
                     width="stretch")

with st.expander("How this works"):
    st.markdown(f"""
The tool uses an **ARDL model of weekly price changes** estimated on all data since 2000: this week's change in
the California price depends on its own last 4 weekly changes, the last 0–8 weekly changes in Brent, a
52-week seasonal pattern, and refinery-outage terms. Because the model includes its own lags, a crude shock keeps
passing through over several weeks. A permanent 10% rise in Brent raises California gasoline by about
3% after 4 weeks and about 4% in the long run, consistent with the VAR results.

**Why not ARIMAX?** The proposal planned ARIMAX for this tool, but in ARIMAX a crude change affects gasoline only
through a few Brent lags and does not build up over time, so it understates how much crude matters.

**Caveats.** Wars affect California prices mainly through crude, so a war scenario = a Brent path. The outage
effect is estimated from only 4 events and is approximate. Intervals come from 1,000 simulated paths
(bootstrapped residuals) and do not include uncertainty about the Brent path itself.
Last Brent price in the data: **\\${last_brent:.1f}/bbl** ({df.index[-1]:%b %d, %Y}).
""")
