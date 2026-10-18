"""Shared paths, data loaders, metrics and chart helpers for the dashboard pages."""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent   # Project/
DATA = ROOT / "data" / "weekly_data.csv"
FC_DIR = ROOT / "forecasts"
RES_DIR = ROOT / "results"

TRAIN_END = pd.Timestamp("2025-06-30")

MODEL_LABELS = {
    "naive": "Naive",
    "smean": "Seasonal mean (10y)",
    "snaive": "Seasonal naive",
    "ets": "ETS",
    "arima_fourier": "ARIMA + Fourier",
    "arima_month": "ARIMA + month dummies",
    "arimax": "ARIMAX",
    "ardl": "ARDL (scenario model)",
    "combo": "Combination",
    "chronos_bolt": "Chronos-Bolt (AI)",
    "chronos2": "Chronos-2 (AI)",
    "timesfm": "TimesFM 2.5 (AI)",
    "llm_claude_anon": "Claude, anonymized (LLM)",
    "llm_claude_context": "Claude, full context (LLM)",
    "llm_openai_anon": "ChatGPT, anonymized (LLM)",
    "llm_openai_context": "ChatGPT, full context (LLM)",
}
BENCHMARKS = {"naive", "snaive", "smean"}
AI_PREFIXES = ("chronos", "timesfm", "llm")

EVENTS = [  # (label, start, end) - shaded on the history charts
    ("Iraq war", "2003-03-24", "2003-04-28"),
    ("COVID", "2020-03-16", "2020-06-30"),
    ("Ukraine war", "2022-02-28", "2022-07-25"),
    ("Iran war", "2026-03-02", None),
]
REFINERY_MARKERS = [  # (label, date) - dashed vertical lines
    ("Torrance explosion", "2015-02-23"),
    ("El Segundo fire", "2025-10-06"),
    ("P66 LA closed", "2025-10-20"),
    ("Benicia closed", "2026-05-04"),
]


def label(model: str) -> str:
    if model == "arimax" and load_summary().get("arimax_crude_form"):
        return f"ARIMAX (crude as {load_summary()['arimax_crude_form']})"
    return MODEL_LABELS.get(model, model)


def family(model: str) -> str:
    if model in BENCHMARKS:
        return "Benchmark"
    if model.startswith(AI_PREFIXES):
        return "AI"
    if model == "combo":
        return "Combination"
    return "Conventional"


FAMILY_COLORS = alt.Scale(domain=["Benchmark", "Conventional", "AI", "Combination"],
                          range=["#9e9e9e", "#1f77b4", "#d62728", "#9467bd"])


# ---------- data ----------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA, index_col="date", parse_dates=True).asfreq("W-MON")


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    """All forecasts/*.csv in the shared format, with actual prices and errors attached."""
    files = sorted(FC_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    fc = pd.concat([pd.read_csv(f, parse_dates=["origin_date", "target_date"]) for f in files], ignore_index=True)
    fc["actual"] = fc["target_date"].map(load_data()["ca_gas"])
    fc["err"] = fc["actual"] - fc["forecast"]
    fc["label"] = fc["model"].map(label)
    fc["family"] = fc["model"].map(family)
    return fc


@st.cache_data
def compute_metrics(horizon: int) -> pd.DataFrame:
    fc = load_forecasts()
    d = fc[(fc["horizon"] == horizon) & fc["actual"].notna()].copy()
    d["covered"] = d["actual"].between(d["lower80"], d["upper80"])
    d["ape"] = (d["err"] / d["actual"]).abs() * 100
    origin_price = d["origin_date"].map(load_data()["ca_gas"])
    f_move, a_move = d["forecast"] - origin_price, d["actual"] - origin_price
    f_dir = np.where(f_move.abs() < 1e-6, 0, np.sign(f_move))   # "no change" forecasts (naive) don't count
    d["dir_ok"] = np.where(f_dir == 0, np.nan, (f_dir == np.sign(a_move)).astype(float))
    m = d.groupby("model").agg(
        n=("err", "size"),
        MAE=("err", lambda e: e.abs().mean()),
        RMSE=("err", lambda e: np.sqrt((e ** 2).mean())),
        MAPE=("ape", "mean"),
        Coverage80=("covered", "mean"),
        DirAcc=("dir_ok", lambda s: s.mean() * 100),
    )
    m["Skill_vs_naive"] = m["MAE"] / m.loc["naive", "MAE"] if "naive" in m.index else np.nan
    m["Model"] = m.index.map(label)
    m["Family"] = m.index.map(family)
    return m.sort_values("MAE")


@st.cache_data
def load_summary() -> dict:
    """Key numbers written by the notebook's Summary section (empty if not run yet)."""
    f = RES_DIR / "summary.json"
    return json.loads(f.read_text()) if f.exists() else {}


@st.cache_data
def load_validation() -> pd.DataFrame:
    """Validation-period scores the notebook used to choose models."""
    f = RES_DIR / "validation_metrics.csv"
    return pd.read_csv(f, index_col="model") if f.exists() else pd.DataFrame()


@st.cache_data
def load_oracle() -> pd.DataFrame:
    """ARIMAX with flat vs realised crude (notebook section 4.3); empty if not computed yet."""
    f = RES_DIR / "oracle_crude.csv"
    return pd.read_csv(f, index_col="horizon") if f.exists() else pd.DataFrame()


def dm_test(e1, e2, h, power=2):
    """Diebold-Mariano test with HAC variance (h-1 lags) and Harvey-Leybourne-Newbold correction."""
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n, dbar = len(d), d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        var = gamma[0] / n
    dm = dbar / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dm, 2 * stats.t.sf(abs(dm), df=n - 1)


@st.cache_data
def compute_dm(horizon: int, benchmark: str) -> pd.DataFrame:
    fc = load_forecasts()
    e = fc[fc["horizon"] == horizon].pivot(index="origin_date", columns="model", values="err")
    rows = []
    for m in e.columns:
        if m != benchmark:
            pair = e[[m, benchmark]].dropna()   # LLMs forecast only every 4th week: align each pair
            if len(pair) < 5:
                continue
            dm, p = dm_test(pair[m].values, pair[benchmark].values, horizon)
            rows.append({"Model": label(m), "vs": label(benchmark), "Weeks": len(pair), "DM statistic": dm, "p-value": p,
                         "Result": ("more accurate" if dm < 0 else "less accurate") + (" (significant)" if p < 0.05 else " (not significant)")})
    return pd.DataFrame(rows).sort_values("DM statistic")


# ---------- chart helpers ----------
def event_layer(start, end, show_labels=True):
    """Grey bands for wars / COVID and dashed lines for refinery events within [start, end]."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    ev = pd.DataFrame([(n, pd.Timestamp(a), pd.Timestamp(b) if b else end) for n, a, b in EVENTS],
                      columns=["event", "start", "end"])
    ev = ev[(ev["end"] >= start) & (ev["start"] <= end)].assign(start=lambda d: d["start"].clip(lower=start),
                                                              end=lambda d: d["end"].clip(upper=end))
    ref = pd.DataFrame([(n, pd.Timestamp(d)) for n, d in REFINERY_MARKERS], columns=["event", "date"])
    ref = ref[(ref["date"] >= start) & (ref["date"] <= end)]
    layers = [
        alt.Chart(ev).mark_rect(color="gray", opacity=0.15).encode(x="start:T", x2="end:T", tooltip=["event"]),
        alt.Chart(ref).mark_rule(color="#555", strokeDash=[4, 3], opacity=0.6).encode(
            x="date:T", tooltip=["event", alt.Tooltip("date:T", format="%Y-%m-%d")]),
    ]
    if show_labels:
        layers.append(alt.Chart(ev).mark_text(align="left", baseline="top", dx=2, dy=2, fontSize=10, color="gray")
                      .encode(x="start:T", y=alt.value(0), text="event"))
    return alt.layer(*layers)


def band_and_line(fc: pd.DataFrame, x: str, color_field: str, title_y="USD/gal", scale=None):
    """80% interval band + forecast line for each value of color_field."""
    scale = scale or alt.Scale(scheme="tableau10")
    base = alt.Chart(fc).encode(x=alt.X(f"{x}:T", title=None))
    band = base.mark_area(opacity=0.15).encode(
        y=alt.Y("lower80:Q", title=title_y, scale=alt.Scale(zero=False)), y2="upper80:Q",
        color=alt.Color(f"{color_field}:N", legend=None, scale=scale))
    line = base.mark_line(strokeWidth=2).encode(
        y="forecast:Q", color=alt.Color(f"{color_field}:N", title=None, scale=scale, legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip(f"{x}:T", format="%Y-%m-%d"), alt.Tooltip(f"{color_field}:N", title="series"),
                 alt.Tooltip("forecast:Q", format="$.2f"), alt.Tooltip("lower80:Q", format="$.2f"),
                 alt.Tooltip("upper80:Q", format="$.2f")])
    return band + line


def actual_line(df: pd.DataFrame, start=None, end=None):
    s = df["ca_gas"].loc[start:end].rename("actual").reset_index()
    return alt.Chart(s).mark_line(color="black", strokeWidth=2.5).encode(
        x="date:T", y=alt.Y("actual:Q", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), alt.Tooltip("actual:Q", format="$.3f", title="actual")])
