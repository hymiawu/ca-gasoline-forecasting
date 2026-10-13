"""
Scenario model used by the dashboard (same as section 8 of forecasting_models.ipynb).

ARDL in weekly log changes:
    dlogP_t = c + sum_i phi_i dlogP_{t-i} + sum_k beta_k dlogBrent_{t-k} + seasonal + outage terms + e_t
Forecast intervals come from bootstrapping the residuals.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

PERIOD = 365.25 / 7
K_SEAS = 3                  # Fourier pairs (chosen by AIC in the notebook)
P_AR, Q_BRENT = 4, 8
SPIKE_W, RECOV_W = 3, 5     # outage: weeks 1-3 prices jump, weeks 4-8 partly reverse

OUTAGES = {  # unplanned California refinery outages (first Monday on/after the event)
    "Chevron Richmond fire": "2012-08-06",
    "ExxonMobil Torrance power outage": "2012-10-01",
    "ExxonMobil Torrance explosion": "2015-02-23",
    "Chevron El Segundo fire": "2025-10-06",
}


def future_dates(data, h):
    return pd.date_range(data.index[-1] + pd.Timedelta(days=7), periods=h, freq="W-MON")


def outage_dummies(index, starts):
    spike = pd.Series(0.0, index=index, name="outage_spike")
    rec = pd.Series(0.0, index=index, name="outage_recovery")
    for d in map(pd.Timestamp, starts):
        if d in index:
            i = index.get_loc(d)
            spike.iloc[i:i + SPIKE_W] = 1
            rec.iloc[i + SPIKE_W:i + SPIKE_W + RECOV_W] = 1
    return spike, rec


def seas_diff(dates, t0):
    t = ((dates - t0).days / 7).values
    return pd.DataFrame({f"{f}{k}": fn(2 * np.pi * k * t / PERIOD) - fn(2 * np.pi * k * (t - 1) / PERIOD)
                         for k in range(1, K_SEAS + 1) for f, fn in (("ds", np.sin), ("dc", np.cos))}, index=dates)


def fit_ardl(data):
    dy, db = data["log_ca_gas"].diff(), data["log_brent"].diff()
    spike, rec = outage_dummies(data.index, OUTAGES.values())
    X = pd.concat([pd.DataFrame({f"dy_l{i}": dy.shift(i) for i in range(1, P_AR + 1)}),
                   pd.DataFrame({f"db_l{k}": db.shift(k) for k in range(Q_BRENT + 1)}),
                   seas_diff(data.index, data.index[0]), spike, rec], axis=1)
    d = pd.concat([dy.rename("dy"), sm.add_constant(X)], axis=1).dropna()
    return sm.OLS(d["dy"], d.drop(columns="dy")).fit(cov_type="HAC", cov_kwds={"maxlags": 4})


def event_scenario(data, res, h=12, brent_path=None, outage_at=None, n_sims=1000, seed=0):
    """
    brent_path: future weekly Brent (USD/bbl), length h. None = Brent stays at its last value.
    outage_at:  horizon week (1..h) when a new refinery outage starts. None = no outage.
    Returns a DataFrame indexed by future date: forecast (median), lower80, upper80 in USD/gal.
    """
    b = res.params
    fut = future_dates(data, h)
    future_lb = np.log(brent_path) if brent_path is not None else data["log_brent"].iloc[-1]
    db = pd.concat([data["log_brent"], pd.Series(future_lb, index=fut)]).diff()
    starts = list(OUTAGES.values()) + ([fut[outage_at - 1]] if outage_at else [])
    spike, rec = outage_dummies(data.index.append(fut), starts)

    det = np.full(h, b["const"])
    for k in range(Q_BRENT + 1):
        det += b[f"db_l{k}"] * db.shift(k).loc[fut].values
    S = seas_diff(fut, data.index[0])
    for c in S:
        det += b[c] * S[c].values
    det += b["outage_spike"] * spike.loc[fut].values + b["outage_recovery"] * rec.loc[fut].values

    phi = b[[f"dy_l{i}" for i in range(1, P_AR + 1)]].values
    lags = np.tile(data["log_ca_gas"].diff().iloc[-P_AR:].values[::-1], (n_sims, 1))
    shocks = np.random.default_rng(seed).choice(res.resid.values, size=(n_sims, h))
    paths = np.zeros((n_sims, h))
    for s in range(h):
        paths[:, s] = det[s] + lags @ phi + shocks[:, s]
        lags = np.column_stack([paths[:, s], lags[:, :-1]])
    lvl = data["log_ca_gas"].iloc[-1] + paths.cumsum(axis=1)
    q = np.exp(np.percentile(lvl, [10, 50, 90], axis=0))
    return pd.DataFrame({"forecast": q[1], "lower80": q[0], "upper80": q[2]}, index=fut)


def scenario(data, res, brent_path, outage_at=None):
    """Dashboard entry point: a Brent price path (USD/bbl) -> California gasoline forecast."""
    return event_scenario(data, res, h=len(brent_path), brent_path=np.asarray(brent_path, float), outage_at=outage_at)
