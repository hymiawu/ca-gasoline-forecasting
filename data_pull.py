"""
data_pull.py - Download all data for the California gasoline forecasting project.

Output:
    data/raw/*.csv          raw downloads, one file per series
    data/weekly_data.csv    one row per week, all variables + event dummies
    data/weekly_data.xlsx   same weekly data + raw series + data dictionary (one sheet each)

Setup:
    pip install pandas requests openpyxl
    Get a free EIA API key: https://www.eia.gov/opendata/register.php
    Then either set an environment variable EIA_API_KEY or save the key in a file named .eia_key
    next to this script (it is git-ignored, so the key never ends up on GitHub).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

KEY_FILE = Path(__file__).resolve().parent / ".eia_key"
EIA_API_KEY = os.environ.get("EIA_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else None)
if not EIA_API_KEY:
    raise SystemExit("Missing EIA API key: set EIA_API_KEY or create .eia_key (see the docstring).")
START = "2000-01-01"

DATA = Path(__file__).resolve().parent / "data"
RAW = DATA / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# EIA: California regular gasoline retail price, USD/gal, weekly (Monday)
EIA_SERIES = {"ca_gas": "PET.EMM_EPMR_PTE_SCA_DPG.W"}

# EIA weekly supply data (week ending Friday, released the following Wednesday)
EIA_SUPPLY_SERIES = {
    "refinery_util_padd5": "PET.W_NA_YUP_R50_PER.W",  # West Coast refinery utilization, %
    "gas_stocks_padd5": "PET.WGTSTP51.W",             # West Coast gasoline stocks, thousand barrels
    "refinery_util_us": "PET.WPULEUS3.W",             # U.S. refinery utilization, %
    "gas_stocks_us": "PET.WGTSTUS1.W",                # U.S. gasoline stocks, thousand barrels
}

# FRED: no API key needed (direct CSV download)
FRED_SERIES = {
    "us_gas": "GASREGW",      # U.S. regular gasoline, USD/gal, weekly (Monday)
    "wti": "DCOILWTICO",      # WTI crude, USD/barrel, daily
    "brent": "DCOILBRENTEU",  # Brent crude, USD/barrel, daily
}


def get_eia(series_id: str) -> pd.Series:
    """Download one weekly series from the EIA API v2 (series ID in APIv1 format, e.g. PET.xxx.W)."""
    url = f"https://api.eia.gov/v2/seriesid/{series_id}"
    params = {
        "api_key": EIA_API_KEY,
        "start": START,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()["response"]["data"]
    if not rows:
        raise ValueError(f"EIA returned no data for {series_id}. Check the series ID and API key.")
    df = pd.DataFrame(rows)[["period", "value"]]
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("period")["value"].sort_index()


def get_fred(series_id: str) -> pd.Series:
    """Download one FRED series as CSV (missing values '.' become NaN)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={START}"
    df = pd.read_csv(url)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].sort_index()


def to_weekly_monday(daily: pd.Series) -> pd.Series:
    """
    Average daily prices over the 7 days BEFORE each Monday, labeled with that Monday.
    This matches the EIA price date and only uses crude prices known at that time
    (no look-ahead).
    """
    return daily.resample("W-MON", closed="left", label="right").mean()


def supply_to_monday(s: pd.Series) -> pd.Series:
    """
    Move Friday-dated EIA supply data to the first Monday after its Wednesday
    release (Friday + 10 days), so each week only uses data already published.
    """
    s = s.copy()
    s.index = s.index + pd.Timedelta(days=10)
    return s


DATA_DICTIONARY = [
    ("date", "Week date (Monday), matches EIA retail price date", "", "EIA"),
    ("ca_gas", "California regular gasoline retail price", "USD/gal", "EIA EMM_EPMR_PTE_SCA_DPG"),
    ("us_gas", "U.S. regular gasoline retail price", "USD/gal", "FRED GASREGW"),
    ("wti", "WTI crude, mean of daily prices in the 7 days before the Monday", "USD/barrel", "FRED DCOILWTICO"),
    ("brent", "Brent crude, mean of daily prices in the 7 days before the Monday", "USD/barrel", "FRED DCOILBRENTEU"),
    ("refinery_util_padd5", "West Coast (PADD 5) refinery utilization, latest released week", "%", "EIA W_NA_YUP_R50_PER"),
    ("gas_stocks_padd5", "West Coast (PADD 5) total gasoline stocks, latest released week", "thousand barrels", "EIA WGTSTP51"),
    ("refinery_util_us", "U.S. refinery utilization, latest released week", "%", "EIA WPULEUS3"),
    ("gas_stocks_us", "U.S. total gasoline stocks, latest released week", "thousand barrels", "EIA WGTSTUS1"),
    ("spread", "ca_gas - us_gas", "USD/gal", "derived"),
    ("log_*", "Natural log of the price (NaN when price <= 0)", "", "derived"),
    ("summer_blend", "1 in April-October (approx. summer-blend season)", "0/1", "team"),
    ("july_tax", "1 in the first 4 weeks of July (CA excise tax adjustment)", "0/1", "team"),
    ("covid", "1 from 2020-03-16 to 2020-06-30", "0/1", "team"),
    ("p66_la_closed", "1 from 2025-10-20 (Phillips 66 LA refinery final crude processing ~Oct 16, 2025)", "0/1", "news"),
    ("benicia_closed", "1 from 2026-05-04 (Valero Benicia stopped fuel production, end of Apr 2026)", "0/1", "news"),
    ("chevron_es_fire", "1 from 2025-10-06 to 2025-11-03 (Chevron El Segundo fire, Oct 2, 2025)", "0/1", "news"),
    ("iraq_war", "1 from 2003-03-24 to 2003-04-28 (Iraq war began Mar 20, 2003)", "0/1", "news"),
    ("ukraine_war", "1 from 2022-02-28 to 2022-07-25 (Russia invaded Ukraine Feb 24, 2022)", "0/1", "news"),
    ("iran_war_2026", "1 from 2026-03-02 onward (US/Israel strikes on Iran Feb 28, 2026; Hormuz blocked)", "0/1", "news"),
]


def add_event_dummies(df: pd.DataFrame) -> pd.DataFrame:
    d = df.index
    # Summer-blend season (approx. April-October; refine using CARB dates if needed)
    df["summer_blend"] = d.month.isin(range(4, 11)).astype(int)
    # First 4 weeks after the July 1 California excise tax adjustment
    df["july_tax"] = ((d.month == 7) & (d.day <= 28)).astype(int)
    # COVID demand collapse
    df["covid"] = ((d >= "2020-03-16") & (d <= "2020-06-30")).astype(int)
    # Refinery closures / major outages: fill in dates from news sources, then
    # add one dummy per event (step dummy = 1 from the event date onward).
    # Dates are moved to the first Monday on/after the event (EIA price date).
    # Step dummies: 1 from this Monday onward
    refinery_events = {
        "p66_la_closed": "2025-10-20",   # Phillips 66 LA final crude processing ~Oct 16, 2025 (company update, Oct 1, 2025)
        "benicia_closed": "2026-05-04",  # Valero Benicia stopped fuel production end of Apr 2026 (Valero 10-Q, Argus)
    }
    for name, date in refinery_events.items():
        df[name] = (d >= date).astype(int)
    # Pulse dummies: 1 between start and end (inclusive); end=None means through the latest week.
    # End dates are approximate (end of the main price spike), not official dates.
    pulse_events = {
        "chevron_es_fire": ("2025-10-06", "2025-11-03"),  # Chevron El Segundo fire, Oct 2, 2025
        "iraq_war": ("2003-03-24", "2003-04-28"),         # Iraq war began Mar 20, 2003
        "ukraine_war": ("2022-02-28", "2022-07-25"),      # Russia invaded Ukraine Feb 24, 2022
        "iran_war_2026": ("2026-03-02", None),            # US/Israel strikes on Iran Feb 28, 2026; Hormuz blocked
    }
    for name, (start, end) in pulse_events.items():
        df[name] = ((d >= start) & (d <= (end or d.max()))).astype(int)
    return df


def main():
    series = {}

    for name, sid in EIA_SERIES.items():
        print(f"Downloading EIA {sid} ...")
        s = get_eia(sid)
        s.to_csv(RAW / f"{name}.csv")
        series[name] = s

    for name, sid in FRED_SERIES.items():
        print(f"Downloading FRED {sid} ...")
        s = get_fred(sid)
        s.to_csv(RAW / f"{name}.csv")
        series[name] = s

    for name, sid in EIA_SUPPLY_SERIES.items():
        print(f"Downloading EIA {sid} ...")
        s = get_eia(sid)
        s.to_csv(RAW / f"{name}.csv")
        series[name] = s

    # Build the weekly dataset on the California price dates
    df = pd.DataFrame({"ca_gas": series["ca_gas"]})
    df = df.join(series["us_gas"].rename("us_gas"), how="left")
    df = df.join(to_weekly_monday(series["wti"]).rename("wti"), how="left")
    df = df.join(to_weekly_monday(series["brent"]).rename("brent"), how="left")
    for name in EIA_SUPPLY_SERIES:
        df = df.join(supply_to_monday(series[name]).rename(name), how="left")

    # Fill short gaps (holidays); keep longer gaps visible
    df = df.ffill(limit=2)

    # Derived variables
    df["spread"] = df["ca_gas"] - df["us_gas"]
    for col in ["ca_gas", "us_gas", "wti", "brent"]:
        df[f"log_{col}"] = np.log(df[col].where(df[col] > 0))  # WTI went negative in Apr 2020

    df = add_event_dummies(df)
    df.index.name = "date"
    df.to_csv(DATA / "weekly_data.csv")

    with pd.ExcelWriter(DATA / "weekly_data.xlsx") as xl:
        df.to_excel(xl, sheet_name="weekly_data")
        pd.DataFrame(DATA_DICTIONARY, columns=["variable", "description", "units", "source"]).to_excel(
            xl, sheet_name="data_dictionary", index=False)
        for name, s in series.items():
            s.rename(name).rename_axis("date").to_frame().to_excel(xl, sheet_name=f"raw_{name}"[:31])

    # Quick checks
    print(f"\nSaved {DATA / 'weekly_data.csv'} and {DATA / 'weekly_data.xlsx'}")
    print(f"Weeks: {len(df)}  |  {df.index.min().date()} to {df.index.max().date()}")
    print("Missing values per column:")
    print(df.isna().sum()[lambda x: x > 0].to_string() or "  none")
    print("\nLast 5 rows:")
    print(df[["ca_gas", "us_gas", "spread", "wti", "brent"]].tail().round(3).to_string())


if __name__ == "__main__":
    main()
