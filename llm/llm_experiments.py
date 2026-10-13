"""
llm_experiments.py - LLM forecasting experiments (proposal section 5).

Design (from the proposal):
  * Every 4th week of the test period (~17 forecast dates), each repeated 3 times; the 3 runs are averaged.
  * Experiment 1, "anon": the LLM sees only the last 156 weekly values rescaled to an index
    (first value = 100) - no series name, no dates.
  * Experiment 2, "context": the LLM is told the series is California regular gasoline, sees dates,
    recent Brent prices and a short news summary of events known on the forecast date.
  * The gap between the two shows whether accuracy comes from pattern recognition or background knowledge.

Outputs:
  forecasts/llm_<provider>_<experiment>.csv   shared forecast format (forecast date subset)
  results/llm_log.jsonl                       every prompt, raw response, model version, token usage

Run (from the Project folder; needs API keys in the environment):
  .venv-ai/bin/python llm/llm_experiments.py --dry-run          # offline test with a fake model
  .venv-ai/bin/python llm/llm_experiments.py --providers claude  # ANTHROPIC_API_KEY
  .venv-ai/bin/python llm/llm_experiments.py --providers openai  # OPENAI_API_KEY (+ OPENAI_MODEL)
Already-logged calls are skipped, so an interrupted run can be restarted without paying twice.
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA = ROOT / "data" / "weekly_data.csv"
TRAIN_END = "2025-06-30"
H, CONTEXT, EVERY, REPEATS = 12, 156, 4, 3
EXPERIMENTS = ["anon", "context"]

# Training-data cutoffs, recorded for the look-ahead check. A cutoff inside the test period
# (July 2025 onward) means the model may have seen part of what it is asked to forecast;
# such models should be judged mainly on the anonymized experiment.
TRAINING_CUTOFF = {
    "claude-opus-5-5": "2026-06 (reliable knowledge cutoff per Anthropic)",
    "gpt-5": "fill in from OpenAI's model page",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "median": {"type": "array", "items": {"type": "number"}},
        "lower80": {"type": "array", "items": {"type": "number"}},
        "upper80": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["median", "lower80", "upper80"],
    "additionalProperties": False,
}

SYSTEM = ("You are an expert time-series forecaster. You will receive a weekly time series and must forecast "
          "the next 12 weekly values. Return JSON with three arrays of exactly 12 numbers each: "
          "'median' (your central forecast), 'lower80' and 'upper80' (an 80% prediction interval: the 10th and "
          "90th percentiles). Use the same units and scale as the input. Return only the JSON.")

# Events dated by when they became public. Only events dated on or before the forecast date are shown,
# and no item mentions anything that happened after its own date.
EVENTS = [
    ("2024-10-16", "Phillips 66 announced it will close its Los Angeles refinery (about 8% of California's "
                   "refining capacity) in the fourth quarter of 2025."),
    ("2025-04-16", "Valero announced plans to idle or close its Benicia refinery (about 170,000 barrels/day) "
                   "by the end of April 2026."),
    ("2025-06-13", "Israel launched strikes on Iran; oil prices jumped."),
    ("2025-06-24", "A ceasefire between Israel and Iran took effect."),
    ("2025-10-02", "A large fire hit a jet-fuel unit at Chevron's El Segundo refinery near Los Angeles; the "
                   "gasoline units were reported unaffected."),
    ("2025-10-01", "Phillips 66 said its Los Angeles refinery received its final crude cargo on Sept 30 and expects "
                   "to stop processing crude around Oct 16; California will rely more on imported gasoline."),
    ("2026-02-28", "The United States and Israel launched strikes on Iran; Iran retaliated and blocked the "
                   "Strait of Hormuz, which carries about 20% of world oil supply."),
    ("2026-03-18", "Iran struck Qatar's Ras Laffan energy complex, cutting Qatar's LNG export capacity."),
    ("2026-04-08", "The US and Iran agreed to a two-week ceasefire, but tanker traffic through Hormuz stayed far "
                   "below normal."),
    ("2026-04-30", "Valero said it has stopped producing fuels at its Benicia refinery; Northern California relies "
                   "more on imported gasoline."),
    ("2026-07-08", "US strikes on Iran resumed after attacks on ships in the Strait of Hormuz."),
    ("2026-09-01", "Renewed attacks on shipping and energy infrastructure; Brent rose to about $109, and talks to "
                   "reopen Hormuz stalled."),
]
BACKGROUND = ("Background: California requires a special cleaner-burning gasoline blend, switches to more expensive "
              "summer-blend gasoline in spring and back to winter blend after October, adjusts its gasoline excise "
              "tax every July 1, and is isolated from other U.S. refining centers, so refinery outages raise "
              "prices quickly.")


def load_data():
    return pd.read_csv(DATA, index_col="date", parse_dates=True).asfreq("W-MON")


def forecast_dates(df):
    test_origins = df.loc[TRAIN_END:].index[:-1]
    return list(test_origins[::EVERY])


def build_prompt(df, origin, experiment):
    hist = df["ca_gas"].loc[:origin].iloc[-CONTEXT:]
    if experiment == "anon":
        scale = 100 / hist.iloc[0]
        values = np.round(hist.values * scale, 2).tolist()
        prompt = (f"Here are {CONTEXT} consecutive weekly observations of a series, rescaled so the first value is "
                  f"100. Forecast the next 12 weekly values on the same scale.\n\nvalues: {json.dumps(values)}")
        return prompt, scale

    brent = df["brent"].loc[:origin].iloc[-12:]
    news = [f"- {d}: {t}" for d, t in EVENTS if pd.Timestamp(d) <= origin][-6:]
    future = pd.date_range(origin + pd.Timedelta(weeks=1), periods=H, freq="W-MON")
    lines = [f"{d:%Y-%m-%d}: {v:.3f}" for d, v in hist.items()]
    prompt = (
        f"Series: California regular gasoline average retail price, USD per gallon, weekly (EIA, Monday prices).\n"
        f"Today is {origin:%Y-%m-%d}. Forecast the prices for these 12 Mondays: "
        f"{', '.join(f'{d:%Y-%m-%d}' for d in future)}.\n\n{BACKGROUND}\n\n"
        f"Recent news known as of today:\n" + ("\n".join(news) if news else "- (none)") + "\n\n"
        f"Brent crude oil, USD/barrel, weekly average (last 12 weeks): "
        f"{', '.join(f'{d:%m-%d} {v:.1f}' for d, v in brent.items())}\n\n"
        f"California gasoline price history (last {CONTEXT} weeks):\n" + "\n".join(lines) +
        f"\n\nvalues: {json.dumps(np.round(hist.values, 3).tolist())}"
    )
    return prompt, 1.0


def validate(parsed):
    arrs = [np.asarray(parsed[k], dtype=float) for k in ("lower80", "median", "upper80")]
    if any(len(a) != H for a in arrs) or any(not np.all(np.isfinite(a)) for a in arrs):
        raise ValueError(f"expected 3 arrays of {H} finite numbers, got lengths {[len(a) for a in arrs]}")
    lo, med, hi = np.sort(np.vstack(arrs), axis=0)   # repair any crossing
    return lo, med, hi


class MockForecaster:
    """Offline stand-in for --dry-run: random walk from the last value, with noise."""
    name = "mock"
    model = "mock-random-walk"

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def forecast(self, system, prompt, schema):
        last = json.loads(re.findall(r"values: (\[.*\])", prompt)[-1])[-1]
        med = last * (1 + self.rng.normal(0, 0.01, H).cumsum())
        out = {"median": med.tolist(), "lower80": (med * 0.95).tolist(), "upper80": (med * 1.05).tolist()}
        return {"parsed": out, "raw": json.dumps(out), "model_returned": self.model, "stop_reason": "end_turn",
                "input_tokens": len(prompt) // 4, "output_tokens": 200, "request_id": ""}


def make_provider(name):
    if name == "claude":
        from claude_client import ClaudeForecaster
        return ClaudeForecaster()
    if name == "openai":
        from openai_client import OpenAIForecaster
        return OpenAIForecaster()
    return MockForecaster()


def run(provider_names, dry_run):
    df = load_data()
    dates = forecast_dates(df)
    out_fc = ROOT / ("results/llm_dryrun" if dry_run else "forecasts")
    out_fc.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / ("results/llm_dryrun/llm_log.jsonl" if dry_run else "results/llm_log.jsonl")
    done = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("ok"):
                done[(r["provider"], r["experiment"], r["origin_date"], r["repeat"])] = r

    n_calls = len(provider_names) * len(EXPERIMENTS) * len(dates) * REPEATS
    print(f"{len(dates)} forecast dates ({dates[0].date()} -> {dates[-1].date()}), {REPEATS} repeats, "
          f"{len(EXPERIMENTS)} experiments, {len(provider_names)} provider(s) = {n_calls} calls "
          f"({len(done)} already logged)")

    for pname in provider_names:
        provider = make_provider(pname)
        for exp in EXPERIMENTS:
            for origin in dates:
                prompt, scale = build_prompt(df, origin, exp)
                for rep in range(1, REPEATS + 1):
                    key = (pname, exp, str(origin.date()), rep)
                    if key in done:
                        continue
                    rec = {"provider": pname, "model_requested": provider.model, "experiment": exp,
                           "origin_date": str(origin.date()), "repeat": rep, "scale": scale,
                           "timestamp": datetime.now().isoformat(timespec="seconds"),
                           "training_cutoff": TRAINING_CUTOFF.get(provider.model, "unknown"),
                           "system": SYSTEM, "prompt": prompt}
                    try:
                        res = provider.forecast(SYSTEM, prompt, SCHEMA)
                        validate(res["parsed"])
                        rec.update(ok=True, **res)
                    except Exception as e:   # log and move on; rerun the script to retry failures
                        rec.update(ok=False, error=f"{type(e).__name__}: {e}")
                        print(f"  {pname}/{exp} {origin.date()} run {rep}: {rec['error']}")
                    with log_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    if rec["ok"]:
                        done[key] = rec
                print(f"  {pname}/{exp} {origin.date()} done")
            save_forecasts(pname, exp, dates, done, out_fc)


def save_forecasts(pname, exp, dates, done, out_dir):
    rows = []
    for origin in dates:
        runs = [done[k] for k in done if k[:3] == (pname, exp, str(origin.date()))]
        if not runs:
            continue
        lo, med, hi = np.mean([validate(r["parsed"]) for r in runs], axis=0) / runs[0]["scale"]
        for h in range(H):
            rows.append({"model": f"llm_{pname}_{exp}", "origin_date": origin.date(), "horizon": h + 1,
                         "target_date": (origin + pd.Timedelta(weeks=h + 1)).date(),
                         "forecast": med[h], "lower80": lo[h], "upper80": hi[h], "n_runs": len(runs)})
    if rows:
        pd.DataFrame(rows).drop(columns="n_runs").to_csv(out_dir / f"llm_{pname}_{exp}.csv", index=False)
        print(f"saved {out_dir / f'llm_{pname}_{exp}.csv'} ({len(rows) // H} forecast dates)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=["claude", "openai"], choices=["claude", "openai"])
    ap.add_argument("--dry-run", action="store_true", help="use a fake model; writes to results/llm_dryrun/")
    a = ap.parse_args()
    run(["mock"] if a.dry_run else a.providers, a.dry_run)
