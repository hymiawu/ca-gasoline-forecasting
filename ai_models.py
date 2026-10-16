"""
ai_models.py - Zero-shot forecasts from time-series foundation models (proposal section 5).

Runs each model at every rolling origin of the test period (same origins as forecasting_models.ipynb),
using the last 512 weeks of California gasoline prices as context, and saves forecasts in the shared
format to forecasts/<model>.csv. Also logs model versions and release dates to results/ai_model_log.csv.

Setup (separate environment, needs PyTorch):
    uv venv .venv-ai --python 3.11
    uv pip install --python .venv-ai/bin/python "chronos-forecasting>=2.2" "timesfm[torch]" pandas
Run:
    .venv-ai/bin/python ai_models.py                 # all models
    .venv-ai/bin/python ai_models.py chronos_bolt    # one model
    .venv-ai/bin/python ai_models.py --validation    # validation origins (2022 - mid 2025), used to choose models
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "weekly_data.csv"
FC_DIR = ROOT / "forecasts"
RES_DIR = ROOT / "results"
TRAIN_END = "2025-06-30"
VAL_START, VAL_STEP = "2022-01-03", 4   # validation origins: every 4th week, targets end by TRAIN_END
H = 12
CONTEXT = 512
QUANTILES = [0.1, 0.5, 0.9]

# Release date (Hugging Face repo creation date) = upper bound on training data. A release inside the test period
# (July 2025 onward) means the model could in principle have seen part of the test data.
MODELS = {
    "chronos_bolt": {"repo": "amazon/chronos-bolt-small", "released": "2024-11-25"},
    "chronos2": {"repo": "amazon/chronos-2", "released": "2025-10-30"},
    "timesfm": {"repo": "google/timesfm-2.5-200m-pytorch", "released": "2025-09-02"},
}


def load_contexts(validation=False):
    df = pd.read_csv(DATA, index_col="date", parse_dates=True).asfreq("W-MON")
    y = df["ca_gas"]
    if validation:
        idx = y.index
        origins = idx[(idx >= VAL_START) & (idx <= pd.Timestamp(TRAIN_END) - pd.Timedelta(weeks=H))][::VAL_STEP]
    else:
        origins = y.loc[TRAIN_END:].index[:-1]
    contexts = [y.loc[:o].values[-CONTEXT:].astype("float32") for o in origins]
    return origins, contexts


def run_chronos(repo, contexts):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(repo, device_map="cpu", torch_dtype=torch.float32)
    if "chronos-2" in repo:
        inputs = [c.reshape(1, -1) for c in contexts]          # (n_variates=1, history)
        q, _ = pipe.predict_quantiles(inputs, prediction_length=H, quantile_levels=QUANTILES)
        return np.stack([t[0].numpy() for t in q])              # (n, H, 3)
    q, _ = pipe.predict_quantiles([torch.tensor(c) for c in contexts], prediction_length=H,
                                  quantile_levels=QUANTILES)
    return q.numpy()                                            # (n, H, 3)


def run_timesfm(repo, contexts):
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo)
    model.compile(timesfm.ForecastConfig(max_context=CONTEXT, max_horizon=H, normalize_inputs=True,
                                         use_continuous_quantile_head=True, fix_quantile_crossing=True,
                                         per_core_batch_size=16))
    _, q = model.forecast(horizon=H, inputs=[c.astype("float64") for c in contexts])
    # q[..., 0] is the mean, then the 10%, 20%, ..., 90% quantiles
    return np.stack([q[:, :H, 1], q[:, :H, 5], q[:, :H, 9]], axis=-1)


def model_revision(repo):
    try:
        from huggingface_hub import model_info
        return model_info(repo).sha[:10]
    except Exception:
        return ""


def main(names, validation=False):
    origins, contexts = load_contexts(validation)
    print(f"{len(origins)} {'validation' if validation else 'test'} origins: {origins[0].date()} -> "
          f"{origins[-1].date()}, context = {CONTEXT} weeks")
    out_dir = ROOT / "forecasts_validation" if validation else FC_DIR
    out_dir.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)
    log_path = RES_DIR / "ai_model_log.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()

    for name in names:
        spec = MODELS[name]
        t0 = time.time()
        q = run_timesfm(spec["repo"], contexts) if name == "timesfm" else run_chronos(spec["repo"], contexts)
        rows = [{"model": name, "origin_date": o.date(), "horizon": h + 1,
                 "target_date": (o + pd.Timedelta(weeks=h + 1)).date(),
                 "forecast": q[i, h, 1], "lower80": q[i, h, 0], "upper80": q[i, h, 2]}
                for i, o in enumerate(origins) for h in range(H)]
        out = pd.DataFrame(rows)
        out.to_csv(out_dir / f"{name}.csv", index=False)
        secs = time.time() - t0
        print(f"{name}: saved {len(out)} rows in {secs:.0f}s")
        if validation:
            continue   # the model log describes the test-period run

        entry = {"model": name, "repo": spec["repo"], "revision": model_revision(spec["repo"]),
                 "released": spec["released"],
                 "release_inside_test_period": pd.Timestamp(spec["released"]) > pd.Timestamp(TRAIN_END),
                 "context_weeks": CONTEXT, "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                 "runtime_s": round(secs)}
        log = pd.concat([log[log.get("model", pd.Series(dtype=str)) != name] if len(log) else log,
                         pd.DataFrame([entry])], ignore_index=True)
        log.to_csv(log_path, index=False)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args or list(MODELS), validation="--validation" in sys.argv)
