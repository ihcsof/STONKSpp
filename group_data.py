#!/usr/bin/env python3
"""
build_tables.py – aggregate simulation summaries AND per-snapshot logs
                   for the LaTeX tables.

✓  Reads .csv, .pkl, .pkl.gz transparently.
✓  --mad / --trust flags as before.
✓  --logdir PATH  to point at the folder that holds snapshot files
                  (default: ./eq_logs).
"""

import argparse, sys, io, pickle, gzip
from pathlib import Path
import pandas as pd

HERE   = Path(__file__).resolve().parent
OUT_REMAP = {"Converged": "Converged",
             "Pseudo-converged": "Pseudo-converged",
             "Not-converged": "Not-conv.",
             "Diverging": "Diverging"}

PATTERNS = {"equilibrate_trades", "gd_balancer"}      # what to extract


# ──────────────────── helpers ──────────────────────
def load_csv(name):
    fp = HERE / name
    if fp.exists() and fp.stat().st_size:
        return pd.read_csv(fp)
    sys.stderr.write(f"[warn] {name} missing/empty – skipped\n")
    return None


def aggregate_grid(df, agg_map, extra_cols):
    key = ["method", "alpha", "attack_prob", "tampering"]
    g   = df.groupby(key, dropna=False).agg(agg_map)
    g.columns = ['_'.join(c) for c in g.columns]
    g.reset_index(inplace=True)
    return g[key + extra_cols]


def aggregate_outcomes(df, class_col):
    key = ["method", "alpha", "attack_prob", "tampering"]
    df2 = df.assign(outcome=df[class_col].map(OUT_REMAP).fillna("Other"))
    return (df2.groupby(key + ["outcome"], dropna=False)
               .size()
               .unstack(fill_value=0)
               .reset_index())


# ─────────────── load pickles / csv transparently ────────────────
def _read_any(fp: Path):
    try:
        if fp.suffix == ".csv":
            return pd.read_csv(fp)

        # *.pkl.gz or *.pkl
        if fp.suffix == ".gz" and fp.with_suffix("").suffix == ".pkl":
            with gzip.open(fp, "rb") as fh:
                return pickle.load(fh)
        if fp.suffix == ".pkl":
            with open(fp, "rb") as fh:
                return pickle.load(fh)
    except Exception as e:
        sys.stderr.write(f"[warn] failed reading {fp}: {e}\n")
    return None


def collect_eq_logs(logdir: Path, pattern: str):
    frames = []
    for fp in logdir.rglob("*"):
        if pattern in fp.name and fp.suffix in {".csv", ".pkl", ".gz"}:
            obj = _read_any(fp)
            if obj is None:
                continue
            if isinstance(obj, pd.DataFrame):
                frames.append(obj)
            elif isinstance(obj, dict) and pattern in obj:
                frames.append(obj[pattern])
    return pd.concat(frames, ignore_index=True) if frames else None


def summarise_snapshot(df, outfile):
    if df is None or df.empty:
        sys.stderr.write(f"[warn] no data for {outfile}\n")
        return
    pct = (df["slope_label"]
           .value_counts(normalize=True)
           .mul(100)
           .round(1)
           .reindex(["Converged", "Not-converged", "Diverging"], fill_value=0))
    pct.to_csv(outfile, header=["percent"])
    print(f"✔  wrote {outfile}")


# ───────────────────── MAD tables ─────────────────────
def build_mad_tables(logdir: Path):
    sim  = load_csv("simulation_results.csv")
    biny = load_csv("binary_summary.csv")
    if sim is None or biny is None:
        return

    sim_mad = sim[sim["tag"].str.contains("_MAD_", regex=False)]
    metrics = aggregate_grid(
        sim_mad,
        agg_map={"iterations":       ["mean", "std"],
                 "mitigation_count": ["mean", "std"]},
        extra_cols=["iterations_mean", "iterations_std",
                    "mitigation_count_mean", "mitigation_count_std"]
    ).rename(columns={"iterations_mean":"iters_mean",
                      "iterations_std":"iters_std",
                      "mitigation_count_mean":"corr_mean",
                      "mitigation_count_std":"corr_std"})
    outcome = aggregate_outcomes(biny, "conv_class")

    metrics.to_csv("grid_metrics.csv", index=False)
    outcome.to_csv("grid_outcome.csv", index=False)
    print("✔  wrote grid_metrics.csv  &  grid_outcome.csv")

    # snapshot summaries
    for pat, out in [("equilibrate_trades", "equilibrate_table.csv"),
                     ("gd_balancer",       "gd_balancer_table.csv")]:
        summarise_snapshot(
            collect_eq_logs(logdir, pat), out)


# ─────────────────── TRUST tables (no pickle logs) ──────────────────
def build_trust_tables():
    simt = (load_csv("simulation_results_trust.csv")
            or load_csv("simulation_results_trust_20250703_144417.csv"))
    biny = load_csv("binary_summary.csv")
    if simt is None or biny is None:
        return

    metrics = aggregate_grid(
        simt,
        agg_map={"iterations":       ["mean", "std"],
                 "trust_violations": ["mean", "std"]},
        extra_cols=["iterations_mean", "iterations_std",
                    "trust_violations_mean", "trust_violations_std"]
    ).rename(columns={"iterations_mean":"iters_mean",
                      "iterations_std":"iters_std",
                      "trust_violations_mean":"trust_violations_mean",
                      "trust_violations_std":"trust_violations_std"})
    outcome = aggregate_outcomes(biny, "conv_class")

    metrics.to_csv("trust_grid_metrics.csv",  index=False)
    outcome.to_csv("trust_grid_outcome.csv", index=False)
    print("✔  wrote trust_grid_metrics.csv  &  trust_grid_outcome.csv")


# ─────────────────────── CLI ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mad",      action="store_true", help="MAD tables only")
    ap.add_argument("--trust",    action="store_true", help="TRUST tables only")
    ap.add_argument("--logdir",   default="eq_logs",
                    help="folder with per-snapshot logs (default: ./eq_logs)")
    args = ap.parse_args()

    logdir = Path(args.logdir).resolve()
    if not logdir.exists():
        sys.exit(f"[error] log directory {logdir} not found")

    run_mad   = args.mad   or (not args.mad and not args.trust)
    run_trust = args.trust or (not args.mad and not args.trust)

    if run_mad:
        build_mad_tables(logdir)
    if run_trust:
        build_trust_tables()


if __name__ == "__main__":
    main()

