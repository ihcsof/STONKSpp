#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch‑runner for the **Trust‑based** robustness mechanism only.

* Sweeps over a grid of configuration parameters (method, attack_prob, …).
* Executes each configuration **N** times, storing per‑run statistics in
  `simulation_results_trust.csv` and an aggregated pivot table in
  `simulation_summary_trust.csv`.
* Log files are written under `logs/`.

Assumes you have **trust_conv.py** (or a package) exposing
`trust_conv.Simulator` that understands the following config keys:
    - trust_threshold  : float         (robustness cut‑off)
    - iter_update_method, byzantine_*, …   (same as MAD version)
    - log_mitigation_file, local_conv_log_file, iter_log_file

Usage
-----
$ python batch_run_trust.py           # runs with default grid below
$ python batch_run_trust.py 5         # runs 5 repetitions instead of N=10

Feel free to trim the parameter lists (attack_probs, multipliers, …) to keep
experiments short.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import importlib
import pandas as pd

# ----------------------------- 0. CLI ---------------------------------------
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10  # repetitions per cfg

# ----------------------------- 1. import simulator --------------------------
import local_conv
from local_conv import Simulator

# ----------------------------- 2. paths -------------------------------------
for folder in ("logs/mitigation", "logs/local_conv", "logs/iter_stats", "logs/binaries"):
    Path(folder).mkdir(parents=True, exist_ok=True)

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ----------------------------- 3. parameter grid ----------------------------
methods             = ["method1", "method2"]
subgraph_nodes_list = [[0, 2, 3]]
alphas              = [0.15, 0.5]          # Only for method2
byzantine_ids_list  = [[2]]
attack_probs        = [0.01, 0.1, 0.5]
multipliers         = [(0.5, 1.5)]         # (lower, upper)
tampering_counts    = [1, 25, float("inf")]
trust_options       = {"yes": 30}  # threshold values

graph_files = {
    "short_p2p": "graphs/examples/P2P_model_reduced.pyp2p",
}

# ----------------------------- 4. helper: log parsing -----------------------

def parse_trust_log(path):
    if not os.path.exists(path):
        return {"trust_violations": 0, "final_trust": 0}

    violations = 0
    last_score = 0
    with open(path) as fh:
        for ln in fh:
            if "[Flag]" in ln:          # <-- counts flag events
                violations += 1
            if "score=" in ln and "flags partner" in ln:
                # optional: capture most recent score
                try:
                    last_score = float(ln.split("score=")[1].split()[0])
                except Exception:
                    pass
    return {"trust_violations": violations, "final_trust": last_score}

# ----------------------------- 5. run one simulation ------------------------

def run_sim(cfg: dict):
    """Instantiate Simulator with *cfg*, run, collect stats."""
    # fresh log files
    for key in ("local_conv_log_file", "log_mitigation_file"):
        if key in cfg:
            Path(cfg[key]).parent.mkdir(parents=True, exist_ok=True)
            open(cfg[key], "w").close()

    sim = Simulator(config=cfg)

    #if hasattr(sim, "log_mitigation_file"):
    #sim.log_mitigation_file = cfg["log_mitigation_file"]

    # allow quick overrides via cfg
    for k, v in cfg.items():
        if hasattr(sim, k):
            setattr(sim, k, v)

    sim.run()

    row = {"iterations": sim.iteration}
    row.update(parse_trust_log(cfg["log_mitigation_file"]))

    if "binary_state_file" in cfg:
        sim.SaveBinaryState(cfg["binary_state_file"])
    return row

# ----------------------------- 6. sweeps ------------------------------------
all_rows = []

# Baseline (no adversary, robustness off) ------------------------------------
for g_label, g_path in graph_files.items():
    tag = f"trust_default_{g_label}"
    cfg = dict(
        graph_file            = g_path,
        iter_update_method    = "method1",
        trust_threshold       = 1.0,  # effectively disables mitigation
        byzantine_ids         = [[2]],
        non_interactive       = True,
        subgraph_nodes        = subgraph_nodes_list[0],
        maximum_iteration     = 1000,
        penaltyfactor         = 0.01,
        residual_primal       = 1e-3,
        residual_dual         = 1e-3,
        # log files
        log_mitigation_file   = f"logs/mitigation/log_{tag}.txt",
        local_conv_log_file   = f"logs/local_conv/local_conv_{tag}.log",
        iter_log_file         = f"logs/iter_stats/iter_{tag}.csv",
        binary_state_file     = f"logs/binaries/state_{tag}.pkl.gz",
    )
    print(f"[TRUST] Running baseline {tag}")
    res = run_sim(cfg)
    res.update(dict(
        graph           = g_label,
        method          = "method1",
        alpha           = 0,
        attack_prob     = 0,
        tampering_count = 0,
        opt_label       = "no",
        run             = 0,
    ))
    all_rows.append(res)

# Full grid --------------------------------------------------------------
for g_label, g_path in graph_files.items():
    for method in methods:
        for byz_ids in byzantine_ids_list:
            for prob in attack_probs:
                for lo, hi in multipliers:
                    for tamp_lim in tampering_counts:
                        for opt_label, thr in trust_options.items():
                            for nodes in subgraph_nodes_list:

                                def make_cfg(**extra):
                                    base = dict(
                                        graph_file            = g_path,
                                        iter_update_method    = method,
                                        trust_threshold       = thr,
                                        byzantine_ids         = byz_ids,
                                        byzantine_attack_probability = prob,
                                        byzantine_multiplier_lower   = lo,
                                        byzantine_multiplier_upper   = hi,
                                        tampering_count       = tamp_lim,
                                        subgraph_nodes        = nodes,
                                        non_interactive       = True,
                                        maximum_iteration     = 1000,
                                        penaltyfactor         = 0.01,
                                        residual_primal       = 1e-3,
                                        residual_dual         = 1e-3,
                                    )
                                    base.update(extra)
                                    return base

                                def make_tag(alpha_txt=""):
                                    trust_tag = "_TRUST" if opt_label == "yes" else ""
                                    return (f"trust_{g_label}{trust_tag}_{method}{alpha_txt}"
                                            f"_prob{prob}_mult{hi}_t{tamp_lim}")

                                runs = range(N_REPS)
                                if method == "method2":
                                    for alpha in alphas:
                                        for run in runs:
                                            tag = make_tag(f"_alpha{alpha}")
                                            cfg = make_cfg(alpha=alpha)
                                            cfg.update(
                                                log_mitigation_file=f"logs/mitigation/log_{tag}_run{run}.txt",
                                                local_conv_log_file=f"logs/local_conv/local_conv_{tag}_run{run}.log",
                                                iter_log_file      =f"logs/iter_stats/iter_{tag}_run{run}.csv",
                                                binary_state_file  =f"logs/binaries/state_{tag}_run{run}.pkl.gz",
                                            )
                                            print(f"[TRUST] {tag} (run {run})")
                                            row = run_sim(cfg)
                                            row.update(dict(
                                                graph           = g_label,
                                                method          = method,
                                                alpha           = alpha,
                                                attack_prob     = prob,
                                                tampering_count = tamp_lim,
                                                opt_label       = opt_label,
                                                run             = run,
                                            ))
                                            all_rows.append(row)
                                else:  # method1
                                    for run in runs:
                                        tag = make_tag()
                                        cfg = make_cfg()
                                        cfg.update(
                                            log_mitigation_file=f"logs/mitigation/log_{tag}_run{run}.txt",
                                            local_conv_log_file=f"logs/local_conv/local_conv_{tag}_run{run}.log",
                                            iter_log_file      =f"logs/iter_stats/iter_{tag}_run{run}.csv",
                                            binary_state_file  =f"logs/binaries/state_{tag}_run{run}.pkl.gz",
                                        )
                                        print(f"[TRUST] {tag} (run {run})")
                                        row = run_sim(cfg)
                                        row.update(dict(
                                            graph           = g_label,
                                            method          = method,
                                            alpha           = 0,
                                            attack_prob     = prob,
                                            tampering_count = tamp_lim,
                                            opt_label       = opt_label,
                                            run             = run,
                                        ))
                                        all_rows.append(row)

# ----------------------------- 7. save output ---------------------------

df = pd.DataFrame(all_rows)
results_file = f"simulation_results_trust_{STAMP}.csv"
df.to_csv(results_file, index=False)
print(f"\nSaved raw results -> {results_file}")
print(df.head())

pivot_cols = ["graph", "method", "alpha", "opt_label", "attack_prob", "tampering_count"]
summary = (df.groupby(pivot_cols)
             .agg(iter_mean=("iterations", "mean"),
                  iter_std =("iterations", "std"),
                  trust_viol_mean=("trust_violations", "mean"))
             .reset_index())
summary_file = f"simulation_summary_trust_{STAMP}.csv"
summary.to_csv(summary_file, index=False)
print(f"Summary table -> {summary_file}\n")
print(summary.head())
