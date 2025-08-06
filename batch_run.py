#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
batch_run.py

This script:
  - Sweeps over multiple parameter configurations (method, tampering_count, attack_prob, etc.).
  - Runs each configuration N times, saving raw results to "simulation_results.csv."
  - Writes mitigation logs into logs/mitigation and local-convergence logs into logs/local_conv.
  - Saves a pivot summary table to "simulation_summary_table.csv."
  - (All plotting code has been removed in this stripped-down version.)
"""

import os
import re
import numpy as np
import pandas as pd

import local_conv
from local_conv import Simulator

import random
import numpy as np

random.seed(42)
np.random.seed(42)

# Ensure log directories exist once, up front
os.makedirs("logs/mitigation", exist_ok=True)
os.makedirs("logs/local_conv",  exist_ok=True)
os.makedirs("logs/iter_stats",  exist_ok=True)
os.makedirs("logs/binaries",    exist_ok=True)

############################################
# Helper Functions
############################################

def parse_log_file(file_name):
    """
    Parse the mitigation log file.
    """
    count = 0
    total_weight = 0.0
    total_deviation = 0.0
    if not os.path.exists(file_name):
        return {"mitigation_count": 0, "avg_weight": 0, "avg_deviation": 0}
    with open(file_name, 'r') as f:
        for line in f:
            count += 1
            try:
                parts = line.split("deviation=")
                if len(parts) > 1:
                    total_deviation += float(parts[1].split(",")[0])
                parts = line.split("weight=")
                if len(parts) > 1:
                    total_weight += float(parts[1].split(",")[0])
            except Exception:
                continue
    avg_weight    = total_weight   / count if count > 0 else 0
    avg_deviation = total_deviation / count if count > 0 else 0
    return {"mitigation_count": count,
            "avg_weight":     avg_weight,
            "avg_deviation":  avg_deviation}


def run_simulation(config):
    """
    Instantiate and run the Simulator, clear both log files, then parse mitigation logs.
    """
    # Update wrapper defaults if needed
    if "subgraph_nodes" in config:
        local_conv.DEFAULT_SUBGRAPH_NODES = config["subgraph_nodes"]

    # Prepare/clear log files
    for key in ("local_conv_log_file", "log_mitigation_file"):  
        if key in config:
            path = config[key]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, 'w').close()

    # Instantiate simulator
    sim = Simulator(config=config)
    for key, value in config.items():
        if hasattr(sim, key):
            setattr(sim, key, value)

    sim.run()

    # Gather results
    result = {"iterations": sim.iteration}
    result.update(parse_log_file(config["log_mitigation_file"]))

    # Binary snapshot
    if "binary_state_file" in config:
        sim.SaveBinaryState(config["binary_state_file"])

    return result

def main():
    # Number of executions per configuration
    N = 10

    methods = ["method1", "method2"]
    #subgraph_nodes_list = [[list(range(i, i+3)) for i in range(0, 51, 3)]]
    subgraph_nodes_list = [
        [
            [0, 5, 8, 39, 44, 49],                              # Community 0
            [1, 17, 20, 35, 48],                                # Community 1
            [2, 3, 7, 10, 18, 19, 25, 32, 34, 40, 43, 46, 47],  # Community 2
            [4, 9, 12, 21, 22, 28, 30, 31, 36, 37, 42, 50],     # Community 3
            [6, 15, 16, 26, 29, 41],                            # Community 4
            [11, 13, 14, 23, 24, 27, 33, 38, 45]                # Community 5
        ]
    ]
    alphas              = [0.15, 0.5]  # only for method2
    byzantine_ids_list  = [[0,5,1,17,2,3,4,9,6,15,11,13]]
    attack_probs        = [0.01, 0.1, 0.5]
    multipliers         = [(0.5, 1.5)]  # (0.5, 1.1)
    tampering_counts    = [1, 25, float('inf')]
    mad_options         = {"yes": 4.1,
                            "no":  1e12}
    graph_files = {
        "short_p2p":  "graphs/examples/P2P_model_pruned.pyp2p",
    }

    # --------------------------------------------------
    # Resume logic: scan logs/mitigation for completed runs
    # --------------------------------------------------
    resume_dir = "logs/mitigation"
    completed = set()
    nodes_str = str(subgraph_nodes_list[0])

    if os.path.isdir(resume_dir):
        for fname in os.listdir(resume_dir):
            if not fname.startswith("log_") or not fname.endswith(".txt"):
                continue
            tag_part = fname[len("log_"):-4]  # strip 'log_' and '.txt'

            # Default (baseline) run
            if tag_part.startswith("default_"):
                g_label = tag_part[len("default_"):]
                completed.add((g_label, "no", "method1", 0, 0, 0, 0, "[]", 0))
                continue

            # Other runs have '_run' suffix
            if "_run" not in tag_part:
                continue
            tag, run_str = tag_part.rsplit("_run", 1)
            try:
                run_idx = int(run_str)
            except ValueError:
                continue

            # Extract fields via regex
            pattern = (
                r"(?P<g_label>[^_]+_[^_]+)(?P<mad>_MAD)?_"
                r"(?P<method>method1|method2)"
                r"(?:_alpha(?P<alpha>[\d\.]+))?_"
                r"prob(?P<prob>[\d\.]+)_mult(?P<mult>[\d\.]+)_t(?P<t>inf|\d+)"
            )
            m = re.match(pattern, tag)
            if not m:
                continue

            g_label    = m.group("g_label")
            mad_label  = "yes" if m.group("mad") else "no"
            method     = m.group("method")
            alpha      = float(m.group("alpha")) if m.group("alpha") else 0
            prob       = float(m.group("prob"))
            mult_upper = float(m.group("mult"))
            t_str      = m.group("t")
            tamplimit  = float('inf') if t_str == 'inf' else int(t_str)

            completed.add((
                g_label, mad_label, method, alpha,
                prob, mult_upper, tamplimit, nodes_str, run_idx
            ))

    print(f"Found {len(completed)} completed runs in {resume_dir}")

    # --------------------------------------------------
    # Execute sweeps, skipping completed
    # --------------------------------------------------
    new_results = []

    # Baseline defaults
    for g_label, g_path in graph_files.items():
        meta = (g_label, 'no', 'method1', 0, 0, 0, 0, '[]', 0)
        if meta in completed:
            print(f"Skipping default_{g_label}, already done")
        else:
            tag = f"default_{g_label}"
            cfg = {
                "graph_file":           g_path,
                "iter_update_method":   "method1",
                "byzantine_ids":        [],
                "mad_threshold":        1e12,
                "non_interactive":      True,
                "subgraph_nodes":       subgraph_nodes_list[0],
                "maximum_iteration":    1000,
                "penaltyfactor":        0.01,
                "residual_primal":      1e-2,
                "residual_dual":        1e-2,
                "log_mitigation_file":  f"logs/mitigation/log_{tag}.txt",
                "local_conv_log_file":  f"logs/local_conv/local_conv_{tag}.log",
                "iter_log_file":        f"logs/iter_stats/iter_{tag}.csv",
                "binary_state_file":    f"logs/binaries/state_{tag}.pkl.gz",
            }
            print(f"Running default simulation {tag}")
            res = run_simulation(cfg)
            res.update({
                "graph":           g_label,
                "MAD":             "no",
                "method":          "method1",
                "alpha":           0,
                "attack_prob":     0,
                "multiplier_upper":0,
                "tampering_count": 0,
                "nodes":           "[]",
                "run":             0
            })
            new_results.append(res)

    # Full parameter sweeps
    for g_label, g_path in graph_files.items():
        for method in methods:
            for byz_ids in byzantine_ids_list:
                for prob in attack_probs:
                    for lower, upper in multipliers:
                        for tamplimit in tampering_counts:
                            for mad_label, mad_thr in mad_options.items():
                                for nodes in subgraph_nodes_list:
                                    def build_config(**extra):
                                        base = {
                                            "graph_file":                   g_path,
                                            "iter_update_method":           method,
                                            "byzantine_ids":                byz_ids,
                                            "byzantine_attack_probability": prob,
                                            "byzantine_multiplier_lower":   lower,
                                            "byzantine_multiplier_upper":   upper,
                                            "tampering_count":              tamplimit,
                                            "subgraph_nodes":               nodes,
                                            "scale_factor":                 15.0,
                                            "mad_threshold":                mad_thr,
                                            "non_interactive":              True,
                                            "maximum_iteration":            1000,
                                            "penaltyfactor":                0.01,
                                            "residual_primal":              1e-2,
                                            "residual_dual":                1e-2
                                        }
                                        base.update(extra)
                                        return base

                                    def make_tag(alpha_tag=""):
                                        mad_tag = "_MAD" if mad_label == "yes" else ""
                                        return (
                                            f"{g_label}{mad_tag}_{method}{alpha_tag}"
                                            f"_prob{prob}_mult{upper}_t{tamplimit}"
                                        )

                                    if method == "method2":
                                        for alpha in alphas:
                                            for run in range(N):
                                                tag = make_tag(f"_alpha{alpha}")
                                                meta = (
                                                    g_label, mad_label, method,
                                                    alpha, prob, upper,
                                                    tamplimit, nodes_str, run
                                                )
                                                if meta in completed:
                                                    print(f"Skipping {tag}_run{run}, done")
                                                    continue
                                                cfg = build_config(
                                                    alpha=alpha,
                                                    local_conv_log_file=f"logs/local_conv/local_conv_{tag}_run{run}.log",
                                                    log_mitigation_file=f"logs/mitigation/log_{tag}_run{run}.txt",
                                                    iter_log_file    =f"logs/iter_stats/iter_{tag}_run{run}.csv",
                                                    binary_state_file=f"logs/binaries/state_{tag}_run{run}.pkl.gz"
                                                )
                                                print(f"Running {tag} (run {run})")
                                                res = run_simulation(cfg)
                                                res.update({
                                                    "graph":           g_label,
                                                    "MAD":             mad_label,
                                                    "method":          method,
                                                    "alpha":           alpha,
                                                    "attack_prob":     prob,
                                                    "multiplier_upper":upper,
                                                    "tampering_count": tamplimit,
                                                    "nodes":           nodes_str,
                                                    "run":             run
                                                })
                                                new_results.append(res)
                                    else:
                                        for run in range(N):
                                            tag = make_tag()
                                            meta = (
                                                g_label, mad_label, method,
                                                0, prob, upper,
                                                tamplimit, nodes_str, run
                                            )
                                            if meta in completed:
                                                print(f"Skipping {tag}_run{run}, done")
                                                continue
                                            cfg = build_config(
                                                local_conv_log_file=f"logs/local_conv/local_conv_{tag}_run{run}.log",
                                                log_mitigation_file=f"logs/mitigation/log_{tag}_run{run}.txt",
                                                iter_log_file    =f"logs/iter_stats/iter_{tag}_run{run}.csv",
                                                binary_state_file=f"logs/binaries/state_{tag}_run{run}.pkl.gz"
                                            )
                                            print(f"Running {tag} (run {run})")
                                            res = run_simulation(cfg)
                                            res.update({
                                                "graph":           g_label,
                                                "MAD":             mad_label,
                                                "method":          method,
                                                "alpha":           0,
                                                "attack_prob":     prob,
                                                "multiplier_upper":upper,
                                                "tampering_count": tamplimit,
                                                "nodes":           nodes_str,
                                                "run":             run
                                            })
                                            new_results.append(res)

    # Merge with existing results and save
    results_file = "simulation_results.csv"
    if os.path.exists(results_file):
        old_df = pd.read_csv(results_file)
        df = pd.concat([old_df, pd.DataFrame(new_results)], ignore_index=True)
    else:
        df = pd.DataFrame(new_results)

    df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Summary table
    pivot_cols = ["graph", "MAD", "method", "alpha", "attack_prob", "tampering_count"]
    summary_table = (
        df.groupby(pivot_cols)
          .agg(iterations_mean=("iterations", "mean"),
               iterations_std=("iterations", "std"),
               mitigations_mean=("mitigation_count", "mean"),
               mitigations_std=("mitigation_count", "std"))
          .reset_index()
    )
    summary_table.to_csv("simulation_summary_table.csv", index=False)
    print("Summary table saved to simulation_summary_table.csv")

if __name__ == "__main__":
    main()
