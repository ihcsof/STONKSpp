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

############################################
def main():
    # Number of executions per configuration
    N = 10

    random.seed(42)
    np.random.seed(42)

    # Parameter sweeps
    methods             = ["method1", "method2"]
    subgraph_nodes_list = [[0, 2, 3]]
    alphas              = [0.15, 0.5]  # only for method2
    byzantine_ids_list  = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
    attack_probs        = [0.01, 0.1, 0.5]
    multipliers         = [(0.5, 1.5)]  # (0.5, 1.1)
    tampering_counts    = [1, 25, float('inf')]
    mad_options         = {"yes": 4.1,
                            "no":  1e12}
    graph_files = {
        "short_p2p":  "graphs/examples/P2P_model.pyp2p",
    }

    results = []

    # ------------------------------------------------------------------
    # DEFAULT (baseline) RUNS
    # ------------------------------------------------------------------
    for g_label, g_path in graph_files.items():

        default_tag = f"default_{g_label}"

        default_cfg = {
            "graph_file":        g_path,
            "iter_update_method": "method1",
            "byzantine_ids":      [],
            "mad_threshold":      1e12,
            "non_interactive":    True,
            "subgraph_nodes":     subgraph_nodes_list[0],

            # convergence / solver settings
            "maximum_iteration":  1000,
            "penaltyfactor":      0.01,
            "residual_primal":    1e-2,
            "residual_dual":      1e-2,

            # log / dump filenames
            "log_mitigation_file": f"logs/mitigation/log_{default_tag}.txt",
            "local_conv_log_file": f"logs/local_conv/local_conv_{default_tag}.log",
            "iter_log_file":       f"logs/iter_stats/iter_{default_tag}.csv",
            "binary_state_file":   f"logs/binaries/state_{default_tag}.pkl.gz",
        }

        print(f"Running default simulation {default_tag}")
        res = run_simulation(default_cfg)

        # annotate result
        res.update({
            "graph":             g_label,
            "MAD":               "no",
            "method":            "method1",
            "alpha":             0,
            "byzantine_ids":     "[]",
            "attack_prob":       0,
            "multiplier_upper":  0,
            "tampering_count":   0,
            "nodes":             "[]",
            "run":               0,
        })
        results.append(res)
    # ------------------------------------------------------------------

    for g_label, g_path in graph_files.items():
        for method in methods:
            for byz_ids in byzantine_ids_list:
                for prob in attack_probs:
                    for lower, upper in multipliers:
                        for tamplimit in tampering_counts:
                            for mad_label, mad_thr in mad_options.items(): 
                                for nodes in subgraph_nodes_list:

                                    def build_common_config(**extra):
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

                                    def tags(alpha_tag=""):  
                                        mad_tag = "_MAD" if mad_label == "yes" else ""
                                        return (
                                            f"{g_label}{mad_tag}_{method}{alpha_tag}"
                                            f"_prob{prob}_mult{upper}_t{tamplimit}"
                                        )

                                    if method == "method2":
                                        for alpha in alphas:
                                            for run in range(N):
                                                tag = tags(f"_alpha{alpha}")
                                                # include run index in filenames
                                                mit_log = os.path.join(
                                                    "logs", "mitigation",
                                                    f"log_{tag}_run{run}.txt"
                                                )
                                                lc_log  = os.path.join(
                                                    "logs", "local_conv",
                                                    f"local_conv_{tag}_run{run}.log"
                                                )
                                                iter_csv = os.path.join(
                                                    "logs", "iter_stats",
                                                    f"iter_{tag}_run{run}.csv"
                                                )
                                                bin_file = os.path.join(
                                                    "logs", "binaries",
                                                    f"state_{tag}_run{run}.pkl.gz"
                                                )
                                                cfg = build_common_config(
                                                    alpha=alpha,
                                                    local_conv_log_file=lc_log,
                                                    log_mitigation_file=mit_log,
                                                    iter_log_file=iter_csv,
                                                    binary_state_file=bin_file
                                                )
                                                print(f"Running simulation {tag} (run {run})")
                                                res = run_simulation(cfg)
                                                res.update({
                                                    "graph":             g_label,
                                                    "MAD":               mad_label,
                                                    "method":            method,
                                                    "alpha":             alpha,
                                                    "byzantine_ids":     str(byz_ids),
                                                    "attack_prob":       prob,
                                                    "multiplier_upper":  upper,
                                                    "tampering_count":   tamplimit,
                                                    "nodes":             str(nodes),
                                                    "run":               run
                                                })
                                                results.append(res)
                                    else:  # method1
                                        for run in range(N):
                                            tag = tags()
                                            # include run index in filenames
                                            mit_log = os.path.join(
                                                "logs", "mitigation",
                                                f"log_{tag}_run{run}.txt"
                                            )
                                            lc_log  = os.path.join(
                                                "logs", "local_conv",
                                                f"local_conv_{tag}_run{run}.log"
                                            )
                                            iter_csv = os.path.join(
                                                "logs", "iter_stats",
                                                f"iter_{tag}_run{run}.csv"
                                            )
                                            bin_file = os.path.join(
                                                "logs", "binaries",
                                                f"state_{tag}_run{run}.pkl.gz"
                                            )
                                            cfg = build_common_config(
                                                local_conv_log_file=lc_log,
                                                log_mitigation_file=mit_log,
                                                iter_log_file=iter_csv,
                                                binary_state_file=bin_file
                                            )
                                            print(f"Running simulation {tag} (run {run})")
                                            res = run_simulation(cfg)
                                            res.update({
                                                "graph":             g_label,
                                                "MAD":               mad_label,
                                                "method":            method,
                                                "alpha":             0,
                                                "byzantine_ids":     str(byz_ids),
                                                "attack_prob":       prob,
                                                "multiplier_upper":  upper,
                                                "tampering_count":   tamplimit,
                                                "nodes":             str(nodes),
                                                "run":               run
                                            })
                                            results.append(res)

    # Save raw results
    df = pd.DataFrame(results)
    df.to_csv("simulation_results.csv", index=False)
    print("Results saved to simulation_results.csv")
    print(df)

    # ----- Summary Table -----
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
    print(summary_table)

if __name__ == "__main__":
    main()
