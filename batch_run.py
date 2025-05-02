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
import random
import pandas as pd

import local_conv
from local_conv import Simulator

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
    N = 1

    # Parameter sweeps
    methods             = ["method1", "method2"]
    subgraph_nodes_list = [[2, 3, 4]]
    alphas              = [0.15, 0.5, 0.9]       # only for method2
    byzantine_ids_list  = [[2]]
    attack_probs        = [0.01, 0.1, 0.5]
    multipliers         = [(0.5, 1.1), (0.5, 1.5)]
    tampering_counts    = [1, 25, float('inf')]

    results = []

    for method in methods:
        for byz_ids in byzantine_ids_list:
            for prob in attack_probs:
                for lower, upper in multipliers:
                    for tamplimit in tampering_counts:
                        for nodes in subgraph_nodes_list:

                            def build_common_config(**extra):
                                base = {
                                    "iter_update_method":           method,
                                    "byzantine_ids":                byz_ids,
                                    "byzantine_attack_probability": prob,
                                    "byzantine_multiplier_lower":   lower,
                                    "byzantine_multiplier_upper":   upper,
                                    "tampering_count":              tamplimit,
                                    "subgraph_nodes":               nodes,
                                    "scale_factor":                 15.0,
                                    "mad_threshold":                4.1,
                                    "non_interactive":              True,
                                    "maximum_iteration":            1000,
                                    "penaltyfactor":                0.01,
                                    "residual_primal":              4e-3,
                                    "residual_dual":                4e-3
                                }
                                base.update(extra)
                                return base

                            def tags(alpha_tag=""):
                                return (
                                    f"{method}{alpha_tag}"
                                    f"_prob{prob}_mult{upper}_t{tamplimit}"
                                )

                            if method == "method2":
                                for alpha in alphas:
                                    for run in range(N):
                                        tag = tags(f"_alpha{alpha}")
                                        mit_log = os.path.join("logs", "mitigation", f"log_{tag}.txt")
                                        lc_log  = os.path.join("logs", "local_conv", f"local_conv_{tag}.log")
                                        cfg = build_common_config(
                                            alpha=alpha,
                                            local_conv_log_file=lc_log,
                                            log_mitigation_file=mit_log,
                                            iter_log_file=os.path.join("logs", "iter_stats", f"iter_{tag}.csv"),
                                            binary_state_file=os.path.join("logs", "binaries", f"state_{tag}.pkl.gz")
                                        )
                                        print(f"Running simulation {tag} (run {run})")
                                        res = run_simulation(cfg)
                                        res.update({
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
                                    mit_log = os.path.join("logs", "mitigation", f"log_{tag}.txt")
                                    lc_log  = os.path.join("logs", "local_conv", f"local_conv_{tag}.log")
                                    cfg = build_common_config(
                                        local_conv_log_file=lc_log,
                                        log_mitigation_file=mit_log,
                                        iter_log_file=os.path.join("logs", "iter_stats", f"iter_{tag}.csv"),
                                        binary_state_file=os.path.join("logs", "binaries", f"state_{tag}.pkl.gz")
                                    )
                                    print(f"Running simulation {tag} (run {run})")
                                    res = run_simulation(cfg)
                                    res.update({
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
    pivot_cols = ["method", "alpha", "attack_prob", "tampering_count"]
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
