#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
batch_run.py

This script:
  - Sweeps over multiple parameter configurations (method, tampering_count, attack_prob, etc.).
  - Runs each configuration N times, saving raw results to "simulation_results.csv."
  - Writes mitigation logs into logs/mitigation and local-convergence logs into logs/local_conv.
  - Produces bar charts and grouped-bar charts (iterations vs. tampering_count, etc.).
  - Saves a pivot summary table to "simulation_summary_table.csv."
"""

import os
import time
import random
import pandas as pd
import matplotlib.pyplot as plt

import local_conv
from local_conv import Simulator

# Ensure our two log directories exist once, up front
os.makedirs("logs/mitigation", exist_ok=True)
os.makedirs("logs/local_conv",  exist_ok=True)

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
    Instantiate and run the Simulator (wrapped by local_conv),
    clear both log files, then parse mitigation logs.
    """
    # Update wrapper defaults if needed
    if "subgraph_nodes" in config:
        local_conv.DEFAULT_SUBGRAPH_NODES = config["subgraph_nodes"]

    # Clear the local‐convergence log file (ensure its dir exists)
    if "local_conv_log_file" in config:
        d = os.path.dirname(config["local_conv_log_file"])
        os.makedirs(d, exist_ok=True)
        open(config["local_conv_log_file"], 'w').close()

    # Clear the mitigation log file (ensure its dir exists)
    if "log_mitigation_file" in config:
        d = os.path.dirname(config["log_mitigation_file"])
        os.makedirs(d, exist_ok=True)
        open(config["log_mitigation_file"], 'w').close()

    # Instantiate and configure simulator
    sim = Simulator(config=config)
    for key, value in config.items():
        if hasattr(sim, key):
            setattr(sim, key, value)

    sim.run()

    # Gather results
    result = {"iterations": sim.iteration}
    mitigation_data = parse_log_file(config["log_mitigation_file"])
    result.update(mitigation_data)
    return result

def clamp_errorbars_at_zero(means, errs):
    import numpy as np
    lower = means - errs
    upper = means + errs
    lower = np.maximum(lower, 0)
    return [means - lower, upper - means]

def plot_grouped_bar_chart(df, x_col, group_col, value_col, ylabel, title, filename):
    import numpy as np
    df = df.copy()
    try:
        df[x_col] = df[x_col].astype(float)
    except:
        pass
    df[group_col] = df[group_col].astype(str)

    grouped    = df.groupby([x_col, group_col])[value_col].agg(['mean','std']).reset_index()
    grouped    = grouped.sort_values(by=x_col)
    pivot_mean = grouped.pivot(index=x_col, columns=group_col, values='mean').sort_index()
    pivot_std  = grouped.pivot(index=x_col, columns=group_col, values='std').sort_index()

    fig, ax = plt.subplots(figsize=(8,6))
    num_groups = len(pivot_mean.columns)
    x_vals     = np.arange(len(pivot_mean.index))
    bar_width  = 0.8 / num_groups

    for i, col in enumerate(pivot_mean.columns):
        means = pivot_mean[col].values
        errs  = pivot_std[col].values
        yerr  = clamp_errorbars_at_zero(means, errs)
        positions = x_vals + (i - num_groups/2) * bar_width + bar_width/2
        ax.bar(positions, means, yerr=yerr, width=bar_width, label=str(col), capsize=5)

    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(idx) for idx in pivot_mean.index])
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=group_col)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_bar_chart(df, group_col, value_col, ylabel, title, filename):
    import numpy as np
    df = df.copy()
    try:
        df[group_col] = df[group_col].astype(float)
    except:
        pass
    grouped = df.groupby(group_col)[value_col].agg(['mean','std']).reset_index().sort_values(by=group_col)
    means   = grouped['mean'].values
    errs    = grouped['std'].values
    yerr    = clamp_errorbars_at_zero(means, errs)

    plt.figure()
    plt.bar(range(len(means)), means, yerr=yerr, capsize=5)
    plt.xticks(range(len(means)), grouped[group_col].astype(str).values)
    plt.xlabel(group_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

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
                for (lower, upper) in multipliers:
                    for tamplimit in tampering_counts:
                        for nodes in subgraph_nodes_list:
                            if method == "method2":
                                # Relaxed ADMM: vary alpha
                                for alpha in alphas:
                                    for run in range(N):
                                        mit_log = os.path.join(
                                            "logs","mitigation",
                                            f"log_{method}_alpha{alpha}"
                                            f"_ids{'-'.join(map(str,byz_ids))}"
                                            f"_prob{prob}_mult{upper}"
                                            f"_tcount{tamplimit}.txt"
                                        )
                                        lc_log = os.path.join(
                                            "logs","local_conv",
                                            f"local_conv_{method}_alpha{alpha}"
                                            f"_nodes{'-'.join(map(str,nodes))}"
                                            f"_prob{prob}_mult{upper}"
                                            f"_tcount{tamplimit}.log"
                                        )
                                        config = {
                                            "iter_update_method":            method,
                                            "alpha":                         alpha,
                                            "byzantine_ids":                 byz_ids,
                                            "byzantine_attack_probability": prob,
                                            "byzantine_multiplier_lower":   lower,
                                            "byzantine_multiplier_upper":   upper,
                                            "tampering_count":               tamplimit,
                                            "subgraph_nodes":                nodes,
                                            "local_conv_log_file":           lc_log,
                                            "log_mitigation_file":           mit_log,
                                            "scale_factor":                  15.0,
                                            "mad_threshold":                 4.1,
                                            "non_interactive":               True,
                                            "maximum_iteration":             1000,
                                            "penaltyfactor":                 0.01,
                                            "residual_primal":               4e-3,
                                            "residual_dual":                 4e-3
                                        }
                                        print(f"Running: {method}, α={alpha}, tamper={tamplimit}, "
                                              f"prob={prob}, mult={upper}, nodes={nodes}")
                                        result = run_simulation(config)
                                        result.update({
                                            "method":                       method,
                                            "alpha":                        alpha,
                                            "byzantine_ids":                str(byz_ids),
                                            "byzantine_attack_probability": prob,
                                            "byzantine_multiplier_upper":   upper,
                                            "tampering_count":              tamplimit,
                                            "nodes":                        str(nodes),
                                            "run":                          run
                                        })
                                        results.append(result)
                            else:
                                # Classical ADMM: no alpha
                                for run in range(N):
                                    mit_log = os.path.join(
                                        "logs","mitigation",
                                        f"log_{method}_ids{'-'.join(map(str,byz_ids))}"
                                        f"_prob{prob}_mult{upper}"
                                        f"_tcount{tamplimit}.txt"
                                    )
                                    lc_log = os.path.join(
                                        "logs","local_conv",
                                        f"local_conv_{method}"
                                        f"_nodes{'-'.join(map(str,nodes))}"
                                        f"_prob{prob}_mult{upper}"
                                        f"_tcount{tamplimit}.log"
                                    )
                                    config = {
                                        "iter_update_method":            method,
                                        "byzantine_ids":                 byz_ids,
                                        "byzantine_attack_probability": prob,
                                        "byzantine_multiplier_lower":   lower,
                                        "byzantine_multiplier_upper":   upper,
                                        "tampering_count":               tamplimit,
                                        "subgraph_nodes":                nodes,
                                        "local_conv_log_file":           lc_log,
                                        "log_mitigation_file":           mit_log,
                                        "scale_factor":                  15.0,
                                        "mad_threshold":                 4.1,
                                        "non_interactive":               True,
                                        "maximum_iteration":             1000,
                                        "penaltyfactor":                 0.01,
                                        "residual_primal":               4e-3,
                                        "residual_dual":                 4e-3
                                    }
                                    print(f"Running: {method}, tamper={tamplimit}, "
                                          f"prob={prob}, mult={upper}, nodes={nodes}")
                                    result = run_simulation(config)
                                    result.update({
                                        "method":                       method,
                                        "alpha":                        "0",
                                        "byzantine_ids":                str(byz_ids),
                                        "byzantine_attack_probability": prob,
                                        "byzantine_multiplier_upper":   upper,
                                        "tampering_count":              tamplimit,
                                        "nodes":                        str(nodes),
                                        "run":                          run
                                    })
                                    results.append(result)

    # Save and display results
    df = pd.DataFrame(results)
    df.to_csv("simulation_results.csv", index=False)
    print("Results saved to simulation_results.csv")
    print(df)

    # ----- Standard Plots -----
    plot_grouped_bar_chart(
        df=df[df["method"]=="method2"],
        x_col="tampering_count",
        group_col="alpha",
        value_col="iterations",
        ylabel="Average Iterations",
        title="Iterations vs. Tampering Count (Relaxed ADMM, varying alpha)",
        filename="iterations_vs_tamperingcount_relaxed.png"
    )
    plot_grouped_bar_chart(
        df=df[df["method"]=="method2"],
        x_col="tampering_count",
        group_col="alpha",
        value_col="mitigation_count",
        ylabel="Average Mitigation Events",
        title="Mitigation Events vs. Tampering Count (Relaxed ADMM, varying alpha)",
        filename="mitigations_vs_tamperingcount_relaxed.png"
    )
    plot_bar_chart(
        df=df[df["method"]=="method1"],
        group_col="tampering_count",
        value_col="iterations",
        ylabel="Average Iterations",
        title="Iterations vs. Tampering Count (Classical ADMM)",
        filename="iterations_vs_tamperingcount_classical.png"
    )

    # ----- Summary Table -----
    pivot_cols    = ["method", "alpha", "byzantine_attack_probability", "tampering_count"]
    summary_table = df.groupby(pivot_cols).agg({
        "iterations":       ["mean","std"],
        "mitigation_count": ["mean","std"]
    }).reset_index()
    summary_table.columns = [
        "method", "alpha", "attack_prob", "tampering_count",
        "iter_mean", "iter_std", "mitig_mean", "mitig_std"
    ]
    summary_table.to_csv("simulation_summary_table.csv", index=False)
    print("Final summary table saved to simulation_summary_table.csv")
    print(summary_table)

if __name__ == "__main__":
    main()
