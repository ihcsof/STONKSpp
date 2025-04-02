#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from SimulatorDiscrete import Simulator

############################################
# Helper Functions
############################################

def parse_log_file(file_name):
    """
    Parse the log file generated during simulation.
    Extract the number of mitigation events, average mitigation weight, and average deviation.
    """
    count = 0
    total_weight = 0.0
    total_deviation = 0.0
    if not os.path.exists(file_name):
        return {"mitigation_count": 0, "avg_weight": 0, "avg_deviation": 0}
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            count += 1
            try:
                parts = line.split("deviation=")
                if len(parts) > 1:
                    deviation_str = parts[1].split(",")[0]
                    deviation = float(deviation_str)
                    total_deviation += deviation
                parts = line.split("weight=")
                if len(parts) > 1:
                    weight_str = parts[1].split(",")[0]
                    weight = float(weight_str)
                    total_weight += weight
            except Exception:
                continue
    avg_weight = total_weight / count if count > 0 else 0
    avg_deviation = total_deviation / count if count > 0 else 0
    return {"mitigation_count": count, "avg_weight": avg_weight, "avg_deviation": avg_deviation}

def run_simulation(config):
    """
    Instantiate the Simulator with the given config, run it, and then parse its log file.
    Returns a dictionary of results.
    """
    sim = Simulator(config=config)
    for key, value in config.items():
        if hasattr(sim, key):
            setattr(sim, key, value)
    sim.run()
    
    result = {
        "iterations": sim.iteration
    }
    log_file = config.get("log_mitigation_file", "log_mitigation.txt")
    mitigation_data = parse_log_file(log_file)
    result.update(mitigation_data)
    return result

def clamp_errorbars_at_zero(means, errs):
    import numpy as np
    lower = means - errs
    upper = means + errs
    lower = np.maximum(lower, 0)
    negative_error = means - lower
    positive_error = upper - means
    yerr = [negative_error, positive_error]
    return yerr

def plot_grouped_bar_chart(df, x_col, group_col, value_col, ylabel, title, filename):
    """
    Grouped bar chart with error bars, clamped at 0.
    """
    import numpy as np
    df = df.copy()
    try:
        df[x_col] = df[x_col].astype(float)
    except:
        pass
    df[group_col] = df[group_col].astype(str)
    
    grouped = df.groupby([x_col, group_col])[value_col].agg(['mean','std']).reset_index()
    grouped = grouped.sort_values(by=x_col)
    
    pivot_mean = grouped.pivot(index=x_col, columns=group_col, values='mean')
    pivot_std  = grouped.pivot(index=x_col, columns=group_col, values='std')
    pivot_mean = pivot_mean.sort_index()
    pivot_std  = pivot_std.sort_index()

    import numpy as np
    fig, ax = plt.subplots(figsize=(8,6))
    num_groups = len(pivot_mean.columns)
    x_vals = np.arange(len(pivot_mean.index))
    bar_width = 0.8 / num_groups

    for i, col in enumerate(pivot_mean.columns):
        means = pivot_mean[col].values
        errs  = pivot_std[col].values
        yerr  = clamp_errorbars_at_zero(means, errs)
        bar_positions = x_vals + (i - num_groups/2) * bar_width + bar_width/2
        ax.bar(bar_positions, means, yerr=yerr, width=bar_width, label=str(col), capsize=5)
    
    x_labels = [str(idx) for idx in pivot_mean.index]
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=group_col)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_bar_chart(df, group_col, value_col, ylabel, title, filename):
    """
    Simple bar chart with error bars, clamped at 0.
    """
    import numpy as np
    df = df.copy()
    try:
        df[group_col] = df[group_col].astype(float)
    except:
        pass
    grouped = df.groupby(group_col)[value_col].agg(['mean','std']).reset_index()
    grouped = grouped.sort_values(by=group_col)
    means = grouped['mean'].values
    errs  = grouped['std'].values
    x_vals = range(len(grouped[group_col]))
    x_labels = grouped[group_col].astype(str).values

    yerr = clamp_errorbars_at_zero(means, errs)

    plt.figure()
    plt.bar(x_vals, means, yerr=yerr, capsize=5)
    plt.xticks(x_vals, x_labels)
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
    N = 3

    # Parameter sweeps
    methods = ["method1", "method2"]
    alphas = [0.25, 0.5, 0.75, 0.9]          # only for method2
    byzantine_ids_list = [[2]]              # fixed for simplicity
    attack_probs = [0.01, 0.05, 0.1, 0.5]
    multipliers = [(0.5, 1.2), (0.5, 1.3), (0.5, 1.5)]
    tampering_counts = [1, 5, 10, 30, 50]

    results = []

    for method in methods:
        for byz_ids in byzantine_ids_list:
            for prob in attack_probs:
                for (lower, upper) in multipliers:
                    for tamplimit in tampering_counts:
                        if method == "method2":
                            # Vary alpha
                            for alpha in alphas:
                                for run in range(N):
                                    config = {
                                        "iter_update_method": method,
                                        "alpha": alpha,
                                        "byzantine_ids": byz_ids,
                                        "byzantine_attack_probability": prob,
                                        "byzantine_multiplier_lower": lower,
                                        "byzantine_multiplier_upper": upper,
                                        "tampering_count": tamplimit,
                                        "scale_factor": 15.0,
                                        "mad_threshold": 4.1,
                                        "log_mitigation_file": (
                                            f"log_{method}_alpha{alpha}_ids{'-'.join(map(str, byz_ids))}"
                                            f"_prob{prob}_mult{upper}_tcount{tamplimit}.txt"
                                        ),
                                        "non_interactive": True,
                                        "maximum_iteration": 500,
                                        "penaltyfactor": 0.01,
                                        "residual_primal": 4e-3,
                                        "residual_dual": 4e-3
                                    }
                                    print(f"Running: method={method}, alpha={alpha}, tampering_count={tamplimit}, prob={prob}, mult={upper}")
                                    sim_result = run_simulation(config)
                                    sim_result.update({
                                        "method": method,
                                        "alpha": alpha,
                                        "byzantine_ids": str(byz_ids),
                                        "byzantine_attack_probability": prob,
                                        "byzantine_multiplier_upper": upper,
                                        "tampering_count": tamplimit,
                                        "run": run
                                    })
                                    results.append(sim_result)
                        else:
                            # method1 (classical ADMM); alpha not used, store alpha as "N/A"
                            for run in range(N):
                                config = {
                                    "iter_update_method": method,
                                    "byzantine_ids": byz_ids,
                                    "byzantine_attack_probability": prob,
                                    "byzantine_multiplier_lower": lower,
                                    "byzantine_multiplier_upper": upper,
                                    "tampering_count": tamplimit,
                                    "scale_factor": 15.0,
                                    "mad_threshold": 4.1,
                                    "log_mitigation_file": (
                                        f"log_{method}_ids{'-'.join(map(str, byz_ids))}"
                                        f"_prob{prob}_mult{upper}_tcount{tamplimit}.txt"
                                    ),
                                    "non_interactive": True,
                                    "maximum_iteration": 500,
                                    "penaltyfactor": 0.01,
                                    "residual_primal": 4e-3,
                                    "residual_dual": 4e-3
                                }
                                print(f"Running: method={method}, tampering_count={tamplimit}, prob={prob}, mult={upper}")
                                sim_result = run_simulation(config)
                                sim_result.update({
                                    "method": method,
                                    "alpha": "N/A",
                                    "byzantine_ids": str(byz_ids),
                                    "byzantine_attack_probability": prob,
                                    "byzantine_multiplier_upper": upper,
                                    "tampering_count": tamplimit,
                                    "run": run
                                })
                                results.append(sim_result)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("simulation_results.csv", index=False)
    print("Results saved to simulation_results.csv")

    # 1) For method2, group by alpha, x-axis = tampering_count, y=iterations
    plot_grouped_bar_chart(
        df=df[df["method"]=="method2"],
        x_col="tampering_count",
        group_col="alpha",
        value_col="iterations",
        ylabel="Average Iterations",
        title="Iterations vs. Tampering Count (Relaxed ADMM, varying alpha)",
        filename="iterations_vs_tamperingcount_relaxed.png"
    )

    # 2) For method2, group by alpha, x-axis = tampering_count, y=mitigation_count
    plot_grouped_bar_chart(
        df=df[df["method"]=="method2"],
        x_col="tampering_count",
        group_col="alpha",
        value_col="mitigation_count",
        ylabel="Average Mitigation Events",
        title="Mitigation Events vs. Tampering Count (Relaxed ADMM, varying alpha)",
        filename="mitigations_vs_tamperingcount_relaxed.png"
    )

    # 3) For classical ADMM, show iterations vs. tampering_count (simple bar)
    plot_bar_chart(
        df=df[df["method"]=="method1"],
        group_col="tampering_count",
        value_col="iterations",
        ylabel="Average Iterations",
        title="Iterations vs. Tampering Count (Classical ADMM)",
        filename="iterations_vs_tamperingcount_classical.png"
    )

    # Produce a meaningful summary table with everything
    # We'll pivot on method, alpha, byzantine_attack_probability, tampering_count
    pivot_cols = ["method", "alpha", "byzantine_attack_probability", "tampering_count"]
    summary_table = df.groupby(pivot_cols).agg({
        "iterations": ["mean","std"],
        "mitigation_count": ["mean","std"]
    }).reset_index()

    # Flatten column names
    summary_table.columns = [
        "method", "alpha", "attack_prob", "tampering_count",
        "iter_mean", "iter_std", "mitig_mean", "mitig_std"
    ]
    summary_table.to_csv("simulation_summary_table.csv", index=False)
    print("Final summary table saved to simulation_summary_table.csv")
    print(summary_table)

if __name__ == "__main__":
    main()
