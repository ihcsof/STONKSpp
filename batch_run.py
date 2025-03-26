#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_run.py

This script performs N executions over different configurations:
  - iter_update_method: "method1" (classical ADMM) and "method2" (relaxed ADMM with tunable alpha)
  - For method2, we vary the relaxation parameter "alpha" (e.g. 0.7, 0.85, 0.95, 1.0)
  - byzantine_attack_probability: different probabilities (e.g. 0.05, 0.1, 0.2)
  - byzantine_multiplier: different tampering magnitudes (using the upper multiplier, e.g. 1.2, 1.3, 1.5, 2.0)
  
For each simulation, it collects:
  - iteration count
  - mitigation data (from the log file):
      • mitigation_count: number of mitigation events
      • avg_weight: average mitigation weight used
      • avg_deviation: average deviation in mitigated trades

Results are saved to a CSV file and several graphs are generated.
"""

import os
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from SimulatorDiscrete import Simulator

def parse_log_file(file_name):
    """
    Parse the log file generated during simulation.
    Extract the number of mitigation events, average mitigation weight, and average deviation.
    Assumes each line is formatted as:
      "YYYY-MM-DD HH:MM:SS - Mitigated agent {j}: deviation={deviation:.2f}, median={median_trade:.2f}, weight={weight:.2f}, threshold={adaptive_threshold:.2f}, original={original:.2f}, new={new_value:.2f}"
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
    sim = Simulator()
    sim.config = config
    for key, value in config.items():
        if hasattr(sim, key):
            setattr(sim, key, value)
    sim.run()
    
    result = {"iterations": sim.iteration}
    log_file = config.get("log_mitigation_file", "log_mitigation.txt")
    mitigation_data = parse_log_file(log_file)
    result.update(mitigation_data)
    return result

def plot_bar_chart(df, group_col, value_col, ylabel, title, filename):
    grouped = df.groupby(group_col)[value_col].agg(['mean', 'std']).reset_index()
    plt.figure()
    plt.bar(grouped[group_col], grouped['mean'], yerr=grouped['std'], capsize=5)
    plt.xlabel(group_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_line_chart(df, x_col, y_col, group_col, ylabel, title, filename):
    plt.figure()
    for key, grp in df.groupby(group_col):
        grouped = grp.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
        plt.errorbar(grouped[x_col], grouped['mean'], yerr=grouped['std'],
                     label=str(key), marker='o', capsize=5)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title=group_col)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def main():
    N = 3  # executions per configuration

    methods = ["method1", "method2"]
    # For method2, we vary alpha; for method1, alpha is not used.
    alphas = [0.25, 0.5, 0.75, 0.9]
    byzantine_ids_list = [[2]]  # Fixed for simplicity
    attack_probs = [0.01, 0.05, 0.1, 0.5]
    multipliers = [(0.5, 1.2), (0.5, 1.3), (0.5, 1.5)]

    results = []

    for method in methods:
        for byz_ids in byzantine_ids_list:
            for prob in attack_probs:
                for (lower, upper) in multipliers:
                    if method == "method2":
                        for alpha in alphas:
                            for run in range(N):
                                config = {
                                    "iter_update_method": method,
                                    "alpha": alpha,
                                    "byzantine_ids": byz_ids,
                                    "byzantine_attack_probability": prob,
                                    "byzantine_multiplier_lower": lower,
                                    "byzantine_multiplier_upper": upper,
                                    "scale_factor": 15.0,
                                    "mad_threshold": 4.1,
                                    "log_mitigation_file": f"log_{method}_alpha{alpha}_ids{'-'.join(map(str, byz_ids))}_prob{prob}_mult{upper}.txt",
                                    "non_interactive": True,
                                    "maximum_iteration": 500,
                                    "penaltyfactor": 0.01,
                                    "residual_primal": 4e-3,
                                    "residual_dual": 4e-3
                                }
                                print(f"Running: method={method}, alpha={alpha}, byzantine_ids={byz_ids}, prob={prob}, upper multiplier={upper}, run={run}")
                                sim_result = run_simulation(config)
                                sim_result.update({
                                    "method": method,
                                    "alpha": alpha,
                                    "byzantine_ids": str(byz_ids),
                                    "byzantine_attack_probability": prob,
                                    "byzantine_multiplier_upper": upper,
                                    "run": run
                                })
                                results.append(sim_result)
                    else:  # method1 (classical ADMM; no alpha used)
                        for run in range(N):
                            config = {
                                "iter_update_method": method,
                                "byzantine_ids": byz_ids,
                                "byzantine_attack_probability": prob,
                                "byzantine_multiplier_lower": lower,
                                "byzantine_multiplier_upper": upper,
                                "scale_factor": 15.0,
                                "mad_threshold": 4.1,
                                "log_mitigation_file": f"log_{method}_ids{'-'.join(map(str, byz_ids))}_prob{prob}_mult{upper}.txt",
                                "non_interactive": True,
                                "maximum_iteration": 500,
                                "penaltyfactor": 0.01,
                                "residual_primal": 4e-3,
                                "residual_dual": 4e-3
                            }
                            print(f"Running: method={method}, byzantine_ids={byz_ids}, prob={prob}, upper multiplier={upper}, run={run}")
                            sim_result = run_simulation(config)
                            sim_result.update({
                                "method": method,
                                "alpha": None,
                                "byzantine_ids": str(byz_ids),
                                "byzantine_attack_probability": prob,
                                "byzantine_multiplier_upper": upper,
                                "run": run
                            })
                            results.append(sim_result)
    
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("simulation_results.csv", index=False)
    print("Results saved to simulation_results.csv")
    
    # --- Generate Graphs ---
    # Graph: Iteration count vs. Byzantine attack probability (separate lines for method and alpha if applicable)
    plot_line_chart(
        df=df[df["method"]=="method2"],
        x_col="byzantine_attack_probability",
        y_col="iterations",
        group_col="alpha",
        ylabel="Average Iterations",
        title="Iterations vs. Attack Probability (Relaxed ADMM, varying alpha)",
        filename="iterations_vs_prob_relaxed.png"
    )
    
    # Graph: Iteration count vs. Upper Tampering Multiplier for method2 (grouped by alpha)
    plot_line_chart(
        df=df[df["method"]=="method2"],
        x_col="byzantine_multiplier_upper",
        y_col="iterations",
        group_col="alpha",
        ylabel="Average Iterations",
        title="Iterations vs. Tampering Multiplier (Relaxed ADMM, varying alpha)",
        filename="iterations_vs_multiplier_relaxed.png"
    )
    
    # Graph: Mitigation event count vs. Byzantine attack probability for method2 (grouped by alpha)
    plot_line_chart(
        df=df[df["method"]=="method2"],
        x_col="byzantine_attack_probability",
        y_col="mitigation_count",
        group_col="alpha",
        ylabel="Average Mitigation Events",
        title="Mitigation Events vs. Attack Probability (Relaxed ADMM, varying alpha)",
        filename="mitigations_vs_prob_relaxed.png"
    )
    
    # Graph: For classical ADMM (method1) as baseline
    plot_bar_chart(
        df=df[df["method"]=="method1"],
        group_col="byzantine_attack_probability",
        value_col="iterations",
        ylabel="Average Iterations",
        title="Iterations vs. Attack Probability (Classical ADMM)",
        filename="iterations_vs_prob_classical.png"
    )

if __name__ == "__main__":
    main()