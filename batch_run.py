#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_run.py

This script performs N executions over different configurations:
  - iter_update_method: "method1" and "method2"
  - byzantine_ids: various lists (e.g., [2], [2,5], [2,5,8])
  - byzantine_attack_probability: different probabilities (e.g., 0.05 and 0.1)
  - byzantine_multiplier_lower and byzantine_multiplier_upper: different multipliers

For each simulation, it collects:
  - iteration count
  - mitigation data (read from the log file specified in the config):
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
                # Extract deviation:
                parts = line.split("deviation=")
                if len(parts) > 1:
                    deviation_str = parts[1].split(",")[0]
                    deviation = float(deviation_str)
                    total_deviation += deviation
                # Extract weight:
                parts = line.split("weight=")
                if len(parts) > 1:
                    weight_str = parts[1].split(",")[0]
                    weight = float(weight_str)
                    total_weight += weight
            except Exception as e:
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
    # Pass configuration to simulator (overwriting defaults)
    sim.config = config
    for key, value in config.items():
        if hasattr(sim, key):
            setattr(sim, key, value)
    # Run the simulation (assumes sim.run() runs to termination)
    sim.run()
    
    result = {
        "iterations": sim.iteration,
        # We can add additional simulation metrics if needed.
    }
    
    # Parse the mitigation log file from the config.
    log_file = config.get("log_mitigation_file", "log_mitigation.txt")
    mitigation_data = parse_log_file(log_file)
    result.update(mitigation_data)
    return result

def plot_bar_chart(df, group_col, value_col, ylabel, title, filename):
    """
    Create a bar chart (with error bars) for the mean of value_col grouped by group_col.
    """
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
    """
    Create a line chart showing y_col versus x_col for each group in group_col, with error bars.
    """
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
    # Number of executions for each configuration combination.
    N = 3  # Adjust as needed

    # Parameter sweeps.
    methods = ["method1", "method2"]
    byzantine_ids_list = [[2], [3], [5]]
    byzantine_probabilities = [0.05, 0.1]
    multipliers = [(0.1, 1.2), (0.5, 1.2)]

    results = []

    # Loop over all configuration combinations.
    for method in methods:
        for byz_ids in byzantine_ids_list:
            for prob in byzantine_probabilities:
                for (lower, upper) in multipliers:
                    for run in range(N):
                        # Build configuration dictionary.
                        config = {
                            "iter_update_method": method,
                            "byzantine_ids": byz_ids,
                            "byzantine_attack_probability": prob,
                            "byzantine_multiplier_lower": lower,
                            "byzantine_multiplier_upper": upper,
                            "scale_factor": 15.0,
                            "mad_threshold": 5,
                            "log_mitigation_file": f"log_{method}_ids{'-'.join(map(str, byz_ids))}_prob{prob}_mult{lower}-{upper}.txt",
                            "non_interactive": True
                        }
                        print(f"Running: method={method}, byzantine_ids={byz_ids}, prob={prob}, multipliers=({lower}, {upper}), run={run}")
                        
                        sim_result = run_simulation(config)
                        sim_result.update({
                            "method": method,
                            "byzantine_ids": str(byz_ids),
                            "byzantine_attack_probability": prob,
                            "byzantine_multiplier_lower": lower,
                            "byzantine_multiplier_upper": upper,
                            "run": run
                        })
                        results.append(sim_result)
    
    # Save results to CSV.
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("simulation_results.csv", index=False)
    print("Results saved to simulation_results.csv")
    
    # --- Generate Graphs ---
    # Graph 1: Bar chart of iteration count by method.
    plot_bar_chart(
        df=df,
        group_col="method",
        value_col="iterations",
        ylabel="Average Iterations",
        title="Average Iterations by Iterative Update Method",
        filename="avg_iterations_by_method.png"
    )
    
    # Graph 2: Bar chart of mitigation event count by method.
    plot_bar_chart(
        df=df,
        group_col="method",
        value_col="mitigation_count",
        ylabel="Average Mitigation Events",
        title="Average Mitigation Events by Iterative Update Method",
        filename="avg_mitigations_by_method.png"
    )
    
    # Graph 3: Line chart of iteration count vs. Byzantine attack probability for each method.
    plot_line_chart(
        df=df,
        x_col="byzantine_attack_probability",
        y_col="iterations",
        group_col="method",
        ylabel="Average Iterations",
        title="Iterations vs. Byzantine Attack Probability",
        filename="iterations_vs_probability.png"
    )
    
    # Graph 4: Line chart of mitigation count vs. Byzantine attack probability for each method.
    plot_line_chart(
        df=df,
        x_col="byzantine_attack_probability",
        y_col="mitigation_count",
        group_col="method",
        ylabel="Average Mitigation Events",
        title="Mitigation Events vs. Byzantine Attack Probability",
        filename="mitigations_vs_probability.png"
    )
    
    # Graph 5: Line chart of average mitigation weight vs. Byzantine attack probability.
    plot_line_chart(
        df=df,
        x_col="byzantine_attack_probability",
        y_col="avg_weight",
        group_col="method",
        ylabel="Average Mitigation Weight",
        title="Avg Mitigation Weight vs. Byzantine Attack Probability",
        filename="avg_weight_vs_probability.png"
    )
    
    # Graph 6: Line chart of average mitigation deviation vs. Byzantine attack probability.
    plot_line_chart(
        df=df,
        x_col="byzantine_attack_probability",
        y_col="avg_deviation",
        group_col="method",
        ylabel="Average Mitigation Deviation",
        title="Avg Mitigation Deviation vs. Byzantine Attack Probability",
        filename="avg_deviation_vs_probability.png"
    )

if __name__ == "__main__":
    main()
