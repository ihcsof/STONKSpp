#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_graphs.py

This script processes existing log files (without re-running simulations) and produces:
  1) "simulation_results.csv" with all raw runs
  2) Several grouped bar charts (histogram style)
  3) A "simulation_summary_table.csv" pivot with aggregated data
  4) Additional charts for analyzing how iteration & mitigation vary with
     attack probability and multiplier, grouped by tampering count or alpha, etc.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Import plotting helpers from batch_run.py (ensure batch_run.py is in the same folder or adjust imports)
from batch_run import plot_bar_chart, plot_grouped_bar_chart

def parse_log_file(file_name):
    """
    Parse the log file generated during simulation.
    Return a dict with:
      - mitigation_count: total lines parsed
      - avg_weight: average 'weight=' field across lines
      - avg_deviation: average 'deviation=' field
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
                deviation_str = line.split("deviation=")[1].split(",")[0]
                weight_str = line.split("weight=")[1].split(",")[0]
                deviation = float(deviation_str)
                weight = float(weight_str)
                total_deviation += deviation
                total_weight += weight
            except:
                # If parsing fails for any line, just skip it
                continue
    
    avg_weight = total_weight / count if count > 0 else 0
    avg_deviation = total_deviation / count if count > 0 else 0
    return {"mitigation_count": count, "avg_weight": avg_weight, "avg_deviation": avg_deviation}

def clamp_errorbars_at_zero(means, errs):
    """
    Helper for clamping error bars so that they never dip below 0.
    """
    import numpy as np
    lower = means - errs
    upper = means + errs
    lower = np.maximum(lower, 0)
    negative_error = means - lower
    positive_error = upper - means
    return [negative_error, positive_error]

def main():
    """
    1) Scans all log_*.txt files.
    2) Extracts method, alpha, etc. from the filename via regex.
    3) Reads line counts and mitigation stats from the logs.
    4) Saves the combined results to 'simulation_results.csv'.
    5) Plots grouped bar charts (and additional charts) for iteration & mitigation.
    6) Builds a pivoted summary table, saved to 'simulation_summary_table.csv'.
    """
    # Find all log files matching our naming scheme
    files = [f for f in os.listdir('.') if re.match(r'log_.*\.txt', f)]
    results = []

    # Regex pattern to capture method, alpha (optional), IDs, probability, multiplier, tampering_count
    # e.g. log_method2_alpha0.5_ids2_prob0.1_mult1.5_tcount30.txt
    pattern = r"log_(method\d)(?:_alpha([\d\.]+))?_ids([\d\-]+)_prob([\d\.]+)_mult([\d\.]+)_tcount(\S+)\.txt"

    for log_file in files:
        params = re.match(pattern, log_file)
        if params:
            method, alpha_str, byz_ids_str, prob_str, mult_str, tcount_str = params.groups()

            # Convert alpha to "N/A" if it's not present (i.e., method1)
            if alpha_str is None:
                alpha = "N/A"
            else:
                alpha = alpha_str

            # Process the rest
            byz_ids = [int(i) for i in byz_ids_str.split('-')]
            prob = float(prob_str)
            mult = float(mult_str)
            # tampering_count might be "inf" if you're using float('inf')
            if tcount_str == 'inf':
                tcount = float('inf')
            else:
                tcount = float(tcount_str)

            # Parse mitigation stats
            mitigation_data = parse_log_file(log_file)

            # Example assumption: # of "iterations" = # of lines in log 
            # This might not match exactly if your code logs multiple lines per iteration.
            iterations = len(open(log_file).readlines())

            results.append({
                "method": method,
                "alpha": alpha,
                "byzantine_ids": byz_ids,
                "byzantine_attack_probability": prob,
                "byzantine_multiplier_upper": mult,
                "tampering_count": tcount,
                "iterations": iterations,
                "mitigation_count": mitigation_data["mitigation_count"],
                "avg_weight": mitigation_data["avg_weight"],
                "avg_deviation": mitigation_data["avg_deviation"]
            })

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv("simulation_results.csv", index=False)
    print("simulation_results.csv saved")
    print(df.head())

    # Create grouped bar charts
    # 1) For method2, group by alpha, x-axis = tampering_count, y=iterations
    df_method2 = df[df["method"] == "method2"]
    if not df_method2.empty:
        plot_grouped_bar_chart(
            df=df_method2,
            x_col="tampering_count",
            group_col="alpha",
            value_col="iterations",
            ylabel="Average Iterations",
            title="Iterations vs. Tampering Count (Relaxed ADMM)",
            filename="iterations_vs_tamperingcount_relaxed.png"
        )

        # 2) For method2, group by alpha, x-axis = tampering_count, y=mitigation_count
        plot_grouped_bar_chart(
            df=df_method2,
            x_col="tampering_count",
            group_col="alpha",
            value_col="mitigation_count",
            ylabel="Average Mitigation Events",
            title="Mitigation Events vs. Tampering Count (Relaxed ADMM)",
            filename="mitigations_vs_tamperingcount_relaxed.png"
        )

    # 3) For classical ADMM, show iterations vs. tampering_count (simple bar)
    df_method1 = df[df["method"] == "method1"]
    if not df_method1.empty:
        plot_bar_chart(
            df=df_method1,
            group_col="tampering_count",
            value_col="iterations",
            ylabel="Average Iterations",
            title="Iterations vs. Tampering Count (Classical ADMM)",
            filename="iterations_vs_tamperingcount_classical.png"
        )

    # 4) Iterations vs. Attack Probability (Grouped by tampering_count)
    if not df_method2.empty:
        plot_grouped_bar_chart(
            df=df_method2,
            x_col="byzantine_attack_probability",
            group_col="tampering_count",
            value_col="iterations",
            ylabel="Average Iterations",
            title="Iterations vs. Attack Probability (Relaxed ADMM)",
            filename="iterations_vs_attackprob_relaxed.png"
        )

    # 5) Mitigations vs. Attack Probability (Grouped by tampering_count)
    if not df_method2.empty:
        plot_grouped_bar_chart(
            df=df_method2,
            x_col="byzantine_attack_probability",
            group_col="tampering_count",
            value_col="mitigation_count",
            ylabel="Average Mitigation Events",
            title="Mitigations vs. Attack Probability (Relaxed ADMM)",
            filename="mitigations_vs_attackprob_relaxed.png"
        )

    # 6) Iterations vs. Multiplier (Grouped by tampering_count)
    if not df_method2.empty:
        plot_grouped_bar_chart(
            df=df_method2,
            x_col="byzantine_multiplier_upper",
            group_col="tampering_count",
            value_col="iterations",
            ylabel="Average Iterations",
            title="Iterations vs. Multiplier (Relaxed ADMM)",
            filename="iterations_vs_multiplier_relaxed.png"
        )

    # 7) Mitigations vs. Multiplier (Grouped by tampering_count)
    if not df_method1.empty:
        plot_grouped_bar_chart(
            df=df_method1,
            x_col="byzantine_multiplier_upper",
            group_col="tampering_count",
            value_col="mitigation_count",
            ylabel="Average Mitigation Events",
            title="Mitigations vs. Multiplier (Classical ADMM)",
            filename="mitigations_vs_multiplier_classical.png"
        )

    # ---- Summary Table ----
    summary_table = df.groupby(["method", "alpha", "byzantine_attack_probability", "tampering_count"]).agg({
        "iterations": ["mean", "std"],
        "mitigation_count": ["mean", "std"]
    }).reset_index()

    # Flatten the multi-index columns
    summary_table.columns = [
        "method", "alpha", "attack_prob", "tampering_count",
        "iter_mean", "iter_std", "mitig_mean", "mitig_std"
    ]

    summary_table.to_csv("simulation_summary_table.csv", index=False)
    print("simulation_summary_table.csv saved")
    print(summary_table.head())

if __name__ == "__main__":
    main()
