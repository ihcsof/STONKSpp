#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
process_logs.py

This script processes existing log files (without re-running simulations) and produces:
  1) "simulation_results.csv" with all raw runs
  2) Several grouped bar charts (histogram style) for:
       - Iterations vs. Tampering Count
       - Mitigations vs. Tampering Count
       - Iterations vs. Attack Probability
       - Mitigations vs. Attack Probability
       - Iterations vs. Multiplier
       - Mitigations vs. Multiplier
  3) A "simulation_summary_table.csv" pivot with aggregated data
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Import the plotting helpers from batch_run.py (ensure batch_run.py is in the same folder)
# The following must contain definitions of:
#   - plot_bar_chart
#   - plot_grouped_bar_chart
#   - clamp_errorbars_at_zero
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
                deviation = float(line.split("deviation=")[1].split(",")[0])
                weight = float(line.split("weight=")[1].split(",")[0])
                total_deviation += deviation
                total_weight += weight
            except:
                # If parsing fails for any line, just skip it
                continue
    
    avg_weight = total_weight / count if count > 0 else 0
    avg_deviation = total_deviation / count if count > 0 else 0
    return {"mitigation_count": count, "avg_weight": avg_weight, "avg_deviation": avg_deviation}

def main():
    """
    1) Scans all log_*.txt files
    2) Extracts method, alpha, etc. from the filename via regex
    3) Reads line counts and mitigation stats
    4) Saves the combined results to 'simulation_results.csv'
    5) Plots all relevant bar/grouped-bar charts
    6) Builds a pivoted summary table, saved to 'simulation_summary_table.csv'
    """
    # Find all log files matching our naming scheme
    files = [f for f in os.listdir('.') if re.match(r'log_.*\.txt', f)]
    results = []

    # Regex pattern to capture method, alpha (optional), IDs, probability, multiplier, and tampering count
    pattern = r"log_(method\d)(?:_alpha([\d\.]+))?_ids([\d\-]+)_prob([\d\.]+)_mult([\d\.]+)_tcount(\d+)\.txt"

    for log_file in files:
        params = re.match(pattern, log_file)
        if params:
            method, alpha_str, byz_ids_str, prob_str, mult_str, tcount_str = params.groups()

            # Convert alpha to "N/A" if it's not present (i.e., method1)
            if alpha_str is None:
                alpha = "N/A"
            else:
                alpha = alpha_str  # keep as string; could also convert to float if desired

            # Process the rest
            byz_ids = [int(i) for i in byz_ids_str.split('-')]
            prob = float(prob_str)
            mult = float(mult_str)
            tcount = int(tcount_str)

            # Parse log file for mitigation stats
            mitigation_data = parse_log_file(log_file)

            # For "iterations," we'll assume one log line per iteration
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

    # We will now generate all the same plots that appear in batch_run.py
    # ------------------------------------------------------------------------------------------
    # 1) For method2, group by alpha, x-axis = tampering_count, y=iterations
    df_method2 = df[df["method"] == "method2"]
    if not df_method2.empty:
        plot_grouped_bar_chart(
            df=df_method2,
            x_col="tampering_count",
            group_col="alpha",
            value_col="iterations",
            ylabel="Average Iterations",
            title="Iterations vs. Tampering Count (faster ADMM, varying alpha)",
            filename="iterations_vs_tamperingcount_faster.png"
        )

        # 2) For method2, group by alpha, x-axis = tampering_count, y=mitigation_count
        plot_grouped_bar_chart(
            df=df_method2,
            x_col="tampering_count",
            group_col="alpha",
            value_col="mitigation_count",
            ylabel="Average Mitigation Events",
            title="Mitigation Events vs. Tampering Count (faster ADMM, varying alpha)",
            filename="mitigations_vs_tamperingcount_faster.png"
        )

    # 3) For classical ADMM (method1), show iterations vs. tampering_count (simple bar)
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

    # ---- Additional Charts (mirroring batch_run.py) ----
    # 4) Iterations vs. Attack Probability (Grouped by tampering_count)
    df_method2_ap = df_method2  # only method2 for these plots, as per batch_run
    if not df_method2_ap.empty:
        plot_grouped_bar_chart(
            df=df_method2_ap,
            x_col="byzantine_attack_probability",
            group_col="tampering_count",
            value_col="iterations",
            ylabel="Average Iterations",
            title="Iterations vs. Attack Probability (faster ADMM)",
            filename="iterations_vs_attackprob_faster.png"
        )

        # 5) Mitigations vs. Attack Probability (Grouped by tampering_count)
        plot_grouped_bar_chart(
            df=df_method2_ap,
            x_col="byzantine_attack_probability",
            group_col="tampering_count",
            value_col="mitigation_count",
            ylabel="Average Mitigation Events",
            title="Mitigations vs. Attack Probability (faster ADMM)",
            filename="mitigations_vs_attackprob_faster.png"
        )

        # 6) Iterations vs. Multiplier (Grouped by tampering_count)
        plot_grouped_bar_chart(
            df=df_method2_ap,
            x_col="byzantine_multiplier_upper",
            group_col="tampering_count",
            value_col="iterations",
            ylabel="Average Iterations",
            title="Iterations vs. Multiplier (faster ADMM)",
            filename="iterations_vs_multiplier_faster.png"
        )

    # 7) Mitigations vs. Multiplier (Grouped by tampering_count) - for method1 (classical ADMM)
    df_method1_mult = df_method1
    if not df_method1_mult.empty:
        plot_grouped_bar_chart(
            df=df_method1_mult,
            x_col="byzantine_multiplier_upper",
            group_col="tampering_count",
            value_col="mitigation_count",
            ylabel="Average Mitigation Events",
            title="Mitigations vs. Multiplier (Classical ADMM)",
            filename="mitigations_vs_multiplier_classical.png"
        )

    # ------------------------------------------------------------------------------------------
    # Build a summary pivot table that includes both method1 and method2
    # Group by method, alpha, probability, and tampering_count
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
