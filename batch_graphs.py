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

# Import plotting helpers from batch_run.py
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
        for line in f:
            count += 1
            try:
                deviation_str = line.split("deviation=")[1].split(",")[0]
                weight_str    = line.split("weight=")[1].split(",")[0]
                total_deviation += float(deviation_str)
                total_weight    += float(weight_str)
            except:
                continue
    
    avg_weight    = total_weight   / count if count > 0 else 0
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
    return [means - lower, upper - means]

def main():
    """
    1) Scans all log_*.txt files under logs/mitigation.
    2) Extracts method, alpha, etc. from the filename via regex.
    3) Reads line counts and mitigation stats from the logs.
    4) Saves the combined results to 'simulation_results.csv'.
    5) Plots grouped bar charts (and additional charts) for iteration & mitigation.
    6) Builds a pivoted summary table, saved to 'simulation_summary_table.csv'.
    """
    # directory containing mitigation logs
    logs_dir = "logs/mitigation"
    os.makedirs(logs_dir, exist_ok=True)

    # find all mitigation log files
    files = [
        os.path.join(logs_dir, f)
        for f in os.listdir(logs_dir)
        if re.match(r'log_.*\.txt', f)
    ]

    results = []

    # pattern to parse filenames
    pattern = r"log_(method\d)(?:_alpha([\d\.]+))?_ids([\d\-]+)_prob([\d\.]+)_mult([\d\.]+)_tcount(\S+)\.txt"

    for log_file in files:
        name = os.path.basename(log_file)
        m = re.match(pattern, name)
        if not m:
            continue

        method, alpha_str, byz_ids_str, prob_str, mult_str, tcount_str = m.groups()
        alpha = alpha_str if alpha_str is not None else "N/A"
        byz_ids = [int(i) for i in byz_ids_str.split('-')]
        prob    = float(prob_str)
        mult    = float(mult_str)
        tcount  = float('inf') if tcount_str == 'inf' else float(tcount_str)

        # parse mitigation stats
        mitigation_data = parse_log_file(log_file)
        # assume iterations ~= number of lines
        iterations = len(open(log_file).readlines())

        results.append({
            "method":                       method,
            "alpha":                        alpha,
            "byzantine_ids":                byz_ids,
            "byzantine_attack_probability": prob,
            "byzantine_multiplier_upper":   mult,
            "tampering_count":              tcount,
            "iterations":                   iterations,
            "mitigation_count":             mitigation_data["mitigation_count"],
            "avg_weight":                   mitigation_data["avg_weight"],
            "avg_deviation":                mitigation_data["avg_deviation"]
        })

    # save raw results
    df = pd.DataFrame(results)
    df.to_csv("simulation_results.csv", index=False)
    print("simulation_results.csv saved")
    print(df.head())

    # split by method
    df2 = df[df["method"] == "method2"]
    df1 = df[df["method"] == "method1"]

    # method2: iterations vs tampering_count (grouped by alpha)
    if not df2.empty:
        plot_grouped_bar_chart(
            df=df2, x_col="tampering_count", group_col="alpha",
            value_col="iterations",
            ylabel="Average Iterations",
            title="Iterations vs. Tampering Count (Relaxed ADMM)",
            filename="iterations_vs_tamperingcount_relaxed.png"
        )
        plot_grouped_bar_chart(
            df=df2, x_col="tampering_count", group_col="alpha",
            value_col="mitigation_count",
            ylabel="Average Mitigation Events",
            title="Mitigation Events vs. Tampering Count (Relaxed ADMM)",
            filename="mitigations_vs_tamperingcount_relaxed.png"
        )

    # method1: simple bar iterations vs tampering_count
    if not df1.empty:
        plot_bar_chart(
            df=df1, group_col="tampering_count",
            value_col="iterations",
            ylabel="Average Iterations",
            title="Iterations vs. Tampering Count (Classical ADMM)",
            filename="iterations_vs_tamperingcount_classical.png"
        )

    # more method2 charts
    if not df2.empty:
        plot_grouped_bar_chart(
            df=df2, x_col="byzantine_attack_probability",
            group_col="tampering_count",
            value_col="iterations",
            ylabel="Average Iterations",
            title="Iterations vs. Attack Probability (Relaxed ADMM)",
            filename="iterations_vs_attackprob_relaxed.png"
        )
        plot_grouped_bar_chart(
            df=df2, x_col="byzantine_attack_probability",
            group_col="tampering_count",
            value_col="mitigation_count",
            ylabel="Average Mitigation Events",
            title="Mitigations vs. Attack Probability (Relaxed ADMM)",
            filename="mitigations_vs_attackprob_relaxed.png"
        )
        plot_grouped_bar_chart(
            df=df2, x_col="byzantine_multiplier_upper",
            group_col="tampering_count",
            value_col="iterations",
            ylabel="Average Iterations",
            title="Iterations vs. Multiplier (Relaxed ADMM)",
            filename="iterations_vs_multiplier_relaxed.png"
        )

    # method1: mitigations vs multiplier
    if not df1.empty:
        plot_grouped_bar_chart(
            df=df1, x_col="byzantine_multiplier_upper",
            group_col="tampering_count",
            value_col="mitigation_count",
            ylabel="Average Mitigation Events",
            title="Mitigations vs. Multiplier (Classical ADMM)",
            filename="mitigations_vs_multiplier_classical.png"
        )

    # summary table
    summary = df.groupby(
        ["method", "alpha", "byzantine_attack_probability", "tampering_count"]
    ).agg({"iterations": ["mean", "std"], "mitigation_count": ["mean", "std"]}).reset_index()
    summary.columns = [
        "method", "alpha", "attack_prob", "tampering_count",
        "iter_mean", "iter_std", "mitig_mean", "mitig_std"
    ]
    summary.to_csv("simulation_summary_table.csv", index=False)
    print("simulation_summary_table.csv saved")
    print(summary.head())

if __name__ == "__main__":
    main()