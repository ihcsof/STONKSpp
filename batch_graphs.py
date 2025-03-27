#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
process_logs.py

This script processes existing log files (without re-running simulations) and produces:
  1) "simulation_results.csv" with all raw runs
  2) Several grouped bar charts (histogram style)
  3) A "simulation_summary_table.csv" pivot with aggregated data
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import re

from batch_run import plot_bar_chart, plot_grouped_bar_chart

# Reuse parse_log_file and plotting functions provided before.

def parse_log_file(file_name):
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
                continue
    avg_weight = total_weight / count if count > 0 else 0
    avg_deviation = total_deviation / count if count > 0 else 0
    return {"mitigation_count": count, "avg_weight": avg_weight, "avg_deviation": avg_deviation}

def clamp_errorbars_at_zero(means, errs):
    import numpy as np
    lower = means - errs
    upper = means + errs
    lower = np.maximum(lower, 0)
    negative_error = means - lower
    positive_error = upper - means
    return [negative_error, positive_error]

# Include your previous plotting functions here (plot_grouped_bar_chart, plot_bar_chart)
# Due to space constraints, they are assumed already included here.

# Main extraction process
def main():
    files = [f for f in os.listdir('.') if re.match(r'log_.*\.txt', f)]
    results = []

    for log_file in files:
        params = re.match(r"log_(method\d)(_alpha([\d\.]+))?_ids([\d\-]+)_prob([\d\.]+)_mult([\d\.]+)_maxT(\d+)\.txt", log_file)
        if params:
            method, _, alpha, byz_ids, prob, mult, maxT = params.groups()
            alpha = float(alpha) if alpha else None
            byz_ids = [int(i) for i in byz_ids.split('-')]
            prob = float(prob)
            mult = float(mult)
            maxT = int(maxT)

            mitigation_data = parse_log_file(log_file)

            iterations = len(open(log_file).readlines())
            
            results.append({
                "method": method,
                "alpha": alpha,
                "byzantine_ids": byz_ids,
                "byzantine_attack_probability": prob,
                "byzantine_multiplier_upper": mult,
                "byzantine_max_tampering": maxT,
                "iterations": iterations,
                "mitigation_count": mitigation_data["mitigation_count"],
                "avg_weight": mitigation_data["avg_weight"],
                "avg_deviation": mitigation_data["avg_deviation"]
            })

    df = pd.DataFrame(results)
    df.to_csv("simulation_results.csv", index=False)
    print("simulation_results.csv saved")

    # Generate grouped bar charts
    plot_grouped_bar_chart(
        df=df[df["method"]=="method2"],
        x_col="byzantine_max_tampering",
        group_col="alpha",
        value_col="iterations",
        ylabel="Average Iterations",
        title="Iterations vs. Max Tampering (Relaxed ADMM, varying alpha)",
        filename="iterations_vs_maxtampering_relaxed.png"
    )

    plot_grouped_bar_chart(
        df=df[df["method"]=="method2"],
        x_col="byzantine_max_tampering",
        group_col="alpha",
        value_col="mitigation_count",
        ylabel="Average Mitigation Events",
        title="Mitigation Events vs. Max Tampering (Relaxed ADMM, varying alpha)",
        filename="mitigations_vs_maxtampering_relaxed.png"
    )

    plot_bar_chart(
        df=df[df["method"]=="method1"],
        group_col="byzantine_max_tampering",
        value_col="iterations",
        ylabel="Average Iterations",
        title="Iterations vs. Max Tampering (Classical ADMM)",
        filename="iterations_vs_maxtampering_classical.png"
    )

    summary_table = df.groupby(["method", "alpha", "byzantine_attack_probability", "byzantine_max_tampering"]).agg({
        "iterations": ["mean", "std"],
        "mitigation_count": ["mean", "std"]
    }).reset_index()
    summary_table.columns = [
        "method", "alpha", "attack_prob", "max_tampering",
        "iter_mean", "iter_std", "mitig_mean", "mitig_std"
    ]

    summary_table.to_csv("simulation_summary_table.csv", index=False)
    print("simulation_summary_table.csv saved")

if __name__ == "__main__":
    main()
