#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_graphs.py

This script looks only at what is ALREADY on disk (CSV logs written by
batch_run.py) and produces every plot we care about, namely:

1)  Aggregate grouped-bar / bar charts (iterations & mitigations) that used
    to live in batch_run.py.
2)  Per-run *iteration* plots created from each CSV in logs/iter_stats:
       • SW vs iteration
       • Avg-price vs iteration
       • Primal & dual residuals vs iteration
       • SW vs price scatter
       • Histogram of price
3)  A fresh "simulation_results.csv" by parsing mitigation logs (so anyone
    can rebuild plots without re-running simulations).
4)  A pivoted "simulation_summary_table.csv" with means/stds.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
#               ----------  Generic Plot Helpers ----------          #
# ------------------------------------------------------------------ #

def clamp_errorbars_at_zero(means, errs):
    lower = means - errs
    upper = means + errs
    lower = np.maximum(lower, 0)
    return [means - lower, upper - means]

def plot_grouped_bar_chart(df, x_col, group_col, value_col,
                           ylabel, title, filename):
    df = df.copy()
    df[group_col] = df[group_col].astype(str)
    try:
        df[x_col] = df[x_col].astype(float)
    except Exception:
        pass

    grouped    = (df.groupby([x_col, group_col])[value_col]
                    .agg(['mean', 'std']).reset_index())
    grouped    = grouped.sort_values(by=x_col)
    pivot_mean = grouped.pivot(index=x_col, columns=group_col,
                               values='mean').sort_index()
    pivot_std  = grouped.pivot(index=x_col, columns=group_col,
                               values='std').sort_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    num_groups = len(pivot_mean.columns)
    x_vals     = np.arange(len(pivot_mean.index))
    bar_width  = 0.8 / num_groups

    for i, col in enumerate(pivot_mean.columns):
        means = pivot_mean[col].values
        errs  = pivot_std[col].values
        yerr  = clamp_errorbars_at_zero(means, errs)
        positions = x_vals + (i - num_groups/2)*bar_width + bar_width/2
        ax.bar(positions, means, yerr=yerr, width=bar_width,
               label=str(col), capsize=5)

    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(idx) for idx in pivot_mean.index])
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=group_col)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

def plot_bar_chart(df, group_col, value_col,
                   ylabel, title, filename):
    df = df.copy()
    try:
        df[group_col] = df[group_col].astype(float)
    except Exception:
        pass

    grouped = (df.groupby(group_col)[value_col]
                 .agg(['mean', 'std']).reset_index()
                 .sort_values(by=group_col))
    means   = grouped['mean'].values
    errs    = grouped['std'].values
    yerr    = clamp_errorbars_at_zero(means, errs)

    fig, ax = plt.subplots()
    ax.bar(range(len(means)), means, yerr=yerr, capsize=5)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(grouped[group_col].astype(str).values)
    ax.set_xlabel(group_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

# ------------------------------------------------------------------ #
#               ----------  Mitigation-log Parsing ----------        #
# ------------------------------------------------------------------ #

def parse_log_file(file_name):
    """Return mitigation_count, avg_weight, avg_deviation from a log file."""
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
            except Exception:
                continue

    avg_weight    = total_weight   / count if count > 0 else 0
    avg_deviation = total_deviation / count if count > 0 else 0
    return {"mitigation_count": count,
            "avg_weight": avg_weight,
            "avg_deviation": avg_deviation}

# ------------------------------------------------------------------ #
#             ----------  Per-Iteration Plot Suite ----------        #
# ------------------------------------------------------------------ #

def generate_iteration_plots(iter_csv, outdir="logs/plots"):
    """
    For a single iter_*.csv file create:
        1) SW vs iteration
        2) Price vs iteration
        3) Primal & dual vs iteration
        4) SW vs price scatter
        5) Histogram of price
    Filenames derive from the CSV basename.
    """
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(iter_csv)
    tag = os.path.splitext(os.path.basename(iter_csv))[0].replace("iter_", "")

    # 1) SW vs iteration
    fig, ax = plt.subplots()
    ax.plot(df["iter"], df["SW"])
    ax.set_xlabel("Iteration"); ax.set_ylabel("SW")
    ax.set_title("SW vs Iteration")
    fig.tight_layout()
    fig.savefig(f"{outdir}/SW_{tag}.png"); plt.close(fig)

    # 2) Price vs iteration
    fig, ax = plt.subplots()
    ax.plot(df["iter"], df["avg_price"])
    ax.set_xlabel("Iteration"); ax.set_ylabel("Average Price")
    ax.set_title("Price vs Iteration")
    fig.tight_layout()
    fig.savefig(f"{outdir}/Price_{tag}.png"); plt.close(fig)

    # 3) Primal & dual residuals vs iteration
    fig, ax = plt.subplots()
    ax.plot(df["iter"], df["prim"], label="Primal")
    ax.plot(df["iter"], df["dual"], label="Dual")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Residual value")
    ax.set_title("Primal & Dual Residuals vs Iteration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/Residuals_{tag}.png"); plt.close(fig)

    # 4) SW vs price scatter
    fig, ax = plt.subplots()
    ax.scatter(df["avg_price"], df["SW"])
    ax.set_xlabel("Average Price"); ax.set_ylabel("SW")
    ax.set_title("SW vs Price")
    fig.tight_layout()
    fig.savefig(f"{outdir}/SW_vs_Price_{tag}.png"); plt.close(fig)

    # 5) Histogram of price
    fig, ax = plt.subplots()
    ax.hist(df["avg_price"], bins=20)
    ax.set_xlabel("Average Price"); ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Average Price")
    fig.tight_layout()
    fig.savefig(f"{outdir}/Price_hist_{tag}.png"); plt.close(fig)

# ------------------------------------------------------------------ #
#                           ----------  Main ----------              #
# ------------------------------------------------------------------ #

def main():
    os.makedirs("logs/plots", exist_ok=True)

    # ----------------------------------------------------------------
    # 1)  Build / rebuild simulation_results.csv from mitigation logs
    # ----------------------------------------------------------------
    mit_dir = "logs/mitigation"
    os.makedirs(mit_dir, exist_ok=True)

    pattern = (
        r"log_(method\d)"
        r"(?:_alpha([\d\.]+))?"
        r"_prob([\d\.]+)"
        r"_mult([\d\.]+)"
        r"_t(\S+)\.txt"
    )

    results = []
    for log_file in glob.glob(os.path.join(mit_dir, "log_*.txt")):
        name = os.path.basename(log_file)
        m = re.match(pattern, name)
        if not m:
            continue

        method, alpha_str, prob_str, mult_str, t_str = m.groups()
        alpha  = float(alpha_str) if alpha_str else 0
        prob   = float(prob_str)
        mult   = float(mult_str)
        tcount = float('inf') if t_str == 'inf' else float(t_str)

        mitig = parse_log_file(log_file)
        iterations = mitig["mitigation_count"]  # crude proxy

        results.append({
            "method":            method,
            "alpha":             alpha,
            "attack_prob":       prob,
            "multiplier_upper":  mult,
            "tampering_count":   tcount,
            "iterations":        iterations,
            "mitigation_count":  mitig["mitigation_count"],
            "avg_weight":        mitig["avg_weight"],
            "avg_deviation":     mitig["avg_deviation"]
        })

    df = pd.DataFrame(results)
    df.to_csv("simulation_results.csv", index=False)
    print("simulation_results.csv saved")

    if df.empty:
        print("No mitigation data found – check the log filenames / pattern.")
        return

    # ---------------------------- 2) Aggregate Charts ---------------

    df2 = df[df["method"] == "method2"]
    df1 = df[df["method"] == "method1"]

    if not df2.empty:
        plot_grouped_bar_chart(
            df2, "tampering_count", "alpha", "iterations",
            "Average Iterations",
            "Iterations vs. Tampering Count (Relaxed ADMM)",
            "iterations_vs_tamperingcount_relaxed.png"
        )
        plot_grouped_bar_chart(
            df2, "tampering_count", "alpha", "mitigation_count",
            "Average Mitigation Events",
            "Mitigations vs. Tampering Count (Relaxed ADMM)",
            "mitigations_vs_tamperingcount_relaxed.png"
        )
        plot_grouped_bar_chart(
            df2, "attack_prob", "tampering_count", "iterations",
            "Average Iterations",
            "Iterations vs. Attack Probability (Relaxed ADMM)",
            "iterations_vs_attackprob_relaxed.png"
        )
        plot_grouped_bar_chart(
            df2, "attack_prob", "tampering_count", "mitigation_count",
            "Average Mitigation Events",
            "Mitigations vs. Attack Probability (Relaxed ADMM)",
            "mitigations_vs_attackprob_relaxed.png"
        )
        plot_grouped_bar_chart(
            df2, "multiplier_upper", "tampering_count", "iterations",
            "Average Iterations",
            "Iterations vs. Multiplier (Relaxed ADMM)",
            "iterations_vs_multiplier_relaxed.png"
        )

    if not df1.empty:
        plot_bar_chart(
            df1, "tampering_count", "iterations",
            "Average Iterations",
            "Iterations vs. Tampering Count (Classical ADMM)",
            "iterations_vs_tamperingcount_classical.png"
        )
        plot_grouped_bar_chart(
            df1, "multiplier_upper", "tampering_count", "mitigation_count",
            "Average Mitigation Events",
            "Mitigations vs. Multiplier (Classical ADMM)",
            "mitigations_vs_multiplier_classical.png"
        )

    # ------------------------- 3) Per-Iteration Charts --------------

    for csv_file in glob.glob("logs/iter_stats/iter_*.csv"):
        generate_iteration_plots(csv_file, outdir="logs/plots")

    # ------------------------- 4) Summary Pivot ---------------------

    summary = (df.groupby(["method", "alpha", "attack_prob", "tampering_count"])
                 .agg(iter_mean=("iterations", "mean"),
                      iter_std=("iterations", "std"),
                      mitig_mean=("mitigation_count", "mean"),
                      mitig_std=("mitigation_count", "std"))
                 .reset_index())
    summary.to_csv("simulation_summary_table.csv", index=False)
    print("simulation_summary_table.csv saved")

if __name__ == "__main__":
    main()
