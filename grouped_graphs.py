#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""subset_plotter.py – variable‑only legends + log‑scaled iterations

Generates nine overlay plots requested by the paper, but **legend labels now
show only the parameters that *differ* inside each group** (no repetitive
constants, no run‑ids).  Example:

* **Group 1** (`t` constant, `α` absent → only *p* varies)
    → legend labels:  ``p=0.01``, ``p=0.10``
* **Group 9** (`α` constant at 0.5, `p` & `t` vary)
    → legend labels:  ``t=1 • p=0.10``, ``t=25 • p=0.01`` …

All figures are saved to ``logs/plots/special``; call with ``--show`` for an
interactive preview.
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────────────────── configuration ──────────────────────────────────────

GROUPS: List[Dict[str, object]] = [
    {  # 1 ─ Price, t=1 (method1)
        "metric": "Price",
        "tags": [
            "short_p2p_MAD_method1_prob0.01_mult1.5_t1_run0",
            "short_p2p_MAD_method1_prob0.1_mult1.5_t1_run1",
            "short_p2p_MAD_method1_prob0.1_mult1.5_t1_run5",
        ],
        "title": "Price • method1 • tampering 1",
    },
    {  # 2 ─ SW, t=1
        "metric": "SW",
        "tags": [
            "short_p2p_MAD_method1_prob0.01_mult1.5_t1_run0",
            "short_p2p_MAD_method1_prob0.1_mult1.5_t1_run1",
            "short_p2p_MAD_method1_prob0.1_mult1.5_t1_run5",
        ],
        "title": "Social‑welfare • method1 • tampering 1",
    },
    {  # 3 ─ Residuals, t=1
        "metric": "Residuals",
        "tags": [
            "short_p2p_MAD_method1_prob0.01_mult1.5_t1_run0",
            "short_p2p_MAD_method1_prob0.1_mult1.5_t1_run1",
            "short_p2p_MAD_method1_prob0.1_mult1.5_t1_run5",
        ],
        "title": "Residuals • method1 • tampering 1",
    },
    {  # 4 ─ Price, t=25
        "metric": "Price",
        "tags": [
            "short_p2p_MAD_method1_prob0.1_mult1.5_t25_run1",
            "short_p2p_MAD_method1_prob0.1_mult1.5_t25_run4",
            "short_p2p_MAD_method1_prob0.01_mult1.5_t25_run2",
        ],
        "title": "Price • method1 • tampering 25",
    },
    {  # 5 ─ SW, t=25
        "metric": "SW",
        "tags": [
            "short_p2p_MAD_method1_prob0.1_mult1.5_t25_run1",
            "short_p2p_MAD_method1_prob0.1_mult1.5_t25_run4",
            "short_p2p_MAD_method1_prob0.01_mult1.5_t25_run2",
        ],
        "title": "Social‑welfare • method1 • tampering 25",
    },
    {  # 6 ─ Residuals, t=25
        "metric": "Residuals",
        "tags": [
            "short_p2p_MAD_method1_prob0.1_mult1.5_t25_run1",
            "short_p2p_MAD_method1_prob0.1_mult1.5_t25_run4",
            "short_p2p_MAD_method1_prob0.01_mult1.5_t25_run2",
        ],
        "title": "Residuals • method1 • tampering 25",
    },
    {  # 7 ─ Price, method2 α=0.5
        "metric": "Price",
        "tags": [
            "short_p2p_MAD_method2_alpha0.5_prob0.1_mult1.5_t1_run0",
            "short_p2p_MAD_method2_alpha0.5_prob0.01_mult1.5_t25_run5",
            "short_p2p_MAD_method2_alpha0.5_prob0.5_mult1.5_t25_run6",
        ],
        "title": "Price • method2 α = 0.5",
    },
    {  # 8 ─ SW, method2 α=0.5
        "metric": "SW",
        "tags": [
            "short_p2p_MAD_method2_alpha0.5_prob0.1_mult1.5_t1_run0",
            "short_p2p_MAD_method2_alpha0.5_prob0.01_mult1.5_t25_run5",
            "short_p2p_MAD_method2_alpha0.5_prob0.5_mult1.5_t25_run6",
        ],
        "title": "Social‑welfare • method2 α = 0.5",
    },
    {  # 9 ─ Residuals, method2 α=0.5
        "metric": "Residuals",
        "tags": [
            "short_p2p_MAD_method2_alpha0.5_prob0.1_mult1.5_t1_run0",
            "short_p2p_MAD_method2_alpha0.5_prob0.01_mult1.5_t25_run5",
            "short_p2p_MAD_method2_alpha0.5_prob0.5_mult1.5_t25_run6",
        ],
        "title": "Residuals • method2 α = 0.5",
    },
]

ITER_DIR = Path("logs/iter_stats")
OUT_DIR  = Path("logs/plots/special")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 9,
    "axes.grid": True,
})

# ────────────────────────── regex helpers ──────────────────────────
_RG = {
    "prob":  re.compile(r"_prob([0-9.]+)"),
    "t":     re.compile(r"_t([^_]+)"),
    "alpha": re.compile(r"_alpha([0-9.]+)"),
}

ParamDict = Dict[str, str]

def extract_params(tag: str) -> ParamDict:
    """Return {'p': '0.01', 't': '1', 'α': '0.5', ...}."""
    out: ParamDict = {}
    if m := _RG["prob"].search(tag):  out["p"] = m.group(1)
    if m := _RG["t"].search(tag):     out["t"] = m.group(1)
    if m := _RG["alpha"].search(tag): out["α"] = m.group(1)
    return out

# ────────────────────────── data loader ────────────────────────────

def load_run(tag: str) -> pd.DataFrame:
    p = ITER_DIR / f"iter_{tag}.csv"
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df["Price"] = df["avg_price"].astype(float) * 100.0
    df["iter_shift"] = df["iter"].astype(float) + 1  # shift for log‑scale
    return df

# ─────────────────────── plotting primitives ───────────────────────

def plot_series(dfs: List[pd.DataFrame], ycol: str, labels: List[str], title: str, out_p: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for df, lbl in zip(dfs, labels):
        ax.plot(df["iter_shift"], df[ycol], label=lbl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ycol); ax.set_title(title); ax.legend(ncol=1)
    plt.tight_layout(); fig.savefig(out_p); plt.close(fig)


def plot_residuals(dfs: List[pd.DataFrame], labels: List[str], title: str, out_p: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for df, lbl in zip(dfs, labels):
        ax.plot(df["iter_shift"], df["prim"], label=f"prim • {lbl}")
        ax.plot(df["iter_shift"], df["dual"], linestyle="--", label=f"dual • {lbl}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual"); ax.set_title(title)
    ax.legend(ncol=2, fontsize="x-small")
    plt.tight_layout(); fig.savefig(out_p); plt.close(fig)

PLOT_FNS = {
    "Price":     lambda d, l, t, p: plot_series(d, "Price", l, t, p),
    "SW":        lambda d, l, t, p: plot_series(d, "SW", l, t, p),
    "Residuals": plot_residuals,
}

# ───────────────────────────── legend logic ────────────────────────────

def make_labels(tags: List[str]) -> List[str]:
    """Return a list of legend strings with only the *varying* parameters."""
    params_per_tag = [extract_params(t) for t in tags]
    # collect which keys vary
    varying = {
        k for k in ("t", "p", "α")
        if len({d.get(k) for d in params_per_tag}) > 1
    }
    labels = []
    for d in params_per_tag:
        parts = []
        for key in ("t", "p", "α"):
            if key in varying and key in d:
                parts.append(f"{key}={d[key]}")
        labels.append(" • ".join(parts) if parts else "(identical)")
    return labels

# ─────────────────────────────── main ────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Overlay plots with variable‑only legends")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    for idx, grp in enumerate(GROUPS, 1):
        metric, tags, title = grp["metric"], grp["tags"], grp["title"]
        print(f"▶  Group {idx}: {metric} – {title}")
        dfs, missing = [], False
        for t in tags:
            try:
                dfs.append(load_run(t))
            except FileNotFoundError as e:
                print(f"  ⚠  {e}; skipping plot …")
                missing = True
        if missing or not dfs:
            print("  ↳  incomplete data, figure skipped.")
            continue
        labels = make_labels(tags)
        out_pdf = OUT_DIR / f"{metric}_grp{idx}.pdf"
        PLOT_FNS[metric](dfs, labels, title, out_pdf)
        print(f"  ↳  saved {out_pdf}")
        if args.show:
            plt.show(block=False)
    print("Done.")


if __name__ == "__main__":
    main()
