#!/usr/bin/env python3
"""
plot_all.py  –  merged & fully robust version, with increased font sizes,
horizontal caps on boxplots, and updated legend title for panel B

(A) Termination iterations           (box-plot, MAD vs TRUST)
(B) MAD corrections / TRUST violations (log-scale box-plot)
(C) Combined MAD+TRUST heat-map     – Classical  (α = 0)
(D) Combined MAD+TRUST heat-map     – Relaxed    (α = 0.50)

Usage: python plot_all.py [pdf|png]
Requires in data/:
  grid_metrics.csv
  grid_outcome.csv
  trust_grid_metrics.csv
  trust_grid_outcome.csv    # now used for panels C & D
"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------- Styling
sns.set_theme(style="whitegrid", font="serif")
sns.set_context("talk", font_scale=1.2)
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 11,
    "legend.title_fontsize": 13,
})

# ---------------------------------------------------------------- CLI
if len(sys.argv) != 2 or sys.argv[1] not in {"pdf", "png"}:
    print("usage: python plot_all.py [pdf|png]")
    sys.exit(1)
ext = sys.argv[1]
out = f"aggregate_boxplots.{ext}"

here = pathlib.Path(__file__).resolve().parent
data = here / "data"

# ------------------------------------------------------------- helpers
def load_csv(fname: str) -> pd.DataFrame:
    return pd.read_csv(data / fname)

def merge_verdicts(metrics: pd.DataFrame, outcome: pd.DataFrame) -> pd.DataFrame:
    """Attach 'verdict' to metrics, dropping bad t's so the join works."""
    def fix(df):
        df2 = df.copy()
        df2["t_num"] = pd.to_numeric(df2["t"], errors="coerce")
        df2 = df2[df2["t_num"].notnull() & np.isfinite(df2["t_num"])]
        df2["t"]        = df2["t_num"].astype(int).astype(str)
        df2["alpha"]    = pd.to_numeric(df2["alpha"],    errors="coerce").astype(float)
        df2["p_attack"] = pd.to_numeric(df2["p_attack"], errors="coerce").astype(float)
        return df2.drop(columns=["t_num"])
    key = ["method","alpha","p_attack","t"]
    m2 = fix(metrics)
    o2 = fix(outcome)[key + ["verdict"]]
    return m2.merge(o2, on=key, how="left")

def combined_heatmap(mad_out: pd.DataFrame,
                     trust_out: pd.DataFrame,
                     alpha_val: float,
                     ax, title: str):
    """Draw 2×3×3 grid showing which of MAD/TRUST succeed per cell."""
    m = mad_out.query(f"alpha == {alpha_val}")
    t = trust_out.query(f"alpha == {alpha_val}")
    m_ok = m.assign(ok=lambda d: d.verdict != "Fail")
    t_ok = t.assign(ok=lambda d: d.verdict != "Fail")
    comb = (m_ok[["p_attack","t","ok"]]
            .merge(t_ok[["p_attack","t","ok"]],
                   on=["p_attack","t"],
                   suffixes=("_mad","_trust")))
    comb["cat"] = comb.apply(
        lambda r: "BOTH" if r.ok_mad and r.ok_trust
                  else "MAD" if r.ok_mad
                  else "TRUST" if r.ok_trust
                  else "NONE", axis=1)
    pivot = comb.pivot(index="p_attack", columns="t", values="cat")
    code_map = {"NONE":0, "TRUST":1, "MAD":2, "BOTH":3}
    pivot_code = pivot.replace(code_map).astype(int)
    cmap = ListedColormap(["#f0f0f0","#a6cee3","#1f78b4","#33a02c"])
    sns.heatmap(
        pivot_code,
        annot=pivot,
        fmt="",
        cmap=cmap,
        cbar=False,
        linewidths=.5,
        annot_kws={"size": 15},
        ax=ax
    )
    ax.set_xlabel("Tampering window $t$")
    ax.set_ylabel("$p_{\\rm attack}$")
    ax.set_title(title)

# ---------------------------------------------------------- 1. LOAD DATA
grid_m    = load_csv("grid_metrics.csv")
grid_out  = load_csv("grid_outcome.csv")
trust_m   = load_csv("trust_grid_metrics.csv")
trust_out = load_csv("trust_grid_outcome.csv")

# ---------------------------------------------------------- 2. PANEL A – Iterations
mad_iter = (merge_verdicts(grid_m, grid_out)
            .assign(method="MAD", iters=lambda d: d.iters_mean)
            [["method","iters","verdict"]])
trust_iter = (trust_m
              .assign(method="TRUST", iters=lambda d: d.iters_mean)
              [["method","iters","verdict"]])
panelA_df = pd.concat([mad_iter, trust_iter], ignore_index=True)

# ---------------------------------------------------------- 3. PANEL B – Filter activations
panelB_df = pd.concat([
    grid_m.assign(method="MAD",
                  t=lambda d: d.t.astype(str),
                  count=lambda d: d.corr_mean)[["method","t","count"]],
    trust_m.assign(method="TRUST",
                   t=lambda d: d.t.astype(str),
                   count=lambda d: d.trust_violations_mean)[["method","t","count"]],
], ignore_index=True)

# ---------------------------------------------------------- 4. PANELS C & D – Combined heat-maps
# use grid_out and trust_out for outcome verdicts
heat_classical = grid_out.query("alpha == 0")
heat_relaxed   = grid_out.query("alpha == 0.50")

# ----------------------------------------------------------- 5. PLOT ALL
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
(A, B), (C, D) = axes

# (A) Termination iterations with horizontal caps
sns.boxplot(
    data=panelA_df, x="method", y="iters", hue="verdict",
    showcaps=True, fliersize=2, ax=A
)
A.set_title("(A) Termination iterations")
A.set_xlabel("")
A.set_ylabel("Iterations")
A.legend_.set_title("Outcome")

# (B) Filter activations (log scale) with horizontal caps
sns.boxplot(
    data=panelB_df, x="method", y="count", hue="t",
    showcaps=True, fliersize=2, ax=B
)
B.set_yscale("log")
B.set_title("(B) Filter activations")
B.set_xlabel("")
B.set_ylabel("Count (log scale)")
B.legend_.set_title("Window t")

# (C) & (D) Combined MAD+TRUST heat-maps
combined_heatmap(grid_out, trust_out, 0.00, C, "(C) Classical (α = 0)")
combined_heatmap(grid_out, trust_out, 0.50, D, "(D) Relaxed (α = 0.50)")

fig.tight_layout()
fig.savefig(out, dpi=300)
print(f"saved → {out}")

