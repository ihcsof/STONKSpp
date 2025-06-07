#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_graphs.py  –  analytics + plotting + binary deep-dive
"""

import sys, os, re, glob, gzip, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

TAU = 2.5

PLOT_DIR = "logs/plots";       os.makedirs(PLOT_DIR, exist_ok=True)
MIT_DIR  = "logs/mitigation";  os.makedirs(MIT_DIR,  exist_ok=True)
ITER_DIR = "logs/iter_stats"
LC_DIR   = "logs/local_conv"
BIN_DIR  = "logs/binaries"

PLOT_INFO: list[tuple[str, str]] = []

def _add_caption(fname: str, desc: str | None):
    if desc:
        PLOT_INFO.append((fname, desc))

def _copy(df):
    return df.copy(deep=True)

def clamp_err(m, s, upper=1000):
    low = np.maximum(0, m - s)
    if upper is None:
        return [m - low, s]
    high = np.minimum(upper, m + s) - m
    return [m - low, high]

def grouped_bar(df, x, g, y, ylab, ttl, fname, *, desc: str | None = None):
    df = _copy(df); df[g] = df[g].astype(str)
    with pd.option_context("mode.chained_assignment", None):
        try:
            df[x] = df[x].astype(float)
        except Exception:
            pass
    tab = df.groupby([x, g])[y].agg(['mean', 'std']).reset_index()
    mtab = tab.pivot(index=x, columns=g, values='mean')
    stab = tab.pivot(index=x, columns=g, values='std')
    X = np.arange(len(mtab)); bw = 0.8/len(mtab.columns)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, col in enumerate(mtab.columns):
        pos = X + (i-len(mtab.columns)/2)*bw + bw/2
        ax.bar(pos, mtab[col], yerr=clamp_err(mtab[col], stab[col]),
               width=bw, capsize=4, label=str(col))
    ax.set_xticks(X); ax.set_xticklabels(mtab.index.astype(str))
    ax.set_xlabel(x); ax.set_ylabel(ylab); ax.set_title(ttl)
    ax.set_ylim(bottom=0); ax.legend(title=g)
    plt.tight_layout()
    out_path = f"{PLOT_DIR}/{fname}"
    plt.savefig(out_path)
    plt.close(fig)
    _add_caption(fname, desc)

def simple_bar(df, x, y, ylab, ttl, fname, *, desc: str | None = None):
    df = _copy(df)
    with pd.option_context("mode.chained_assignment", None):
        try:
            df[x] = df[x].astype(float)
        except Exception:
            pass
    tab = df.groupby(x)[y].agg(['mean', 'std']).reset_index().sort_values(x)
    fig, ax = plt.subplots()
    ax.bar(tab.index, tab['mean'], yerr=clamp_err(tab['mean'], tab['std']), capsize=4)
    ax.set_xticks(tab.index); ax.set_xticklabels(tab[x].astype(str))
    ax.set_xlabel(x); ax.set_ylabel(ylab); ax.set_title(ttl); ax.set_ylim(bottom=0)
    plt.tight_layout()
    out_path = f"{PLOT_DIR}/{fname}"
    plt.savefig(out_path)
    plt.close(fig)
    _add_caption(fname, desc)

def mean_curve(df, x, y, hue, ttl, fname, *, desc: str | None = None):
    df = _copy(df); fig, ax = plt.subplots()
    for k, g in df.groupby(hue):
        ln = g.groupby(x)[y].agg(['mean', 'std']).reset_index()
        ax.plot(ln[x], ln['mean'], label=str(k))
        ax.fill_between(ln[x], ln['mean']-ln['std'], ln['mean']+ln['std'], alpha=.25)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(ttl); ax.legend(title=hue)
    plt.tight_layout()
    out_path = f"{PLOT_DIR}/{fname}"
    plt.savefig(out_path)
    plt.close(fig)
    _add_caption(fname, desc)

def boxplots(df, x, y, ttl, fname, *, desc: str | None = None,
             log_if_span: bool = True, jitter: float = 0.08, show_n: bool = True):
    import random
    d = df[[x, y]].dropna().copy()
    def sk(v):
        if v is np.inf or (isinstance(v, float) and np.isinf(v)):
            return (2, np.inf)
        try:
            return (0, float(v))
        except Exception:
            return (1, str(v))
    cats = sorted(d[x].unique(), key=sk)
    data, labels, consts, pts, counts = [], [], [], [], []
    for i, c in enumerate(cats):
        arr = d[d[x] == c][y].values
        if arr.size == 0:
            continue
        counts.append(arr.size)
        pts.extend([(i + random.uniform(-jitter, jitter), v) for v in arr])
        if np.allclose(arr, arr[0]):
            consts.append((i, arr[0]))
            data.append([arr[0], arr[0], arr[0]])
        else:
            data.append(arr)
        if c is np.inf or (isinstance(c, float) and np.isinf(c)):
            labels.append("inf")
        elif isinstance(c, (int, float)) and float(c).is_integer():
            labels.append(str(int(c)))
        else:
            labels.append(str(c))
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, positions=range(len(data)),
                    showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("none")
        patch.set_edgecolor("black")
    if pts:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=8, alpha=0.4, color="grey", zorder=0)
    for pos, val in consts:
        ax.hlines(val, pos-0.3, pos+0.3, color="black", lw=1.5)
        ax.plot(pos, val, "o", color="black")
    if show_n:
        for pos, n in enumerate(counts):
            ax.text(pos, ax.get_ylim()[0], f" n={n}", ha="center",
                    va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(ttl)
    if log_if_span and not d[y].empty:
        span = d[y].max() / max(d[y].min(), 1e-12)
        if span > 1e3:
            ax.set_yscale("log")
    plt.tight_layout()
    out_path = f"{PLOT_DIR}/{fname}"
    plt.savefig(out_path)
    plt.close(fig)
    _add_caption(fname, desc)

def overlaid_hists(df, x, cat, bins, ttl, fname, *, alpha=0.35, desc: str | None = None):
    df = _copy(df)
    fig, ax = plt.subplots()
    for c, g in df.groupby(cat):
        ax.hist(g[x], bins=bins, alpha=alpha, label=str(c), density=True)
    ax.set_xlabel(x); ax.set_ylabel("Density"); ax.set_title(ttl)
    ax.legend(title=cat); plt.tight_layout()
    out_path = f"{PLOT_DIR}/{fname}"
    plt.savefig(out_path)
    plt.close(fig)
    _add_caption(fname, desc)

def cdf_plot(df, x, cat, ttl, fname, *, desc: str | None = None):
    df = _copy(df); fig, ax = plt.subplots()
    for c, g in df.groupby(cat):
        data = np.sort(g[x].values)
        yvals = np.arange(1, len(data)+1) / float(len(data))
        ax.step(data, yvals, where="post", label=str(c))
    ax.set_xlabel(x); ax.set_ylabel("CDF"); ax.set_title(ttl)
    ax.legend(title=cat); plt.tight_layout()
    out_path = f"{PLOT_DIR}/{fname}"
    plt.savefig(out_path)
    plt.close(fig)
    _add_caption(fname, desc)

def build_sim_results() -> pd.DataFrame:
    iter_rows = []
    for p in glob.glob(f"{ITER_DIR}/iter_*.csv"):
        fn = os.path.splitext(os.path.basename(p))[0]
        tag_run = fn[5:]
        if "_run" in tag_run:
            tag, run_s = tag_run.rsplit("_run", 1)
            run = int(run_s)
        else:
            tag, run = tag_run, 0
        iters = int(pd.read_csv(p, usecols=["iter"])["iter"].max())
        iter_rows.append((tag, run, iters))
    iter_df = pd.DataFrame(iter_rows, columns=["tag", "run", "iterations"])
    if iter_df.empty:
        raise RuntimeError("No iter_* files in logs/iter_stats.")
    mit_rows = []
    mit_pat = r"log_(.*?)(?:_run(\d+))?\.txt$"
    for p in glob.glob(f"{MIT_DIR}/log_*.txt"):
        m = re.match(mit_pat, os.path.basename(p))
        if not m:
            continue
        tag, run_s = m.groups(); run = int(run_s) if run_s else 0
        cnt_lines = flag_cnt = w_sum = dev_sum = 0
        for ln in open(p, encoding="utf-8", errors="ignore"):
            if "[Flag]" in ln:
                flag_cnt += 1
            if "deviation=" in ln and "weight=" in ln:
                cnt_lines += 1
                try:
                    dev_sum += float(ln.split("deviation=")[1].split(',')[0])
                    w_sum   += float(ln.split("weight=")[1].split(',')[0])
                except Exception:
                    pass
        segs = {s[:4]: s for s in tag.split('_') if s[:4] in ("meth", "alph", "prob", "mult", "t")}
        meth = segs.get("meth", "")
        alpha = float(segs.get("alph", "0")[4:] or 0)
        prob = float(segs.get("prob", "0")[4:] or 0)
        mult = float(segs.get("mult", "0")[4:] or 0)
        tam_s = segs.get("t", "0")[1:]
        tam = np.inf if tam_s == "inf" else float(tam_s or 0)
        mit_rows.append(dict(
            tag=tag, run=run,
            method=meth, alpha=alpha, attack_prob=prob,
            multiplier=mult, tampering=tam,
            mitigation_count=cnt_lines, flag_count=flag_cnt,
            avg_weight=w_sum/cnt_lines if cnt_lines else np.nan,
            avg_deviation=dev_sum/cnt_lines if cnt_lines else np.nan
        ))
    mit_df = pd.DataFrame(mit_rows)
    merged = (iter_df.merge(mit_df, on=["tag", "run"], how="left")
                      .fillna({"mitigation_count": 0, "flag_count": 0}))
    merged.to_csv("simulation_results.csv", index=False)
    return merged

def main():
    df = build_sim_results()
    if df.empty:
        return
    df["tampering"] = df["tampering"].replace({np.inf: "inf"}).astype(str)
    for col in ("multiplier", "attack_prob"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df2 = df[df.method == "method2"]
    df1 = df[df.method == "method1"]
    if not df2.empty:
        grouped_bar(df2, "tampering", "alpha", "iterations",
                    "Avg Iterations",
                    "Iterations vs Tampering (Relaxed ADMM)",
                    "iterations_vs_tamperingcount_relaxed.pdf",
                    desc="Relaxed ADMM")
    if not df1.empty:
        simple_bar(df1, "tampering", "iterations",
                   "Avg Iterations",
                   "Iterations vs Tampering (Classical ADMM)",
                   "iterations_vs_tamperingcount_classical.pdf",
                   desc="Classical ADMM")
    if not df.empty:
        grouped_bar(df, "tampering", "method",
                    "flag_count",
                    "Average Full-Flag Events",
                    f"Honest-only Flags (τ={TAU})",
                    "flags_vs_tampering.pdf",
                    desc="Full partner discards")
        grouped_bar(df, "attack_prob", "tampering",
                    "flag_count",
                    "Average Full-Flag Events",
                    f"Flags vs Attack-prob (τ={TAU})",
                    "flags_vs_attackprob.pdf",
                    desc="Flags vs attack probability")
    with open(f"{PLOT_DIR}/plot_explanations.txt", "w") as fh:
        for fname, txt in PLOT_INFO:
            fh.write(f"{fname} : {txt}\n")

if __name__ == "__main__":
    main()
