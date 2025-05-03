#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_graphs.py  –  offline analytics & visualisation

Run *after* batch_run.py.  Reads everything in logs/ and produces:

• Original grouped-bar / bar charts (iterations & mitigations)
• Extra plots (heat-maps, convergence curves, violins, etc.)
• Per-run iteration figures from logs/iter_stats/*.csv
• Local-convergence histograms & grouped bars (logs/local_conv)
• simulation_results.csv & simulation_summary_table.csv
"""

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless backend
import matplotlib.pyplot as plt

PLOT_DIR = "logs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ───────────────────────────── Plot helpers ────────────────────────── #

def _copy(df): return df.copy(deep=True)

def clamp_err(m, s):
    low = np.maximum(0, m - s)
    return [m - low, s]

def grouped_bar(df, x, g, y, ylab, ttl, fname):
    df = _copy(df); df[g] = df[g].astype(str)
    with pd.option_context('mode.chained_assignment', None):
        try: df[x] = df[x].astype(float)
        except: pass
    tab = (df.groupby([x, g])[y].agg(['mean', 'std']).reset_index())
    mtab = tab.pivot(index=x, columns=g, values='mean').sort_index()
    stab = tab.pivot(index=x, columns=g, values='std').sort_index()

    X = np.arange(len(mtab.index)); bw = 0.8 / len(mtab.columns)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, col in enumerate(mtab.columns):
        pos = X + (i - len(mtab.columns)/2)*bw + bw/2
        ax.bar(pos, mtab[col], yerr=clamp_err(mtab[col], stab[col]),
               width=bw, capsize=4, label=str(col))
    ax.set_xticks(X); ax.set_xticklabels(mtab.index.astype(str))
    ax.set_xlabel(x); ax.set_ylabel(ylab); ax.set_title(ttl)
    ax.set_ylim(bottom=0); ax.legend(title=g)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, fname)); plt.close(fig)

def simple_bar(df, x, y, ylab, ttl, fname):
    df = _copy(df)
    with pd.option_context('mode.chained_assignment', None):
        try: df[x] = df[x].astype(float)
        except: pass
    tmp = (df.groupby(x)[y].agg(['mean', 'std']).reset_index()
             .sort_values(x))
    fig, ax = plt.subplots()
    ax.bar(tmp.index, tmp['mean'],
           yerr=clamp_err(tmp['mean'], tmp['std']), capsize=4)
    ax.set_xticks(tmp.index); ax.set_xticklabels(tmp[x].astype(str))
    ax.set_xlabel(x); ax.set_ylabel(ylab); ax.set_title(ttl)
    ax.set_ylim(bottom=0)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, fname)); plt.close(fig)

def mean_curve(df, x, y, hue, ttl, fname):
    df = _copy(df)
    fig, ax = plt.subplots()
    for key, grp in df.groupby(hue):
        ln = grp.groupby(x)[y].agg(['mean', 'std']).reset_index()
        ax.plot(ln[x], ln['mean'], label=str(key))
        ax.fill_between(ln[x], ln['mean']-ln['std'], ln['mean']+ln['std'], alpha=.25)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(ttl); ax.legend(title=hue)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, fname)); plt.close(fig)

def violins(df, x, y, ttl, fname):
    df = _copy(df)
    cats = sorted(df[x].unique())
    data = [df[df[x]==c][y].values for c in cats]
    fig, ax = plt.subplots()
    ax.boxplot(data, positions=np.arange(len(cats)), showfliers=False)
    for i, d in enumerate(data):
        ax.scatter(np.random.normal(i, 0.04, len(d)), d, s=8, alpha=0.4)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([str(c) for c in cats])
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(ttl)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, fname)); plt.close(fig)

def heatmap(df, x, y, z, ttl, fname):
    df = _copy(df)
    pv = (df.pivot_table(index=y, columns=x, values=z, aggfunc='mean')
            .sort_index().sort_index(axis=1))
    fig, ax = plt.subplots()
    im = ax.imshow(pv.values, aspect='auto', origin='lower')
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
    ax.set_yticks(range(len(pv.index)));   ax.set_yticklabels(pv.index)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(ttl)
    fig.colorbar(im).set_label(f"mean {z}")
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, fname)); plt.close(fig)

# ─────────────────── mitigation-log parser  ───────────────────────── #

def parse_mit_log(path):
    c = tW = tD = 0
    with open(path, 'r') as f:
        for ln in f:
            c += 1
            try:
                tD += float(ln.split("deviation=")[1].split(",")[0])
                tW += float(ln.split("weight=")[1].split(",")[0])
            except: pass
    return c, (tW/c if c else 0), (tD/c if c else 0)

# ─────────────────── per-run iteration figures  ───────────────────── #

def generate_iteration_plots(csv_path, outdir=PLOT_DIR):
    df = pd.read_csv(csv_path)
    tag = os.path.splitext(os.path.basename(csv_path))[0][5:]

    def save(fig, name):
        plt.tight_layout()
        fig.savefig(f"{outdir}/{name}_{tag}.png")
        plt.close(fig)

    fig, ax = plt.subplots(); ax.plot(df["iter"], df["SW"])
    ax.set_xlabel("iter"); ax.set_ylabel("SW"); ax.set_title("SW vs iter")
    save(fig, "SW")

    fig, ax = plt.subplots(); ax.plot(df["iter"], df["avg_price"])
    ax.set_xlabel("iter"); ax.set_ylabel("price"); ax.set_title("Price vs iter")
    save(fig, "Price")

    fig, ax = plt.subplots()
    ax.plot(df["iter"], df["prim"], label="prim")
    ax.plot(df["iter"], df["dual"], label="dual")
    ax.set_xlabel("iter"); ax.set_ylabel("residual"); ax.legend()
    ax.set_title("Primal & dual vs iter")
    save(fig, "Residuals")

    fig, ax = plt.subplots(); ax.scatter(df["avg_price"], df["SW"])
    ax.set_xlabel("price"); ax.set_ylabel("SW"); ax.set_title("SW vs price")
    save(fig, "SWvsPrice")

    fig, ax = plt.subplots(); ax.hist(df["avg_price"], bins=20)
    ax.set_xlabel("price"); ax.set_ylabel("freq"); ax.set_title("Price histogram")
    save(fig, "PriceHist")

def iter_stats_df(csv_path):
    df = pd.read_csv(csv_path)
    tag = os.path.splitext(os.path.basename(csv_path))[0][5:]
    parts = dict(re.findall(r"([a-z]+)([^_]+)", tag))
    df["tag"]     = tag
    df["method"]  = parts.get("method", "")
    df["prob"]    = float(parts.get("prob", 0))
    df["mult"]    = float(parts.get("mult", 0))
    df["tamper"]  = np.inf if parts.get("t")=="inf" else float(parts.get("t",0))
    return df

# ───────────────────────── Local-conv parser ─────────────────────── #

def local_conv_items(path):
    out=[]; pat=r"Sub-graph\s+(\[.*?\])\s+locally converged at iter (\d+)"
    for ln in open(path):
        m=re.search(pat,ln)
        if m: out.append((m.group(1).replace(" ",""), int(m.group(2))))
    return out

# ───────────────────────────────── MAIN ───────────────────────────── #

def main():
    # 1) mitigation logs  ➜  df -------------------------------------
    mit_dir="logs/mitigation"; os.makedirs(mit_dir,exist_ok=True)
    file_pat=(r"log_(method\d)(?:_alpha([\d\.]+))?_prob([\d\.]+)_mult([\d\.]+)_t(\S+)\.txt")
    rows=[]
    for path in glob.glob(os.path.join(mit_dir,"log_*.txt")):
        m=re.match(file_pat, os.path.basename(path))
        if not m: continue
        method,a_s,pr_s,mu_s,t_s = m.groups()
        alpha=float(a_s) if a_s else 0
        prob=float(pr_s); mult=float(mu_s)
        tam=np.inf if t_s=='inf' else float(t_s)
        cnt,w,d=parse_mit_log(path)
        rows.append(dict(method=method, alpha=alpha, attack_prob=prob,
                         multiplier_upper=mult, tampering_count=tam,
                         iterations=cnt, mitigation_count=cnt,
                         avg_weight=w, avg_deviation=d))
    df=pd.DataFrame(rows)
    df.to_csv("simulation_results.csv", index=False)
    print(f"simulation_results.csv saved ({len(df)} rows)")

    if df.empty:
        print("No mitigation data found – aborting generation.")
        return

    # 2) classic plots ----------------------------------------------
    df2, df1 = df[df.method=="method2"], df[df.method=="method1"]
    if not df2.empty:
        grouped_bar(df2,"tampering_count","alpha","iterations",
                    "Avg iterations","Iter vs tamper (relaxed)",
                    "iter_vs_tamper_relaxed.png")
        grouped_bar(df2,"tampering_count","alpha","mitigation_count",
                    "Avg mitigations","Mitig vs tamper (relaxed)",
                    "mitig_vs_tamper_relaxed.png")
        heatmap(df2,"attack_prob","tampering_count","iterations",
                "Mean iterations (method2)","heat_iter_prob_tamper.png")
    if not df1.empty:
        simple_bar(df1,"tampering_count","iterations",
                   "Avg iterations","Iter vs tamper (classic)",
                   "iter_vs_tamper_classic.png")

    # 3) per-run iteration plots & richer analytics -----------------
    it_csv = glob.glob("logs/iter_stats/iter_*.csv")
    if it_csv:
        for p in it_csv:
            generate_iteration_plots(p)
        it_all = pd.concat([iter_stats_df(p) for p in it_csv], ignore_index=True)
        last   = it_all.sort_values("iter").groupby("tag").tail(1)
        violins(last,"tamper","SW","Final SW by tamper","finalSW_violin_tamper.png")
        mean_curve(it_all,"iter","SW","method",
                   "SW convergence (mean±std)","SW_conv_meanstd.png")
        it_all["tamper_b"]=np.where(it_all.tamper==np.inf,"inf",it_all.tamper)
        mean_curve(it_all,"iter","avg_price","tamper_b",
                   "Price conv by tamper","price_conv_tamper.png")
        for c in ("prim","dual"):
            violins(last,"method",c,f"{c} final by method",
                    f"{c}_final_violin_method.png")

    # 4) local-convergence analytics --------------------------------
    lc_rows=[]; lc_pat=(r"local_conv_(method\d)(?:_alpha([\d\.]+))?_prob([\d\.]+)_mult([\d\.]+)_t(\S+)\.log")
    for p in glob.glob("logs/local_conv/local_conv_*.log"):
        m=re.match(lc_pat, os.path.basename(p))
        if not m: continue
        meth,a_s,pr_s,mu_s,t_s=m.groups()
        alpha=float(a_s) if a_s else 0
        prob=float(pr_s); mult=float(mu_s)
        tam=np.inf if t_s=='inf' else float(t_s)
        for sub,it in local_conv_items(p):
            lc_rows.append(dict(method=meth, alpha=alpha, attack_prob=prob,
                                multiplier=mult, tampering=tam,
                                subgraph=sub, conv_iter=it))
    if lc_rows:
        lcd=pd.DataFrame(lc_rows)
        grouped_bar(lcd,"tampering","method","conv_iter",
                    "Avg local-conv iter","localConv vs tamper",
                    "localConv_iter_vs_tamper.png")
        fig, ax = plt.subplots(); ax.hist(lcd.conv_iter, bins=30)
        ax.set_xlabel("iter"); ax.set_ylabel("count")
        ax.set_title("Sub-graph local-conv histogram")
        plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR,"localConv_hist.png")); plt.close(fig)

    # 5) summary pivot ----------------------------------------------
    summary=(df.groupby(["method","alpha","attack_prob","tampering_count"])
               .agg(iter_mean=("iterations","mean"),
                    iter_std=("iterations","std"),
                    mitig_mean=("mitigation_count","mean"),
                    mitig_std=("mitigation_count","std"))
               .reset_index())
    summary.to_csv("simulation_summary_table.csv", index=False)
    print("simulation_summary_table.csv saved")

if __name__ == "__main__":
    main()
