#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_graphs.py   ―   offline analytics (price ×100, no '%' texts)

What changed • 2025-05-05
────────────────────────────────────────────────────────────────────
✓ heat-map removed
✓ price labels no longer say “×100” or “%”
✓ price-convergence curve removed
✓ per-category scatter added when <=3 data points (fixes “all zeros” look)
✓ new box-plots:
      iterations_box_tamper.png
      price_box_method.png
      prim_box_method.png
      dual_box_method.png
      avgWeight_box_tamper.png
────────────────────────────────────────────────────────────────────
"""

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = "logs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ───────── plot helpers ───────── #

def _copy(df): return df.copy(deep=True)

def clamp_err(m, s):
    low = np.maximum(0, m - s)
    return [m - low, s]

def grouped_bar(df, x, g, y, ylab, ttl, fname):
    df=_copy(df); df[g]=df[g].astype(str)
    with pd.option_context('mode.chained_assignment', None):
        try: df[x]=df[x].astype(float)
        except: pass
    stat=df.groupby([x,g])[y].agg(['mean','std']).reset_index()
    mtab=stat.pivot(index=x,columns=g,values='mean')
    stab=stat.pivot(index=x,columns=g,values='std')
    X=np.arange(len(mtab)); bw=0.8/len(mtab.columns)
    fig,ax=plt.subplots(figsize=(8,6))
    for i,col in enumerate(mtab.columns):
        pos=X+(i-len(mtab.columns)/2)*bw+bw/2
        ax.bar(pos, mtab[col], yerr=clamp_err(mtab[col], stab[col]),
               width=bw, capsize=4, label=str(col))
    ax.set_xticks(X); ax.set_xticklabels(mtab.index.astype(str))
    ax.set_xlabel(x); ax.set_ylabel(ylab); ax.set_title(ttl)
    ax.set_ylim(bottom=0); ax.legend(title=g)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR,fname)); plt.close(fig)

def simple_bar(df,x,y,ylab,ttl,fname):
    df=_copy(df)
    with pd.option_context('mode.chained_assignment', None):
        try: df[x]=df[x].astype(float)
        except: pass
    tmp=df.groupby(x)[y].agg(['mean','std']).reset_index().sort_values(x)
    fig,ax=plt.subplots()
    ax.bar(tmp.index,tmp['mean'],yerr=clamp_err(tmp['mean'],tmp['std']),capsize=4)
    ax.set_xticks(tmp.index); ax.set_xticklabels(tmp[x].astype(str))
    ax.set_xlabel(x); ax.set_ylabel(ylab); ax.set_title(ttl); ax.set_ylim(bottom=0)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR,fname)); plt.close(fig)

def mean_curve(df,x,y,hue,ttl,fname):
    df=_copy(df); fig,ax=plt.subplots()
    for key,grp in df.groupby(hue):
        ln=grp.groupby(x)[y].agg(['mean','std']).reset_index()
        ax.plot(ln[x],ln['mean'],label=str(key))
        ax.fill_between(ln[x],ln['mean']-ln['std'],ln['mean']+ln['std'],alpha=.25)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(ttl); ax.legend(title=hue)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR,fname)); plt.close(fig)

def boxplots(df, x, y, ttl, fname):
    """
    Classic box-plots with safe category sorting and point overlay
    when a category has very few samples. Writes to logs/plots/<fname>.
    
    Parameters:
    - df: DataFrame containing the data
    - x: column name for categories (e.g., tampering_count or method)
    - y: column name for values to plot
    - ttl: plot title
    - fname: output filename (under PLOT_DIR)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Work on a deep copy to avoid SettingWithCopyWarning
    df = df.copy(deep=True)

    # Build a robust sort key: numbers first (by value), then strings
    def _key(v):
        if v is np.inf:
            return (2, np.inf)
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, str(v))

    # Determine category order
    cats = sorted(df[x].unique(), key=_key)

    # Gather data per category
    data = [df[df[x] == c][y].values for c in cats]

    # Create the boxplot
    fig, ax = plt.subplots()
    ax.boxplot(data, positions=np.arange(len(cats)), showfliers=False)

    # Overlay raw points if category has ≤ 3 samples
    for i, d in enumerate(data):
        if len(d) <= 3:
            ax.scatter(np.full(len(d), i), d, color="tab:orange", zorder=3)

    # Build clean labels, handling infinity and numeric vs. string
    labels = []
    for c in cats:
        if c is np.inf or (isinstance(c, float) and np.isinf(c)):
            labels.append("inf")
        elif isinstance(c, (int, float)):
            if float(c).is_integer():
                labels.append(str(int(c)))
            else:
                labels.append(str(c))
        else:
            labels.append(str(c))

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(ttl)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, fname))
    plt.close(fig)

# ───── mitigation-log parser ───── #

def parse_mit_log(path):
    c=W=D=0
    with open(path) as f:
        for ln in f:
            c+=1
            try:
                D+=float(ln.split("deviation=")[1].split(",")[0])
                W+=float(ln.split("weight=")[1].split(",")[0])
            except: pass
    return c,(W/c if c else 0),(D/c if c else 0)

# ───── per-run plots ───── #

def gen_iter_plots(csv_path):
    df=pd.read_csv(csv_path); df["price"]=df["avg_price"]*100
    tag=os.path.splitext(os.path.basename(csv_path))[0][5:]

    def sv(fig,stem): plt.tight_layout(); fig.savefig(f"{PLOT_DIR}/{stem}_{tag}.png"); plt.close(fig)

    fig,ax=plt.subplots(); ax.plot(df["iter"],df["SW"])
    ax.set_xlabel("Iteration"); ax.set_ylabel("SW"); ax.set_title("SW vs Iter"); sv(fig,"SW")

    fig,ax=plt.subplots(); ax.plot(df["iter"],df["price"])
    ax.set_xlabel("Iteration"); ax.set_ylabel("Price"); ax.set_title("Price vs Iter"); sv(fig,"Price")

    fig,ax=plt.subplots(); ax.plot(df["iter"],df["prim"],label="prim"); ax.plot(df["iter"],df["dual"],label="dual")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Residual"); ax.legend(); ax.set_title("Primal & Dual vs Iter"); sv(fig,"Residuals")

def iter_df(csv_path):
    df=pd.read_csv(csv_path); df["price"]=df["avg_price"]*100
    tag=os.path.splitext(os.path.basename(csv_path))[0][5:]
    pt=dict(re.findall(r"([a-z]+)([^_]+)",tag))
    df["tag"]=tag; df["method"]=pt.get("method","")
    df["tamper"]=np.inf if pt.get("t")=="inf" else float(pt.get("t",0))
    return df

# ───── local-conv parser ───── #

def local_conv_items(path):
    out=[]; rg=r"Sub-graph\s+(\[.*?\])\s+locally converged at iter (\d+)"
    for ln in open(path):
        m=re.search(rg,ln); 
        if m: out.append((m.group(1).replace(" ",""),int(m.group(2))))
    return out

# ─────────────────── MAIN ─────────────────── #

def main():
    mit="logs/mitigation"; os.makedirs(mit,exist_ok=True)
    pat=r"log_(method\d)(?:_alpha([\d\.]+))?_prob([\d\.]+)_mult([\d\.]+)_t(\S+)\.txt"

    rows=[]
    for p in glob.glob(f"{mit}/log_*.txt"):
        m=re.match(pat,os.path.basename(p)); 
        if not m: continue
        meth,a_s,pr_s,mu_s,t_s=m.groups()
        alpha=float(a_s) if a_s else 0; prob=float(pr_s); mult=float(mu_s)
        tam=np.inf if t_s=="inf" else float(t_s)
        cnt,w,d=parse_mit_log(p)
        rows.append(dict(method=meth,alpha=alpha,attack_prob=prob,
                         multiplier_upper=mult,tampering_count=tam,
                         iterations=cnt,mitigation_count=cnt,
                         avg_weight=w,avg_deviation=d))
    df=pd.DataFrame(rows); df.to_csv("simulation_results.csv",index=False)
    print(f"simulation_results.csv saved ({len(df)})")

    if df.empty: print("No data."); return

    # classic plots
    df2,df1=df[df.method=="method2"],df[df.method=="method1"]
    if not df2.empty:
        grouped_bar(df2,"tampering_count","alpha","iterations",
                    "Avg Iterations","Iterations vs Tampering (Relaxed)",
                    "iter_vs_tamper_relaxed.png")
        grouped_bar(df2,"tampering_count","alpha","mitigation_count",
                    "Avg Mitigations","Mitigations vs Tampering (Relaxed)",
                    "mitig_vs_tamper_relaxed.png")
    if not df1.empty:
        simple_bar(df1,"tampering_count","iterations","Avg Iterations",
                   "Iterations vs Tampering (Classic)","iter_vs_tamper_classic.png")

    # per-run & extra boxes
    it_csv=glob.glob("logs/iter_stats/iter_*.csv")
    if it_csv:
        for p in it_csv: gen_iter_plots(p)
        all_it=pd.concat([iter_df(p) for p in it_csv],ignore_index=True)
        last=all_it.sort_values("iter").groupby("tag").tail(1)

        boxplots(last,"tamper","SW","Final SW by Tampering","finalSW_box_tamper.png")
        mean_curve(all_it,"iter","SW","method",
                   "SW Convergence (Mean±Std)","SW_conv_meanstd.png")

        # Comparison boxes
        boxplots(df,"tampering_count","iterations",
                 "Iterations by Tampering","iterations_box_tamper.png")
        boxplots(last,"method","price","Final Price by Method","price_box_method.png")
        boxplots(last,"method","prim","Prim Final by Method","prim_box_method.png")
        boxplots(last,"method","dual","Dual Final by Method","dual_box_method.png")
        boxplots(df,"tampering_count","avg_weight",
                 "Avg Weight by Tampering","avgWeight_box_tamper.png")

    # local-conv
    lc_rg=r"local_conv_(method\d)(?:_alpha([\d\.]+))?_prob([\d\.]+)_mult([\d\.]+)_t(\S+)\.log"
    lc_rows=[]
    for p in glob.glob("logs/local_conv/local_conv_*.log"):
        m=re.match(lc_rg,os.path.basename(p))
        if not m: continue
        meth,a_s,pr_s,mu_s,t_s=m.groups()
        alpha=float(a_s) if a_s else 0; tam=np.inf if t_s=="inf" else float(t_s)
        for sub,it in local_conv_items(p):
            lc_rows.append(dict(method=meth,alpha=alpha,tampering=tam,
                                subgraph=sub,conv_iter=it))
    if lc_rows:
        lcd=pd.DataFrame(lc_rows)
        grouped_bar(lcd,"tampering","method","conv_iter",
                    "Avg Local-Conv Iter","LocalConv Iter vs Tampering",
                    "localConv_iter_vs_tamper.png")

    # summary
    summary=(df.groupby(["method","alpha","attack_prob","tampering_count"])
               .agg(iter_mean=("iterations","mean"),
                    iter_std=("iterations","std")).reset_index())
    summary.to_csv("simulation_summary_table.csv",index=False)
    print("simulation_summary_table.csv saved")

if __name__=="__main__":
    main()
