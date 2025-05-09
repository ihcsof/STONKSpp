#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_graphs.py  –  analytics + plotting + binary deep-dive
2025-05-10  (extended visual suite – fixed final_SW column name)

 • simulation_results.csv       ← mitigation logs
 • simulation_summary_table.csv ← pivot of means/stds
 • binary_summary.csv           ← ≥1000-iter binaries with residual history
 • PNG figures → logs/plots/
"""

import os, re, glob, gzip, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib;  matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# ────────── directories ──────────
PLOT_DIR = "logs/plots";       os.makedirs(PLOT_DIR,    exist_ok=True)
MIT_DIR  = "logs/mitigation";  os.makedirs(MIT_DIR,     exist_ok=True)
ITER_DIR = "logs/iter_stats"
LC_DIR   = "logs/local_conv"
BIN_DIR  = "logs/binaries"

# ═══════════════════ plotting helpers ═══════════════════
def _copy(df): return df.copy(deep=True)

def clamp_err(m, s):
    low = np.maximum(0, m - s)
    return [m - low, s]

def grouped_bar(df, x, g, y, ylab, ttl, fname):
    df = _copy(df); df[g] = df[g].astype(str)
    with pd.option_context("mode.chained_assignment", None):
        try: df[x] = df[x].astype(float)
        except: pass
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
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/{fname}"); plt.close(fig)

def simple_bar(df, x, y, ylab, ttl, fname):
    df = _copy(df)
    with pd.option_context("mode.chained_assignment", None):
        try: df[x] = df[x].astype(float)
        except: pass
    tab = df.groupby(x)[y].agg(['mean', 'std']).reset_index().sort_values(x)
    fig, ax = plt.subplots()
    ax.bar(tab.index, tab['mean'],
           yerr=clamp_err(tab['mean'], tab['std']), capsize=4)
    ax.set_xticks(tab.index); ax.set_xticklabels(tab[x].astype(str))
    ax.set_xlabel(x); ax.set_ylabel(ylab); ax.set_title(ttl); ax.set_ylim(bottom=0)
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/{fname}"); plt.close(fig)

def mean_curve(df, x, y, hue, ttl, fname):
    df = _copy(df); fig, ax = plt.subplots()
    for k, g in df.groupby(hue):
        ln = g.groupby(x)[y].agg(['mean', 'std']).reset_index()
        ax.plot(ln[x], ln['mean'], label=str(k))
        ax.fill_between(ln[x], ln['mean']-ln['std'], ln['mean']+ln['std'], alpha=.25)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(ttl); ax.legend(title=hue)
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/{fname}"); plt.close(fig)

def boxplots(df, x, y, ttl, fname):
    df = _copy(df)
    def sk(v):
        if v is np.inf or (isinstance(v, float) and np.isinf(v)):
            return (2, np.inf)
        try:  return (0, float(v))
        except: return (1, str(v))
    cats = sorted(df[x].unique(), key=sk)
    data = [df[df[x] == c][y].values for c in cats]

    fig, ax = plt.subplots()
    ax.boxplot(data, positions=np.arange(len(cats)), showfliers=False)
    for i, d in enumerate(data):
        if len(d) <= 3:
            ax.scatter(np.full(len(d), i), d, color="tab:orange", zorder=3)
    labels = []
    for c in cats:
        if c is np.inf or (isinstance(c, float) and np.isinf(c)):
            labels.append("inf")
        elif isinstance(c, (int, float)) and float(c).is_integer():
            labels.append(str(int(c)))
        else:
            labels.append(str(c))
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(labels)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(ttl)
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/{fname}"); plt.close(fig)

def overlaid_hists(df, x, cat, bins, ttl, fname, alpha=0.35):
    df = _copy(df)
    fig, ax = plt.subplots()
    for c, g in df.groupby(cat):
        ax.hist(g[x], bins=bins, alpha=alpha, label=str(c), density=True)
    ax.set_xlabel(x); ax.set_ylabel("Density"); ax.set_title(ttl)
    ax.legend(title=cat); plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}"); plt.close(fig)

def cdf_plot(df, x, cat, ttl, fname):
    df = _copy(df); fig, ax = plt.subplots()
    for c, g in df.groupby(cat):
        data = np.sort(g[x].values)
        yvals = np.arange(1, len(data)+1) / float(len(data))
        ax.step(data, yvals, where="post", label=str(c))
    ax.set_xlabel(x); ax.set_ylabel("CDF"); ax.set_title(ttl)
    ax.legend(title=cat); plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}"); plt.close(fig)

# ═══════════════════ per-run iteration PNGs (unchanged) ════════════════════
def gen_iter_plots(csv_p):
    df = pd.read_csv(csv_p); df["Price"] = df["avg_price"]*100
    tag = os.path.splitext(os.path.basename(csv_p))[0][5:]
    def sv(fig, stem):
        plt.tight_layout(); fig.savefig(f"{PLOT_DIR}/{stem}_{tag}.png"); plt.close(fig)
    for col, ttl in [("SW", "SW vs Iter"), ("Price", "Price vs Iter")]:
        fig, ax = plt.subplots(); ax.plot(df["iter"], df[col])
        ax.set_xlabel("Iteration"); ax.set_ylabel(col); ax.set_title(ttl); sv(fig, col)
    fig, ax = plt.subplots(); ax.plot(df["iter"], df["prim"], label="prim")
    ax.plot(df["iter"], df["dual"], label="dual")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Residual"); ax.legend()
    ax.set_title("Primal & Dual vs Iter"); sv(fig, "Residuals")

def iter_df(csv_p):
    df = pd.read_csv(csv_p); df["Price"] = df["avg_price"]*100
    tag = os.path.splitext(os.path.basename(csv_p))[0][5:]
    parts = dict(re.findall(r"([a-z]+)([^_]+)", tag))
    df["tag"] = tag
    df["method"] = parts.get("method","")
    df["tamper"] = np.inf if parts.get("t")=="inf" else float(parts.get("t",0))
    return df

# ═════════════════ residual-history extraction + classify (unchanged) ══════
def _norm_progress(obj):
    import numpy as _np, pandas as _pd
    rows=[]
    if isinstance(obj,(list,tuple)) and obj and isinstance(obj[0],(list,tuple,_np.ndarray)):
        for i,r in enumerate(obj):
            r=list(r); rows.append(tuple([i]+r) if len(r)==4 else tuple(r[:5]))
    elif isinstance(obj,_np.ndarray) and obj.ndim==2 and obj.shape[1]>=4:
        for i,r in enumerate(obj):
            r=r.tolist(); rows.append(tuple([i]+r) if len(r)==4 else tuple(r[:5]))
    elif isinstance(obj,_pd.DataFrame) and obj.shape[1]>=4:
        for i,(_,r) in enumerate(obj.iterrows()):
            r=r.values.tolist(); rows.append(tuple([i]+r) if len(r)==4 else tuple(r[:5]))
    return rows

KEYS_TRY=("prog_arr","progress","opti_progress","history","iter_list","progress_array")

def extract_progress(bd):
    if isinstance(bd,dict):
        for k in KEYS_TRY:
            if k in bd:
                pts=_norm_progress(bd[k])
                if pts:
                    return pts
    def _shallow(v): return _norm_progress(v)
    if isinstance(bd,dict):
        for v in bd.values():
            pts=_shallow(v)
            if pts:
                return pts
    elif isinstance(bd,(list,tuple)):
        for v in bd:
            pts=_shallow(v)
            if pts:
                return pts
    if isinstance(bd,dict):
        kmap={k.lower():k for k in bd}
        req=["sw","price","prim","dual"]
        if all(any(r in k for k in kmap) for r in req):
            try:
                sw   = np.asarray(bd[next(k for k in kmap if "sw" in k)],float)
                prc  = np.asarray(bd[next(k for k in kmap if "price" in k)],float)
                prim = np.asarray(bd[next(k for k in kmap if "prim" in k)],float)
                dual = np.asarray(bd[next(k for k in kmap if "dual" in k)],float)
                if len({len(sw),len(prc),len(prim),len(dual)})==1:
                    return [(i,sw[i],prc[i],prim[i],dual[i]) for i in range(len(sw))]
            except Exception:
                pass
    return []

def classify_run(prog, eps=1e-3):
    prim=np.asarray([p[3] for p in prog],float)
    dual=np.asarray([p[4] for p in prog],float)
    fp,fd,fsw = prim[-1], dual[-1], prog[-1][1]
    sp=sd=np.nan
    if len(prog)>=3:
        win=min(100,len(prog)-1)
        sp=np.polyfit(range(win),prim[-win:],1)[0]
        sd=np.polyfit(range(win),dual[-win:],1)[0]
    if abs(fp)<eps and abs(fd)<eps:      cls="Converged"
    elif len(prog)>=3 and sp<0 and sd<0: cls="Pseudo-converged"
    else:                                cls="Diverging"
    return cls,fp,fd,fsw,sp,sd

# ═══════════ mitigation logs → simulation_results.csv ════════════════
def build_sim_results():
    pat=(r"log_(method\d)(?:_alpha([\d\.]+))?_prob([\d\.]+)"
         r"_mult([\d\.]+)_t(\S+)\.txt")
    rows=[]
    for p in glob.glob(f"{MIT_DIR}/log_*.txt"):
        m=re.match(pat,os.path.basename(p))
        if m is None:
            continue
        meth,a_s,pr_s,mu_s,t_s=m.groups()
        alpha=float(a_s) if a_s else 0
        prob=float(pr_s); mult=float(mu_s)
        tam=np.inf if t_s=="inf" else float(t_s)
        cnt=w=d=0
        for ln in open(p):
            cnt+=1
            try:
                d+=float(ln.split("deviation=")[1].split(",")[0])
                w+=float(ln.split("weight=")[1].split(",")[0])
            except: pass
        rows.append(dict(method=meth,alpha=alpha,attack_prob=prob,
                         multiplier=mult,tampering=tam,iterations=cnt,
                         mitigation_count=cnt,
                         avg_weight=w/cnt if cnt else 0,
                         avg_deviation=d/cnt if cnt else 0))
    df=pd.DataFrame(rows)
    df.to_csv("simulation_results.csv",index=False)
    print(f"simulation_results.csv saved ({len(df)})")
    return df

# ───────────── local-conv log parser (unchanged) ────────────────────
def local_conv_items(path: str):
    out=[]
    rg=r"Sub-graph\s+(\[.*?\])\s+locally converged at iter (\d+)"
    with open(path) as f:
        for ln in f:
            m=re.search(rg,ln)
            if m:
                sub=m.group(1).replace(" ",""); it=int(m.group(2))
                out.append((sub,it))
    return out

# ═════════════ binaries → binary_summary.csv ═════════════════════════
def analyse_binaries():
    pat=(r"state_(method\d)(?:_alpha([\d\.]+))?_prob([\d\.]+)"
         r"_mult([\d\.]+)_t(\S+)\.pkl\.gz")
    rows=[]
    for bp in glob.glob(f"{BIN_DIR}/state_*.pkl.gz"):
        m=re.match(pat,os.path.basename(bp))
        if m is None:
            continue
        meth,a_s,pr_s,mu_s,t_s=m.groups()
        alpha=float(a_s) if a_s else 0
        prob=float(pr_s); mult=float(mu_s)
        tam=np.inf if t_s=="inf" else float(t_s)

        with gzip.open(bp,"rb") as f:
            bd=pickle.load(f)

        if bd.get("iteration",0)<1000:
            print(f"BIN  {os.path.basename(bp)}  →  iteration {bd.get('iteration',0)} (<1000)  SKIP")
            continue

        prog=extract_progress(bd)
        if not prog:
            print(f"BIN  {os.path.basename(bp)}  →  NO usable history (skipped)")
            continue

        cls,fp,fd,fsw,sp,sd=classify_run(prog)
        print(f"BIN  {os.path.basename(bp)}  →  found {len(prog)} rows   [{cls}]")
        rows.append(dict(method=meth,alpha=alpha,attack_prob=prob,
                         multiplier=mult,tampering=tam,conv_class=cls,
                         final_prim=fp,final_dual=fd,final_SW=fsw,
                         slope_prim=sp,slope_dual=sd))

    if not rows:
        print("No binary yielded history ≥1000 iterations.")
        return None

    bdf=pd.DataFrame(rows)
    bdf.to_csv("binary_summary.csv",index=False)
    print("binary_summary.csv saved")

    # ── additional visuals ─────────────────────────────────────────
    overlaid_hists(bdf,"final_prim","conv_class",bins=30,
                   ttl="Prim residual distribution by class",
                   fname="prim_hist_by_class.png")
    cdf_plot(bdf,"final_prim","conv_class",
             ttl="CDF of prim residuals by class",
             fname="prim_cdf_by_class.png")
    boxplots(bdf,"conv_class","final_SW",
             "Final SW by Convergence Class","finalSW_box_class.png")
    return bdf

# ═════════════════════════════════ main ══════════════════════════════
def main():
    df = build_sim_results()
    if df.empty:
        return

    # aggregated plots on mitigation logs
    df2, df1 = df[df.method=="method2"], df[df.method=="method1"]
    if not df2.empty:
        grouped_bar(df2,"tampering","alpha","iterations",
                    "Avg Iterations","Iterations vs Tampering (Relaxed)",
                    "iter_vs_tamper_relaxed.png")
        grouped_bar(df2,"tampering","alpha","mitigation_count",
                    "Avg Mitigations","Mitigations vs Tampering (Relaxed)",
                    "mitig_vs_tamper_relaxed.png")
    if not df1.empty:
        simple_bar(df1,"tampering","iterations","Avg Iterations",
                   "Iterations vs Tampering (Classic)",
                   "iter_vs_tamper_classic.png")

    # per-run iteration logs
    csvs = glob.glob(f"{ITER_DIR}/iter_*.csv")
    if csvs:
        for p in csvs:
            gen_iter_plots(p)
        all_it = pd.concat([iter_df(p) for p in csvs], ignore_index=True)
        last   = all_it.sort_values("iter").groupby("tag").tail(1)

        boxplots(last,"tamper","SW","Final SW by Tampering",
                 "finalSW_box_tamper.png")
        mean_curve(all_it,"iter","SW","method",
                   "SW Convergence (Mean±Std)","SW_conv_meanstd.png")

        overlaid_hists(last,"SW","tamper",bins=25,
                       ttl="Final SW distribution by tampering",
                       fname="sw_hist_by_tamper.png")
        cdf_plot(last,"SW","tamper",
                 ttl="CDF of final SW by tampering",
                 fname="sw_cdf_by_tamper.png")

        boxplots(df,"tampering","iterations",
                 "Iterations by Tampering","iterations_box_tamper.png")
        boxplots(last,"method","Price","Final Price by Method",
                 "price_box_method.png")
        boxplots(last,"method","prim","Prim Final by Method",
                 "prim_box_method.png")
        boxplots(last,"method","dual","Dual Final by Method",
                 "dual_box_method.png")
        boxplots(df,"tampering","avg_weight",
                 "Avg Weight by Tampering","avgWeight_box_tamper.png")

        overlaid_hists(df,"iterations","tampering",bins=30,
                       ttl="Iterations distribution by tampering",
                       fname="iter_hist_by_tampering.png")
        cdf_plot(df,"iterations","tampering",
                 ttl="CDF of iterations by tampering",
                 fname="iter_cdf_by_tampering.png")

    # local convergence quicklook
    lc_pat=(r"local_conv_(method\d)(?:_alpha([\d\.]+))?_prob([\d\.]+)"
            r"_mult([\d\.]+)_t(\S+)\.log")
    lc_rows=[]
    for lp in glob.glob(f"{LC_DIR}/local_conv_*.log"):
        m=re.match(lc_pat,os.path.basename(lp))
        if m is None:
            continue
        meth,a_s,pr_s,mu_s,t_s=m.groups()
        alpha=float(a_s) if a_s else 0
        tam=np.inf if t_s=="inf" else float(t_s)
        for sub,it in local_conv_items(lp):
            lc_rows.append(dict(method=meth,alpha=alpha,tampering=tam,
                                subgraph=sub,conv_iter=it))
    if lc_rows:
        lcd=pd.DataFrame(lc_rows)
        grouped_bar(lcd,"tampering","method","conv_iter",
                    "Avg Local-Conv Iter","LocalConv Iter vs Tampering",
                    "localConv_iter_vs_tamper.png")

    # binaries
    bdf = analyse_binaries()

    # summary pivot
    summ=(df.groupby(["method","alpha","attack_prob","tampering"])
             .agg(iter_mean=("iterations","mean"),
                  iter_std=("iterations","std")).reset_index())
    summ.to_csv("simulation_summary_table.csv",index=False)
    print("simulation_summary_table.csv saved")

if __name__=="__main__":
    main()
