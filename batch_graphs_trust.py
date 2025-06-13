#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_graphs_trust.py  –  analytics + plotting + binary deep-dive
"""

import sys, argparse
import os
import re
import glob
import gzip
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

_sys_mod = 'numpy._core.numeric'
if _sys_mod not in sys.modules:
    try:
        sys.modules[_sys_mod] = np.core.numeric
    except:
        pass

PLOT_DIR = "logs/plots";       os.makedirs(PLOT_DIR,    exist_ok=True)
MIT_DIR  = "logs/mitigation";  os.makedirs(MIT_DIR,     exist_ok=True)
ITER_DIR = "logs/iter_stats"
LC_DIR   = "logs/local_conv"
BIN_DIR  = "logs/binaries"

matplotlib.rcParams.update({
    # Base font size for text (labels, legend, titles inherit relative to this)
    "font.size":          16,
    # Axes titles (“ttl”)
    "axes.titlesize":     18,
    # Axes labels (x/y)
    "axes.labelsize":     16,
    # Tick labels
    "xtick.labelsize":    14,
    "ytick.labelsize":    14,
    # Legend text
    "legend.fontsize":    12,
    # Figure title if you ever use it
    "figure.titlesize":   20,
})

PLOT_INFO = []

def _add_caption(fname, desc):
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

def grouped_bar(df, x, g, y, ylab, ttl, fname, *, desc=None):
    df = _copy(df); df[g] = df[g].astype(str)
    with pd.option_context("mode.chained_assignment", None):
        try:
            df[x] = df[x].astype(float)
        except:
            pass
    tab = df.groupby([x, g])[y].agg(['mean', 'std']).reset_index()
    mtab = tab.pivot(index=x, columns=g, values='mean')
    stab = tab.pivot(index=x, columns=g, values='std')
    X = np.arange(len(mtab)); bw = 0.8/len(mtab.columns)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, col in enumerate(mtab.columns):
        pos = X + (i - len(mtab.columns)/2)*bw + bw/2
        ax.bar(pos, mtab[col], yerr=clamp_err(mtab[col], stab[col]), width=bw, capsize=4, label=str(col))
    ax.set_xticks(X); ax.set_xticklabels(mtab.index.astype(str))
    ax.set_xlabel(x); ax.set_ylabel(ylab); ax.set_title(ttl)
    ax.set_ylim(bottom=0); ax.legend(title=g)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close(fig)
    _add_caption(fname, desc)

def simple_bar(df, x, y, ylab, ttl, fname, *, desc=None):
    df = _copy(df)
    with pd.option_context("mode.chained_assignment", None):
        try:
            df[x] = df[x].astype(float)
        except:
            pass
    tab = df.groupby(x)[y].agg(['mean','std']).reset_index().sort_values(x)
    fig, ax = plt.subplots()
    ax.bar(tab.index, tab['mean'], yerr=clamp_err(tab['mean'], tab['std']), capsize=4)
    ax.set_xticks(tab.index); ax.set_xticklabels(tab[x].astype(str))
    ax.set_xlabel(x); ax.set_ylabel(ylab); ax.set_title(ttl); ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close(fig)
    _add_caption(fname, desc)

def mean_curve(df, x, y, hue, ttl, fname, *, desc=None):
    df = _copy(df); fig, ax = plt.subplots()
    for k, g in df.groupby(hue):
        ln = g.groupby(x)[y].agg(['mean','std']).reset_index()
        ax.plot(ln[x], ln['mean'], label=str(k))
        ax.fill_between(ln[x], ln['mean']-ln['std'], ln['mean']+ln['std'], alpha=.25)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(ttl); ax.legend(title=hue)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close(fig)
    _add_caption(fname, desc)

def boxplots(df, x, y, ttl, fname, *, desc=None, jitter=0.08, show_n=True):
    d = df[[x,y]].dropna().copy()
    def sk(v):
        if v is np.inf: return (2, np.inf)
        try: return (0, float(v))
        except: return (1, str(v))
    cats = sorted(d[x].unique(), key=sk)
    data, labels, consts, pts, counts = [], [], [], [], []
    for i, c in enumerate(cats):
        arr = d[d[x]==c][y].values
        if arr.size==0: continue
        counts.append(arr.size)
        pts.extend([(i+np.random.uniform(-jitter,jitter), v) for v in arr])
        if np.allclose(arr, arr[0]):
            consts.append((i, arr[0])); data.append([arr[0]]*3)
        else:
            data.append(arr)
        if c is np.inf:
            labels.append("inf")
        elif isinstance(c, (int,float)) and float(c).is_integer():
            labels.append(str(int(c)))
        else:
            labels.append(str(c))
    fig, ax = plt.subplots(figsize=(8,6))
    bp = ax.boxplot(data, positions=range(len(data)), showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("none"); patch.set_edgecolor("black")
    if pts:
        xs, ys = zip(*pts); ax.scatter(xs, ys, s=8, alpha=0.4, color="grey", zorder=0)
    for pos, val in consts:
        ax.hlines(val, pos-0.3, pos+0.3, color="black", lw=1.5); ax.plot(pos, val, "o", color="black")
    if show_n:
        for pos, n in enumerate(counts):
            ax.text(pos, ax.get_ylim()[0], f" n={n}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(ttl)
    span = d[y].max()/max(d[y].min(),1e-12) if not d[y].empty else 1
    if span>1e3: ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close(fig)
    _add_caption(fname, desc)

def overlaid_hists(df, x, cat, bins, ttl, fname, *, alpha=0.35, desc=None):
    df = _copy(df); fig, ax = plt.subplots()
    for c, g in df.groupby(cat):
        ax.hist(g[x], bins=bins, alpha=alpha, label=str(c), density=True)
    ax.set_xlabel(x); ax.set_ylabel("Density"); ax.set_title(ttl)
    ax.legend(title=cat); plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close(fig)
    _add_caption(fname, desc)

def cdf_plot(df, x, cat, ttl, fname, *, desc=None):
    df = _copy(df); fig, ax = plt.subplots()
    for c, g in df.groupby(cat):
        data = np.sort(g[x].values); yvals = np.arange(1,len(data)+1)/float(len(data))
        ax.step(data, yvals, where="post", label=str(c))
    ax.set_xlabel(x); ax.set_ylabel("CDF"); ax.set_title(ttl)
    ax.legend(title=cat); plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close(fig)
    _add_caption(fname, desc)

def gen_iter_plots(csv_p):
    df = pd.read_csv(csv_p); df["Price"]=df["avg_price"]*100
    tag = os.path.splitext(os.path.basename(csv_p))[0][5:]
    def sv(fig, stem):
        fname = f"{stem}_{tag}.pdf"
        fig.savefig(f"{PLOT_DIR}/{fname}"); plt.close(fig)
        _add_caption(fname, None)
    for col, ttl in [("SW","SW vs Iter"),("Price","Price vs Iter")]:
        fig, ax = plt.subplots(); ax.plot(df["iter"], df[col])
        ax.set_xlabel("Iteration"); ax.set_ylabel(col); ax.set_title(ttl)
        sv(fig, col)
    fig, ax = plt.subplots()
    ax.plot(df["iter"], df["prim"], label="prim"); ax.plot(df["iter"], df["dual"], label="dual")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Residual"); ax.legend(); ax.set_title("Primal & Dual vs Iter")
    sv(fig, "Residuals")

def iter_df(csv_p):
    df = pd.read_csv(csv_p); df["Price"]=df["avg_price"]*100
    tag = os.path.splitext(os.path.basename(csv_p))[0][5:]
    parts = dict(re.findall(r"([a-z]+)([^_]+)", tag))
    df["tag"]=tag; df["method"]=parts.get("method","")
    t = parts.get("t","0")
    df["tamper"] = np.inf if t=="inf" else float(t)
    return df

def _norm_progress(obj):
    rows = []
    if isinstance(obj, (list,tuple)) and obj and isinstance(obj[0], (list,tuple,np.ndarray)):
        for i, r in enumerate(obj):
            r = list(r); rows.append(tuple([i] + r[:4]))
    elif isinstance(obj, np.ndarray) and obj.ndim==2:
        for i, r in enumerate(obj):
            r = r.tolist(); rows.append(tuple([i] + r[:4]))
    elif hasattr(obj, "iterrows"):
        for i, (_, r) in enumerate(obj.iterrows()):
            r = r.values.tolist(); rows.append(tuple([i] + r[:4]))
    return rows

def _equilibrate_trades(mat, *, run_id=None, logdir=None, rel_tol=1e-3, scale_to_zero=1e-3):
    T_orig = mat.astype(float, copy=False); T = T_orig.copy()
    for i in range(T.shape[0]):
        for j in range(i+1, T.shape[1]):
            avg = 0.5*(T[i,j] - T[j,i]); T[i,j]=avg; T[j,i]=-avg
    steps = 1
    row_err = abs(T.sum(axis=1)).max()
    max_trade = abs(T_orig).max() if T_orig.size else 0.0
    tol = max(rel_tol*max_trade,1e-3)
    if row_err > tol:
        scale = tol/row_err if row_err else 0.0; T*=scale; steps+=1
        cls = "Scaled-balanced" if scale>=scale_to_zero else "Forced-zero"
    else:
        cls = "Balanced"; scale=1.0
    if logdir and run_id:
        os.makedirs(logdir, exist_ok=True)
        with gzip.open(os.path.join(logdir, f"{run_id}.pkl.gz"), "wb") as fh:
            pickle.dump({"T_orig":T_orig,"T_balanced":T}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return cls, steps, scale

def _gradient_descent_balance(mat, *, lr=0.05, max_iter=500, tol=1e-3):
    """
    Try to zero each row‐sum of the antisymmetric Trades matrix via tiny
    gradient steps, while keeping antisymmetry intact.
    Returns (cls, steps, err):
      cls  ∈ {"GD-balanced","GD-partial","GD-failed","GD-no-matrix"}
      steps = iterations executed
      err   = final max |row-sum|
    """
    if mat is None:
        return "GD-no-matrix", 0, np.nan
    T = mat.astype(float, copy=True); n = T.shape[0]
    for k in range(max_iter):
        r = T.sum(axis=1, keepdims=True)
        err = abs(r).max()
        if err <= tol:
            return "GD-balanced", k, err
        grad = np.broadcast_to(r, T.shape) / n
        for i in range(n):
            for j in range(i+1, n):
                d = lr * (grad[i,j]-grad[j,i])
                T[i,j] -= d; T[j,i] += d
    err = abs(T.sum(axis=1)).max()
    cls = "GD-partial" if err <= 10*tol else "GD-failed"
    return cls, max_iter, err

KEYS_TRY = ("prog_arr","progress","opti_progress","history","iter_list","progress_array")

def extract_progress(bd):
    if isinstance(bd, dict):
        for k in KEYS_TRY:
            if k in bd:
                pts = _norm_progress(bd[k])
                if pts: return pts
    return []

def classify_run(prog, eps=1e-3):
    prim = np.asarray([p[3] for p in prog], float)
    dual = np.asarray([p[4] for p in prog], float)
    fp, fd, fsw = prim[-1], dual[-1], prog[-1][1]
    sp = sd = np.nan
    if len(prog) >= 3:
        win = min(100, len(prog)-1)
        sp = np.polyfit(range(win), prim[-win:], 1)[0]
        sd = np.polyfit(range(win), dual[-win:], 1)[0]
    if abs(fp)<eps and abs(fd)<eps:
        cls="Converged"
    elif len(prog)>=3 and sp<0 and sd<0:
        cls="Pseudo-converged"
    else:
        cls="Diverging"
    return cls, fp, fd, fsw, sp, sd

def build_sim_results():
    iter_rows = []
    for p in glob.glob(f"{ITER_DIR}/iter_*.csv"):
        fn = os.path.splitext(os.path.basename(p))[0][5:]
        if '_run' in fn:
            base, run_s = fn.rsplit('_run',1); run=int(run_s)
        else:
            base, run = fn, 0
        df = pd.read_csv(p, usecols=["iter"])
        iter_rows.append((base, run, int(df["iter"].max())))
    iter_df = pd.DataFrame(iter_rows, columns=["tag","run","iterations"])
    mit_rows = []
    for p in glob.glob(f"{MIT_DIR}/log_*.txt"):
        fn = os.path.basename(p); full = os.path.splitext(fn)[0][4:]
        if '_run' in full:
            tag, run_s = full.rsplit('_run',1); run=int(run_s)
        else:
            tag, run = full, 0
        parts = tag.split('_')
        segs = {}
        for seg in parts:
            if seg.startswith('method'): segs['method'] = seg
            elif seg.startswith('alpha'): segs['alpha'] = seg[5:]
            elif seg.startswith('prob'): segs['prob'] = seg[4:]
            elif seg.startswith('mult'): segs['mult'] = seg[4:]
            elif seg.startswith('t') and (seg[1:].replace('.','',1).isdigit() or seg[1:]=='inf'):
                segs['tam'] = seg[1:]
        meth = segs.get('method','')
        alpha = float(segs.get('alpha','0'))
        prob  = float(segs.get('prob','0'))
        mult  = float(segs.get('mult','0'))
        tam_s = segs.get('tam','0')
        tam   = np.inf if tam_s=='inf' else float(tam_s)
        cnt = w_tot = d_tot = 0.0
        for ln in open(p, encoding='utf-8', errors='ignore'):
            if "[Flag]" in ln: cnt+=1
            try:
                d_tot += float(ln.split('deviation=')[1].split(',')[0])
                w_tot += float(ln.split('weight=')[1].split(',')[0])
            except:
                pass
        mit_rows.append(dict(
            tag=tag, run=run, method=meth, alpha=alpha,
            attack_prob=prob, multiplier=mult, tampering=tam,
            mitigation_count=int(cnt),
            avg_weight=w_tot/cnt if cnt else np.nan,
            avg_deviation=d_tot/cnt if cnt else np.nan
        ))
    mit_df = pd.DataFrame(mit_rows)
    merged = iter_df.merge(mit_df, on=["tag","run"], how="left")
    merged['mitigation_count'] = merged['mitigation_count'].fillna(0).astype(int)
    for col in ['method','alpha','attack_prob','multiplier','tampering']:
        merged[col] = merged[col].fillna('unknown')
    merged.to_csv('simulation_results.csv', index=False)
    return merged

def local_conv_items(path):
    out = []
    rg = r"Sub-graph\s+(\[.*?\])\s+locally converged at iter (\d+)"
    with open(path) as f:
        for ln in f:
            m = re.search(rg, ln)
            if m:
                sub = m.group(1).replace(" ",""); it=int(m.group(2))
                out.append((sub, it))
    return out

def analyse_binaries():
    rows = []
    for bp in glob.glob(f"{BIN_DIR}/state_*.pkl.gz"):
        fn = os.path.basename(bp)          # ① needed for run_id
        with gzip.open(bp, "rb") as fh:
            bd = pickle.load(fh)

        cfg   = bd.get("config", {})
        meth  = cfg.get("iter_update_method", "unknown")
        alpha = float(cfg.get("alpha", 0))
        prob  = float(cfg.get("byzantine_attack_probability", 0))
        mult  = float(cfg.get("byzantine_multiplier_upper", 0))

        tam = cfg.get("tampering_count", 0)
        # ② robust conversion to float / np.inf
        if isinstance(tam, str) and tam.lower() == "inf":
            tam = np.inf
        elif tam == float("inf"):
            tam = np.inf
        else:
            tam = float(tam)

        # ───────────────────────────────────────────────
        gd_cls, gd_steps, gd_err = _gradient_descent_balance(bd.get("Trades"))
        trd_cls, trd_steps, trd_scale = _equilibrate_trades(
            bd.get("Trades"), run_id=fn, logdir="eq_logs"
        )

        prog = extract_progress(bd)
        cls, fp, fd, fsw, sp, sd = ("Converged", 0.0, 0.0, 0.0, 0.0, 0.0)
        if prog:
            cls, fp, fd, fsw, sp, sd = classify_run(prog)

        rows.append(dict(
            method=meth, alpha=alpha, attack_prob=prob,
            multiplier=mult, tampering=tam, conv_class=cls,
            final_prim=fp, final_dual=fd, final_SW=fsw,
            slope_prim=sp, slope_dual=sd,
            gd_class=gd_cls, gd_steps=gd_steps, gd_err=gd_err,
            eq_class=trd_cls, eq_steps=trd_steps, eq_scale=trd_scale
        ))

    bdf = pd.DataFrame(rows)
    if not bdf.empty:
        bdf.to_csv("binary_summary.csv", index=False)
    return bdf

def main():

    ap = argparse.ArgumentParser(description="Analyse mitigation/binary logs")
    ap.add_argument("--skip-plots", action="store_true",
                    help="only build the CSV tables (no PDFs)")
    args = ap.parse_args()

    df = build_sim_results()
    df = df[df["tampering"]!="unknown"]
    df["tampering"] = df["tampering"].replace({"inf":np.inf}).astype(float)
    for col in ("multiplier","attack_prob"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df2 = df[df.method=="method2"]
    df1 = df[df.method=="method1"]
    if not df2.empty:
        grouped_bar(df2, "tampering", "alpha", "iterations", "Avg Iterations", "Iter vs Tampering (Relaxed)", "iterations_vs_tampering_relaxed.pdf")
        grouped_bar(df2, "tampering", "alpha", "mitigation_count", "Avg Mitigations", "Mitigations vs Tampering (Relaxed)", "mitigations_vs_tampering_relaxed.pdf")
    if not df1.empty:
        simple_bar(df1, "tampering", "iterations", "Avg Iterations", "Iter vs Tampering (Classical)", "iterations_vs_tampering_classical.pdf")
        grouped_bar(df1, "multiplier", "tampering", "mitigation_count", "Avg Mitigations", "Mitigations vs Multiplier (Classical)", "mitigations_vs_multiplier_classical.pdf")
        grouped_bar(df1, "attack_prob", "tampering", "iterations", "Avg Iterations", "Iter vs AttackProb (Classical)", "iterations_vs_attackprob_classical.pdf")
    fig, ax = plt.subplots()
    ax.scatter(df['mitigation_count'], df['iterations'], alpha=0.6)
    ax.set_xlabel('Mitigation Events'); ax.set_ylabel('Iterations'); ax.set_title('Iterations vs Mitigations')
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/iterations_vs_mitigations.pdf"); plt.close(fig)
    _add_caption('iterations_vs_mitigations.pdf','Scatter of mitigation events vs convergence iterations.')
    csvs = glob.glob(f"{ITER_DIR}/iter_*.csv")
    if csvs and not args.skip_plots:
        for p in csvs: gen_iter_plots(p)
        all_it = pd.concat([iter_df(p) for p in csvs], ignore_index=True)
        fig, ax = plt.subplots(figsize=(8,6))
        for tag, g in all_it.groupby('tag'):
            ax.plot(g['iter'], g['Price'], alpha=0.3, color='gray')
        for method, g in all_it.groupby('method'):
            ln = g.groupby('iter')['Price'].mean().reset_index()
            ax.plot(ln['iter'], ln['Price'], label=f"mean {method}", linewidth=2)
        ax.set_xlabel('Iteration'); ax.set_ylabel('Price'); ax.set_title('Price vs Iter'); ax.legend(fontsize='small', ncol=2)
        plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/price_per_run.pdf"); plt.close(fig)
        _add_caption('price_per_run.pdf', None)
        mean_curve(all_it, "iter", "Price", "method", "Price Convergence", "price_conv_meanstd.pdf")
    lc_rows = []
    for lp in glob.glob(f"{LC_DIR}/local_conv_*.log"):
        m = re.match(r"local_conv_.*?(method\d)(?:_alpha([\d\.]+))?_prob([\d\.]+)_mult([\d\.]+)_t([^_]+)", os.path.basename(lp))
        if not m: continue
        meth, a_s, pr_s, mu_s, t_s = m.groups()
        alpha = float(a_s) if a_s else 0; tam = np.inf if t_s=="inf" else float(t_s)
        for sub, it in local_conv_items(lp):
            lc_rows.append(dict(method=meth, alpha=alpha, tampering=tam, subgraph=sub, conv_iter=it))
    if lc_rows:
        lcd = pd.DataFrame(lc_rows)
        grouped_bar(lcd, "tampering", "method", "conv_iter", "Avg Local Iter", "Local Conv vs Tamper", "localconv_vs_tamper.pdf")

    bdf = analyse_binaries()
    
    summ = df.groupby(["method","alpha","attack_prob","tampering"]).agg(iter_mean=("iterations","mean"), iter_std=("iterations","std")).reset_index()
    summ.to_csv("simulation_summary_table.csv", index=False)

    with open(f"{PLOT_DIR}/plot_explanations.txt","w") as f:
        for fn, desc in PLOT_INFO:
            f.write(f"{fn} : {desc}\n")

if __name__=="__main__":
    main()
