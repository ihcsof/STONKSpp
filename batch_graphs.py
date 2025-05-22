#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_graphs.py  –  analytics + plotting + binary deep‑dive
2025‑05‑19  (extended visual suite + per‑figure explanations)

 • simulation_results.csv       ← mitigation logs
 • simulation_summary_table.csv ← pivot of means/stds
 • binary_summary.csv           ← ≥1000‑iter binaries with residual history
 • PNG figures → logs/plots/   + logs/plots/plot_explanations.txt  (captions)
"""

import sys
import os, re, glob, gzip, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

_sys_mod = 'numpy._core.numeric'
if _sys_mod not in sys.modules:
    try:
        sys.modules[_sys_mod] = np.core.numeric
    except AttributeError:
        # in case np.core.numeric isn’t available (very old numpy?), skip
        pass

# ────────── directories ──────────
PLOT_DIR = "logs/plots";       os.makedirs(PLOT_DIR,    exist_ok=True)
MIT_DIR  = "logs/mitigation";  os.makedirs(MIT_DIR,     exist_ok=True)
ITER_DIR = "logs/iter_stats"
LC_DIR   = "logs/local_conv"
BIN_DIR  = "logs/binaries"

# ────────── global plot‑caption collector ──────────
PLOT_INFO: list[tuple[str,str]] = []     # (filename, description)

def _add_caption(fname: str, desc: str | None):
    """Register a description for the just‑saved figure."""
    if desc:
        PLOT_INFO.append((fname, desc))

# ═══════════════════ plotting helpers ═══════════════════

def _copy(df):
    return df.copy(deep=True)

def clamp_err(m, s, upper=1000):
    low = np.maximum(0, m - s)
    if upper is None:
        return [m - low, s]
    high = np.minimum(upper, m + s) - m
    return [m - low, high]

def grouped_bar(df, x, g, y, ylab, ttl, fname, *, desc: str | None = None):
    """Grouped bar‑chart with mean±std error bars."""
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
    plt.tight_layout();
    out_path = f"{PLOT_DIR}/{fname}"
    plt.savefig(out_path); plt.close(fig)
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
    plt.tight_layout();
    out_path = f"{PLOT_DIR}/{fname}"
    plt.savefig(out_path); plt.close(fig)
    _add_caption(fname, desc)


def mean_curve(df, x, y, hue, ttl, fname, *, desc: str | None = None):
    df = _copy(df); fig, ax = plt.subplots()
    for k, g in df.groupby(hue):
        ln = g.groupby(x)[y].agg(['mean', 'std']).reset_index()
        ax.plot(ln[x], ln['mean'], label=str(k))
        ax.fill_between(ln[x], ln['mean']-ln['std'], ln['mean']+ln['std'], alpha=.25)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(ttl); ax.legend(title=hue)
    plt.tight_layout();
    out_path = f"{PLOT_DIR}/{fname}"
    plt.savefig(out_path); plt.close(fig)
    _add_caption(fname, desc)

def boxplots(
        df, x, y, ttl, fname,
        *, desc: str | None = None,
        log_if_span: bool = True,
        jitter: float = 0.08,
        show_n: bool = True,
):
    """
    A robust replacement for the old `boxplots()`:

    • skips empty categories completely  
    • if a group is constant-valued, draws a horizontal line + dot  
    • overlays jittered raw points so you can see *n*  
    • auto-switches to log-scale when data span > 1 000×  
    """
    import random

    d = df[[x, y]].dropna().copy()

    # ── category ordering identical to the original helper ──────────
    def sk(v):
        if v is np.inf or (isinstance(v, float) and np.isinf(v)):
            return (2, np.inf)
        try:
            return (0, float(v))
        except Exception:
            return (1, str(v))
    cats = sorted(d[x].unique(), key=sk)

    # ── assemble data buckets & bookkeeping ─────────────────────────
    data, labels, consts, pts, counts = [], [], [], [], []
    for i, c in enumerate(cats):
        arr = d[d[x] == c][y].values
        if arr.size == 0:
            continue                      # skip empty category entirely
        counts.append(arr.size)
        # jittered points
        pts.extend([(i + random.uniform(-jitter, jitter), v) for v in arr])

        # constant bucket?
        if np.allclose(arr, arr[0]):
            consts.append((i, arr[0]))
            # supply a “fake” 3-point spread so boxplot still draws sth.
            data.append([arr[0], arr[0], arr[0]])
        else:
            data.append(arr)

        # pretty x-tick labels
        if c is np.inf or (isinstance(c, float) and np.isinf(c)):
            labels.append("inf")
        elif isinstance(c, (int, float)) and float(c).is_integer():
            labels.append(str(int(c)))
        else:
            labels.append(str(c))

    # ── plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot(
        data,
        positions = range(len(data)),
        showfliers = False,
        patch_artist = True
    )
    # outline only (white fill)
    for patch in bp["boxes"]:
        patch.set_facecolor("none")
        patch.set_edgecolor("black")

    # jittered raw points for context
    if pts:
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=8, alpha=0.4, color="grey", zorder=0)

    # constant groups: emphasise with a thicker line + dot
    for pos, val in consts:
        ax.hlines(val, pos-0.3, pos+0.3, color="black", lw=1.5)
        ax.plot(pos, val, "o", color="black")

    # annotate n if requested
    if show_n:
        for pos, n in enumerate(counts):
            ax.text(pos, ax.get_ylim()[0], f" n={n}", ha="center",
                    va="bottom", fontsize=8, rotation=90)

    # x-ticks
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(ttl)

    # optional log scale
    if log_if_span and not d[y].empty:
        span = d[y].max() / max(d[y].min(), 1e-12)
        if span > 1e3:
            ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
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
    plt.savefig(out_path); plt.close(fig)
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
    plt.savefig(out_path); plt.close(fig)
    _add_caption(fname, desc)

# ═══════════════════ per‑run iteration PNGs (unchanged logic) ════════════════════

def gen_iter_plots(csv_p: str):
    df = pd.read_csv(csv_p); df["Price"] = df["avg_price"]*100
    tag = os.path.splitext(os.path.basename(csv_p))[0][5:]

    def sv(fig, stem, caption):
        plt.tight_layout(); fname = f"{stem}_{tag}.png"
        fig.savefig(f"{PLOT_DIR}/{fname}"); plt.close(fig)
        _add_caption(fname, caption)

    for col, ttl, cap in [("SW", "SW vs Iter", "Trajectory of social‑welfare across iterations for a single run."),
                          ("Price", "Price vs Iter", "Average price evolution across iterations for a single run.")]:
        fig, ax = plt.subplots(); ax.plot(df["iter"], df[col])
        ax.set_xlabel("Iteration"); ax.set_ylabel(col); ax.set_title(ttl); sv(fig, col, cap)

    fig, ax = plt.subplots();
    ax.plot(df["iter"], df["prim"], label="prim")
    ax.plot(df["iter"], df["dual"], label="dual")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Residual"); ax.legend()
    ax.set_title("Primal & Dual vs Iter")
    sv(fig, "Residuals", "Primal and dual residual convergence behaviour for a single run.")


def iter_df(csv_p):
    df = pd.read_csv(csv_p); df["Price"] = df["avg_price"]*100
    tag = os.path.splitext(os.path.basename(csv_p))[0][5:]
    parts = dict(re.findall(r"([a-z]+)([^_]+)", tag))
    df["tag"] = tag
    df["method"] = parts.get("method", "")
    df["tamper"] = np.inf if parts.get("t") == "inf" else float(parts.get("t", 0))
    return df

# ═════════════════ residual‑history extraction + classify (unchanged) ══════

def _norm_progress(obj):
    import numpy as _np, pandas as _pd
    rows = []
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (list, tuple, _np.ndarray)):
        for i, r in enumerate(obj):
            r = list(r); rows.append(tuple([i]+r) if len(r) == 4 else tuple(r[:5]))
    elif isinstance(obj, _np.ndarray) and obj.ndim == 2 and obj.shape[1] >= 4:
        for i, r in enumerate(obj):
            r = r.tolist(); rows.append(tuple([i]+r) if len(r) == 4 else tuple(r[:5]))
    elif isinstance(obj, _pd.DataFrame) and obj.shape[1] >= 4:
        for i, (_, r) in enumerate(obj.iterrows()):
            r = r.values.tolist(); rows.append(tuple([i]+r) if len(r) == 4 else tuple(r[:5]))
    return rows

KEYS_TRY = ("prog_arr", "progress", "opti_progress", "history", "iter_list", "progress_array")

def extract_progress(bd):
    if isinstance(bd, dict):
        for k in KEYS_TRY:
            if k in bd:
                pts = _norm_progress(bd[k])
                if pts:
                    return pts
    def _shallow(v):
        return _norm_progress(v)
    if isinstance(bd, dict):
        for v in bd.values():
            pts = _shallow(v)
            if pts:
                return pts
    elif isinstance(bd, (list, tuple)):
        for v in bd:
            pts = _shallow(v)
            if pts:
                return pts
    if isinstance(bd, dict):
        kmap = {k.lower(): k for k in bd}
        req = ["sw", "price", "prim", "dual"]
        if all(any(r in k for k in kmap) for r in req):
            try:
                sw   = np.asarray(bd[next(k for k in kmap if "sw"    in k)], float)
                prc  = np.asarray(bd[next(k for k in kmap if "price" in k)], float)
                prim = np.asarray(bd[next(k for k in kmap if "prim"  in k)], float)
                dual = np.asarray(bd[next(k for k in kmap if "dual"  in k)], float)
                if len({len(sw), len(prc), len(prim), len(dual)}) == 1:
                    return [(i, sw[i], prc[i], prim[i], dual[i]) for i in range(len(sw))]
            except Exception:
                pass
    return []

def classify_run(prog, eps=1e-3):
    prim = np.asarray([p[3] for p in prog], float)
    dual = np.asarray([p[4] for p in prog], float)
    fp, fd, fsw = prim[-1], dual[-1], prog[-1][1]
    sp = sd = np.nan
    if len(prog) >= 3:
        win = min(100, len(prog)-1)
        sp  = np.polyfit(range(win), prim[-win:], 1)[0]
        sd  = np.polyfit(range(win), dual[-win:], 1)[0]
    if abs(fp) < eps and abs(fd) < eps:
        cls = "Converged"
    elif len(prog) >= 3 and sp < 0 and sd < 0:
        cls = "Pseudo-converged"
    else:
        cls = "Diverging"
    return cls, fp, fd, fsw, sp, sd

# ═══════════ mitigation logs → simulation_results.csv ════════════════

def build_sim_results() -> pd.DataFrame:
    # ---- 1) collect final_iter & run from iter_stats ----
    iter_rows = []
    for p in glob.glob(f"{ITER_DIR}/iter_*.csv"):
        fn       = os.path.splitext(os.path.basename(p))[0]  # e.g. 'iter_tag_run3'
        base_run = fn[5:]  # strip 'iter_'
        if '_run' in base_run:
            base, run_s = base_run.rsplit('_run', 1)
            run = int(run_s)
        else:
            base, run = base_run, 0
        df = pd.read_csv(p, usecols=["iter"])
        iter_rows.append((base, run, int(df["iter"].max())))
    iter_df = pd.DataFrame(iter_rows, columns=["tag", "run", "iterations"])
    if iter_df.empty:
        raise RuntimeError("No iter_*.csv found in logs/iter_stats – cannot continue.")

    # ---- 2) collect mitigation info (may be missing) ----
    mit_pat = (r"log_(.*?)(?:_run\d+)?\.txt$")
    mit_rows = []
    for p in glob.glob(f"{MIT_DIR}/log_*.txt"):
        fn = os.path.basename(p)
        full = os.path.splitext(fn)[0][4:]  # strip 'log_'
        if '_run' in full:
            tag, run_s = full.rsplit('_run', 1)
            run = int(run_s)
        else:
            tag, run = full, 0
        m = re.match(mit_pat, fn)
        if not m:
            continue
        # parse tag details
        parts = tag.split('_')
        # find method, alpha, prob, mult, tampering via regex
        m2 = re.match(
            r"(.*?)(?:_run\d+)?$", tag
        )
        # fallback: parse individual fields
        segs = { }
        for seg in parts:
            if seg.startswith('method'):
                segs['method'] = seg
            elif seg.startswith('alpha'):
                segs['alpha'] = seg[5:]
            elif seg.startswith('prob'):
                segs['prob'] = seg[4:]
            elif seg.startswith('mult'):
                segs['mult'] = seg[4:]
            elif seg.startswith('t'):
                segs['tam'] = seg[1:]
        meth = segs.get('method', '')
        alpha = float(segs.get('alpha', 0))
        prob = float(segs.get('prob', 0))
        mult = float(segs.get('mult', 0))
        tam_s = segs.get('tam', '0')
        tam = np.inf if tam_s == 'inf' else float(tam_s)
        # basic per-line parsing
        cnt = w_tot = d_tot = 0.0
        for ln in open(p, encoding='utf-8', errors='ignore'):
            cnt += 1
            try:
                d_tot += float(ln.split('deviation=')[1].split(',')[0])
                w_tot += float(ln.split('weight=')[1].split(',')[0])
            except Exception:
                pass
        mit_rows.append(dict(
            tag=tag, run=run,
            method=meth, alpha=alpha,
            attack_prob=prob, multiplier=mult,
            tampering=tam, mitigation_count=int(cnt),
            avg_weight=w_tot/cnt if cnt else np.nan,
            avg_deviation=d_tot/cnt if cnt else np.nan
        ))
    mit_df = pd.DataFrame(mit_rows)

    # ---- 3) merge – join on tag and run ----
    merged = iter_df.merge(mit_df, on=["tag", "run"], how="left")
    merged['mitigation_count'] = merged['mitigation_count'].fillna(0).astype(int)
    for col in ['method','alpha','attack_prob','multiplier','tampering']:
        merged[col] = merged[col].fillna('unknown')
    merged.to_csv('simulation_results.csv', index=False)
    print(f"simulation_results.csv saved ({len(merged)})")
    return merged

# ───────────── local‑conv log parser (unchanged) ────────────────────

def local_conv_items(path: str):
    out = []
    rg = r"Sub-graph\s+(\[.*?\])\s+locally converged at iter (\d+)"
    with open(path) as f:
        for ln in f:
            m = re.search(rg, ln)
            if m:
                sub = m.group(1).replace(" ", "")
                it = int(m.group(2))
                out.append((sub, it))
    return out

# ═════════════ binaries → binary_summary.csv ═════════════════════════

def analyse_binaries():
    pat = (
        r"state_.*?(method\d)(?:_alpha([\d\.]+))?_prob([\d\.]+)"
        r"_mult([\d\.]+)_t([^_]+)(?:_run\d+)?\.pkl\.gz"
    )
    rows = []
    for bp in glob.glob(f"{BIN_DIR}/state_*.pkl.gz"):
        fn = os.path.basename(bp)
        m = re.match(pat, fn)
        if not m:
            continue
        meth, a_s, pr_s, mu_s, t_s = m.groups()
        alpha = float(a_s) if a_s else 0
        prob = float(pr_s); mult = float(mu_s)
        tam = np.inf if t_s == 'inf' else float(t_s)
        with gzip.open(bp, 'rb') as f:
            bd = pickle.load(f)
        iters = bd.get('iteration', 0)
        if iters < 1000:
            prog = extract_progress(bd)
            if prog:
                _, fsw, _, fp, fd = prog[-1]
                sp = sd = np.nan
            else:
                fp = fd = fsw = sp = sd = np.nan
            rows.append(dict(
                method=meth, alpha=alpha, attack_prob=prob,
                multiplier=mult, tampering=tam, conv_class='Converged',
                final_prim=fp, final_dual=fd, final_SW=fsw,
                slope_prim=sp, slope_dual=sd
            ))
            continue
        prog = extract_progress(bd)
        if not prog:
            continue
        cls, fp, fd, fsw, sp, sd = classify_run(prog)
        rows.append(dict(
            method=meth, alpha=alpha, attack_prob=prob,
            multiplier=mult, tampering=tam, conv_class=cls,
            final_prim=fp, final_dual=fd, final_SW=fsw,
            slope_prim=sp, slope_dual=sd
        ))
    if not rows:
        print('No binary yielded history ≥1000 iterations.')
        return None
    bdf = pd.DataFrame(rows)
    bdf.to_csv('binary_summary.csv', index=False)
    print('binary_summary.csv saved')

    # ── additional visuals ─────────────────────────────────────────
    overlaid_hists(bdf, "final_prim", "conv_class", bins=30,
                   ttl="Prim residual distribution by class",
                   fname="prim_hist_by_class.png",
                   desc="Distribution of final primal residuals separated by convergence class.")

    cdf_plot(bdf, "final_prim", "conv_class",
             ttl="CDF of prim residuals by class",
             fname="prim_cdf_by_class.png",
             desc="Cumulative distribution of final primal residuals across convergence classes.")

    boxplots(bdf, "conv_class", "final_SW",
             ttl="Final SW by Convergence Class",
             fname="finalSW_box_class.png",
             desc="Box‑plot of final social‑welfare for each detected convergence class.")
    return bdf

# ═════════════════════════════════ main ══════════════════════════════

def main():
    df = build_sim_results()
    if df.empty:
        return
    
    # Remove rows where tampering is genuinely missing
    df = df[df["tampering"] != "unknown"]

    # Convert tampering `"inf"` → np.inf  and make it numeric
    df["tampering"] = df["tampering"].replace({"inf": np.inf}).astype(float)

    # (same trick for multiplier / attack_prob if you filled them with "unknown")
    for col in ("multiplier", "attack_prob"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Split by method
    df2 = df[df.method == "method2"]   # Relaxed ADMM
    df1 = df[df.method == "method1"]   # Classical ADMM

    # ─────────────── Baseline plots (already present) ───────────────
    if not df2.empty:
        grouped_bar(df2, "tampering", "alpha", "iterations",
                    "Avg Iterations",
                    "Iterations vs Tampering (Relaxed ADMM, varying alpha)",
                    "iterations_vs_tamperingcount_relaxed.png",
                    desc="Average iterations required for convergence for the Relaxed ADMM across different tampering intensities and relaxation parameters (alpha).")

        grouped_bar(df2, "tampering", "alpha", "mitigation_count",
                    "Avg Mitigation Events",
                    "Mitigation Events vs Tampering Count (Relaxed ADMM, varying alpha)",
                    "mitigations_vs_tamperingcount_relaxed.png",
                    desc="Average number of mitigation events triggered in Relaxed ADMM under increasing tamper counts, broken‑down by alpha.")
    if not df1.empty:
        simple_bar(df1, "tampering", "iterations",
                   "Avg Iterations",
                   "Iterations vs Tampering Count (Classical ADMM)",
                   "iterations_vs_tamperingcount_classical.png",
                   desc="Impact of tampering intensity on convergence iterations for the Classical ADMM.")

    # ─────────────── NEW comparative plots ───────────────
    # Classic ADMM extra comparisons
    if not df1.empty:
        grouped_bar(df1, "multiplier", "tampering", "mitigation_count",
                    "Average Mitigation Events",
                    "Mitigations vs. Multiplier (Classical ADMM)",
                    "mitigations_vs_multiplier_classical.png",
                    desc="Classical ADMM: how the upper bound on Byzantine multiplier influences the average mitigation count under different tamper settings.")

        grouped_bar(df1, "multiplier", "tampering", "iterations",
                    "Average Iterations",
                    "Iterations vs. Multiplier (Classical ADMM)",
                    "iterations_vs_multiplier_classical.png",
                    desc="Classical ADMM: convergence iterations as a function of multiplier upper bound, stratified by tampering count.")

        grouped_bar(df1, "attack_prob", "tampering", "mitigation_count",
                    "Average Mitigation Events",
                    "Mitigations vs. Attack Probability (Classical ADMM)",
                    "mitigations_vs_attackprob_classical.png",
                    desc="Classical ADMM: sensitivity of mitigation frequency to the probability of a Byzantine attack, for different tamper levels.")

        grouped_bar(df1, "attack_prob", "tampering", "iterations",
                    "Average Iterations",
                    "Iterations vs. Attack Probability (Classical ADMM)",
                    "iterations_vs_attackprob_classical.png",
                    desc="Classical ADMM: how convergence iterations grow with increasing likelihood of a Byzantine attack across tampering counts.")

    # Relaxed ADMM extra comparisons
    if not df2.empty:
        grouped_bar(df2, "multiplier", "tampering", "iterations",
                    "Average Iterations",
                    "Iterations vs. Multiplier (Relaxed ADMM)",
                    "iterations_vs_multiplier_relaxed.png",
                    desc="Relaxed ADMM: convergence cost in iterations against the Byzantine multiplier bound, coloured by tampering intensity.")

        grouped_bar(df2, "multiplier", "tampering", "mitigation_count",
                    "Average Mitigation Events",
                    "Mitigations vs. Multiplier (Relaxed ADMM)",
                    "mitigations_vs_multiplier_relaxed.png",
                    desc="Relaxed ADMM: mitigation event frequency under varying multiplier bounds and tampering counts.")

        grouped_bar(df2, "attack_prob", "tampering", "iterations",
                    "Average Iterations",
                    "Iterations vs. Attack Probability (Relaxed ADMM)",
                    "iterations_vs_attackprob_relaxed.png",
                    desc="Relaxed ADMM: how convergence iterations respond to increasing attack probabilities across tamper settings.")

        grouped_bar(df2, "attack_prob", "tampering", "mitigation_count",
                    "Average Mitigation Events",
                    "Mitigations vs. Attack Probability (Relaxed ADMM)",
                    "mitigations_vs_attackprob_relaxed.png",
                    desc="Relaxed ADMM: mitigation frequency vs attack probability for each tamper level.")

        # ─────────────── per-run iteration plots ───────────────
    csvs = glob.glob(f"{ITER_DIR}/iter_*.csv")
    if csvs:
        # 1) individual run curves
        for p in csvs:
            gen_iter_plots(p)

        # 2) collect all runs into one DataFrame
        all_it = pd.concat([iter_df(p) for p in csvs], ignore_index=True)
        all_it = all_it[ all_it['tag'].str.contains(r"_method[12]_") ]

        # 3) per‐run overlay for Price
        fig, ax = plt.subplots(figsize=(8,6))
        for tag, g in all_it.groupby('tag'):
            ax.plot(g['iter'], g['Price'], alpha=0.3, color='gray')
        # only label the methods:
        for method, g in all_it.groupby('method'):
            ln = g.groupby('iter')['Price'].mean().reset_index()
            ax.plot(ln['iter'], ln['Price'], label=f"mean, method {method}", linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Price')
        ax.set_title('Price vs Iter (per run)')
        ax.legend(fontsize='small', ncol=2)
        plt.tight_layout()
        out = f"{PLOT_DIR}/price_per_run.png"
        fig.savefig(out)
        plt.close(fig)
        _add_caption('price_per_run.png',
                     'Price trajectory for each individual run (no averaging).')

        # 4) mean±std curves
        mean_curve(
            all_it, "iter", "Price", "method",
            ttl="Price Convergence (Mean±Std)",
            fname="Price_conv_meanstd.png",
            desc="Average price trajectory with one-std-dev band, separated by method."
        )
        mean_curve(
            all_it, "iter", "prim", "method",
            ttl="Primal Residual Convergence (Mean±Std)",
            fname="prim_conv_meanstd.png",
            desc="Average primal residual trajectory with one-std-dev band, separated by method."
        )
        mean_curve(
            all_it, "iter", "dual", "method",
            ttl="Dual Residual Convergence (Mean±Std)",
            fname="dual_conv_meanstd.png",
            desc="Average dual residual trajectory with one-std-dev band, separated by method."
        )
        mean_curve(
            all_it, "iter", "SW", "method",
            ttl="SW Convergence (Mean±Std)",
            fname="SW_conv_meanstd.png",
            desc="Average social-welfare trajectory with one-std-dev band, separated by method."
        )

        # 5) snapshot of final values per run
        last = all_it.sort_values("iter").groupby("tag").tail(1)

        boxplots(last, "tamper", "SW",
                 ttl="Final SW by Tampering",
                 fname="finalSW_box_tamper.png",
                 desc="Distribution of final social‑welfare values grouped by tampering count (per‑run view).")

        overlaid_hists(last, "SW", "tamper", bins=25,
                       ttl="Final SW distribution by tampering",
                       fname="sw_hist_by_tamper.png",
                       desc="Overlayed histogram of per‑run final social‑welfare for each tampering level.")

        cdf_plot(last, "SW", "tamper",
                 ttl="CDF of final SW by tampering",
                 fname="sw_cdf_by_tamper.png",
                 desc="Empirical CDF of final social‑welfare across tamper counts.")

        boxplots(df, "tampering", "iterations",
                 ttl="Iterations by Tampering",
                 fname="iterations_box_tamper.png",
                 desc="Box‑plot of total iterations across all runs, grouped by tampering count.")

        boxplots(last, "method", "Price",
                 ttl="Final Price by Method",
                 fname="price_box_method.png",
                 desc="Distribution of final average price outcomes split by optimisation method.")

        boxplots(last, "method", "prim",
                 ttl="Prim Final by Method",
                 fname="prim_box_method.png",
                 desc="Final primal residual distribution per method.")

        boxplots(last, "method", "dual",
                 ttl="Dual Final by Method",
                 fname="dual_box_method.png",
                 desc="Final dual residual distribution per method.")

        boxplots(df, "tampering", "avg_weight",
                 ttl="Avg Weight by Tampering",
                 fname="avgWeight_box_tamper.png",
                 desc="Average (mean) weight parameter recorded in mitigation logs vs tampering.")

        overlaid_hists(df, "iterations", "tampering", bins=30,
                       ttl="Iterations distribution by tampering",
                       fname="iter_hist_by_tampering.png",
                       desc="Histogram of total iterations per run for each tamper level.")

        cdf_plot(df, "iterations", "tampering",
                 ttl="CDF of iterations by tampering",
                 fname="iter_cdf_by_tampering.png",
                 desc="Empirical CDF of convergence iterations under different tamper counts.")

    # ─────────────── local convergence quick‑look ───────────────
    lc_pat = (
        r"local_conv_.*?(method\d)(?:_alpha([\d\.]+))?_prob([\d\.]+)"
        r"_mult([\d\.]+)_t([^_]+)(?:_run\d+)?\.log"
    )

    lc_rows = []
    for lp in glob.glob(f"{LC_DIR}/local_conv_*.log"):
        m = re.match(lc_pat, os.path.basename(lp))
        if m is None:
            continue
        meth, a_s, pr_s, mu_s, t_s = m.groups()
        alpha = float(a_s) if a_s else 0
        tam = np.inf if t_s == "inf" else float(t_s)
        for sub, it in local_conv_items(lp):
            lc_rows.append(dict(method=meth, alpha=alpha, tampering=tam,
                                subgraph=sub, conv_iter=it))
    if lc_rows:
        lcd = pd.DataFrame(lc_rows)
        grouped_bar(lcd, "tampering", "method", "conv_iter",
                    "Avg Local‑Conv Iter",
                    "Local‑Convergence Iterations vs Tampering",
                    "localConv_iter_vs_tamper.png",
                    desc="Average iterations for sub‑graph local convergence, split by method and tamper count.")

    # ─────────────── binaries summary & extra visuals ───────────────
    bdf = analyse_binaries()

    # ─────────────── summary pivot table ───────────────
    summ = (df.groupby(["method", "alpha", "attack_prob", "tampering"])
               .agg(iter_mean=("iterations", "mean"),
                    iter_std=("iterations", "std")).reset_index())
    summ.to_csv("simulation_summary_table.csv", index=False)
    print("simulation_summary_table.csv saved")

    # ─────────────── dump plot explanations ───────────────
    with open(f"{PLOT_DIR}/plot_explanations.txt", "w") as f:
        for fname, desc in PLOT_INFO:
            f.write(f"{fname} : {desc}\n")
    print(f"⮕  Wrote {len(PLOT_INFO)} figure descriptions to logs/plots/plot_explanations.txt")


if __name__ == "__main__":
    main()
