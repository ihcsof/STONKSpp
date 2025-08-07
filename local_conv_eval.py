#!/usr/bin/env python3
import re, glob, os
import csv

# directory containing your MAD logs
LOG_DIR = "logs/local_conv"

# capture method, p_attack, mult, t, run from filename
FNAME_PATT = re.compile(
    r"MAD_method(?P<method>\d+)"
    r"_prob(?P<p>[\d\.]+)"
    r"_mult(?P<mult>[\d\.]+)"
    r"_t(?P<t>[\d_inf]+)"
    r"_run(?P<run>\d+)\.log$"
)

# capture lines like:
# Sub-graph [..] locally converged at iter N (prim=..., dual=...)
LINE_RE = re.compile(
    r"Sub-graph\s+\[(?P<subgraph>[^\]]+)\]\s+"
    r"locally converged at iter\s+(?P<it>\d+)\s+"
    r"\(prim=(?P<prim>[\deE\+\-\.]+),\s*dual=(?P<dual>[\deE\+\-\.]+)\)"
)

# will hold best record per (method,p,t,run,subgraph)
best = {}

def process(path):
    fname = os.path.basename(path)
    m = FNAME_PATT.search(fname)
    if not m:
        return
    md = m.groupdict()
    method   = int(md["method"])
    p_attack = float(md["p"])
    t        = md["t"]
    run      = int(md["run"])
    for L in open(path):
        lm = LINE_RE.search(L)
        if not lm:
            continue
        it = int(lm.group("it"))
        if it < 5:
            continue
        sub = lm.group("subgraph")
        key = (method, p_attack, t, run, sub)
        rec = {
            "method":   method,
            "p_attack": p_attack,
            "t":        t,
            "run":      run,
            "subgraph": sub,
            "iter":     it,
            "prim":     float(lm.group("prim")),
            "dual":     float(lm.group("dual")),
        }
        # keep the record with largest iter per subgraph
        if key not in best or best[key]["iter"] < it:
            best[key] = rec

if __name__=="__main__":
    # scan for MAD logs
    for path in glob.glob(os.path.join(LOG_DIR, "*MAD*.log")):
        process(path)

    if not best:
        print("No valid MAD logs found or all convergences < iter 5.")
        exit(1)

    # write out CSV
    out_fields = ["method","p_attack","t","run","subgraph","iter","prim","dual"]
    with open("local_conv_mad.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for rec in sorted(best.values(),
                          key=lambda r:(r["method"],r["p_attack"],r["t"],r["run"],r["subgraph"])):
            writer.writerow(rec)

    print("Wrote local_conv_mad.csv with {} records.".format(len(best)))

