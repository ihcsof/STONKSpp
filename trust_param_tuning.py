#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd

try:
    from ProsumerGUROBI_FIX import Simulator
except ImportError:
    pass

def parse_log_file(fname):
    if not os.path.exists(fname):
        return {"events": 0, "metric": 0}
    c = 0
    m = 0
    with open(fname, 'r') as f:
        for line in f:
            pass
    return {"events": c, "metric": m}

def run_simulation(cfg):
    sim = Simulator(config=cfg)
    sim.run()
    r = {"iterations": getattr(sim, "iteration", 0), "final_obj": getattr(sim, "final_objective", 0)}
    log_stats = parse_log_file(cfg.get("log_file", "log_sim.txt"))
    r.update(log_stats)
    return r

def main():
    combos = []
    smoothing_factors = [0.1, 0.25, 0.4, 0.8]
    suspicion_decays = [0.1, 0.25, 0.5, 0.95]
    runs = 10
    for s in smoothing_factors:
        for d in suspicion_decays:
            for n in range(runs):
                cfg = {
                    "smoothing_factor": s,
                    "suspicion_decay": d,
                    "log_file": f"log_s{s}_d{d}_run{n}.txt"
                }
                res = run_simulation(cfg)
                res["smoothing_factor"] = s
                res["suspicion_decay"] = d
                res["run"] = n
                combos.append(res)
    df = pd.DataFrame(combos)
    df.to_csv("simulation_results.csv", index=False)
    grouped = df.groupby(["smoothing_factor","suspicion_decay"]).agg({
        "iterations":"mean","final_obj":"mean","events":"mean","metric":"mean"
    }).reset_index()
    grouped["iterations"] = grouped["iterations"].round(2)
    ranking = grouped.sort_values(by="iterations", ascending=True)
    ranking.to_csv("ranking_smoothing_suspicion.csv", index=False)
    print(ranking)

if __name__ == "__main__":
    main()
