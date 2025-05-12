#!/usr/bin/env python3
"""
Randomly delete ("prune") **N** edges from a graph **without isolating any node**, then verify by reloading.

Supports the project-specific *.pyp2p* files (gzipped pickled igraph graphs).

Usage:
    python prune_graph.py -i graphs/examples/P2P_model.pyp2p -n 100 [--inplace]

If --inplace is omitted, writes to <input>_pruned.<ext>.
"""

import argparse
import gzip
import os
import pickle
import random
from typing import Tuple

try:
    from MGraph import MGraph  # type: ignore
    _HAS_MGRAPH = True
except ImportError:
    _HAS_MGRAPH = False

from igraph import Graph

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _read_pyp2p(path: str) -> Graph:
    with gzip.open(path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, Graph):
        return obj
    if _HAS_MGRAPH and isinstance(obj, MGraph):
        try:
            return obj._g  # extract underlying igraph.Graph
        except Exception:
            raise OSError("Cannot extract igraph.Graph from MGraph pickle")
    raise OSError(".pyp2p did not contain an igraph Graph object")

def _write_pyp2p(g: Graph, path: str) -> None:
    with gzip.open(path, "wb") as fh:
        pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)

def _load_graph(path: str) -> Graph:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pyp2p":
        return _read_pyp2p(path)
    if _HAS_MGRAPH:
        try:
            fmt = "picklez" if ext == ".pyp2p" else None
            return MGraph.Load(path, format=fmt)  # type: ignore[arg-type]
        except Exception:
            pass
    try:
        return Graph.Read(path)
    except (KeyError, OSError):
        raise OSError("Unknown format. Install MGraph or convert to GraphML/GML.")

def _save_graph(g: Graph, out_path: str) -> None:
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".pyp2p":
        _write_pyp2p(g, out_path); return
    if _HAS_MGRAPH and isinstance(g, MGraph):
        fmt = "picklez" if ext == ".pyp2p" else None
        MGraph.Save(g, out_path, format=fmt); return
    g.write(out_path)

# ---------------------------------------------------------------------------
# Core pruning logic
# ---------------------------------------------------------------------------
def _random_delete_edges(g: Graph, n: int) -> Tuple[Graph, int]:
    removed = 0
    ids = list(range(g.ecount())); random.shuffle(ids)
    for eid in ids:
        if removed >= n: break
        if eid >= g.ecount(): continue
        v1, v2 = g.es[eid].tuple
        if g.degree(v1) > 1 and g.degree(v2) > 1:
            g.delete_edges([eid]); removed += 1
    if removed < n:
        print(f"⚠️  Only removed {removed}/{n} without isolating nodes.")
    return g, removed

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(prog="prune_graph.py",
        description="Prune edges and verify by reload.")
    parser.add_argument("--input", "-i", required=True, help="Input graph file")
    parser.add_argument("--number", "-n", type=int, required=True, help="Edges to delete")
    parser.add_argument("--inplace", "-r", action="store_true",
                        help="Overwrite input")
    args = parser.parse_args()

    graph = _load_graph(args.input)
    original = graph.ecount()
    graph, removed = _random_delete_edges(graph, args.number)

    if args.inplace:
        out_path = args.input
    else:
        out_path = f"{os.path.splitext(args.input)[0]}_pruned{os.path.splitext(args.input)[1]}"

    _save_graph(graph, out_path)

    # Reload to verify
    reloaded = _load_graph(out_path)
    new_count = reloaded.ecount()

    print(f"✅ Saved to {out_path}")
    print(f"🔍 Original edges: {original}")
    print(f"🔍 Removed edges:  {removed}")
    print(f"🔍 Edges after reload: {new_count}")

if __name__ == "__main__":
    main()
