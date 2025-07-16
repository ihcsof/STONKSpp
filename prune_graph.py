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
def _random_delete_edges(g: Graph, n_pairs: int) -> Tuple[Graph, int]:
    """
    Remove n_pairs bidirectional links (two directed edges each) without
    isolating any vertex. Returns the graph and the total number of edges
    actually deleted (2 * n_pairs, unless degree constraints prevent full removal).
    """
    removed_pairs = 0

    # Build a list of unique undirected links
    undirected = list({tuple(sorted(e.tuple)) for e in g.es})
    random.shuffle(undirected)

    for v1, v2 in undirected:
        if removed_pairs >= n_pairs:
            break

        # Both directions must exist
        if not (g.are_connected(v1, v2) and g.are_connected(v2, v1)):
            continue

        # Ensure neither endpoint would become isolated (needs degree > 2)
        if g.degree(v1) > 2 and g.degree(v2) > 2:
            eid1 = g.get_eid(v1, v2, directed=True)
            eid2 = g.get_eid(v2, v1, directed=True)
            g.delete_edges([eid1, eid2])
            removed_pairs += 1

    if removed_pairs < n_pairs:
        print(f"⚠️  Only removed {removed_pairs}/{n_pairs} edge pairs without isolating nodes.")

    # Return the modified graph and the number of edges removed
    return g, removed_pairs * 2

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
