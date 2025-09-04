# local_conv.py
from SimulatorDiscrete import Simulator

# defaults (can be overridden via self.config)
DEFAULT_SUBGRAPH_NODES = [[0,1,2], [4,5,6], [1,4,5], [0,1]]
DEFAULT_LOG_FILE = "local_conv.log"
DEFAULT_SCALE_BY_SIZE = True   # <<< NEW: scale epsilons by |S|/|V|

def _total_nodes(sim) -> int:
    # be permissive about how the simulator exposes players
    if hasattr(sim, "players") and sim.players is not None:
        try:
            return len(sim.players)
        except Exception:
            pass
    if hasattr(sim, "n_players"):
        return int(sim.n_players)
    # last resort to avoid ZeroDivision
    return 1

def _epsilons(sim):
    # grab the same epsilons used by the global stop condition
    eps_p = getattr(sim, "residual_primal", 1e-3)
    eps_d = getattr(sim, "residual_dual",   1e-3)
    return float(eps_p), float(eps_d)

def _normalize_subgraphs(raw):
    # raw can be a flat list [0,1,2] or a list of lists [[...],[...]]
    if len(raw) > 0 and isinstance(raw[0], int):
        return [raw]
    return raw

def _local_residuals(sim):
    nodesets = sim.config.get("subgraph_nodes", DEFAULT_SUBGRAPH_NODES)
    subgraphs = _normalize_subgraphs(nodesets)
    # Return (for the first subgraph) to preserve the original function’s signature
    nodes = subgraphs[0] if subgraphs else []
    prim = sum(sim.players[i].Res_primal for i in nodes)
    dual = sum(sim.players[i].Res_dual  for i in nodes)
    return prim, dual

def _local_has_converged(sim):
    # For backward compatibility, check the first configured subgraph only
    nodesets = sim.config.get("subgraph_nodes", DEFAULT_SUBGRAPH_NODES)
    subgraphs = _normalize_subgraphs(nodesets)
    if not subgraphs:
        return False

    nodes = subgraphs[0]
    prim = sum(sim.players[i].Res_primal for i in nodes)
    dual = sum(sim.players[i].Res_dual  for i in nodes)

    eps_p, eps_d = _epsilons(sim)
    V = _total_nodes(sim)
    S = len(nodes)

    scale_by_size = sim.config.get("scale_thresholds_by_size", DEFAULT_SCALE_BY_SIZE)
    scale = (S / V) if scale_by_size else 1.0

    return (prim <= scale * eps_p) and (dual <= scale * eps_d)

# keep a reference to the original state method
_orig_state = Simulator.Opti_LocDec_State

def _state_with_local_monitor(self, out):
    # 1) call original state update
    _orig_state(self, out)

    # 2) fetch config and normalize subgraphs
    raw = self.config.get("subgraph_nodes", DEFAULT_SUBGRAPH_NODES)
    subgraphs = _normalize_subgraphs(raw)

    # initialize the per‐subgraph flag dict on first call
    if not hasattr(self, "_subgraphs_converged"):
        self._subgraphs_converged = {}

    eps_p, eps_d = _epsilons(self)
    V = _total_nodes(self)
    scale_by_size = self.config.get("scale_thresholds_by_size", DEFAULT_SCALE_BY_SIZE)
    log_file = self.config.get("local_conv_log_file", DEFAULT_LOG_FILE)

    # for each sub-graph, check residuals and log once
    for nodes in subgraphs:
        key = tuple(nodes)
        S = len(nodes)
        scale = (S / V) if scale_by_size else 1.0

        prim = sum(self.players[i].Res_primal for i in nodes)
        dual = sum(self.players[i].Res_dual   for i in nodes)

        thr_p = scale * eps_p
        thr_d = scale * eps_d

        has_conv = (prim <= thr_p) and (dual <= thr_d)

        # if converged *and* not yet reported
        if has_conv and not self._subgraphs_converged.get(key, False):
            msg = (
                f"Sub-graph {list(nodes)} locally converged at iter {self.iteration} "
                f"(prim={prim:.3e} ≤ {thr_p:.3e}, dual={dual:.3e} ≤ {thr_d:.3e}; "
            )
            print("  ↳ " + msg)
            try:
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
            except IOError:
                pass
            self._subgraphs_converged[key] = True

# monkey-patch the simulator
Simulator.Opti_LocDec_State = _state_with_local_monitor

# allow running standalone if you like
def main():
    sim = Simulator()
    sim.run()

if __name__ == "__main__":
    main()
