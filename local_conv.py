# local_conv.py
from SimulatorDiscrete import Simulator

# default, if not overridden via config
DEFAULT_SUBGRAPH_NODES = [[0,1,2], [4,5,6], [1,4,5], [0,1]]
DEFAULT_LOG_FILE = "local_conv.log"

def _local_residuals(sim):
    nodes = sim.config.get("subgraph_nodes", DEFAULT_SUBGRAPH_NODES)
    prim = sum(sim.players[i].Res_primal for i in nodes)
    dual = sum(sim.players[i].Res_dual  for i in nodes)
    return prim, dual

def _local_has_converged(sim):
    prim, dual = _local_residuals(sim)
    return prim <= sim.residual_primal and dual <= sim.residual_dual

# keep a reference to the original state method
_orig_state = Simulator.Opti_LocDec_State

def _state_with_local_monitor(self, out):
    # 1) call original state update
    _orig_state(self, out)

    # 2) fetch raw config: could be a flat list [0,1,2] or a list of lists [[0,1,2],[3,4,5]]
    raw = self.config.get("subgraph_nodes", DEFAULT_SUBGRAPH_NODES)
    # normalize to a list of lists
    if len(raw) > 0 and isinstance(raw[0], int):
        subgraphs = [raw]
    else:
        subgraphs = raw

    # initialize the per‐subgraph flag dict on first call
    if not hasattr(self, "_subgraphs_converged"):
        self._subgraphs_converged = {}

    # for each sub-graph, check/residuals and log once
    for nodes in subgraphs:
        key = tuple(nodes)
        prim = sum(self.players[i].Res_primal for i in nodes)
        dual = sum(self.players[i].Res_dual   for i in nodes)
        has_conv = (prim <= self.residual_primal and dual <= self.residual_dual)

        # if converged *and* not yet reported
        if has_conv and not self._subgraphs_converged.get(key, False):
            msg = (
                f"Sub-graph {nodes} locally converged at iter {self.iteration} "
                f"(prim={prim:.3e}, dual={dual:.3e})"
            )
            print("  ↳ " + msg)

            log_file = self.config.get("local_conv_log_file", DEFAULT_LOG_FILE)
            try:
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
            except IOError:
                pass

            # mark this subgraph as done
            self._subgraphs_converged[key] = True


# monkey-patch the simulator
Simulator.Opti_LocDec_State = _state_with_local_monitor

# allow running standalone if you like
def main():
    sim = Simulator()
    sim.run()

if __name__ == "__main__":
    main()