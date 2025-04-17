from SimulatorDiscrete import Simulator

SUBGRAPH_NODES = [0, 1, 2]

def _local_residuals(sim):
    prim = sum(sim.players[i].Res_primal for i in SUBGRAPH_NODES)
    dual = sum(sim.players[i].Res_dual for i in SUBGRAPH_NODES)
    return prim, dual

def _local_has_converged(sim):
    prim, dual = _local_residuals(sim)
    return prim <= sim.residual_primal and dual <= sim.residual_dual

_orig_state = Simulator.Opti_LocDec_State

def _state_with_local_monitor(self, out):
    _orig_state(self, out)
    prim, dual = _local_residuals(self)
    if _local_has_converged(self):
        if not getattr(self, '_subgraph_converged', False):
            print(
                f"  ↳ Sub‑graph {SUBGRAPH_NODES} locally converged at "
                f"iter {self.iteration} "
                f"(prim={prim:.3e}, dual={dual:.3e})"
            )
            self._subgraph_converged = True

Simulator.Opti_LocDec_State = _state_with_local_monitor

def main():
    sim = Simulator()
    sim.run()

if __name__ == '__main__':
    main()
