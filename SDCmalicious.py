# -*- coding: utf-8 -*-

META = {
    'api_version': '3.0',
    'type': 'event-based',
    'models': {
        'ProsumerSim': {
            'public': True,
            'params': [],
            'attrs': ['message'],  # 'stats' if needed
        }
    }
}

from typing import List, Tuple
import json
import math
import sys
import time
import random
import numpy as np
from igraph import Graph
from ProsumerGUROBI_FIX import Prosumer, Manager
from cosimaSim import Simulation, Event
import mosaik_api_v3 as mosaik
from cosima_core.util.general_config import CONNECT_ATTR
import re
import gzip, pickle

class Simulator(Simulation):
    def __init__(self):
        super().__init__(META)

        # ---- General/co-sim defaults ----
        self.logname = 'collectorLogs'
        self.simulation_on = False
        self.simulation_message = ""
        self.force_stop = True

        # ---- Core algorithm params ----
        self.timeout = 3600000
        self.Interval = 3
        self.Add_Commission_Fees = 'Yes'
        self.Commission_Fees_P2P = 1      # c$/kWh
        self.Commission_Fees_Community = 0
        self.algorithm = 'Decentralized'
        self.target = 'CPU'
        self.location = 'local'
        self.account = 'AWS'
        self.account_token = ''
        self.maximum_iteration = 2_000_000
        self.penaltyfactor = 0.01
        self.residual_primal = 1e-4
        self.residual_dual = 1e-4
        self.communications = 'Synchronous'

        # ---- Malicious/mitigation config ----
        self.config = {}
        self.log_mitigation_file = "log_mitigation.txt"
        self.iter_log_file = None   # e.g. "iter_log.csv"
        self.isLatency = False
        self.scale_factor = 1.0
        self.mad_threshold = 4.1
        self.mad_scale = 15.0
        self.min_threshold = 0.01

        # ---- Network/cosim state ----
        self._sid = None
        self._client_name = None
        self._msg_counter = 0
        self._msg_inbox = []   # will JSON-serialize to list of messages
        self._msg_outbox = []  # per-step messages to send
        self._outbox = []      # mosaik get_data outbox
        self._output_time = 0
        self.has_finished = False
        self.step_Size = 1000

        # ---- Graph / agents ----
        self.MGraph = None
        self.players = {}
        self.Trades = None
        self.Prices = None
        self.temps = None
        self.nag = 0

        # ---- Partner accounting ----
        self.partners = {}
        self.npartners = {}
        self.n_optimized_partners = {}
        self.n_updated_partners = {}

        # ---- Iteration/progress ----
        self.iteration = 0
        self.iteration_last = -1
        self.SW = 0
        self.prim = float("inf")
        self.dual = float("inf")
        self.Price_avg = 0
        self.opti_progress = []

    # ---------------------- Utility / Config ----------------------
    def Registered_Token(self, account='AWS'):
        if self.account_token == '':
            self.account_token = ''
        return

    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as f:
                cfg = json.load(f)

            # Keep a copy and expose a few keys directly
            self.config = cfg

            # Optional knobs
            self.iter_log_file = cfg.get("iter_log_file", self.iter_log_file)
            self.log_mitigation_file = cfg.get("log_mitigation_file", self.log_mitigation_file)
            self.isLatency = cfg.get("isLatency", self.isLatency)
            self.scale_factor = cfg.get("scale_factor", self.scale_factor)
            self.mad_threshold = cfg.get("mad_threshold", self.mad_threshold)
            self.mad_scale = cfg.get("scale_factor_mad", cfg.get("mad_scale", self.mad_scale))
            self.min_threshold = cfg.get("min_threshold", self.min_threshold)
            self.maximum_iteration = cfg.get("maximum_iteration", self.maximum_iteration)
            self.residual_primal = cfg.get("residual_primal", self.residual_primal)
            self.residual_dual = cfg.get("residual_dual", self.residual_dual)

            # Generic passthrough: copy any attribute if it exists on self
            for k, v in cfg.items():
                if hasattr(self, k):
                    setattr(self, k, v)

            print("Parameters updated from config file successfully.")
        except FileNotFoundError:
            print(f"Config file '{config_file}' not found.")
        except json.JSONDecodeError:
            print("Invalid JSON format in config file.")

    def SaveBinaryState(self, filename):
        payload = {
            "config":    self.config,
            "iteration": self.iteration,
            "Trades":    self.Trades,
            "Prices":    self.Prices,
            "progress":  self.opti_progress,
            "players": {
                i: {
                    "Res_primal": p.Res_primal,
                    "Res_dual":   p.Res_dual,
                    "SW":         p.SW
                }
                for i, p in self.players.items()
            }
        }
        with gzip.open(filename, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---------------------- Mosaik hooks ----------------------
    def init(self, sid, **sim_params):
        self._sid = sid
        if 'run' in sim_params:
            self._run_id = sim_params['run']
        if 'client_name' in sim_params:
            self.meta['models']['ProsumerSim']['attrs'].append(f'{CONNECT_ATTR}{sim_params["client_name"]}')
            self._client_name = sim_params['client_name']
        if 'step_size' in sim_params:
            self.step_Size = sim_params['step_size']
        if 'scale_factor' in sim_params:
            self.scale_factor = sim_params['scale_factor']
            print(f"Scale factor: {self.scale_factor}")
        if 'graph' in sim_params:
            self.MGraph = Graph.Load(sim_params['graph'], format='picklez')
        else:
            raise RuntimeError("Graph not provided to co-sim init().")
        if 'name' in sim_params:
            self.logname = sim_params['name']
        if 'config' in sim_params:
            # allow passing an external JSON file path
            self.load_config(sim_params['config'])

        self.Opti_LocDec_Init()
        self.Opti_LocDec_InitModel()
        self.temps = np.zeros([self.nag, self.nag])
        self.initialize_partners()
        self.Opti_LocDec_Start()
        return META

    def create(self, num, model, **model_conf):
        return [{'eid': self._sid, 'type': model}]

    def step(self, time, inputs, max_advance):
        content = 'Simulation has finished.'
        if self.has_finished:
            time = float('inf')
        else:
            if inputs:
                # Messages are packed by communication simulator; unpack and append to inbox
                received = inputs["Simulator-0"][f'message_with_delay_for_client{self.nag}']['CommunicationSimulator-0.CommunicationSimulator'][0]
                msg_id = received['msg_id']
                lat = self._get_multi_msg(msg_id)
                start_time = received['creation_time']
                new_data = json.loads(received['content'])

                # Append latency logs (aggregated)
                with open(f'{self.logname}/collector_log_{self._run_id}', 'a') as f:
                    for content_item in new_data:
                        # (sim_time + lat) - start + content_item["real_time"]*1000
                        f.write(f'{msg_id},{((time + lat) - start_time) + (content_item["real_time"] * 1000)},{content_item["trade"]},{content_item["prim"]},{content_item["dual"]}\n')

                if isinstance(self._msg_inbox, list):
                    data = self._msg_inbox
                else:
                    data = json.loads(self._msg_inbox)
                data.extend(new_data)
                self._msg_inbox = json.dumps(data)

            # Drive the discrete-event engine until we have outbound messages
            while not self._msg_outbox:
                self.run()

            content = json.dumps(self._msg_outbox)
            towhom = self._msg_outbox[0]['src']
            self._msg_outbox = []

        self._outbox.append({
            'msg_id': f'{self._client_name}_{self._msg_counter}',
            'max_advance': max_advance,
            'sim_time': time + 1,
            'sender': self._client_name,
            'receiver': f'client{towhom}',
            'content': content,
            'creation_time': time,
        })
        self._msg_counter += 1
        self._output_time = time + 1
        return None

    def get_data(self, outputs):
        data = {}
        if self._outbox:
            data = {self._sid: {'message': self._outbox}, 'time': self._output_time}
            self._outbox = []
        return data

    # ---------------------- Core optimization wiring ----------------------
    def Opti_LocDec_Init(self):
        nag = len(self.MGraph.vs)
        self.nag = nag
        self.Trades = np.zeros([nag, nag])
        self.Prices = np.zeros([nag, nag])
        self.iteration = 0
        self.iteration_last = -1
        self.SW = 0
        self.prim = float("inf")
        self.dual = float("inf")
        self.Price_avg = 0
        self.opti_progress = []

    def Opti_LocDec_InitModel(self):
        self.Communities = {}
        for x in self.MGraph.vs.select(Type='Manager'):
            self.Communities[x.index] = []

        part = np.zeros(self.Trades.shape)
        pref = np.zeros(self.Trades.shape)

        for es in self.MGraph.es:
            part[es.source][es.target] = 1
            if self.MGraph.vs[es.target]['ID'] in self.MGraph.vs[es.source]['Partners']:
                pref[es.source][es.target] = es['weight'] + max(self.Commission_Fees_P2P / 100, 0)
                if self.MGraph.vs[es.source]['Type'] == 'Manager' and self.MGraph.vs[es.source]['CommGoal'] == 'Autonomy':
                    pref[es.source][es.target] += max(self.MGraph.vs[self.MGraph.vs[es.source]['ID']]['ImpFee'] if 'ImpFee' in self.MGraph.vs[self.MGraph.vs[es.source]['ID']] else 0, 0)
            elif self.MGraph.vs[es.target]['ID'] in self.MGraph.vs[es.source]['Community']:
                if self.MGraph.vs[es.source]['Type'] == 'Manager':
                    self.Communities[es.source].append(es.target)
                else:
                    pref[es.source][es.target] = es['weight'] + max(self.Commission_Fees_Community / 100, 0)
            else:
                pref[es.source][es.target] = es['weight']

        # Create agents passing config (so byzantine/mitigation knobs propagate)
        self.players = {}
        for x in self.MGraph.vs:
            if x['Type'] == 'Manager':
                self.players[x.index] = Manager(agent=x, partners=part[x.index], preferences=pref[x.index], rho=self.penaltyfactor, config=self.config)
            else:
                self.players[x.index] = Prosumer(agent=x, partners=part[x.index], preferences=pref[x.index], rho=self.penaltyfactor, config=self.config)

        self.part = part
        self.pref = pref

    def initialize_partners(self):
        self.partners = {v.index: [] for v in self.MGraph.vs}
        for e in self.MGraph.es:
            self.partners[e.source].append(e.target)
        for v in self.MGraph.vs:
            self.npartners[v.index] = len(self.partners[v.index])
            self.n_optimized_partners[v.index] = 0
            self.n_updated_partners[v.index] = len(self.partners[v.index])

    def Opti_LocDec_Start(self):
        for i in range(self.nag):
            self.schedule(0, PlayerUpdateMsg(i))
        self.schedule(0, CheckStateEvent())

    def Opti_LocDec_Stop(self):
        self.simulation_on_tab = False
        self.simulation_on = False

    def Opti_LocDec_State(self, out: bool):
        # iteration book-keeping
        self.iteration += 1
        if self.Prices[self.Prices != 0].size != 0:
            self.Price_avg = self.Prices[self.Prices != 0].mean()
        else:
            self.Price_avg = 0
        self.SW = sum(self.players[i].SW for i in range(self.nag))

        # Optional iteration log
        if self.iter_log_file:
            header = "iter,SW,avg_price,prim,dual\n"
            line = f"{self.iteration},{self.SW:.6g},{self.Price_avg:.6g},{self.prim:.6g},{self.dual:.6g}\n"
            mode = "a" if self.iteration > 1 else "w"
            with open(self.iter_log_file, mode) as f:
                if self.iteration == 1:
                    f.write(header)
                f.write(line)

        if self.iteration_last < self.iteration:
            self.iteration_last = self.iteration
            print(f"Iteration: {self.iteration}, SW: {self.SW:.3g}, Primal: {self.prim:.3g}, Dual: {self.dual:.3g}, Avg Price: {self.Price_avg * 100:.2f}")

        if out:
            print("Optimization stopped.")

    # ---------------------- Networking helpers ----------------------
    def _get_multi_msg(self, msg_id: str) -> int:
        # Extract trailing *_<n> optional counter
        match = re.match(r"^(.*?_\d+)(?:_(\d+))?$", msg_id)
        if match:
            last_number = match.group(2)
            return int(last_number) if last_number is not None else 0
        raise ValueError("msg_id format is incorrect")

    def check_partners(self, agent: int) -> bool:
        data = json.loads(self._msg_inbox) if not isinstance(self._msg_inbox, list) else self._msg_inbox
        src_set = set()
        for message in data:
            if message.get('dest') == -1:  # lost message
                return False
            if message.get('dest') == agent:
                src_set.add(message.get('src'))
        missing = [p for p in self.partners[agent] if p not in src_set]
        return not missing

    def update_trades(self, agent: int):
        partners_set = set(self.partners[agent])
        data = json.loads(self._msg_inbox) if not isinstance(self._msg_inbox, list) else self._msg_inbox
        trades_map = {}
        to_remove = []

        for message in data:
            if message.get('dest') == agent and message.get('src') in partners_set:
                trades_map[message['src']] = message['trade']
                partners_set.remove(message['src'])
                to_remove.append(message)
                if not partners_set:
                    break

        # prune processed messages
        for m in to_remove:
            data.remove(m)
        self._msg_inbox = json.dumps(data)

        # apply trades
        for partner in self.partners[agent]:
            if partner in trades_map:
                self.Trades[agent, partner] = trades_map[partner]
                # Assert network consistency with temps
                # (cols are temps[:, self.i], row is Trades[self.i, :]
                # For clarity, we don't hard-assert exit; we trust mitigation to damp anomalies.
            else:
                # missing value shouldn't happen after check_partners()
                pass

    # ---------------------- Robust/MAD helpers ----------------------
    def _robust_center(self, values: np.ndarray) -> Tuple[float, float, float]:
        m = len(values)
        if m == 0:
            return 0.0, 0.0, float('inf')

        k = int(math.log2(m))
        if k > m:
            k = m
        if (m - k) % 2 != 0:
            k -= 1
        if k < 1:
            center = float(np.median(values))
        else:
            sv = np.sort(values)
            start = (m - k) // 2
            window = sv[start:start + k]
            center = float(np.mean(window))

        mad = float(np.median(np.abs(values - center)))
        if mad < self.mad_threshold:
            threshold = float('inf')
        else:
            threshold = max(self.mad_scale * mad, self.min_threshold)
        return center, mad, threshold

    def _mitigate_row(self, row_vals: np.ndarray, partner_ids: List[int], tag: str, agent_i: int):
        if len(row_vals) == 0:
            return row_vals
        center, mad, threshold = self._robust_center(row_vals.copy())
        mitigated = row_vals.copy()
        for idx, _ in enumerate(row_vals):
            deviation = abs(row_vals[idx] - center)
            if deviation > threshold:
                weight = min((deviation - threshold) / max(deviation, 1e-12), 0.9)
                new_val = (1 - weight) * row_vals[idx] + weight * center
                mitigated[idx] = new_val
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                log_line = (f"{ts} - [{tag}] Agent {agent_i} partner {partner_ids[idx]}: "
                            f"dev={deviation:.4f}, center={center:.4f}, thresh={threshold:.4f}, "
                            f"orig={row_vals[idx]:.4f}, new={new_val:.4f}\n")
                with open(self.log_mitigation_file, "a") as f:
                    f.write(log_line)
        return mitigated

    # ---------------------- Events ----------------------
class PlayerOptimizationMsg(Event):
    def __init__(self, player_i):
        super().__init__()
        self.i = player_i
        self.wait_less = 0
        self.wait_more = 0

    def process(self, sim: Simulator):
        # Need all partners to have optimized AND messages received
        if sim.n_optimized_partners[self.i] < (sim.npartners[self.i] - self.wait_less):
            return
        if not sim.check_partners(self.i):
            return
        if random.random() < self.wait_more:
            return

        sim.n_optimized_partners[self.i] = 0

        # Start from last temps (proposed trades)
        proposed = np.copy(sim.temps)
        original = np.copy(sim.Trades)

        # Keep non-partner rows as original (only update links to my partners)
        for j in range(len(proposed)):
            if j not in sim.partners[self.i]:
                proposed[j] = original[j]

        # Apply network-delivered trades for row i (from partners)
        sim.Trades = proposed  # provisional
        sim.update_trades(self.i)

        # Mitigate row i *after* ingesting network trades (robust against malicious inputs)
        partner_ids = sim.partners[self.i]
        if partner_ids:
            row_vals = np.array([sim.Trades[self.i, j] for j in partner_ids], dtype=float)
            row_vals = sim._mitigate_row(row_vals, partner_ids, "OptMsg", self.i)
            for idx, pj in enumerate(partner_ids):
                sim.Trades[self.i, pj] = row_vals[idx]

        # Update residuals scoped to partners
        sim.prim = sum(sim.players[j].Res_primal for j in sim.partners[self.i])
        sim.dual = sum(sim.players[j].Res_dual for j in sim.partners[self.i])

        # Schedule next updates
        max_delay = 10 + (random.randint(0, 2) if sim.isLatency else 0)
        for j in sim.partners[self.i]:
            sim.n_updated_partners[j] += 1
            ratio = sim.n_updated_partners[j] / max(sim.npartners[j], 1)
            delay = max_delay - (ratio * (max_delay - 6))
            sim.schedule(int(delay), PlayerUpdateMsg(j))


class PlayerUpdateMsg(Event):
    def __init__(self, player_i):
        super().__init__()
        self.i = player_i
        self.wait_less = 0
        self.wait_more = 0

    def process(self, sim: Simulator):
        if sim.n_updated_partners[self.i] < (sim.npartners[self.i] - self.wait_less):
            return
        if random.random() < self.wait_more:
            return

        sim.n_updated_partners[self.i] = 0

        # Build a robust view of the row used by optimizer (protect against outliers)
        partner_ids = sim.partners[self.i]
        robust_row = np.copy(sim.Trades[self.i, :])
        if partner_ids:
            trades = np.array([sim.Trades[self.i, j] for j in partner_ids], dtype=float)
            mitigated = sim._mitigate_row(trades, partner_ids, "UpdMsg", self.i)
            for idx, pj in enumerate(partner_ids):
                robust_row[pj] = mitigated[idx]

        # Optimize column self.i using robust row
        start_time = time.time()
        sim.temps[:, self.i] = sim.players[self.i].optimize(robust_row)
        end_time = time.time()
        real_time = (end_time - start_time) * sim.scale_factor

        sim.Prices[:, self.i][partner_ids] = sim.players[self.i].y

        # Fan-out messages to partners with timing
        max_delay = 10 + (random.randint(0, 2) if sim.isLatency else 0)
        for j in partner_ids:
            sim.n_optimized_partners[j] += 1
            ratio = sim.n_optimized_partners[j] / max(sim.npartners[j], 1)
            delay = max_delay - (ratio * (max_delay - 6))
            sim.schedule(int(delay), PlayerOptimizationMsg(j))
            sim._msg_outbox.append({
                'src': self.i,
                'dest': j,
                'real_time': real_time,
                'trade': float(sim.temps[j, self.i]),
                'prim': sim.prim,
                'dual': sim.dual
            })


class CheckStateEvent(Event):
    def __init__(self):
        super().__init__()

    def process(self, sim: Simulator):
        # stopping criteria
        if sim.prim <= sim.residual_primal and sim.dual <= sim.residual_dual:
            sim.simulation_message = 1
        elif sim.iteration >= sim.maximum_iteration:
            sim.simulation_message = -1
        else:
            sim.simulation_message = 0

        if sim.simulation_message:
            sim.Opti_LocDec_Stop()
            sim.Opti_LocDec_State(True)
            # show final metrics succinctly (no interactive menu in co-sim)
            total_trade = np.abs(sim.Trades).sum()
            print(f"[RESULT] iters={sim.iteration} SW={sim.SW:.1f} price_avg={sim.Price_avg*100:.2f}c$/kWh trade_sum={total_trade:.1f}")
            sim.events = []   # allow profiler/driver to exit cleanly
            sim.has_finished = True
        else:
            sim.Opti_LocDec_State(False)
            sim.schedule(100, CheckStateEvent())


def main():
    # Co-simulation by default
    return mosaik.start_simulation(Simulator())

if __name__ == "__main__":
    main()
