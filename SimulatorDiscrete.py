# simulator_simple_trust_ignore.py
# -*- coding: utf-8 -*-
"""
@originalAuthor: Thomas
Modified approach to flagging truly bad (Byzantine) agents only after repeated suspicious trades.
"""

import copy
import json
import sys
import time
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from igraph import Graph, plot
from ProsumerGUROBI_FIX import Prosumer, Manager
from discrete_event_sim import Simulation, Event
import logging

class Simulator(Simulation):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.log_mitigation_file = self.config.get("log_mitigation_file", "debug_log.txt")
        self.simulation_on = False
        self.simulation_message = ""
        self.force_stop = False
        
        # Load graph
        default_graph = "graphs/examples/P2P_model_reduced.pyp2p"
        graph_path    = self.config.get("graph_file", default_graph)
        self.MGraph = Graph.Load(graph_path, format='picklez')
        
        logging.info("Loaded graph from file.")
        self.timeout = 3600
        self.Interval = 3
        self.Add_Commission_Fees = 'Yes'
        self.Commission_Fees_P2P = 1
        self.Commission_Fees_Community = 0
        self.algorithm = 'Decentralized'
        self.target = 'CPU'
        self.location = 'local'
        self.account = 'AWS'
        self.account_token = ''
        self.Registered_Token()
        self.maximum_iteration = 500
        self.penaltyfactor = 0.01
        self.residual_primal = 1e-3
        self.residual_dual = 1e-3
        self.communications = 'Synchronous'
        self.isLatency = False
        self.latency_times = []
        self.iter_update_method = "method2"
        self.players = {}
        self.Trades = 0
        self.Opti_LocDec_Init()
        self.Opti_LocDec_InitModel()
        self.temps = np.zeros([self.nag, self.nag])
        self.partners = {}
        self.npartners = {}
        self.n_optimized_partners = {}
        self.n_updated_partners = {}
        self.initialize_partners()

        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
        logging.basicConfig(filename=self.log_mitigation_file,
                            level=logging.DEBUG,
                            format="%(message)s",
                            filemode="w")

        logging.info("Initialization complete: created %d agents.", self.nag)
        #plot(self.MGraph, "graph.png", layout=self.MGraph.layout("kk"))
        logging.info("Graph plotted and saved to graph.png.")
        self.trust_threshold = self.config.get("trust_threshold", 30)
        self.byz_score = {}
        for i in range(self.nag):
            self.byz_score[i] = {}
        logging.info("Trust parameters set: trust_threshold=%d", self.trust_threshold)

        self.Opti_LocDec_Start()
        logging.info("Initial events scheduled. Starting simulation.")
        return

    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            self.config = config_data
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"Ignoring unknown parameter: {key}")
            print("Parameters updated from config file successfully.")
            logging.info("Config file loaded and parameters updated.")
        except FileNotFoundError:
            print("Config file not found.")
            logging.error("Config file not found.")
        except json.JSONDecodeError:
            print("Invalid JSON format in config file.")
            logging.error("Invalid JSON format in config file.")

    def Parameters_Test(self):
        if self.location != 'local':
            print("Simulation on an external server is not possible yet. Using local")
            self.location = 'local'
        if self.algorithm != 'Decentralized':
            print("Centralized simulation is not possible yet. Using decentralized")
            self.algorithm = 'Decentralized'
        if self.target != 'CPU':
            print("Simulation on GPU is not possible yet. Using CPU")
            self.target = 'CPU'
        logging.info("Parameters_Test completed.")

    def Registered_Token(self, account='AWS'):
        if self.account_token == '':
            self.account_token = ''
        return
    
    def SaveBinaryState(self, filename):
        """
        Store all the heavy data we might want later, compressed.
        """
        import gzip, pickle

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
        return
    
    #%% Optimization
    def Opti_LocDec_Init(self):
        nag = len(self.MGraph.vs)
        self.nag = nag
        self.trust_flags = [False for _ in range(nag)]
        self.Trades = np.zeros([nag, nag])
        self.Prices = np.zeros([nag, nag])
        self.iteration = 0
        self.iteration_last = -1
        self.SW = 0
        self.prim = float("inf")
        self.dual = float("inf")
        self.Price_avg = 0
        self.simulation_time = 0
        self.opti_progress = []
        logging.info("Opti_LocDec_Init: Initialized trades, prices, and iteration counters.")
        return

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
            elif self.MGraph.vs[es.target]['ID'] in self.MGraph.vs[es.source]['Community']:
                if self.MGraph.vs[es.source]['Type'] == 'Manager':
                    self.Communities[es.source].append(es.target)
                else:
                    pref[es.source][es.target] = es['weight'] + max(self.Commission_Fees_Community / 100, 0)
            else:
                pref[es.source][es.target] = es['weight']
        for x in self.MGraph.vs:
            if x['Type'] == 'Manager':
                self.players[x.index] = Manager(agent=x, partners=part[x.index],
                                                preferences=pref[x.index],
                                                rho=self.penaltyfactor, config=self.config)
            else:
                self.players[x.index] = Prosumer(agent=x, partners=part[x.index],
                                                 preferences=pref[x.index],
                                                 rho=self.penaltyfactor, config=self.config)
        self.part = part
        self.pref = pref
        logging.info("Opti_LocDec_InitModel: Model built for %d players.", self.nag)
        return

    def initialize_partners(self):
        for vertex in self.MGraph.vs:
            self.partners[vertex.index] = []
        for edge in self.MGraph.es:
            self.partners[edge.source].append(edge.target)
        for vertex in self.MGraph.vs:
            self.npartners[vertex.index] = len(self.partners[vertex.index])
            self.n_optimized_partners[vertex.index] = 0
            self.n_updated_partners[vertex.index] = len(self.partners[vertex.index])
        logging.info("initialize_partners: Partners initialized.")
    
    def Opti_LocDec_Start(self):
        for i in range(self.nag):
            self.schedule(0, PlayerUpdateMsg(i))
        self.schedule(0, CheckStateEvent())
        logging.info("Opti_LocDec_Start: Initial scheduling done.")

    def Opti_LocDec_State(self, out):
        if self.iteration >= self.maximum_iteration:
            return
            
        self.iteration += 1
        if self.Prices[self.Prices != 0].size != 0:
            self.Price_avg = self.Prices[self.Prices != 0].mean()
        else:
            self.Price_avg = 0
        self.SW = sum([self.players[i].SW for i in range(self.nag)])

        iter_log = self.config.get("iter_log_file", None)
        if iter_log:
            header = "iter,SW,avg_price,prim,dual\n"
            line = (
                f"{self.iteration},{self.SW:.6g},{self.Price_avg:.6g},"
                f"{self.prim:.6g},{self.dual:.6g}\n"
            )
            mode = "a" if self.iteration > 1 else "w"
            with open(iter_log, mode) as f:
                if self.iteration == 1:
                    f.write(header)
                f.write(line)

        if self.iteration_last < self.iteration:
            self.iteration_last = self.iteration
            print(f"Iteration: {self.iteration}, SW: {self.SW:.3g}, Primal: {self.prim:.3g}, Dual: {self.dual:.3g}, Avg Price: {self.Price_avg * 100:.2f}")
        if out:
            print("Optimization stopped.")
            logging.info("Opti_LocDec_State: Simulation stopping at iteration %d.", self.iteration)

    def Opti_LocDec_Stop(self):
        self.simulation_on_tab = False
        self.simulation_on = False
        logging.info("Opti_LocDec_Stop: Simulation stopped.")
        return

    def Infos(self):
        self.tot_trade = np.zeros(self.Trades.shape)
        for es in self.MGraph.es:
            if self.MGraph.vs[es.source]['Type'] != 'Manager':
                if self.MGraph.vs[es.target]['Type'] == 'Manager':
                    self.tot_trade[es.source][es.target] = abs(self.Trades[es.source][es.target])
                else:
                    self.tot_trade[es.source][es.target] = abs(self.Trades[es.source][es.target]) / 2
        self.tot_prod = np.zeros(self.nag)
        self.tot_cons = np.zeros(self.nag)
        for i in range(self.nag):
            prod, cons = self.players[i].production_consumption()
            self.tot_prod[i] = prod
            self.tot_cons[i] = cons
        logging.info("Infos: Totals computed for trade, production, and consumption.")

    def ErrorMessages(self):
        if self.simulation_message == 1:
            self.Infos()
            print(f"Simulation converged after {self.iteration} iterations")
            print(f"The total social welfare is {self.SW:.0f} $.")
            print(f"The total amount of power exchanged is {self.tot_trade.sum():.0f} kW.")
            print(f"The total amount of power produced is {self.tot_prod.sum():.0f} kW.")
            print(f"The total amount of power consumed is {self.tot_cons.sum():.0f} kW.")
            print(f"With an average energy/trading price of {self.Price_avg * 100:.2f} c$/kWh.")
            logging.info("ErrorMessages: Simulation converged.")
        else:
            if self.simulation_message == -1:
                print("Maximum number of iterations reached.")
                logging.info("ErrorMessages: Maximum iterations reached.")
            else:
                print("Something went wrong.")
                logging.error("ErrorMessages: Unknown error.")

    def ShowResults(self):
        self.Infos()
        self.ErrorMessages()
        #if (hasattr(self, "non_interactive") and self.non_interactive) or (hasattr(self, "config") and self.config.get("non_interactive", False)):
        print("Non-interactive mode: Exiting simulator.")
        logging.info("ShowResults: Non-interactive mode - exit.")
        return
        while True:
            print("What do you want to do next?")
            print("1. Save results")
            print("2. Create report")
            print("3. Exit")
            choice = input("Enter your choice (1, 2 or 3): ")
            if choice == "1":
                self.SaveResults()
            elif choice == "2":
                self.CreateReport()
            elif choice == "3":
                print("Exiting the simulator.")
                logging.info("ShowResults: Exiting simulator after user request.")
                return
            else:
                print("Invalid option.")

    def SaveResults(self):
        print("\tNot implemented yet")
        logging.info("SaveResults: Not implemented yet.")

    def CreateReport(self):
        Perceived = np.zeros([self.nag, self.nag])
        for i in range(self.nag):
            for j in range(self.players[i].data.num_partners):
                m = self.players[i].data.partners[j]
                if self.Trades[i][m] < 0:
                    Perceived[i][m] = self.Prices[i][m] + self.players[i].data.pref[j]
                elif self.Trades[i][m] > 0:
                    Perceived[i][m] = self.Prices[i][m] - self.players[i].data.pref[j]
        if Perceived[self.Trades < 0].size > 0:
            Selling_avg = Perceived[self.Trades < 0].mean()
            print(f"\tAverage selling price: {Selling_avg * 100:.2f} c$/kWh")
        if Perceived[self.Trades > 0].size > 0:
            Buying_avg = Perceived[self.Trades > 0].mean()
            print(f"\tAverage buying price: {Buying_avg * 100:.2f} c$/kWh")
        logging.info("CreateReport: Report generated.")

    def ConfirmAction(self, action):
        confirmation = input(f"Are you sure you want to {action}? (yes/no): ").lower()
        if confirmation == "yes":
            if action == "start a new simulation":
                self.StartNewSimulation()
            elif action == "save the results":
                self.SaveResults()
        else:
            print("Action canceled.")
            logging.info("ConfirmAction: Action canceled by user.")


class PlayerOptimizationMsg(Event):
    def __init__(self, player_i):
        super().__init__()
        self.i = player_i
        self.wait_less = 0
        self.wait_more = 0

    def process(self, sim: Simulator):
        decay = sim.config.get("suspicion_decay", 0.95)
        for pj in list(sim.byz_score[self.i].keys()):
            sim.byz_score[self.i][pj] *= decay
        if sim.n_optimized_partners[self.i] < (sim.npartners[self.i] - self.wait_less):
            return
        if random.random() < self.wait_more:
            return
        sim.n_optimized_partners[self.i] = 0
        new_trades = np.copy(sim.temps)
        for row_i in range(sim.nag):
            if row_i != self.i:
                new_trades[row_i] = sim.Trades[row_i]
        partner_list = sim.partners[self.i]
        if partner_list and not sim.players[self.i].data.isByzantine:
            row_values = new_trades[self.i, partner_list]
            median_val = np.median(row_values)
            mad_val = np.median(np.abs(row_values - median_val))
            scale_factor = sim.config.get("scale_factor", 15.0)
            min_threshold = sim.config.get("min_threshold", 0.01)
            mad_threshold = sim.config.get("mad_threshold", 4.1)
            smoothing_factor = sim.config.get("smoothing_factor", 0.4)
            if mad_val < mad_threshold:
                dev_threshold = float('inf')
            else:
                dev_threshold = max(scale_factor * mad_val, min_threshold)
            logging.debug(f"[PlayerOptMsg] Agent {self.i}: median={median_val:.2f}, mad={mad_val:.2f}, threshold={dev_threshold:.2f}")
            for idx, partner_j in enumerate(partner_list):
                if sim.trust_flags[partner_j]:
                    logging.debug(f"Agent {self.i} skipping already flagged partner {partner_j}.")
                    new_trades[self.i, partner_j] = 0.0
                    continue
                if self.i == partner_j:
                    continue
                deviation = abs(row_values[idx] - median_val)
                if deviation > dev_threshold:
                    if partner_j not in sim.byz_score[self.i]:
                        sim.byz_score[self.i][partner_j] = 0
                    sim.byz_score[self.i][partner_j] += 1
                    score = sim.byz_score[self.i][partner_j]
                    if score < sim.trust_threshold:
                        factor = (score / sim.trust_threshold)
                        old_val = row_values[idx]
                        new_val = old_val - (smoothing_factor * factor) * (old_val - median_val)
                        new_trades[self.i, partner_j] = new_val
                        logging.info(f"Agent {self.i} partially corrects partner {partner_j}: {old_val:.2f}->{new_val:.2f} (score={score:.2f})")
                    else:
                        sim.trust_flags[partner_j] = True
                        new_trades[self.i, partner_j] = 0.0
                        logging.info(f"[Flag] Agent {self.i} flags partner {partner_j} (score={score}>=threshold)")
        sim.Trades = new_trades
        sim.prim = sum([sim.players[p].Res_primal for p in sim.partners[self.i]])
        sim.dual = sum([sim.players[p].Res_dual for p in sim.partners[self.i]])
        max_delay = 10
        for p_j in sim.partners[self.i]:
            sim.n_updated_partners[p_j] += 1
            ratio = sim.n_updated_partners[p_j] / sim.npartners[p_j] if sim.npartners[p_j] > 0 else 1.0
            delay = max_delay - (ratio * (max_delay - 6))
            sim.latency_times.append(delay)
            sim.schedule(int(delay), PlayerUpdateMsg(p_j))

class PlayerUpdateMsg(Event):
    def __init__(self, player_i):
        super().__init__()
        self.i = player_i
        self.wait_less = 0
        self.wait_more = 0

    def process(self, sim: Simulator):
        decay = sim.config.get("suspicion_decay", 0.95)
        for pj in list(sim.byz_score[self.i].keys()):
            sim.byz_score[self.i][pj] *= decay
        if sim.n_updated_partners[self.i] < (sim.npartners[self.i] - self.wait_less):
            return
        sim.n_updated_partners[self.i] = 0
        if len(sim.partners[self.i]) == 0:
            return
        trades_i = np.copy(sim.Trades[self.i, :])
        partner_list = sim.partners[self.i]
        if partner_list and not sim.players[self.i].data.isByzantine:
            row_values = trades_i[partner_list]
            median_val = np.median(row_values)
            mad_val = np.median(np.abs(row_values - median_val))
            scale_factor = sim.config.get("scale_factor", 15.0)
            min_threshold = sim.config.get("min_threshold", 0.01)
            mad_threshold = sim.config.get("mad_threshold", 4.1)
            smoothing_factor = sim.config.get("smoothing_factor", 0.4)
            if mad_val < mad_threshold:
                dev_threshold = float('inf')
            else:
                dev_threshold = max(scale_factor * mad_val, min_threshold)
            logging.debug(f"[PlayerUpdateMsg] Agent {self.i}: median={median_val:.2f}, MAD={mad_val:.2f}, threshold={dev_threshold:.2f}")
            for idx, partner_j in enumerate(partner_list):
                if sim.trust_flags[partner_j]:
                    logging.debug(f"Agent {self.i} skipping flagged partner {partner_j} (update).")
                    trades_i[partner_j] = 0.0
                    continue
                if self.i == partner_j:
                    continue
                deviation = abs(row_values[idx] - median_val)
                if deviation > dev_threshold:
                    if partner_j not in sim.byz_score[self.i]:
                        sim.byz_score[self.i][partner_j] = 0
                    sim.byz_score[self.i][partner_j] += 1
                    score = sim.byz_score[self.i][partner_j]
                    if score < sim.trust_threshold:
                        factor = (score / sim.trust_threshold)
                        old_val = row_values[idx]
                        new_val = old_val - (smoothing_factor * factor) * (old_val - median_val)
                        trades_i[partner_j] = new_val
                        logging.info(f"Agent {self.i} partially corrects partner {partner_j} (update): {old_val:.2f}->{new_val:.2f} (score={score:.2f})")
                    else:
                        sim.trust_flags[partner_j] = True
                        trades_i[partner_j] = 0.0
                        logging.info(f"[Flag] Agent {self.i} flags partner {partner_j} as Byzantine (update) (score={score}>=threshold)")
        sim.temps[:, self.i] = sim.players[self.i].optimize(trades_i)
        for p_j in sim.partners[self.i]:
            partner_array = sim.players[self.i].data.partners
            loc_idx = np.where(partner_array == p_j)[0][0]
            sim.Prices[p_j, self.i] = sim.players[self.i].y[loc_idx]
        max_delay = 10
        for p_j in sim.partners[self.i]:
            sim.n_optimized_partners[p_j] += 1
            ratio = sim.n_optimized_partners[p_j] / sim.npartners[p_j] if sim.npartners[p_j] > 0 else 1.0
            delay = max_delay - (ratio * (max_delay - 6))
            sim.latency_times.append(delay)
            sim.schedule(int(delay), PlayerOptimizationMsg(p_j))

class CheckStateEvent(Event):
    def __init__(self):
        super().__init__()

    def process(self, sim: Simulator):
        if sim.prim <= sim.residual_primal and sim.dual <= sim.residual_dual:
            sim.simulation_message = 1
        elif sim.iteration >= sim.maximum_iteration:
            sim.simulation_message = -1
        else:
            sim.simulation_message = 0
        if sim.simulation_message:
            sim.Opti_LocDec_Stop()
            sim.Opti_LocDec_State(True)
            sim.ShowResults()
            sim.events = []
            logging.info("CheckStateEvent: Simulation terminated.")
            return
        else:
            sim.Opti_LocDec_State(False)
            sim.schedule(100, CheckStateEvent())


def main():
    sim = Simulator()
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        sim.load_config(config_file)
        sim.Parameters_Test()
    else:
        print("No configuration file provided. Using default parameters.")
    sim.run()


if __name__ == "__main__":
    main()
