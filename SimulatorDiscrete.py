# -*- coding: utf-8 -*-
"""
@originalAuthor: Thomas
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


class Simulator(Simulation):
    def __init__(self): 
        super().__init__()
        self.simulation_on = False
        self.simulation_message = ""
        self.force_stop = False

        self.MGraph = Graph.Load('graphs/examples/P2P_model_reduced.pyp2p', format='picklez')

        self.timeout = 3600  # UNUSED
        self.Interval = 3  # in s
        # Default optimization parameters
        self.Add_Commission_Fees = 'Yes'
        self.Commission_Fees_P2P = 1  # in c$/kWh
        self.Commission_Fees_Community = 0  # in c$/kWh
        self.algorithm = 'Decentralized'
        self.target = 'CPU'
        self.location = 'local'
        self.account = 'AWS'
        self.account_token = ''
        self.Registered_Token()
        self.maximum_iteration = 100
        self.penaltyfactor = 0.01
        self.residual_primal = 1e-4
        self.residual_dual = 1e-4
        self.communications = 'Synchronous'

        # Latency
        self.isLatency = False
        self.latency_times = []

        # Optimization model
        self.players = {}
        self.Trades = 0
        self.Opti_LocDec_Init()
        self.Opti_LocDec_InitModel()
        self.temps = np.zeros([self.nag, self.nag]) # Temporary trades matrix
    
        self.partners = {}
        self.npartners = {} # Number of partners for each player
        self.n_optimized_partners = {} # Number of partners that has optimized for each player
        self.n_updated_partners = {} # Number of partners that has updated for each player
        self.initialize_partners()

        plot(self.MGraph, "graph.png", layout=self.MGraph.layout("kk"))

        self.Opti_LocDec_Start()

        return
    
    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value) 
                else:
                    print(f"Ignoring unknown parameter: {key}")
            print("Parameters updated from config file successfully.")
        except FileNotFoundError:
            print("Config file not found.")
        except json.JSONDecodeError:
            print("Invalid JSON format in config file.")
    
    def Parameters_Test(self):
        if not self.location == 'local':
            print("Simulation on an external server is not possible yet. Using local")
            self.location = 'local'
        if not self.algorithm == 'Decentralized':
            print("Centralized simulation is not possible yet. Using decentralized")
            self.algorithm = 'Decentralized'
        if not self.target == 'CPU':
            print("Simulation on GPU is not possible yet. Using CPU")
            self.target = 'CPU'
    
    def Registered_Token(self, account='AWS'):
        # Look into pre-registered tokens
        if self.account_token == '':
            self.account_token = ''
        return
    
    #%% Optimization
    def Opti_LocDec_Init(self):
        nag = len(self.MGraph.vs)
        self.nag = nag
        self.Trades = np.zeros([nag,nag])
        self.Prices = np.zeros([nag,nag])
        self.iteration = 0
        self.iteration_last = -1
        self.SW = 0
        self.prim = float("inf")
        self.dual = float("inf")
        self.Price_avg = 0
        self.simulation_time = 0 # NOW UNUSED
        self.opti_progress = []
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
                pref[es.source][es.target] = es['weight'] + max(self.Commission_Fees_P2P/100,0)
                if self.MGraph.vs[es.source]['Type']=='Manager' and self.MGraph.vs[es.source]['CommGoal']=='Autonomy':
                    pref[es.source][es.target] += max(self.MGraph.vs[self.AgentID]['ImpFee'],0)
            elif self.MGraph.vs[es.target]['ID'] in self.MGraph.vs[es.source]['Community']:
                if self.MGraph.vs[es.source]['Type']=='Manager':
                    self.Communities[es.source].append(es.target)
                else:
                    pref[es.source][es.target] = es['weight'] + max(self.Commission_Fees_Community/100,0)
            else:
                pref[es.source][es.target] = es['weight']
        for x in self.MGraph.vs:
            if x['Type']=='Manager':
                self.players[x.index] = Manager(agent=x, partners=part[x.index], preferences=pref[x.index], rho=self.penaltyfactor)
            else:
                self.players[x.index] = Prosumer(agent=x, partners=part[x.index], preferences=pref[x.index], rho=self.penaltyfactor)
        self.part = part
        self.pref = pref
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
    
    def Opti_LocDec_Start(self):
        for i in range(self.nag):
            self.schedule(0, PlayerUpdateMsg(i))

        self.schedule(0, CheckStateEvent())
    
    def Opti_LocDec_State(self, out):
        self.iteration += 1
        
        if(self.Prices[self.Prices!=0].size!=0):
            self.Price_avg = self.Prices[self.Prices!=0].mean()
        else:
            self.Price_avg = 0
        self.SW = sum([self.players[i].SW for i in range(self.nag)])

        if self.iteration_last < self.iteration:
            self.iteration_last = self.iteration
            print(f"Iteration: {self.iteration}, SW: {self.SW:.3g}, Primal: {self.prim:.3g}, Dual: {self.dual:.3g}, Avg Price: {self.Price_avg * 100:.2f}")

        # In the last version there was the time calculation
        if out:
            print("Optimization stopped.")

    def Opti_LocDec_Stop(self):
        self.simulation_on_tab = False
        self.simulation_on = False
        return
    
    #%% Results gathering
    def Infos(self):
        self.tot_trade = np.zeros(self.Trades.shape)
        for es in self.MGraph.es:
            if self.MGraph.vs[es.source]['Type']!='Manager':
                if self.MGraph.vs[es.target]['Type']=='Manager':
                    self.tot_trade[es.source][es.target] = abs(self.Trades[es.source][es.target])
                else:
                    self.tot_trade[es.source][es.target] = abs(self.Trades[es.source][es.target])/2
        self.tot_prod = np.zeros(self.nag)
        self.tot_cons = np.zeros(self.nag)
        for i in range(self.nag):
            prod,cons = self.players[i].production_consumption()
            self.tot_prod[i] = prod
            self.tot_cons[i] = cons
    
    def ErrorMessages(self):
        if self.simulation_message == 1:
            self.Infos()
            print(f"Simulation converged after {self.iteration} iterations")
            print(f"The total social welfare is {self.SW:.0f} $.")
            print(f"The total amount of power exchanged is {self.tot_trade.sum():.0f} kW.")
            print(f"The total amount of power produced is {self.tot_prod.sum():.0f} kW.")
            print(f"The total amount of power consumed is {self.tot_cons.sum():.0f} kW.")
            print(f"With an average energy/trading price of {self.Price_avg * 100:.2f} c$/kWh.")
        else:
            if self.simulation_message == -1:
                print("Maximum number of iterations reached.")
            else:
                print("Something went wrong.")
                
    def ShowResults(self):
        self.Infos()  # Ensure all totals are calculated for display
        self.ErrorMessages()  # Display results or errors

        if self.force_stop:
            print("Simulation stopped by parameter change.")
            return
        
        while(True):
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
                return
            else:
                print("Invalid option. Please enter a valid choice.")

    def SaveResults(self):
        # NOTIMPLEMENTED: saving results logic here (e.g., save to a file or database)
        print("\tNot implemented yet")

    def CreateReport(self):
        # MOCK EXAMPLE: Displaying some report data
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

        # TODO: Add more report data as needed

    def ConfirmAction(self, action):
        confirmation = input(f"Are you sure you want to {action}? (yes/no): ").lower()
        if confirmation == "yes":
            if action == "start a new simulation":
                self.StartNewSimulation()
            elif action == "save the results":
                self.SaveResults()
        else:
            print("Action canceled.")

class PlayerOptimizationMsg(Event):
    def __init__(self, player_i):
        super().__init__()
        self.i = player_i
        self.wait_less = 0
        self.wait_more = 0
    
    def process(self, sim: Simulator):
        if sim.n_optimized_partners[self.i] < (sim.npartners[self.i] - self.wait_less):
            return

        if random.random() < self.wait_more:
            return

        sim.n_optimized_partners[self.i] = 0

        original_values = np.copy(sim.Trades)
        proposed_trades = np.copy(sim.temps)

        for j in range(len(proposed_trades)):
            if j not in sim.partners[self.i]:
                proposed_trades[j] = original_values[j]

        row_values = proposed_trades[self.i, sim.partners[self.i]]

        if len(row_values) > 0:
            row_median = np.median(row_values)
            row_mad = np.median(np.abs(row_values - row_median))

            scale_factor = 15.0
            min_threshold = 0.01

            if row_mad < 4.1:
                adaptive_threshold = float('inf')
            else:
                adaptive_threshold = max(scale_factor * row_mad, min_threshold)

            for idx, j in enumerate(sim.partners[self.i]):
                deviation = abs(row_values[idx] - row_median)
                if deviation > adaptive_threshold:
                    weight = min((deviation - adaptive_threshold)/deviation, 0.8)
                    new_value = (1 - weight)*row_values[idx] + weight*row_median
                    row_values[idx] = new_value
                    with open("log_mitigation.txt", "a") as f:
                        f.write((
                            f"[Mitigation in PlayerOptimizationMsgMitigated] Agent {self.i} -> Partner {j}"
                            f": deviation={deviation:.2f}, median={row_median:.2f}, "
                            f"threshold={adaptive_threshold:.2f}, corrected={new_value:.2f}\n"
                        ))

            proposed_trades[self.i, sim.partners[self.i]] = row_values

        sim.Trades = proposed_trades

        sim.prim = sum([sim.players[j].Res_primal for j in sim.partners[self.i]])
        sim.dual = sum([sim.players[j].Res_dual for j in sim.partners[self.i]])

        max_delay = 10 + random.randint(0, 2) if sim.isLatency else 10
        for j in sim.partners[self.i]:
            sim.n_updated_partners[j] += 1
            ratio = sim.n_updated_partners[j] / sim.npartners[j]
            delay = max_delay - (ratio * (max_delay - 6))
            sim.latency_times.append(delay)
            sim.schedule(int(delay), PlayerUpdateMsg(j))

class PlayerUpdateMsg(Event):
    def __init__(self, player_i):
        super().__init__()
        self.i = player_i
        self.wait_less = 0
        self.wait_more = 0
    
    def process(self, sim: Simulator):
        # Only proceed when enough partners have updated
        if sim.n_updated_partners[self.i] < (sim.npartners[self.i] - self.wait_less):
            return

        # Reset the counter for updated partners
        sim.n_updated_partners[self.i] = 0

        # Make a copy of the current trade vector for self.i
        robust_trade = np.copy(sim.Trades[self.i, :])

        # Get partner indices
        partner_indices = sim.partners[self.i]
        
        if partner_indices:
            # Gather all trade values from partners
            partner_trades = [sim.Trades[self.i, j] for j in partner_indices]
            
            # Calculate median and median absolute deviation (MAD)
            median_trade = np.median(partner_trades)
            mad = np.median(np.abs(np.array(partner_trades) - median_trade))
            
            # Set adaptive threshold with minimum value
            min_threshold = 0.01  # Absolute minimum threshold 
            scale_factor = 15.0    # How many MADs to allow (THE HIGHER IS THIS THE LOWER CAN BE THE CHECK BELOW)

            if mad < 4.1:  # or another small threshold
                adaptive_threshold = float('inf')  # so no value is filtered
            else:
                adaptive_threshold = max(scale_factor * mad, min_threshold)
            
            # Apply robust filtering to ALL partners
            for j in partner_indices:
                deviation = abs(sim.Trades[self.i, j] - median_trade)
                if deviation > adaptive_threshold:
                    # Calculate weight that decreases as deviation increases
                    weight = min((deviation - adaptive_threshold) / deviation, 0.8)  # Cap at 80% replacement
                    # Blend the reported value with the median
                    new_value = (1 - weight) * sim.Trades[self.i, j] + weight * median_trade
                    robust_trade[j] = new_value
                    # Log the mitigation
                    with open("log_mitigation.txt", "a") as f:
                        f.write(f"Mitigated agent {j}: deviation={deviation:.2f}, median={median_trade:.2f}, " +
                                f"weight={weight:.2f}, threshold={adaptive_threshold:.2f}, " +
                                f"original={sim.Trades[self.i, j]:.2f}, new={new_value:.2f}\n")

        # Use the robust trade vector for optimization
        sim.temps[:, self.i] = sim.players[self.i].optimize(robust_trade)
        sim.Prices[:, self.i][sim.partners[self.i]] = sim.players[self.i].y

        # schedule optimization for partners
        maxval = 10 + random.randint(0, 2) if sim.isLatency else 10
        for j in sim.partners[self.i]:
            sim.n_optimized_partners[j] += 1
            ratio = sim.n_optimized_partners[j] / sim.npartners[j]
            delay = maxval - (ratio * (maxval - 6))
            sim.latency_times.append(delay)
            sim.schedule(int(delay), PlayerOptimizationMsg(j))

class CheckStateEvent(Event):
    def __init__(self):
        super().__init__()
    
    def process(self, sim: Simulator):
        if sim.prim<=sim.residual_primal and sim.dual<=sim.residual_dual:
            sim.simulation_message = 1
        elif sim.iteration>=sim.maximum_iteration:
            sim.simulation_message = -1
        else:
            sim.simulation_message = 0

        if sim.simulation_message:
            sim.Opti_LocDec_Stop()
            sim.Opti_LocDec_State(True)
            sim.ShowResults()
            sim.events = [] # like doing exit() but allowing the profiler
            return
        else:
            sim.Opti_LocDec_State(False)
            sim.schedule(100, CheckStateEvent())

def main():
    # Initialize the simulator
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