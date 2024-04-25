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
from igraph import Graph, plot
from ProsumerGUROBI_FIX import Prosumer, Manager
from discrete_event_sim import Simulation, Event


class Simulator(Simulation):
    def __init__(self): 
        super().__init__()
        self.simulation_on = False
        self.simulation_message = ""
        self.force_stop = False

        self.MGraph = Graph.Load('graphs/examples/P2P_model.pyp2p', format='picklez')

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
        self.maximum_iteration = 2000
        self.penaltyfactor = 0.01
        self.residual_primal = 1e-4
        self.residual_dual = 1e-4
        self.communications = 'Synchronous'
        # Optimization model
        self.players = {}
        self.Trades = 0

        self.partners = {}
        self.npart = {} # Number of partners for each player
        self.npartopt = {} # Number of partners that has optimized for each player
        self.initialize_partners()
        # print all the partners for each player
        print("Neigbors debugging: ")
        for vertex in self.MGraph.vs:
            print(f"Player {vertex.index} has partners {self.partners[vertex.index]}")
            print(f"\t it has {self.npart[vertex.index]} partners")

        plot(self.MGraph, "graph.png", layout=self.MGraph.layout("kk"))

        self.Opti_LocDec_Init()
        self.Opti_LocDec_InitModel()
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
            self.npart[vertex.index] = len(self.partners[vertex.index])
            # the very first time, all partners have optimized (otherwise nothing happens)
            self.npartopt[vertex.index] = len(self.partners[vertex.index])
    
    def Opti_LocDec_Start(self):
        for i in range(self.nag):
            self.schedule(0, PlayerOptimizationMsg(i))

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
    
    def process(self, sim: Simulator):
        #print(f"Player {self.i} is optimizing...")
        #if (sim.prim > sim.residual_primal or sim.dual > sim.residual_dual) and sim.iteration < sim.maximum_iteration and not (np.isnan(sim.prim) or np.isnan(sim.dual)):
        
        # SYNC: if not all partners have optimized, skip the turn
        if sim.npartopt[self.i] < sim.npart[self.i]:
            sim.schedule(8, PlayerOptimizationMsg(self.i))
            return
        
        sim.npartopt[self.i] = 0 # Reset the number of partners that have optimized

        sim.Trades[:, self.i] = sim.players[self.i].optimize(sim.Trades[self.i, :])
        sim.Prices[:, self.i][sim.partners[self.i]] = sim.players[self.i].y

        local_primal = sum(sim.players[j].Res_primal for j in sim.partners[self.i] if j != self.i)
        local_dual = sum(sim.players[j].Res_dual for j in sim.partners[self.i] if j != self.i)

        sim.prim = min(sim.prim, local_primal)
        sim.dual = min(sim.dual, local_dual)

        for j in sim.partners[self.i]:
            sim.npartopt[j] += 1
            sim.schedule(8, PlayerOptimizationMsg(j))

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
            # TODO: maybe there are still event to be processed?
            exit()
        else:
            sim.Opti_LocDec_State(False)
            sim.schedule(5, CheckStateEvent())

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