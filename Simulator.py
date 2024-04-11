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
from igraph import Graph
from ProsumerCVX import Prosumer, Manager


class Simulator:
    def __init__(self): 
        self.simulation_on = False
        self.optimizer_on = False
        self.simulation_message = ""
        self.Stopped = False

        self.MGraph = Graph.Load('graphs/examples/Pool_model.pyp2p', format='picklez')

        self.timeout = 3600  # in s
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
        self.maximum_iteration = 5000
        self.penaltyfactor = 0.01
        self.residual_primal = 1e-4
        self.residual_dual = 1e-4
        self.communications = 'Synchronous'
        # Optimization model
        self.players = {}
        self.Trades = 0
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
    
    def Progress_Optimize(self):
        self.start_sim = time.time()  # Updated to time.time() for current Python versions
        print("Optimization started...")
        print("Press 'Ctrl + C' to stop the simulation at any time.")
        try:
            self.Opti_LocDec_State()
        except KeyboardInterrupt:
            print("Simulation stopped by user.")
            self.Stopped = True
    
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
        self.simulation_time = 0
        self.opti_progress = []
        self.Stopped = False
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
    
    def Opti_LocDec_State(self, out=None):
        if self.iteration_last < self.iteration:
            self.iteration_last = self.iteration
            print(f"Iteration: {self.iteration}, SW: {self.SW:.3g}, Primal: {self.prim:.3g}, Dual: {self.dual:.3g}, Avg Price: {self.Price_avg * 100:.2f}")
        
        if out is None:
            out = self.Opti_End_Test()

        if out:
            print(f"Total simulation time: {self.simulation_time:.1f} s")
            print("Optimization stopped.")
        else:
            print(f"...Running time: {self.simulation_time:.1f} s")

    def Opti_LocDec_Start(self):
        if not self.optimizer_on:
            self.optimizer_on = True
            self.start_sim = time.perf_counter()
            self.simulation_time = 0
        lapsed = 0
        start_time = time.perf_counter()
        # check if self.prim is not Nan
        if np.isnan(self.prim) or np.isnan(self.dual):
            self.Stopped = True
        while (self.prim > self.residual_primal or self.dual > self.residual_dual) and self.iteration < self.maximum_iteration and lapsed <= self.Interval and not self.Stopped:
            self.iteration += 1
            temp = np.copy(self.Trades)
            for i in range(self.nag):
                temp[:, i] = self.players[i].optimize(self.Trades[i, :])
                self.Prices[:, i][self.part[i, :].nonzero()] = self.players[i].y
            self.Trades = np.copy(temp)
            self.prim = sum([self.players[i].Res_primal for i in range(self.nag)])
            self.dual = sum([self.players[i].Res_dual for i in range(self.nag)])
            lapsed = time.perf_counter() - start_time

        self.simulation_time += lapsed
        if(self.Prices[self.Prices!=0].size!=0):
            self.Price_avg = self.Prices[self.Prices!=0].mean()
        else:
            self.Price_avg = 0
        self.SW = sum([self.players[i].SW for i in range(self.nag)])
        
        if self.Opti_End_Test():
            self.Opti_LocDec_Stop()
            return self.Opti_LocDec_State(True)
        else:
            return self.Opti_LocDec_State(False)
    
    def Opti_LocDec_Stop(self):
        self.optimizer_on = False
        self.simulation_on_tab = False
        self.simulation_on = False
        return
    
    def Opti_End_Test(self):
        if self.prim<=self.residual_primal and self.dual<=self.residual_dual:
            self.simulation_message = 1
        elif self.iteration>=self.maximum_iteration:
            self.simulation_message = -1
        elif self.simulation_time>=self.timeout:
            self.simulation_message = -2
        elif self.Stopped:
            self.simulation_message = -3
        else:
            self.simulation_message = 0
        return self.simulation_message
    
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
            print(f"Simulation converged after {self.iteration} iterations in {self.simulation_time:.1f} seconds.")
            print(f"The total social welfare is {self.SW:.0f} $.")
            print(f"The total amount of power exchanged is {self.tot_trade.sum():.0f} kW.")
            print(f"The total amount of power produced is {self.tot_prod.sum():.0f} kW.")
            print(f"The total amount of power consumed is {self.tot_cons.sum():.0f} kW.")
            print(f"With an average energy/trading price of {self.Price_avg * 100:.2f} c$/kWh.")
        else:
            if self.simulation_message == -1:
                print("Maximum number of iterations reached.")
            elif self.simulation_message == -2:
                print("Simulation time exceeded timeout.")
            elif self.simulation_message == -3:
                print("Simulation stopped by user.")
            else:
                print("Something went wrong.")
                
    def ShowResults(self):
        self.Infos()  # Ensure all totals are calculated for display
        self.ErrorMessages()  # Display results or errors
        
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

def main():
    # Initialize the simulator
    sim = Simulator()
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        sim.load_config(config_file)
        sim.Parameters_Test()
    else:
        print("No configuration file provided. Using default parameters.")
    
    sim.Opti_LocDec_Init()
    sim.Opti_LocDec_InitModel()
    sim.Progress_Optimize()
    while(True):
        if sim.simulation_message:
            break
        sim.Opti_LocDec_Start()
    
    sim.ShowResults()

if __name__ == "__main__":
    main()