# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:56:13 2018

@author: Thomas
"""

import copy
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
        self.init_test()
        self.verbose = True
        self.timeout = 3600  # in s
        self.Interval = 3  # in s
        self.full_progress = []
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
        self.show = True
        self.progress = 'Partial'
        # Optimization model
        self.players = {}
        self.Trades = 0
        print("Initialization complete.")
        return
    
    def init_test(self):
        # FOR TESTING
        self.simulation_on = True

        if self.simulation_on:
            self.MGraph = Graph.Load('graphs/examples/P2P_model.pyp2p', format='picklez')
            #self.MGraph.BuildGraphOfMarketGraph(True)
        return
    
    def Parameters_Save(self):
        print("Option unavailable yet.")
        return ["Option unavailable yet."]
    
    def Parameters_Test(self):
        test_loc = self.location == 'local'
        test_algo = self.algorithm == 'Decentralized'
        test_target = self.target == 'CPU'
        test = test_loc and test_algo and test_target
        message = []
        if not test_loc:
            message.append("Simulation on an external server is not possible yet.")
        if not test_algo:
            message.append("Centralized simulation is not possible yet.")
        if not test_target:
            message.append("Simulation on GPU is not possible yet.")
        return test, message
    
    def Registered_Token(self, account='AWS'):
        # Look into pre-registered tokens
        if self.account_token == '':
            self.account_token = ''
        return
    
    '''def LoadMGraph(self, Graph):
        self.MGraph = Graph
        self.MGraph.Save(self.MGraph, 'temp/presim.pyp2p', format='picklez')
        return'''
    
    def ShowProgress(self, In=True):
        test, out = self.Parameters_Test()
        if test:
            print("Simulation in progress...")
            if self.progress == 'Full':
                self.Graph_Progress(In)
            else:
                print("Market graph update...")
            print("Main progress...")
        else:
            for message in out:
                print(message)
            return out
    
    def Progress_Main(self):
        self.full_progress = []
        print("Optimizer feedbacks: Init ...")
        print("Initializing parameters ...")
        return "Initializing..."
    
    def Progress_Start(self):
        print("Constructing model ...")
        return "Constructing model..."

    def Progress_Optimize(self, click=None):
        if click is not None:
            self.start_sim = time.time()  # Updated to time.time() for current Python versions
            print("Optimization started...")
            #self.Opti_LocDec_InitModel()
            print("Press 'Ctrl + C' to stop the simulation at any time.")
            try:
                self.Opti_LocDec_State()
            except KeyboardInterrupt:
                print("Simulation stopped by user.")
                self.Stopped = True
        else:
            print("Preparing for optimization...")
            # Instead of returning HTML content, you might set up and display initial optimization status
            print("Iteration | Social Welfare | Primal Residual | Dual Residual | Average Price")
            print("--------------------------------------------------------------------------------")
    
    def Graph_Progress(self, In=True, click=None):
        print("Graph progress update...")
    
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
            self.iteration_last = self.iteration # TOCHECK
            print(f"Iteration: {self.iteration}, SW: {self.SW:.3g}, Primal: {self.prim:.3g}, Dual: {self.dual:.3g}, Avg Price: {self.Price_avg * 100:.2f}")
        
        if out is None:
            out = self.Opti_End_Test()
        
        if out:
            print(f"Total simulation time: {self.simulation_time:.1f} s")
            print("Optimization stopped.")
            self.ErrorMessages()
        else:
            print("...")
            print(f"Running time: {self.simulation_time:.1f} s")

    def Opti_LocDec_Start(self, click=None):
        if click is not None:
            if not self.optimizer_on:
                self.optimizer_on = True
                self.start_sim = time.perf_counter()
                self.simulation_time = 0
            lapsed = 0
            start_time = time.perf_counter()
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
                self.Price_avg = self.Prices[self.Prices!=0].mean()
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
    
    def Button_Stop(self,click=None):
        if click is not None and click!=0:
            self.Stopped = True
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
    
    #%% Results display
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
            print("Simulation did not converge.")
            if self.simulation_message == -1:
                print("Maximum number of iterations reached.")
            elif self.simulation_message == -2:
                print("Simulation time exceeded timeout.")
            elif self.simulation_message == -3:
                print("Simulation stopped by user.")
            else:
                print("Something went wrong.")
                
    def ShowResults(self, click=None):
        self.Infos()  # Ensure all totals are calculated for display
        self.ErrorMessages()  # Display results or errors
        
        # Next steps options
        print("What do you want to do next?")
        print("1. Save results")
        print("2. Create report")
        print("3. Do another simulation")
        choice = input("Enter your choice (1, 2, or 3): ")
        if choice == "1":
            self.SaveResults()
        elif choice == "2":
            self.CreateReport()
        elif choice == "3":
            self.StartNewSimulation()
        else:
            print("Invalid option. Please enter a valid choice.")

    def SaveResults(self):
        # TOCHEK: saving results logic here (e.g., save to a file or database)
        print("Results saved (functionality to be implemented).")

    def CreateReport(self):
        # Summarize and display a report based on the simulation data
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
            print(f"Average selling price: {Selling_avg * 100:.2f} c$/kWh")
        if Perceived[self.Trades > 0].size > 0:
            Buying_avg = Perceived[self.Trades > 0].mean()
            print(f"Average buying price: {Buying_avg * 100:.2f} c$/kWh")

    def StartNewSimulation(self):
        # TOCHEK: Logic to reset the environment and start a new simulation
        print("Starting a new simulation (reset environment and reinitialize).")

    def ConfirmAction(self, action):
        confirmation = input(f"Are you sure you want to {action}? (yes/no): ").lower()
        if confirmation == "yes":
            if action == "start a new simulation":
                self.StartNewSimulation()
            elif action == "save the results":
                self.SaveResults()
        else:
            print("Action canceled.")
    
    def Exit_Simulator(self):
        print("Exiting the simulator and resetting for a new simulation...")
        self.simulation_message = False
        self.Opti_LocDec_Init()
        print("Simulator reset complete. Ready for new simulation.")

if __name__ == "__main__":
    sim = Simulator()
    sim.Opti_LocDec_Init()
    sim.Opti_LocDec_InitModel()
    sim.Opti_LocDec_Start()
    sim.ShowResults()
    sim.Exit_Simulator()
    print("Simulation complete.")
