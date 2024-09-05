# -*- coding: utf-8 -*-

META = {
    'api_version': '3.0',
    'type': 'event-based',
    'models': {
        'ProsumerSim': {
            'public': True,
            'params': [],
            'attrs': ['message'], # 'stats'
        }
    }
}

import copy
import json
import sys
import time
import random
import pandas as pd
import numpy as np
from igraph import Graph, plot
from ProsumerGUROBI_FIX import Prosumer, Manager
from cosimaSim import Simulation, Event
import mosaik_api_v3 as mosaik
from cosima_core.util.general_config import CONNECT_ATTR
from cosima_core.util.util_functions import log
import re

#import logging
#logging.basicConfig(filename='sim.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

class Simulator(Simulation):
    def __init__(self): 
        super().__init__(META)
        self.simulation_on = False
        self.simulation_message = ""
        self.force_stop = True

        self.MGraph = Graph.Load('P2P_model_reduced.pyp2p', format='picklez')

        self.timeout = 3600000  # UNUSED
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
        self.maximum_iteration = 2000000
        self.penaltyfactor = 0.01
        self.residual_primal = 1e-4
        self.residual_dual = 1e-4
        self.communications = 'Synchronous'

        # Mosaik and Cosima parameters
        self._sid = None
        self._client_name = None
        self._msg_counter = 0
        self._msg_inbox = []
        self._msg_outbox = []
        self._outbox = []
        self._output_time = 0
        self.has_finished = False 
        self.step_Size = 1000
        self.scale_factor = 1
        
        # Optimization model
        self.players = {}
        self.Trades = 0
        self.Opti_LocDec_Init()
        self.Opti_LocDec_InitModel()
        self.temps = np.zeros([self.nag, self.nag]) # Temporary trades matrix

        # SDC ADVANCED (for decentralized convergence criteria)
        self.prims = {}
        self.duals = {}
        self.converged = []
        self.converged_threshold = 1e-4
    
        self.partners = {}
        self.npartners = {} # Number of partners for each player
        self.n_optimized_partners = {} # Number of partners that has optimized for each player
        self.n_updated_partners = {} # Number of partners that has updated for each player
        self.initialize_partners()

        #plot(self.MGraph, "graph.png", layout=self.MGraph.layout("kk"))

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
        
    def get_multi_msg(self, msg_id):
        # Regex to find the last number after the second underscore
        match = re.match(r"^(.*?_\d+)(?:_(\d+))?$", msg_id)
        
        if match:
            last_number = match.group(2)  # the last number after the second underscore
            return int(last_number) if last_number is not None else 0
        else:
            # In case the msg_id doesn't match the expected pattern
            raise ValueError("msg_id format is incorrect")
    
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
            self.prims[vertex.index] = float("inf")
            self.duals[vertex.index] = float("inf")

        for edge in self.MGraph.es:
            self.partners[edge.source].append(edge.target)

        for vertex in self.MGraph.vs:
            self.npartners[vertex.index] = len(self.partners[vertex.index])
            self.n_optimized_partners[vertex.index] = 0
            self.n_updated_partners[vertex.index] = len(self.partners[vertex.index])

        print(self.partners)

    def init(self, sid, **sim_params):
        self._sid = sid
        if 'run' in sim_params.keys():
            self._run_id = sim_params['run']
        if 'client_name' in sim_params.keys():
            self.meta['models']['ProsumerSim']['attrs'].append(f'{CONNECT_ATTR}{sim_params["client_name"]}')
            self._client_name = sim_params['client_name']
        if 'step_size' in sim_params.keys():
            self.step_Size = sim_params['step_size']
        return META

    def create(self, num, model, **model_conf):
        return [{'eid': self._sid, 'type': model}]

    def step(self, time, inputs, max_advance): 
        content = 'Simulation has finished.'
        if self.has_finished:
            time = float('inf')
        else:
            # get the received messages to use them in the simulation
            if(inputs):
                data = self._msg_inbox if isinstance(self._msg_inbox, list) else json.loads(self._msg_inbox)

                # Load the new data from the inputs dictionary
                received = inputs["Simulator-0"][f'message_with_delay_for_client{self.nag}']['CommunicationSimulator-0.CommunicationSimulator'][0]
                msg_id = received['msg_id']
                lat = self.get_multi_msg(msg_id)
                start_time = received['creation_time']
                new_data = json.loads(received['content'])

                # Save latency logs (aggregated)
                with open(f'collectorLogs/collector_log_{self._run_id }', 'a') as f:
                    for content_item in new_data:
                        f.write(f'{msg_id},{((time + lat) - start_time) + (content_item["real_time"] * 1000)},{content_item["trade"]},{content_item["prim"]},{content_item["dual"]}\n')
                        #f.write(f'{msg_id},{time + lat},{start_time},{content_item["real_time"]},{content_item["trade"]}\n')

                # Update self._msg_inbox with the updated list
                data.extend(new_data)
                self._msg_inbox = json.dumps(data)

            # run the simulation
            while(self._msg_outbox == []):
                self.run()

            content = json.dumps(self._msg_outbox)
            towhom = self._msg_outbox[0]['src']
            self._msg_outbox = []

        self._outbox.append({'msg_id': f'{self._client_name}_{self._msg_counter}',
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
            data = {self._sid: {f'message': self._outbox}, 'time': self._output_time}
            self._outbox = []
        return data

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
            exit()
        
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

    # Function to check if the partners for a specific agent are present in the messages
    def check_partners(self, agent):
        data = json.loads(self._msg_inbox)
        src_set = set() # set due to presence of possible duplicates
        for message in data:
            if message['dest'] == -1: # lost message
                return False
            if message['dest'] == agent:
                src_set.add(message['src'])
        
        # Check if each partner is present in the sources
        missing_partners = [partner for partner in self.partners[agent] if partner not in src_set]
        if missing_partners:
            return False
        return True

    def update_trades(self, agent):
        partners_set = set(self.partners[agent])
        data = json.loads(self._msg_inbox)
        trades_map = {}
        to_remove = []

        for message in data:
            if message['dest'] == agent and message['src'] in partners_set:
                trades_map[message['src']] = message['trade']
                # Remove the message from the partners set and the data list
                partners_set.remove(message['src'])
                to_remove.append(message)
                # if the set is empty, break the loop
                if not partners_set:
                    break

        # Update the inbox with the remaining messages after processing
        for message in to_remove:
            data.remove(message)
        self._msg_inbox = json.dumps(data) 

        # touch carefully :)
        complete_lost_prob = 0
        if random.random() < complete_lost_prob:
            return
        
        # Update sim.Trades with the extracted trade values
        for partner in self.partners[agent]:
            if partner in trades_map:
                self.Trades[agent, partner] = trades_map[partner]
                if(self.temps[agent, partner] != trades_map[partner]): #  DEBUG: should never happen
                    print(f"Assert: Network failed")
                    exit()
            else: #  DEBUG: should never happen (this function is always called after check_partners() returns True)
                print(f"Assert: No trade value for agent {agent} and src {partner}")
                exit()

class PlayerOptimizationMsg(Event):
    def __init__(self, player_i):
        super().__init__()
        self.i = player_i
        self.wait_less = 0
        self.wait_more = 0.1
    
    def process(self, sim: Simulator):
        # if not all partners have optimized, skip the turn
        if sim.n_optimized_partners[self.i] < (sim.npartners[self.i] - self.wait_less):
            return

        # if I haven't received all the messages yet, skip the turn
        if not sim.check_partners(self.i):
            return

        if random.random() < self.wait_more:
            sim.schedule(random.randint(5, 10), PlayerOptimizationMsg(self.i))
            return

        # if the player has already converged, skip the turn
        if self.i in sim.converged:
            return

        sim.n_optimized_partners[self.i] = 0 # Reset the number of partners that have optimized

        original_values = np.copy(sim.Trades)
        sim.Trades = np.copy(sim.temps) 
        # Restore original values for players that are not partners of the current player
        for j in range(len(sim.Trades)):
            if j not in sim.partners[self.i]:
                sim.Trades[j] = original_values[j]

        sim.update_trades(self.i)

        sim.prims[self.i] = sum([sim.players[j].Res_primal for j in sim.partners[self.i]])
        sim.duals[self.i] = sum([sim.players[j].Res_dual for j in sim.partners[self.i]])

        if sim.prims[self.i] <= sim.residual_primal and sim.duals[self.i] <= sim.residual_dual:
            sim.converged.append(self.i)

        # schedule optimization for partners
        for j in sim.partners[self.i]:
            sim.n_updated_partners[j] += 1
            ratio = sim.n_updated_partners[j] / sim.npartners[j]
            delay = 10 - (ratio * (10- 6))
            sim.schedule(int(delay), PlayerUpdateMsg(j))

class PlayerUpdateMsg(Event):
    def __init__(self, player_i):
        super().__init__()
        self.i = player_i
        self.wait_less = 0
        self.wait_more = 0.1
    
    def process(self, sim: Simulator):
        # if not all partners have updated, skip the turn
        if (sim.n_updated_partners[self.i] < (sim.npartners[self.i] - self.wait_less)):
            return

        if random.random() < self.wait_more:
            sim.schedule(random.randint(5, 10), PlayerUpdateMsg(self.i))
            return

        # if the player has already converged, skip the turn
        if self.i in sim.converged:
            return
        
        # reset the number of partners that have updated
        sim.n_updated_partners[self.i] = 0

        start_time = time.time()
        sim.temps[:, self.i] = sim.players[self.i].optimize(sim.Trades[self.i, :])
        end_time = time.time()
        real_time = (end_time - start_time) * sim.scale_factor

        sim.Prices[:, self.i][sim.partners[self.i]] = sim.players[self.i].y

        # schedule optimization for partners
        for j in sim.partners[self.i]:
            sim.n_optimized_partners[j] += 1
            ratio = sim.n_optimized_partners[j] / sim.npartners[j]
            delay = 10 - (ratio * (10 - 6))
            sim.schedule(int(delay), PlayerOptimizationMsg(j))
            sim._msg_outbox.append({'src': self.i, 'dest': j, 'real_time' : real_time, 'trade':  sim.temps[j, self.i], 'prim': sim.prims[self.i], 'dual': sim.duals[self.i]})

class CheckStateEvent(Event):
    def __init__(self):
        super().__init__()
    
    def process(self, sim: Simulator):
        if len(sim.converged) == sim.nag:
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
            sim.has_finished = True
        else:
            sim.Opti_LocDec_State(False)
            sim.schedule(100, CheckStateEvent())

def main():
    
    '''if len(sys.argv) > 1:
        config_file = sys.argv[1]
        sim.load_config(config_file)
        sim.Parameters_Test()
    else:
        print("No configuration file provided. Using default parameters.")'''
    
    return mosaik.start_simulation(Simulator())

if __name__ == "__main__":
    main()
