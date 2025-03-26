# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:11:54 2018

Original Author: fmoret
Updated by: [Your Name]
"""

import random
import gurobipy as gb
import numpy as np

# Class for flexible attribute storage.
class expando(object):
    pass

# Subproblem: Prosumer
class Prosumer:
    def __init__(self, agent=None, partners=None, preferences=None, rho=1, config=None):
        self.data = expando()
        self.Who()
        # Store config for use in the agent (default to empty dict if None)
        self.config = config if config is not None else {}
        
        # Data -- Agent and its assets
        if agent is not None:
            self.data.type = agent['Type']
            self.data.id = agent['ID']
            if "byzantine_ids" in self.config:
                self.data.isByzantine = (self.data.id in self.config["byzantine_ids"])
                print("Byzantine flag set to", self.data.isByzantine, "for agent", self.data.id)
            else:
                self.data.isByzantine = (self.data.id == 2)
                print("Byzantine flag set to", self.data.isByzantine, "for agent", self.data.id)
            self.data.tampered = 0
            self.data.CM = (agent['Type'] == 'Manager')
            if agent['AssetsNum'] <= len(agent['Assets']):
                self.data.num_assets = agent['AssetsNum']
                self.data.a = np.zeros([self.data.num_assets])
                self.data.b = np.zeros([self.data.num_assets])
                self.data.Pmin = np.zeros([self.data.num_assets])
                self.data.Pmax = np.zeros([self.data.num_assets])
                for i in range(self.data.num_assets):
                    if agent['Assets'][i]['costfct'] == 'Quadratic':
                        self.data.a[i] = agent['Assets'][i]['costfct_coeff'][0]
                        self.data.b[i] = agent['Assets'][i]['costfct_coeff'][1]
                        self.data.Pmax[i] = agent['Assets'][i]['p_bounds_up']
                        self.data.Pmin[i] = agent['Assets'][i]['p_bounds_low']
            else:
                self.data.num_assets = 0
        
        # Data -- Partnerships
        if partners is not None:
            self.data.partners = partners.nonzero()[0]
        else:
            self.data.partners = np.zeros([0])
        self.data.num_partners = len(self.data.partners)
        
        # Data -- Preferences
        if preferences is not None:
            self.data.pref = preferences[partners.nonzero()]
        else:
            self.data.pref = np.zeros([0])
        
        # Data -- Penalty factor
        self.data.rho = rho
        
        # Data -- Progress
        self.SW = 0
        self.Res_primal = 0
        self.Res_dual = 0
        
        # Model variables
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._build_model()
        
        # Initialize iteration variables
        self.t_old = np.zeros(self.data.num_partners)
        self.t_new = np.zeros(self.data.num_partners)
        self.y = np.zeros(self.data.num_partners)
        self.y0 = np.zeros(self.data.num_partners)
        
        # Choose the iterative update method based on config (default is "method2")
        method = self.config.get("iter_update_method", "method2")
        if method == "method1":
            self.iter_update_method = self._iter_update_method1
        elif method == "method2":
            self.iter_update_method = self._iter_update_method2
        else:
            raise ValueError("Unknown iter_update_method in config: " + method)
        return

    def optimize(self, trade):
        # Use the chosen iterative update method
        self.iter_update_method(trade)
        self._update_objective()
        self.model.optimize()
        if self.model.Status == gb.GRB.Status.OPTIMAL:
            self._opti_status(trade)
            
            val = self.t_old.copy()
            # Check for Byzantine behavior
            if self.data.isByzantine and self.data.tampered < 1:
                chance = self.config.get("byzantine_attack_probability", 0.05)
                lower = self.config.get("byzantine_multiplier_lower", 0.5)
                upper = self.config.get("byzantine_multiplier_upper", 1.2)
                if random.random() < chance:
                    self.data.tampered += 1
                    # For tampering, we use the upper multiplier (tampering magnitude)
                    multiplier = upper
                    val *= multiplier

            trade[self.data.partners] = val
        return trade
    
    def production_consumption(self):
        if self.model.Status == gb.GRB.Status.OPTIMAL:
            prod = abs(np.array([self.variables.p[i].X for i in range(self.data.num_assets) if self.variables.p[i].X > 0]).sum())
            cons = abs(np.array([self.variables.p[i].X for i in range(self.data.num_assets) if self.variables.p[i].X < 0]).sum())
        else:
            prod = 0
            cons = 0
            print("Cannot compute production and consumption because the model did not solve optimally.")
        return prod, cons

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self.model.setParam('OutputFlag', False)
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()
        return

    def _build_variables(self):
        m = self.model
        self.variables.p = np.array([m.addVar(lb=self.data.Pmin[i], ub=self.data.Pmax[i], name='p') for i in range(self.data.num_assets)])
        self.variables.t = np.array([m.addVar(lb=-gb.GRB.INFINITY, name='t') for i in range(self.data.num_partners)])
        self.variables.t_pos = np.array([m.addVar(name='t_pos') for i in range(self.data.num_partners)])
        m.update()
        return
        
    def _build_constraints(self):
        self.constraints.pow_bal = self.model.addConstr(sum(self.variables.p) == sum(self.variables.t))
        for i in range(self.data.num_partners):
            self.model.addConstr(self.variables.t[i] <= self.variables.t_pos[i])
            self.model.addConstr(self.variables.t[i] >= -self.variables.t_pos[i])
        return
        
    def _build_objective(self):
        self.obj_assets = (sum(self.data.b * self.variables.p + self.data.a * self.variables.p * self.variables.p / 2) +
                           sum(self.data.pref * self.variables.t_pos))
        return
        
    def _update_objective(self):
        augm_lag = (-sum(self.y * (self.variables.t - self.t_average)) + 
                    self.data.rho / 2 * sum((self.variables.t - self.t_average) * (self.variables.t - self.t_average)))
        self.model.setObjective(self.obj_assets + augm_lag)
        self.model.update()
        return
        
    ###
    #   Interchangeable Iterative Updates
    ###
    def _iter_update_method1(self, trade):
        # Classical ADMM update (effectively alpha = 0)
        self.t_average = (self.t_old - trade[self.data.partners]) / 2
        self.y -= self.data.rho * (self.t_old - self.t_average)
        return

    def _iter_update_method2(self, trade):
        # Relaxed ADMM update with tunable alpha (default 0.95)
        self.t_average = (self.t_old - trade[self.data.partners]) / 2
        if self.model.Status == gb.GRB.Status.OPTIMAL:
            t_new = np.array([self.variables.t[i].X for i in range(self.data.num_partners)])
        else:
            t_new = self.t_old.copy()
        alpha = self.config.get("alpha", 0.95)  # Tunable parameter
        t_relaxed = alpha * t_new + (1 - alpha) * self.t_old
        self.y -= self.data.rho * (t_relaxed - self.t_average)
        self.t_old = t_new.copy()
        return

    ###
    #   Optimization status
    ###    
    def _opti_status(self, trade):
        for i in range(self.data.num_partners):
            self.t_new[i] = self.variables.t[i].X
        self.SW = -self.model.objVal
        self.Res_primal = sum((self.t_new + trade[self.data.partners])**2)
        self.Res_dual = sum((self.t_new - self.t_old)**2)
        self.t_old = np.copy(self.t_new)
        return
    
    def Who(self):
        self.who = 'Prosumer'
        return

# Subproblem: Manager inherits from Prosumer.
class Manager(Prosumer):
    def Who(self):
        self.who = 'Manager'
        return