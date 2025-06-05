# ProsumerGUROBI_FIX.py
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:11:54 2018

Original Author: fmoret
Updated by: [Your Name] with logging for debugging trust and tampering.
"""

import random
import gurobipy as gb
import numpy as np
import logging

# Setup logging with only the message (no time stamps)
#logging.basicConfig(filename="debug_log.txt", level=logging.DEBUG, format="%(message)s", filemode="a")

class expando(object):
    pass

class Prosumer:
    def __init__(self, agent=None, partners=None, preferences=None, rho=1, config=None):
        self.data = expando()
        self.Who()
        self.config = config if config is not None else {}
        if agent is not None:
            self.data.type = agent['Type']
            self.data.id = agent['ID']
            if "byzantine_ids" in self.config:
                self.data.isByzantine = (self.data.id in self.config["byzantine_ids"])
            else:
                self.data.isByzantine = (self.data.id == 2)
            print("Byzantine flag set to", self.data.isByzantine, "for agent", self.data.id)
            #logging.info(f"Prosumer {self.data.id}: Byzantine flag set to {self.data.isByzantine}")
            self.data.tampered = 0
            self.data.max_tampering = self.config.get("tampering_count", 1)
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
        if partners is not None:
            self.data.partners = partners.nonzero()[0]
        else:
            self.data.partners = np.zeros([0])
        self.data.num_partners = len(self.data.partners)
        if preferences is not None:
            self.data.pref = preferences[partners.nonzero()]
        else:
            self.data.pref = np.zeros([0])
        self.data.rho = rho
        self.SW = 0
        self.Res_primal = 0
        self.Res_dual = 0
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self._build_model()
        self.t_old = np.zeros(self.data.num_partners)
        self.t_new = np.zeros(self.data.num_partners)
        self.y = np.zeros(self.data.num_partners)
        self.y0 = np.zeros(self.data.num_partners)
        method = self.config.get("iter_update_method", "method2")
        if method == "method1":
            self.iter_update_method = self._iter_update_method1
        elif method == "method2":
            self.iter_update_method = self._iter_update_method2
        else:
            raise ValueError("Unknown iter_update_method in config: " + method)
        #logging.debug(f"Prosumer {self.data.id} initialized. Partners: {self.data.partners}")

    def optimize(self, trade):
        #logging.debug(f"Prosumer {self.data.id} optimize: starting with trade vector shape {trade.shape}")
        self.iter_update_method(trade)
        self._update_objective()
        self.model.optimize()
        if self.model.Status == gb.GRB.Status.OPTIMAL:
            self._opti_status(trade)
            val = self.t_old.copy()
            if self.data.isByzantine and (self.data.tampered < self.data.max_tampering):
                chance = self.config.get("byzantine_attack_probability", 0.05)
                lower = self.config.get("byzantine_multiplier_lower", 0.5)
                upper = self.config.get("byzantine_multiplier_upper", 1.5)
                if random.random() < chance:
                    self.data.tampered += 1
                    multiplier = upper
                    #logging.debug(f"Prosumer {self.data.id} TAMPERING: old val {val}, multiplier {multiplier}")
                    val *= multiplier
            #logging.debug(f"Prosumer {self.data.id} optimize: final trade val {val}")
            trade[self.data.partners] = val
        else:
            pass
            #logging.debug(f"Prosumer {self.data.id} optimize: model not optimal, status {self.model.Status}")
        return trade

    def production_consumption(self):
        if self.model.Status == gb.GRB.Status.OPTIMAL:
            prod = abs(np.array([self.variables.p[i].X for i in range(self.data.num_assets) if self.variables.p[i].X > 0]).sum())
            cons = abs(np.array([self.variables.p[i].X for i in range(self.data.num_assets) if self.variables.p[i].X < 0]).sum())
        else:
            prod = 0
            cons = 0
            print("Cannot compute production and consumption because the model did not solve optimally.")
            #logging.warning(f"Prosumer {self.data.id} production_consumption: model not optimal.")
        return prod, cons

    def _build_model(self):
        self.model = gb.Model()
        self.model.setParam('OutputFlag', False)
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()

    def _build_variables(self):
        m = self.model
        self.variables.p = np.array([m.addVar(lb=self.data.Pmin[i], ub=self.data.Pmax[i], name='p')
                                     for i in range(self.data.num_assets)])
        self.variables.t = np.array([m.addVar(lb=-gb.GRB.INFINITY, name='t')
                                     for i in range(self.data.num_partners)])
        self.variables.t_pos = np.array([m.addVar(name='t_pos')
                                         for i in range(self.data.num_partners)])
        m.update()

    def _build_constraints(self):
        self.constraints.pow_bal = self.model.addConstr(sum(self.variables.p) == sum(self.variables.t))
        for i in range(self.data.num_partners):
            self.model.addConstr(self.variables.t[i] <= self.variables.t_pos[i])
            self.model.addConstr(self.variables.t[i] >= -self.variables.t_pos[i])

    def _build_objective(self):
        self.obj_assets = (sum(self.data.b * self.variables.p +
                               self.data.a * self.variables.p * self.variables.p / 2)
                           + sum(self.data.pref * self.variables.t_pos))

    def _update_objective(self):
        augm_lag = (-sum(self.y * (self.variables.t - self.t_average))
                    + self.data.rho / 2 * sum((self.variables.t - self.t_average) ** 2))
        self.model.setObjective(self.obj_assets + augm_lag)
        self.model.update()

    def _iter_update_method1(self, trade):
        self.t_average = (self.t_old - trade[self.data.partners]) / 2
        self.y -= self.data.rho * (self.t_old - self.t_average)
        #logging.debug(f"Prosumer {self.data.id} method1: t_old={self.t_old}, trade={trade[self.data.partners]}, t_avg={self.t_average}, y={self.y}")

    def _iter_update_method2(self, trade):
        self.t_average = (self.t_old - trade[self.data.partners]) / 2
        if self.model.Status == gb.GRB.Status.OPTIMAL:
            t_new = np.array([self.variables.t[i].X for i in range(self.data.num_partners)])
        else:
            t_new = self.t_old.copy()
        alpha = self.config.get("alpha", 0)
        t_relaxed = alpha * t_new + (1 - alpha) * self.t_old
        self.y -= self.data.rho * (t_relaxed - self.t_average)
        #logging.debug(f"Prosumer {self.data.id} method2: t_old={self.t_old}, t_new={t_new}, t_relaxed={t_relaxed}, t_avg={self.t_average}, y={self.y}")
        self.t_old = t_new.copy()

    def _opti_status(self, trade):
        for i in range(self.data.num_partners):
            self.t_new[i] = self.variables.t[i].X
        self.SW = -self.model.objVal
        self.Res_primal = sum((self.t_new + trade[self.data.partners]) ** 2)
        self.Res_dual = sum((self.t_new - self.t_old) ** 2)
        self.t_old = np.copy(self.t_new)
        #logging.debug(f"Prosumer {self.data.id} status: SW={self.SW}, Res_primal={self.Res_primal}, Res_dual={self.Res_dual}")

    def Who(self):
        self.who = 'Prosumer'

class Manager(Prosumer):
    def Who(self):
        self.who = 'Manager'
