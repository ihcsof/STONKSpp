# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:11:54 2018

Original Author: fmoret
Updated by: [Your Name]
"""

import random
import gurobipy as gb
import logging
import numpy as np

class expando(object):
    pass

logging.basicConfig(
    filename="beta_gamma.log",
    level=logging.DEBUG,
    format="%(message)s",
    filemode="w"
)
logger = logging.getLogger(__name__)

def apply_beta_gamma_weights(
    trades_array,
    neighbor_indices,
    trust_dict,
    local_malicious,
    beta=0.15,
    gamma=4,
    outlier_factor=3.0,
    trust_penalty=0.1,
    min_trust_for_nonzero=0.2
):
    n_partners = len(neighbor_indices)
    if n_partners == 0:
        return trades_array.copy(), np.array([])
    median_val = np.median(trades_array)
    abs_devs = np.abs(trades_array - median_val)
    mad = np.median(abs_devs)
    if mad < 1e-9:
        threshold = float('inf')
    else:
        threshold = outlier_factor * mad
    for idx, nb in enumerate(neighbor_indices):
        cur_trade = trades_array[idx]
        diff_from_median = abs(cur_trade - median_val)
        old_trust = trust_dict.get(nb, 1.0)
        if diff_from_median > threshold:
            new_trust = max(0.0, old_trust - trust_penalty)
            trust_dict[nb] = new_trust
        else:
            new_trust = min(1.0, old_trust + 0.01)
            trust_dict[nb] = new_trust
    alpha = np.zeros(n_partners, dtype=float)
    good_candidates = []
    for idx, nb in enumerate(neighbor_indices):
        if nb in local_malicious:
            alpha[idx] = 0.0
        else:
            t = trust_dict.get(nb, 1.0)
            if t >= min_trust_for_nonzero:
                good_candidates.append((t, idx, nb))
    good_candidates.sort(key=lambda x: x[0], reverse=True)
    for rank, (score, idx, nb) in enumerate(good_candidates):
        if rank < gamma:
            alpha[idx] = beta
        else:
            alpha[idx] = 0.0
    total_assigned = alpha.sum()
    leftover = 1.0 - total_assigned
    if leftover <= 1e-10:
        if total_assigned > 1e-9:
            alpha /= total_assigned
    else:
        active_idx = [i for i in range(n_partners) if alpha[i] > 0]
        if len(active_idx) == 0:
            active_idx = [i for i in range(n_partners)
                          if trust_dict.get(neighbor_indices[i],1.0) >= min_trust_for_nonzero]
        if len(active_idx) == 0:
            alpha[:] = 1.0 / n_partners
        else:
            leftover_each = leftover / len(active_idx)
            for i in active_idx:
                alpha[i] += leftover_each
    sum_alpha = alpha.sum()
    if sum_alpha > 1e-9:
        alpha /= sum_alpha
    else:
        alpha[:] = 1.0 / n_partners
    new_trades = trades_array * alpha
    return new_trades, alpha

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
                print("Byzantine flag set to", self.data.isByzantine, "for agent", self.data.id)
            else:
                self.data.isByzantine = (self.data.id == 2)
                print("Byzantine flag set to", self.data.isByzantine, "for agent", self.data.id)
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
        self.simulator = None
        if 'simulator' in self.config:
            self.simulator = self.config['simulator']

    def optimize(self, trade):
        self.iter_update_method(trade)
        self._update_objective()
        self.model.optimize()
        if self.model.Status == gb.GRB.Status.OPTIMAL:
            self._opti_status(trade)
            val = self.t_old.copy()
            if self.data.isByzantine and (self.data.tampered < self.data.max_tampering):
                chance = self.config.get("byzantine_attack_probability", 0.05)
                lower = self.config.get("byzantine_multiplier_lower", 0.5)
                upper = self.config.get("byzantine_multiplier_upper", 1.2)
                if random.random() < chance:
                    self.data.tampered += 1
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

    def _build_model(self):
        self.model = gb.Model()
        self.model.setParam('OutputFlag', False)
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()

    def _build_variables(self):
        m = self.model
        self.variables.p = np.array([m.addVar(lb=self.data.Pmin[i], ub=self.data.Pmax[i], name='p') for i in range(self.data.num_assets)])
        self.variables.t = np.array([m.addVar(lb=-gb.GRB.INFINITY, name='t') for i in range(self.data.num_partners)])
        self.variables.t_pos = np.array([m.addVar(name='t_pos') for i in range(self.data.num_partners)])
        m.update()

    def _build_constraints(self):
        self.constraints.pow_bal = self.model.addConstr(sum(self.variables.p) == sum(self.variables.t))
        for i in range(self.data.num_partners):
            self.model.addConstr(self.variables.t[i] <= self.variables.t_pos[i])
            self.model.addConstr(self.variables.t[i] >= -self.variables.t_pos[i])

    def _build_objective(self):
        self.obj_assets = (sum(self.data.b * self.variables.p + self.data.a * self.variables.p * self.variables.p / 2) + sum(self.data.pref * self.variables.t_pos))

    def _update_objective(self):
        augm_lag = (-sum(self.y * (self.variables.t - self.t_average)) + self.data.rho / 2 * sum((self.variables.t - self.t_average) ** 2))
        self.model.setObjective(self.obj_assets + augm_lag)
        self.model.update()

    def _iter_update_method1(self, trade):
        self.t_average = (self.t_old - trade[self.data.partners]) / 2
        self.y -= self.data.rho * (self.t_old - self.t_average)

    def _iter_update_method2(self, trade):
        sim = self.simulator
        if sim is None:
            trust_dict = {}
        else:
            trust_dict = sim.trust_scores[self.data.id]
        beta_cfg = self.config.get("beta_admissible", 0.15)
        gamma_cfg = self.config.get("gamma_admissible", 4)
        weighted_trade, _ = apply_beta_gamma_weights(
            trade[self.data.partners],
            self.data.partners,
            trust_dict=trust_dict,
            local_malicious=set(),
            beta=beta_cfg,
            gamma=gamma_cfg,
            outlier_factor=3.0,
            trust_penalty=0.1,
            min_trust_for_nonzero=0.2
        )
        eta = self.config.get("beta_mix", 0.1)
        blended_trade = (1 - eta) * trade[self.data.partners] + eta * weighted_trade
        self.t_average = (self.t_old - blended_trade) / 2
        if self.model.Status == gb.GRB.Status.OPTIMAL:
            t_new = np.array([self.variables.t[i].X for i in range(self.data.num_partners)])
        else:
            t_new = self.t_old.copy()
        with open("weighted_trade.txt", "a") as f:
            f.write(str(blended_trade))
            f.write("---")
            f.write(str(trade[self.data.partners]))
            f.write("\n")
        self.y -= self.data.rho * (self.t_old - self.t_average)
        self.t_old = t_new.copy()

    def _opti_status(self, trade):
        for i in range(self.data.num_partners):
            self.t_new[i] = self.variables.t[i].X
        self.SW = -self.model.objVal
        self.Res_primal = sum((self.t_new + trade[self.data.partners]) ** 2)
        self.Res_dual = sum((self.t_new - self.t_old) ** 2)
        self.t_old = np.copy(self.t_new)

    def Who(self):
        self.who = 'Prosumer'

class Manager(Prosumer):
    def Who(self):
        self.who = 'Manager'
