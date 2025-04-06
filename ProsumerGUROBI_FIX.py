# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:11:54 2018

Original Author: fmoret
"""

import random
import gurobipy as gb
import logging
import numpy as np

logging.basicConfig(
    filename="beta_gamma.log",
    level=logging.DEBUG,
    format="%(message)s",
    filemode="w"
)
logger = logging.getLogger(__name__)

class expando(object):
    pass

def apply_beta_gamma_weights(
    trades_array,
    neighbor_indices,
    trust_dict,
    local_malicious,
    beta=0.15,
    gamma=4,
    outlier_factor=1.5,
    trust_penalty=0.2,
    min_trust_for_nonzero=0.2,
    max_trade=30.0
):
    logger.debug("=== apply_beta_gamma_weights start ===")
    logger.debug("Input trades_array: %s", trades_array.tolist())
    logger.debug("Neighbor indices: %s", neighbor_indices)
    logger.debug("Trust dict (start): %s", {k: f"{v:.3f}" for k,v in trust_dict.items()})
    logger.debug("Local malicious: %s", local_malicious)
    logger.debug("beta=%.3f, gamma=%d, outlier_factor=%.3f, trust_penalty=%.3f, min_trust_for_nonzero=%.3f",
                 beta, gamma, outlier_factor, trust_penalty, min_trust_for_nonzero)

    n_partners = len(neighbor_indices)
    if n_partners == 0:
        logger.debug("No partners found; returning original trades.")
        return trades_array.copy(), np.array([])

    trades_array = np.clip(trades_array, -max_trade, max_trade)

    median_val = np.median(trades_array)
    abs_devs = np.abs(trades_array - median_val)
    mad = np.median(abs_devs)
    if mad < 1e-9:
        threshold = float('inf')
    else:
        threshold = outlier_factor * mad
    logger.debug("Median: %.3f, MAD: %.3f, Threshold: %.3f", median_val, mad, threshold)

    for idx, nb in enumerate(neighbor_indices):
        cur_trade = trades_array[idx]
        diff = abs(cur_trade - median_val)
        old_t = trust_dict.get(nb, 1.0)
        if diff > threshold:
            new_t = max(0.0, old_t - trust_penalty)
            trust_dict[nb] = new_t
            logger.debug("Outlier detected for neighbor %d: trade=%.3f, old_trust=%.3f -> new_trust=%.3f", nb, cur_trade, old_t, new_t)
        else:
            new_t = min(1.0, old_t + 0.01)
            trust_dict[nb] = new_t
            logger.debug("Neighbor %d within threshold: trade=%.3f, old_trust=%.3f -> new_trust=%.3f", nb, cur_trade, old_t, new_t)

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

    logger.debug("Initial alpha after top-gamma: %s", alpha.tolist())

    total_assigned = alpha.sum()
    leftover = 1.0 - total_assigned
    if leftover <= 1e-10:
        if total_assigned > 1e-9:
            alpha /= total_assigned
    else:
        active_idx = [i for i in range(n_partners) if alpha[i] > 0]
        if len(active_idx) == 0:
            active_idx = [i for i in range(n_partners)
                          if trust_dict.get(neighbor_indices[i], 1.0) >= min_trust_for_nonzero]
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
    logger.debug("Final alpha vector: %s", alpha.tolist())
    logger.debug("New trades: %s", new_trades.tolist())
    logger.debug("=== apply_beta_gamma_weights end ===\n")
    return new_trades, alpha

class Prosumer:
    def __init__(self, agent=None, partners=None, preferences=None, rho=1e-5, config=None):
        self.data = expando()
        self.Who()
        self.config = config if config is not None else {}
        self.SW = 0
        self.Res_primal = 0
        self.Res_dual = 0

        if agent is not None:
            self.data.type = agent['Type']
            self.data.id = agent['ID']
            if "byzantine_ids" in self.config:
                self.data.isByzantine = (self.data.id in self.config["byzantine_ids"])
            else:
                self.data.isByzantine = (self.data.id == 2)
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

        logger.info("Prosumer init: ID=%s, isByzantine=%s, num_partners=%d, num_assets=%d",
                    str(getattr(self.data, 'id', '?')),
                    str(self.data.isByzantine),
                    self.data.num_partners,
                    getattr(self.data, 'num_assets', 0))

    def optimize(self, trade):
        logger.info("Optimize called for ID=%s", str(getattr(self.data, 'id', '?')))
        self.iter_update_method(trade)
        self._update_objective()
        logger.info("Gurobi solve about to start for ID=%s", str(getattr(self.data, 'id', '?')))
        self.model.optimize()
        logger.info("Gurobi status for ID=%s = %d", str(getattr(self.data, 'id', '?')), self.model.Status)
        if self.model.Status == gb.GRB.Status.OPTIMAL:
            self._opti_status(trade)
            val = self.t_old.copy()
            if self.data.isByzantine and (self.data.tampered < self.data.max_tampering):
                chance = self.config.get("byzantine_attack_probability", 0.05)
                lower = self.config.get("byzantine_multiplier_lower", 0.5)
                upper = self.config.get("byzantine_multiplier_upper", 1.2)
                if random.random() < chance:
                    self.data.tampered += 1
                    val *= upper
                    logger.warning("Byzantine agent %s tampered trades by %.3f multiplier", str(self.data.id), upper)
            logger.info("Finalize trade for ID=%s: %s", str(self.data.id), val.tolist())
            trade[self.data.partners] = val
        else:
            logger.warning("Not optimal => ignoring update for ID=%s", str(self.data.id))
        return trade

    def production_consumption(self):
        if self.model.Status == gb.GRB.Status.OPTIMAL:
            prod = abs(np.array([self.variables.p[i].X for i in range(self.data.num_assets) if self.variables.p[i].X > 0]).sum())
            cons = abs(np.array([self.variables.p[i].X for i in range(self.data.num_assets) if self.variables.p[i].X < 0]).sum())
        else:
            prod = 0
            cons = 0
            logger.warning("Cannot compute production/consumption, model not optimal for ID=%s",
                           str(getattr(self.data, 'id', '?')))
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
        self.obj_assets = sum(self.data.b * self.variables.p + self.data.a * self.variables.p * self.variables.p / 2) \
                          + sum(self.data.pref * self.variables.t_pos)

    def _update_objective(self):
        augm_lag = -sum(self.y * (self.variables.t - self.t_average)) \
                   + self.data.rho / 2 * sum((self.variables.t - self.t_average) ** 2)
        self.model.setObjective(self.obj_assets + augm_lag)
        self.model.update()

    def _iter_update_method1(self, trade):
        logger.debug("Iter_update_method1 called ID=%s trade=%s",
                     str(getattr(self.data, 'id', '?')),
                     trade[self.data.partners].tolist())
        self.t_average = (self.t_old - trade[self.data.partners]) / 2
        self.y -= self.data.rho * (self.t_old - self.t_average)
        logger.debug("ID=%s: t_old=%s, t_average=%s, y=%s",
                     str(self.data.id),
                     self.t_old.tolist(),
                     self.t_average.tolist(),
                     self.y.tolist())

    def _iter_update_method2(self, trade):
        logger.debug("Iter_update_method2 called ID=%s trade=%s",
                     str(getattr(self.data, 'id', '?')),
                     trade[self.data.partners].tolist())
        sim = self.simulator
        if sim is None:
            trust_dict = {}
        else:
            trust_dict = sim.trust_scores[self.data.id]

        beta_cfg = self.config.get("beta_admissible", 0.15)
        gamma_cfg = self.config.get("gamma_admissible", 4)
        outlier_factor = 1.5
        trust_penalty = 0.2
        min_trust = 0.2
        max_trade = 30.0

        weighted_trade, _ = apply_beta_gamma_weights(
            trade[self.data.partners],
            self.data.partners,
            trust_dict=trust_dict,
            local_malicious=set(),
            beta=beta_cfg,
            gamma=gamma_cfg,
            outlier_factor=outlier_factor,
            trust_penalty=trust_penalty,
            min_trust_for_nonzero=min_trust,
            max_trade=max_trade
        )

        eta = self.config.get("beta_mix", 0.02)
        blended_trade = (1 - eta) * trade[self.data.partners] + eta * weighted_trade
        blended_trade = np.clip(blended_trade, -max_trade, max_trade)

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
        logger.debug("ID=%s: t_old(before)=%s, blended_trade=%s, t_average=%s",
                     str(self.data.id),
                     self.t_old.tolist(),
                     blended_trade.tolist(),
                     self.t_average.tolist())
        self.t_old = t_new.copy()
        logger.debug("ID=%s: t_new(after)=%s, y=%s",
                     str(self.data.id),
                     self.t_old.tolist(),
                     self.y.tolist())

    def _opti_status(self, trade):
        for i in range(self.data.num_partners):
            self.t_new[i] = self.variables.t[i].X
        self.SW = -self.model.objVal
        self.Res_primal = sum((self.t_new + trade[self.data.partners]) ** 2)
        self.Res_dual = sum((self.t_new - self.t_old) ** 2)
        logger.info("Post-optim ID=%s SW=%.3f Res_primal=%.6g Res_dual=%.6g",
                    str(self.data.id),
                    self.SW,
                    self.Res_primal,
                    self.Res_dual)
        self.t_old = np.copy(self.t_new)

    def Who(self):
        self.who = 'Prosumer'

class Manager(Prosumer):
    def Who(self):
        self.who = 'Manager'
