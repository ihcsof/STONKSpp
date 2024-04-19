# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:11:54 2018

@author: fmoret
"""

import re

from typing import Any, Optional

import numpy as np
import cvxpy as cp

# Class which can have attributes set.
class expando(object): pass

# Subproblem
class Prosumer:
    def __init__(self,agent=None,partners=None,preferences=None,rho=1):
        self.data = expando()
        self.Who()
        # Data -- Agent and its assets
        if agent is not None:
            self.data.type = agent['Type']
            self.data.CM = (agent['Type']=='Manager')
            if agent['AssetsNum']<=len(agent['Assets']):
                self.data.num_assets = agent['AssetsNum']
                self.data.a = np.zeros([self.data.num_assets])
                self.data.b = np.zeros([self.data.num_assets])
                self.data.Pmin = np.zeros([self.data.num_assets])
                self.data.Pmax = np.zeros([self.data.num_assets])
                for i in range(self.data.num_assets):
                    if agent['Assets'][i]['costfct']=='Quadratic':
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
        return

    def optimize(self, trade):
        self._iter_update(trade)
        self._update_objective()

        problem = cp.Problem(cp.Minimize(self.obj), self.constraints_repo)
        self.optimization_result = problem.solve(verbose = False)

        self._opti_status(trade)
        trade[self.data.partners] = self.t_old
        return trade
    
    def production_consumption(self):
        prod = abs(np.array([self.variables.p[i].value for i in range(self.data.num_assets) if self.variables.p[i].value is not None and self.variables.p[i].value > 0]).sum())
        cons = abs(np.array([self.variables.p[i].value for i in range(self.data.num_assets) if self.variables.p[i].value is not None and self.variables.p[i].value < 0]).sum())
        return prod, cons

    ###
    #   Model Building
    ###
    def _add_model_constraint(self, constraint: cp.Constraint) -> cp.Constraint:
        # print('Added constraint', constraint)
        self.constraints_repo.append(constraint)
        return constraint

    _variable_family_re = re.compile(r'(?P<family>[^0-9]+)([0-9]+)$')

    def _add_model_variable(self, variable: cp.Variable, lb: Optional[float] = None, ub: Optional[float] = None) -> cp.Variable:
        # print('Added variable', variable.name())
        self.variables_repo.append(variable)
        # eventually add to family
        if (m := self._variable_family_re.match(variable.name())) is not None:
            family_name = m.group('family')
            if family_name in self.variables_families:
                self.variables_families[family_name].append(variable)
            else:
                self.variables_families[family_name] = [ variable ]
        # handle variable constraints
        if lb is not None:
            self._add_model_constraint(variable >= lb)
        if ub is not None:
            self._add_model_constraint(variable <= ub)
        return variable

    def _build_model(self):
        # prepare storage
        self.constraints_repo: list[cp.Constraint] = []
        self.variables_repo: list[cp.Variable] = []
        self.variables_families: dict[str, list[cp.Variable]] = {}
        #
        # Variables
        #
        for i in range(self.data.num_assets):
            self._add_model_variable(cp.Variable(name = f'p{i}'), lb = self.data.Pmin[i], ub = self.data.Pmax[i])
        for i in range(self.data.num_partners):
            self._add_model_variable(cp.Variable(name = f't{i}'))
            self._add_model_variable(cp.Variable(name = f't_pos{i}'))
        self.t_old = np.zeros(self.data.num_partners)
        self.t_new = np.zeros(self.data.num_partners)
        self.y = np.zeros(self.data.num_partners)
        self.y0 = np.zeros(self.data.num_partners)
        #
        # Constraints
        #
        self._add_model_constraint(cp.sum(self.variables_families['p']) == cp.sum(self.variables_families['t']))
        for t, t_pos in zip(self.variables_families['t'], self.variables_families['t_pos']):
            self._add_model_constraint(t <= t_pos)
            self._add_model_constraint(t >= -t_pos)
        #
        # Objectives
        #
        self.obj_assets = cp.sum([
            cp.sum([ self.data.b[i] * self.variables_families['p'][i] for i in range(len(self.data.b)) ]),
            cp.sum([ self.data.a[i] * cp.square(self.variables_families['p'][i]) / 2. for i in range(len(self.data.a)) ]),
            cp.sum([ self.data.pref[i] * self.variables_families['t_pos'][i] for i in range(len(self.data.pref)) ]),
        ])
        # self._build_variables()
        # self._build_constraints()
        # self._build_objective()
        return

    def _build_variables(self):
        #
        # P
        #
        self.variables.p = [cp.Variable(name=f'p{i}') for i in range(self.data.num_assets)]
        #-------------------------------
        # HERE    
        #-------------------------------
        #self.variables.t = [cp.Variable(name='t{}'.format(i)) for i in range(self.data.num_partners)]
        self.variables.t = cp.Variable((self.data.num_partners,), name='t')

        self.variables.t_pos = [cp.Variable(name='t_pos{}'.format(i)) for i in range(self.data.num_partners)]
        self.t_old = np.zeros(self.data.num_partners)
        self.t_new = np.zeros(self.data.num_partners)
        self.y = np.zeros(self.data.num_partners)
        self.y0 = np.zeros(self.data.num_partners)
        self.t_average = np.zeros(self.data.num_partners) # Define t_average here
        return
        
    def _build_constraints(self):
        constraints = [sum(self.variables.p) == sum(self.variables.t)]
        for i in range(self.data.num_partners):
            constraints += [self.variables.t[i] <= self.variables.t_pos[i], self.variables.t[i] >= -self.variables.t_pos[i]]
        self.constraints.pow_bal = constraints
        return
            
    def _build_objective(self):
        return

        #-------------------------------
        # HERE    
        #-------------------------------

        quad_form_assets = sum(cp.square(self.variables.p[i]) * self.data.a[i] for i in range(self.data.num_assets))

        trading_costs = cp.sum(self.variables.t @ self.data.pref)

        aug_lag = cp.sum(self.y @ (self.variables.t - self.t_average)) + (self.data.rho / 2) * cp.sum_squares(self.variables.t - self.t_average)

        self.model = cp.Problem(cp.Minimize(quad_form_assets + trading_costs + aug_lag), self.constraints.pow_bal)


    ###
    #   Model Updating
    ###    
    def _update_objective(self):
        augm_lag = cp.sum([
            -cp.sum([self.y[i] * (self.variables_families['t'][i] - self.t_average[i]) for i in range(len(self.t_average))]),
            self.data.rho / 2. * cp.sum([
                cp.square(self.variables_families['t'][i] - self.t_average[i])
                for i in range(len(self.variables_families['t']))
            ])
        ])
        self.obj = self.obj_assets + augm_lag
        return
        
    ###
    #   Iteration Update
    ###    
    def _iter_update(self, trade):
        self.t_average = (self.t_old - trade[self.data.partners])/2
        self.y -= self.data.rho*(self.t_old - self.t_average)
        return
        
    ###
    #   Optimization status
    ###    
    def _opti_status(self,trade):
        for i in range(self.data.num_partners):
            self.t_new[i] = self.variables_families['t'][i].value
        self.SW = self.optimization_result
        self.Res_primal = sum( (self.t_new + trade[self.data.partners])*(self.t_new + trade[self.data.partners]) )
        self.Res_dual = sum( (self.t_new-self.t_old)*(self.t_new-self.t_old) )
        self.t_old = np.copy(self.t_new)
        return
    
    def Who(self):
        self.who = 'Prosumer'
        return


# Subproblem
class Manager(Prosumer):
    def Who(self):
        self.who = 'Manager'
        return