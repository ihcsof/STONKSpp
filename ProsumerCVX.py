# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:11:54 2018

@author: fmoret
"""

import numpy as np
import cvxpy as cp

# Class which can have attributes set.
class expando(object):
    pass

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
        self.model.solve(cp.CVXOPT)
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
    def _build_model(self):
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        return

    def _build_variables(self):
        self.variables.p = [cp.Variable(name='p{}'.format(i)) for i in range(self.data.num_assets)]
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
        # Quadratic term for assets' operation cost
        quad_form_assets = sum(cp.square(self.variables.p[i]) * self.data.a[i] for i in range(self.data.num_assets))

        # Linear term for trading costs/preference
        # RROBLEM: Ensure self.data.pref is compatible in dimension and formulated correctly????
        trading_costs = cp.sum(self.variables.t @ self.data.pref)

        # Augmented Lagrangian terms
        # Assuming y is the dual variable, and t_average?
        # Does rho parameter should ensure convexity;??
        aug_lag = cp.sum(self.y @ (self.variables.t - self.t_average)) + (self.data.rho / 2) * cp.sum_squares(self.variables.t - self.t_average)

        # Combine all parts into the objective
        self.model = cp.Problem(cp.Minimize(quad_form_assets + trading_costs + aug_lag), self.constraints.pow_bal)


    ###
    #   Model Updating
    ###    
    def _update_objective(self):
        self._build_objective()
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
            self.t_new[i] = self.variables.t[i].value
        self.SW = self.model.value
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