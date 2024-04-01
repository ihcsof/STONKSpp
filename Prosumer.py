# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:11:54 2018

@author: fmoret
"""

# Import Gurobi Library
import gurobipy as gb
import numpy as np

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
                #print('a:'+str(self.data.a)+', b:'+str(self.data.b)+', max:'+str(self.data.Pmax)+', min:'+str(self.data.Pmin))
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
        self.model.optimize()
        self._opti_status(trade)
        trade[self.data.partners] = self.t_old
        return trade
    
    def production_consumption(self):
        prod = abs( np.array([self.variables.p[i].x for i in range(self.data.num_assets) if self.variables.p[i].x>0]).sum() )
        cons = abs( np.array([self.variables.p[i].x for i in range(self.data.num_assets) if self.variables.p[i].x<0]).sum() )
        return prod,cons

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self.model.setParam( 'OutputFlag', False )
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()
        return

    def _build_variables(self):
        m = self.model
        self.variables.p = np.array([m.addVar(lb = self.data.Pmin[i], ub = self.data.Pmax[i], name = 'p') for i in range(self.data.num_assets)])
        self.variables.t = np.array([m.addVar(lb = -gb.GRB.INFINITY, name = 't') for i in range(self.data.num_partners)])
        self.variables.t_pos = np.array([m.addVar(name = 't_pos') for i in range(self.data.num_partners)]) #obj=self.data.pref[i], 
        self.t_old = np.zeros(self.data.num_partners)
        self.t_new = np.zeros(self.data.num_partners)
        self.y = np.zeros(self.data.num_partners)
        self.y0 = np.zeros(self.data.num_partners)
        m.update()
        return
        
    def _build_constraints(self):
        self.constraints.pow_bal = self.model.addConstr(sum(self.variables.p) == sum(self.variables.t))
        for i in range(self.data.num_partners):
            self.model.addConstr(self.variables.t[i] <= self.variables.t_pos[i])
            self.model.addConstr(self.variables.t[i] >= -self.variables.t_pos[i])
        return
        
    def _build_objective(self):
        self.obj_assets = (sum(self.data.b*self.variables.p + self.data.a*self.variables.p*self.variables.p/2) +
                           sum(self.data.pref*self.variables.t_pos) )
        return
        
    ###
    #   Model Updating
    ###    
    def _update_objective(self):
        augm_lag = (-sum(self.y*( self.variables.t - self.t_average )) + 
                    self.data.rho/2*sum( ( self.variables.t - self.t_average )
                                            *( self.variables.t - self.t_average ))
                   )
        self.model.setObjective(self.obj_assets + augm_lag)
        self.model.update()
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
            self.t_new[i] = self.variables.t[i].x
        self.SW = -self.model.objVal
        self.Res_primal = sum( (self.t_new + trade[self.data.partners])*(self.t_new + trade[self.data.partners]) )
        self.Res_dual = sum( (self.t_new-self.t_old)*(self.t_new-self.t_old) )
        self.t_old = np.copy(self.t_new)
        return
    
    def Who(self):
        #print('Prosumer')
        self.who = 'Prosumer'
        return


# Subproblem
class Manager(Prosumer):
    def Who(self):
        #print('Manager')
        self.who = 'Manager'
        return