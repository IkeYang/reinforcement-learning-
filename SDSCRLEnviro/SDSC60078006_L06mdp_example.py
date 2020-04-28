#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:25:34 2020

@author: Klint
"""

import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt

# basic info and setup
S        = 9
nStates  = S + 1
nActions = int(S/3) + 1
gamma    = 0.995

purchase_cost = 1.0
price         = 1.6
hold_cost     = 0.1
backlog_cost  = 0.2

maxInventory = int(2*S/3)
maxBacklog   = int(S/3) 

States  = np.arange(-maxBacklog,maxInventory+1)
Actions = np.arange(0,nActions)
assert(len(States) == nStates)

# Get demand 
nordist_mean = S/3
nordist_sd   = S/10
Demands      = np.arange(0,int(2*S/3)+1)
Demand_prob  = np.array([norm.pdf(d,nordist_mean,nordist_sd) for d in Demands])
Demand_prob  = Demand_prob/sum(Demand_prob)

# plotting
#plt.plot(Demands,Demand_prob)
#plt.show()


def GetNextState(state,action,demand):
    inventory      = min(state + action,maxInventory)
    next_state     = max(inventory - demand, -maxBacklog)
    next_state_idx = next_state + maxBacklog
    return next_state, next_state_idx



def GetReward(state,action,demand):
    
    next_state, idx = GetNextState(state,action,demand)

    # puchase cost based on our action
    next_purchase_cost = action * purchase_cost
    
    # compute the holding cost and the backlogging cost
    if next_state < 0:
        next_backlog_cost = abs(next_state * backlog_cost)
        next_holding_cost = 0.0
    else:
        next_backlog_cost = 0.0
        next_holding_cost = next_state * hold_cost

    # compute the revenue 
    revenue = 0.0
    # consider the inventory before demand
    inventory = min(state + action,maxInventory)
    if state < 0:
        if inventory >= 0:
            revenue += abs(state)* price
        else: 
            revenue += action * price
    # consider the revenue from current demand
    if inventory > 0 and next_state <= 0:
        revenue += inventory * price
    elif inventory > 0 and next_state > 0:
        revenue += price * (inventory - next_state)

    next_reward = revenue - next_purchase_cost - next_holding_cost - next_backlog_cost
    
    return next_reward



def BellmanUpdate(state,v):
    assert(len(v) == nStates)
    
    v_s   = float('-inf')
    a_opt = Actions[0]

    for a in Actions:
        v_temp = 0.0
        for i, d in enumerate(Demands):
            next_state, s_idx = GetNextState(state,a,d)
            next_reward       = GetReward(state,a,d)
            
            v_temp += Demand_prob[i]*(next_reward + gamma * v[s_idx])
        
        if v_s < v_temp:
            v_s   = v_temp
            a_opt = a
    
    return v_s, a_opt


def value_iteration(v0):
    # make a copy of v0
    v1 = v0
    opt_a = np.zeros(nStates)
    
    # for each iteration
    for k in range(0,500):
        v0 = v1
        for i,s in enumerate(States):
            v1[i], opt_a[i] = BellmanUpdate(s,v0)
    
    return v1, opt_a


v0 = np.arange(0,nStates)
v, policy = value_iteration(v0)

print(States)
print(policy)


















