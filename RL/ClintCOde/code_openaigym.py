#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:26:41 2020

@author: Klint
"""

import gym
import random


env = gym.make('CartPole-v0')

class Agent():
    def __init__(self,env):
        self.action_size = env.action_space.n
        
    def get_action(self, observation):
        # action = random.choice(range(self.action_size))
        angle = observation[2]
        action = 0 if angle < 0 else 1
        return action


agent = Agent(env)
observation = env.reset()


for t in range(200):
    # action = env.action_space.sample() # random action
    
    action = agent.get_action(observation)
    
    observation, reward, done, info = env.step(action)
    
    env.render()


env.close()


