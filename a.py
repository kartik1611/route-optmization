# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 05:04:47 2019

@author: home
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 00:44:41 2019

@author: home
"""

import numpy as np

# Q learning  Parmaeters

gamma =.75
alpha =.9

#  Building AI

# States

location_to_state={'A':0,
                   'B':1,
                   'C':2,
                   'D':3,
                   'E':4,
                   "F":5,
                   'G':6,
                   'H':7,
                   'I':8,
                   'J':9,
                   'K':10,
                   'L':11
                   }

# Actions
action=[0,1,2,3,4,5,6,7,8,9,10,11]

# Rewards

R=np.array([[0,1,0,0,0,0,0,0,0,0,0,0], # A
            [1,0,1,0,0,1,0,0,0,0,0,0], # B
            [0,1,0,0,0,0,1,0,0,0,0,0], # C
            [0,0,0,0,0,0,0,1,0,0,0,0], # D
            [0,0,0,0,0,0,0,0,1,0,0,0], # E
            [0,1,0,0,0,0,0,0,0,1,0,0], # F
            [0,0,1,0,0,0,1,1,0,0,0,0], # G
            [0,0,0,1,0,0,1,0,0,0,0,1], # H
            [0,0,0,0,1,0,0,0,0,1,0,0], # I
            [0,0,0,0,0,1,0,0,1,0,1,0], # J
            [0,0,0,0,0,0,0,0,0,1,0,1], # k
            [0,0,0,0,0,0,0,1,0,0,1,0]]) #l

# Building AI
# Intializes Q values


  



# Making a mapping fromstate to loaction
 
state_to_locatino={state:location for location,state in location_to_state.items()}  
    
def route(start,end):
    R_new=np.copy(R)
    ending_state=location_to_state[end]
    R_new[ending_state,ending_state]=1000
    Q =np.array(np.zeros([12,12]))
    for i in range(1000):
        current_state=np.random.randint(0,12)
        playable_action=[]
        for k in range(12):
            if R_new[current_state,k] > 0:
                playable_action.append(k)
        next_state=np.random.choice(playable_action)
        TD = R_new[current_state,next_state] + gamma* Q[next_state,np.argmax(Q[next_state,])]  -Q[current_state,next_state] 
        Q[current_state,next_state]+=alpha*TD
    route = [start]
    next_location = start
    while (next_location != end):
        starting_state = location_to_state[start]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_locatino[next_state]
        route.append(next_location)
        start = next_location
        return route


print('Route:')
route('E','G')



