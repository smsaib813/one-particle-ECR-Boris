#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 10:17:40 2022

@author: sarasaib

veloctity kick given each 
"""

import math
import numpy as np
import random
from scipy.spatial.transform import Rotation
from numpy.linalg import norm

v = [1,1,1]
v_new_size = np.linalg.norm(v)

f_c = 10E9 #cyclotron frequency is about 1GHz
delta_t = 1/f_c/10
collision_freq = 1E6 #set to 1MHz
kick_prob = delta_t * collision_freq

i=0
while i < 100000:
    a, b = random.uniform(0,math.pi), random.uniform(0,2*math.pi)  #random selection of angles
    # random selection of rotational vector based on the probability
    kick = random.choices([[math.sin(a)*math.cos(b),math.sin(a)*math.sin(b), math.cos(a)],np.ones(3)],[kick_prob,1-kick_prob])
    
    theta = random.uniform(0,2*math.pi)
    kick = kick / norm(kick)  # normalize the rotation vector first
    
    rot = Rotation.from_rotvec(theta * kick)
    
    new_v = rot.apply(v)  
    if kick.all() != np.ones(3).all():
        print("kicked")
        print(new_v)    
        print(np.linalg.norm(new_v))
    i+=1