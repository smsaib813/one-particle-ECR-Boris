#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 10:17:40 2022

@author: sarasaib

veloctity kick given each 
"""
print('running random kick.py...')
import math
import numpy as np
import random
from scipy.spatial.transform import Rotation
from numpy.linalg import norm
import matplotlib.pyplot as plt
v = np.array([[1,1,3]])
v_new_size = np.linalg.norm(v)

f_c = 10E9 #cyclotron frequency is about 1GHz
delta_t = 1/f_c/20
collision_freq = 1E9 #collision frequency set to 1GHz
kick_prob = delta_t * collision_freq

ar = []
i=0
while i < 0:
    print(i)
    a, b = random.uniform(0,math.pi), random.uniform(0,2*math.pi)  #random selection of angles
    # random selection of rotational vector based on the probability
    kick = random.choices([[math.sin(a)*math.cos(b),math.sin(a)*math.sin(b), math.cos(a)],np.ones(3)],[kick_prob,1-kick_prob])
    
    #print(kick)
    theta = random.uniform(0,2*math.pi)
    #print(theta)
    kick = kick / norm(kick)  # normalize the rotation vector first
    
    rot = Rotation.from_rotvec(theta * kick)
    
    new_v = rot.apply(v)  
    ar.append(new_v)
    i+=1
    #print(abs(norm(new_v)- norm(np.ones(3)))<1E-17)
    print('v and norm',new_v,norm(new_v)) 
    '''
    if not abs(norm(new_v)- norm(np.ones(3)))<1E-10:
        print(i," kicked")
        print(norm(new_v),norm(np.ones(3)))    
        #print(np.linalg.norm(new_v))
'''

def random_kick_false(v,delta_t,collision_freq = 1E11):
    kick_prob = delta_t * collision_freq
    a, b = random.uniform(0,math.pi), random.uniform(0,2*math.pi)  #random selection of angles
    # random selection of rotational vector based on the probability
    kick = random.choices([[math.sin(a)*math.cos(b),math.sin(a)*math.sin(b), math.cos(a)],np.ones(3)],[kick_prob,1-kick_prob])
    theta = random.uniform(0,2*math.pi)
    kick = kick / norm(kick)  # normalize the rotation vector first
    rot = Rotation.from_rotvec(theta * kick)
    new_v = rot.apply(v) 
    print('v1 = ',new_v)
    return new_v

def random_kick(v,delta_t,collision_freq = 1E11):
    kick_prob = delta_t * collision_freq
    a, b = random.uniform(0,math.pi), random.uniform(0,2*math.pi)  #random selection of angles
    # random selection of rotational vector based on the probability
    kick = [math.sin(a)*math.cos(b),math.sin(a)*math.sin(b), math.cos(a)]
    theta = random.uniform(0,2*math.pi)
    kick = kick / norm(kick)  # normalize the rotation vector first
    rot = Rotation.from_rotvec(theta * kick)
    new_v = random.choices([rot.apply(v),v],[kick_prob,1-kick_prob])  #kick with prob or not
    #print('v2 = ',new_v, norm(new_v))
    return np.array(new_v)[0,:]

while i<10:
    v1 = random_kick_false(v,delta_t,collision_freq = 1E11)
    v2 = random_kick(v,delta_t,collision_freq = 1E11)
    ar.append(v2)
    i+=1
