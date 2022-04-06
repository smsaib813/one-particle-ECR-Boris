#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:25:35 2022

@author: sarasaib

takes spectra for only 1 particle
simplified version of one_particle_power_spectra.py
Use this with oneparticle_boris_simulation.py
"""
print('onepartilce_power_spectra.py')
import numpy as np
import oneparticle_boris_simulation as boris
import matplotlib.pyplot as plt
import scipy.constants as cst
from math import pi
import time

'''
q = -1*cst.e #not sure what to do with the increase in speed and the particle reaching the speed higher than c
m = cst.electron_mass
N = 1 # number of particles

mass_array = np.transpose(np.ones(N))*m
mass_array = mass_array[:,None]   #making an array of N x 1
print('mass_array',mass_array)

q_array = np.transpose(np.ones(N))*q
q_array = q_array[:,None]
print('q_array',q_array)

E_ext = np.array([0,0,0])  #needs to be updated according the penning trap potential
B_ext = np.array([0,0,1]) *0.7

#v_array_old = np.array([[10,0,0]])
v_array_old = np.ones(shape = (N,3))*14215 #This is the mean velocity when the total mean v = 24600 m/s from v=sqrt(2KT/m)
print('v_array_old',v_array_old)

v_array_new = np.zeros(shape = (N,3))

if N==2:
    r_array_old = np.array([[1,1,1],[-1,-1,-1]])  #for 2 particles
#for 1 particle only:
r_array_old = np.array([[0.0001,0.0001,0.0001]])

print('r_array_old',r_array_old)
r_array_new = np.zeros(shape = (N,3))
#calculating B field
B_array = B_ext
print('B_array = ',B_array)
'''


q = -1*cst.e
m = cst.m_e
inj = 1E-10#0.5E-6

[N, q_array, mass_array, E_ext, B_array, r_array_old, r_array_new, v_array_old, v_array_new] = boris.initial_conditions()

Lamda = abs(cst.c/(q*0.7/m/(2*pi))) #resonance
frequency = abs(q*0.7/m/(2*pi))   #=19594742910.633125

def power(r_array_old,v_array_old,B_array,E_ext,freq):
    #Should we take the E_fin - E_ini? Or average over?
    lamda=cst.c/freq
    print('input to the power and calcRVE function lamda=',lamda)
    [v, r, t, Time_max, E, totalKE, trapU, intU] = boris.calc_rvE(N, q_array, mass_array, E_ext, B_array, r_array_old, r_array_new, v_array_old, v_array_new,lamda,injection_time = inj)
    totalE1 = totalKE + trapU + intU

    P1 = (totalE1[-2] - totalE1[1])/Time_max   #E_final/ total calc time
    return Time_max, P1, [v, r, t, Time_max, E, totalKE, trapU, intU]

def main():   
    start = time.time()
    power_array1 = []  #tracks power for each frequency of microwave
    #power_array2 = []
    f_left = 19.55E9 #frequency*0.5
    f_right = 19.6E9 #frequency*1.5
    f_array = np.linspace(f_left,f_right,20) #resonance frequency for B=0.7 is lamda = 0.015321487301949693
    
    count = 1
    for freq in f_array: 
        #plot p vs w
        Time_max, P1,ar = power(r_array_old,v_array_old,B_array,E_ext,freq)
        power_array1.append(P1)
        #power_array2.append(P2)
        power_calc_time = time.time()
        print('count = ', count, ' /', str(np.size(f_array)),'i am gooooooood')
        print('time until now ='+str(power_calc_time - start))
        count+=1
        
    spectra(f_array,power_array1)
    #spectra(f_array,power_array2,label = 'no interaction')
    end = time.time()
    print('time took = ' + str(end - start))
    return power_array1#,power_array2
        
def power_spectra():
    start = time.time()
    power_array1 = []  #tracks power for each frequency of microwave
    #power_array2 = []
    data_ar = []  #contains data such as r,v, etc for each freq
    
    f_left = 19.55E9 #frequency*0.5
    f_right = 19.6E9 #frequency*1.5
    f_array = np.linspace(f_left,f_right,20) #resonance frequency for B=0.7 is lamda = 0.015321487301949693
    
    count = 1
    for freq in f_array: 
        #plot p vs w
        Time_max, P1,ar = power(r_array_old,v_array_old,B_array,E_ext,freq)
        power_array1.append(P1)
        #power_array2.append(P2)
        data_ar.append(ar)
        power_calc_time = time.time()
        print('count = ', count, ' /', str(np.size(f_array)),'heloooooo')
        print('time until now ='+str(power_calc_time - start))
        count+=1
        
    spectra(f_array,power_array1)
    #spectra(f_array,power_array2,label = 'no interaction')
    end = time.time()
    print('time took = ' + str(end - start))
    print('returned: power spectra, data_array for each f')
    return power_array1, data_ar

def spectra(f_array,power_array,label = '', f = cst.e*0.7/cst.electron_mass/(2*pi)):
    plt.plot(f_array,power_array,label = label)
    plt.axvline(x = f, color = 'r', label = 'cyclotron frequency')
    plt.title('Power Spectra',y=1.08)
    plt.xlabel('microwave frequency (Hz)')
    plt.ylabel('power')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
    


# In[ ]:




