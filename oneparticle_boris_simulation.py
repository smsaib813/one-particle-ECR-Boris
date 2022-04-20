#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:50:34 2022

@author: sarasaib

This is only for one particle simulation. Uses Boris Algorithm
"""

print('oneparticle_boris_simulation.py')
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.constants as cst
import random 
import time
import velocity_sample 

start_time = time.time()

#constants
eps0 = 8.8541878128E-12
k = 1/(4*math.pi*eps0)
q = cst.e
m = cst.m_e
N = 1 #number of particles
inj = 1E-11 #microwave injection time
Lamda = abs(cst.c/(q*0.7/m/(2*math.pi)))#resonance

def circular_microwave_zt(r,t,lamda = 1E-3):  #refer to my derivation in paper "Microwave derivation 2"
    z = r[:,2:3]
    
    c = cst.c
    #lamda = 1E-3 #(set 0.001 to 0.3 m for microwave wavelength)
    power = 10E-3  #10dbm=10mW is standard in ECR paper
    area = math.pi*(2E-2)**2 #2cm radius
    frac = 1/math.sqrt(2)
    E_0 = np.sqrt(power/(2*eps0*area*c))*1000 #Readme this is a huge exaggeration. I just added *1000
    Ex = E_0 * (frac + math.sqrt(1 - frac**2))* np.cos(2*math.pi/lamda* (z - c*t))
    Ey = -E_0 * (frac - math.sqrt(1 - frac**2))* np.sin(2*math.pi/lamda* (z - c*t))
    Ez = z*0
    E = np.concatenate((Ex,Ey,Ez),axis=1)
    
    '''
    freq = c/lamda
    w = circular_microwave_zt_freq()
    if t%0.2==0: 
        print('freq',freq)
        print('microwave_omega', w)
        print('E_0',E_0)   
    '''
    return E


def circular_microwave_zt_freq(lamda = 1E-3):
    #(set lamda 0.001 to 0.3 m for microwave wavelength)
    c = cst.c
    freq = c/lamda
    #w = 2*math.pi*freq
    return freq


#Kinetic Energy vs. time graph
def total_kinetic(m,v):
    return np.sum(1/2*m*v**2)

#Kinetic parallel to B field (assuming B is in z direction only)
def par_kinetic(m,v):
    v_par = v[:,2:3]
    K_par = np.sum(1/2*m*v_par**2)
    return K_par

#Kinetic perpendicular to B field (assuming B is in z direction only)
def perp_kinetic(m,v):
    v_perp = v[:,0:2]
    K_perp = np.sum(1/2*m*v_perp**2)
    return K_perp

def trap_potential(r):
    k_2 = 70
    r=np.copy(r)
    r[:,0:1],r[:,1:2],r[:,2:3] = 1/4*np.square(r[:,0:1]),1/4*np.square(r[:,1:2]), -1/2*np.square(r[:,2:3])
    U = q*k_2*np.sum(r)
    return U

def interaction_potential(dr_mag_array):
    U = k*q/dr_mag_array
    return U

def E_trap(r):
    k_2 = 70
    r=np.copy(r)
    r[:,0:1],r[:,1:2],r[:,2:3] = -1/2*r[:,0:1],-1/2*r[:,1:2],r[:,2:3]
    E= r*k_2
    return E


def initial_conditions(N=1): #number of particles
    q = -1*cst.e #not sure what to do with the increase in speed and the particle reaching the speed higher than c
    m = cst.m_e
    
    mass_array = np.transpose(np.ones(N))*m
    mass_array = mass_array[:,None]   #making an array of N x 1
    
    q_array = np.transpose(np.ones(N))*q
    q_array = q_array[:,None]
    
    E_ext = np.array([0,0,0])  #needs to be updated according the penning trap potential
    Bmag = 0.7
    B_ext = np.array([0,0,1]) *Bmag
    
    #v_array_old = np.array([[10,0,0]])
    v_array_old = np.ones(shape = (N,3))*14215 #for N particles with v = 0
    print('v_array_old',v_array_old)
    
    v_array_new = np.zeros(shape = (N,3))
    
    #r_array_old = np.array([[1,0,0],[-1,0,0]])*8.37E-6 #for 2 particles when intU ~ KE
    #for 1 particle only:
    r_array_old = np.random.uniform(low=0.0, high=1.0, size=(N,3))*8.37E-6   #uniformly distributed array between 0 and 8.37E-6 
    print('r_array_old',r_array_old)
    r_array_new = np.zeros(shape = (N,3))
    #calculating B field
    B_array = B_ext
    print('B_array = ',B_array)
    return [N, q_array, mass_array, E_ext, B_array, r_array_old, r_array_new, v_array_old, v_array_new] 

def energy_investigation(injection_time = inj,lamda=cst.c/(cst.e*0.7/cst.m_e/(2*math.pi)), plot=True):
    [N, q_array, mass_array, E_ext, B_array, r_array_old, r_array_new, v_array_old, v_array_new]=initial_conditions()
    [v_track, r_track, t_track, Time_max, E_track, totalKE, trapU, intU]=calc_rvE(N, q_array, mass_array, E_ext, B_array, r_array_old, r_array_new, v_array_old, v_array_new,lamda,injection_time)
    if plot:
        plot_totE(t_track, Time_max, totalKE, trapU, intU, injection_time=injection_time)
    #plt.subplot(223)
    #plot_potential(t_track,trapU,intU)
    return [N,v_track, r_track, t_track, Time_max, E_track, totalKE, trapU, intU]

#v_array_old = ranmdom.choices()
def cyclotron_freq(q,B_array,m):
    w_c = q*B_array/m
    return w_c

def step_Max(q,B,m):
    w_c = cyclotron_freq(q,B,m)
    f_c = np.linalg.norm(w_c/(2*math.pi))
    #print('ECR freq f_c = ', f_c)
    step_max = 1/f_c
    if f_c == 0:
        step_max = 1e-10
        print('fc=0')
    #print('step_max = ', step_max)
    return step_max

def calc_rvE(N, q_array, mass_array, E_ext, B_array, r_array_old, r_array_new, v_array_old, v_array_new,lamda,injection_time = inj):
    step_max = step_Max(q,B_array,mass_array)   #this is the full cyclotron rotation period

    start_op_time = time.time()
    #delta_t = step_max/100
    delta_t = step_max/10
    T = 0
    Time_max = step_max*2500#1E-7#delta_t*100
    #Time_max = 1.05E-6 #(1.05 us)
    
    dr_array = np.array([np.zeros(shape=(N-1,3)),]*N)
    dr_mag_array = np.zeros(shape=(N,1))
    #E_array = np.array([np.array([0.,0.,0.]) for i in range (N)])   #initialized E_array
    #E_array_new = np.array([np.array([0.,0.,0.]) for i in range (N)])

    E_track = []
    v_track = []
    r_track = []
    t_track = []
    trapU = []
    #interactionU = []
    totalKE = []
    parE = []
    perpE = []

    test_r_final = []  # this list stores final positions after each loop for comparison
    test_t_final = []
    
    
    print('---',N, q_array, mass_array, E_ext, B_array, 'r_old =',r_array_old, r_array_new, v_array_old, v_array_new,lamda,injection_time)
    count  = 0
    while(T<Time_max):#0):#0.5*1/f_c):
        count += 1
        if T==5*delta_t:
            print('time in calc_time= ',T)
            print('inj = ',injection_time)
        micro_start = (Time_max - injection_time)/2
        micro_end = micro_start + injection_time
        
        if ((T> micro_start) and (T<(micro_end))):  #turn on microwave from t =  T/2 to 3T/4 #README if possible, change this to a period I can extract in other codes
            E_microwave = circular_microwave_zt(r_array_old,T-micro_start,lamda)
            #print('----')
            #print(E_microwave)
        else:
            E_microwave = 0
        #print(E_microwave)
        E_array_new = E_microwave + np.copy(E_trap(r_array_old))
        E_track.append(E_microwave + np.copy(E_trap(r_array_old)))
 

        #At the end of the array, we have as a main thing: E-field whose rows correspond to each particle
        totalKE.append(total_kinetic(m,v_array_old))
        trapU.append(trap_potential(r_array_old))
        '''
        if N>1:
            interactionU.append(interaction_potential(dr_mag_array))
        '''
        parE.append(par_kinetic(m,v_array_old))
        perpE.append(perp_kinetic(m,v_array_old))

        v_minus = v_array_old + delta_t/2*(q_array/mass_array)*E_array_new
        
        for i in range(N): 
            
            c = float(-(q_array[i]*delta_t)/(2*mass_array[i]))

            B1= B_array[0]
            B2 = B_array[1]
            B3 = B_array[2]

            a1 = np.array([[1, c*B3, -c*B2], [-c*B3, 1, -c*B1], [c*B2, c*B1, 1]])
            b1 = np.array(-c*np.cross(v_minus[i],B_array) + v_minus[i])
            v_plus =np.linalg.solve(a1, b1)

            v_array_new[i] = v_plus + (q_array[i]*delta_t)/(2*mass_array[i])*E_array_new[i]
        
        r_array_new = v_array_new * delta_t + r_array_old
        #print(r_array_old)
        v_array_old = v_array_new
        r_array_old = r_array_new
        v_track.append(np.copy(v_array_old))
        r_track.append(r_array_old)
        t_track.append(T)

        T += delta_t

    print('---end of calculation for all particles---')
    totalKE = np.array(totalKE)
    #interactionU = np.array(interactionU)

    '''
    if N == 1:
        interactionU = totalKE*0   #just to make an array of same size as totalKE but elements all being 0 for no interaction
    #print(interactionU[0:10])

    intU = [np.linalg.norm(interactionU[i]) for i in range(np.size(totalKE))] 
    '''

    intU = 0
    totalE = np.array(totalKE) + np.array(trapU) + np.array(intU)
    #totalU = np.array(trapU) + np.array(intU)
    trapU = np.array(trapU)
    v_track = np.array(v_track)
    r_track = np.array(r_track)
    t_track = np.array(t_track)
    E_track = np.array(E_track)

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time

    print('current time = ', current_time)
    #print('time took = ', time_took)
    print('total points',count, np.size(t_track))
    return [v_track, r_track, t_track, Time_max, E_track, totalKE, trapU, intU]

#plotting x vs y position
def plot_xy(r_track):  
    start_op_time = time.time()
    fig, (ax0,ax01) = plt.subplots(1,2)
    fig.tight_layout()

    x1 = r_track[:,:,0]
    y1 = r_track[:,:,1]

    size = np.size(y1)
    #ax1.margins(-0.1,-0.1)
    print(np.size(x1))
    #1st particle
    #ax01 = plt.subplot(132)
    ax0.plot(x1,y1,'r-')
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('y (m)')
    ax0.title.set_text('particles position')

    ax01.plot(x1[0:1000],y1[0:1000],'-')
    ax01.set_xlabel('x (m)')
    ax01.title.set_text('particles position')
    ax01.text(0.5, 0.5, '1st 1000 points', horizontalalignment='center',
     verticalalignment='center', transform=ax01.transAxes)
    #plt.gca().set_aspect('equal')
    plt.show()

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)


#plotting z vs t
def plot_position(t_track,r_track):  #readme needs modifying
    start_op_time = time.time()
    
    plt.subplot(3,1,1)
    x1 = r_track[:,:,0]
    y1 = r_track[:,:,1]
    z1 = r_track[:,:,2]

    plt.xlabel('time (s)')
    plt.ylabel('x (m)')
    plt.plot(t_track,x1)
    plt.title('1st particle x vs t')
    plt.show()

    #plotting y vs t
    plt.subplot(3,1,2)

    plt.xlabel('time (s)')
    plt.ylabel('y (m)')
    plt.plot(t_track,y1)
    plt.title('1st particle y vs t,',y=1.1)
    plt.show()

    #plotting z vs t
    plt.subplot(3,1,3)

    plt.xlabel('time (s)')
    plt.ylabel('z (m)')
    plt.plot(t_track,z1)
    plt.title('1st particle z vs t',y=1.1)
    plt.show()

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)


# In[14]:


#plotting Vx vs t position
def plot_Vx_t(t_track, v_track):
    start_op_time = time.time()

    fig1, (ax1,ax11) = plt.subplots(1,2)
    fig1.tight_layout()
    l = 500001
    v1_track = v_track[:,0,:]
    vx1 = v1_track[:,0]
    vy1 = v1_track[:,1]
    vz1 = v1_track[:,2]
    print(np.size(vy1))
    ax1 = plt.subplot(221)
    #ax1.margins(-0.1,-0.1)
    Vx_graph, = ax1.plot(t_track[0:l],vx1[0:l],'-',label = 'Vx')
    Vy_graph, = ax1.plot(t_track[0:l],vy1[0:l],'-',label = 'Vy')
    Vz_graph, = ax1.plot(t_track[0:l],vz1[0:l],'-',label = 'Vz')

    plt.xlabel('t')
    plt.ylabel('v (m/s)')
    ax1.legend(handles=[Vx_graph, Vy_graph,Vz_graph])
    plt.title('1st particle velocities',y=1.05)

    ax11 = plt.subplot(222)
    Vtotal_graph, = ax11.plot(t_track[0:l], np.sqrt(vx1**2 + vy1**2 + vz1**2)[0:l],'g-', label = 'V total magnitude')
    plt.xlabel('t')
    plt.ylabel('v (m/s)')
    ax11.legend(handles=[Vtotal_graph])
    plt.title('1st particle v mag',y=1.05)
    #plt.gca().set_aspect('equal')


    plt.show()

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)

from operator import add
def plot_KE(t_track, Time_max, totalKE, trapU, intU,l=250000, injection_time = inj): #readme needs modifying
    l =  np.size(t_track)
    start_op_time = time.time()
    totalE = totalKE + trapU + intU
    
    #plot for totalKE
    plt.subplot(121)
    plt.plot(t_track[0:l],totalKE[0:l])
    if injection_time>0:
        micro_start = (Time_max - injection_time)/2
        micro_end = micro_start+injection_time
        plt.axvline(x = micro_start, color = 'r')#, label = 'start of microwave heating')
        plt.axvline(x = micro_end, color = 'r')#, label = 'end of microwave heating')
    plt.ylabel('energy')
    plt.xlabel('time')
    plt.title('Total KE',y=1.05)
    
    #plot for total E
    plt.subplot(122)
    plt.plot(t_track[0:l],totalE[0:l])
    if injection_time>0:
        micro_start = (Time_max - injection_time)/2
        micro_end = micro_start+injection_time
        plt.axvline(x = micro_start, color = 'r')#, label = 'start of microwave heating')
        plt.axvline(x = micro_end, color = 'r')#, label = 'end of microwave heating')
    plt.title('Total E',y=1.05)
    plt.xlabel('time')

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)

#plotting total energy
def plot_totE(t_track, Time_max, totalKE, trapU, intU, l=500000, injection_time = inj): 
    l = np.size(t_track)
    start_op_time = time.time()
    
    totalU = trapU + intU
    totalE = totalKE + totalU
    fig_energy, (ax_energy1, ax_energy2) = plt.subplots(1,2)

    #l = 500001
    ax_energy1 = plt.subplot(121)
    totalKE_graph, = ax_energy1.plot(t_track[0:l],totalKE[0:l],'-',label = 'total KE')
    totalU_graph, = ax_energy1.plot(t_track[0:l],totalU[0:l],'-',label = 'total U')
    #totalE_graph, = ax_energy1.plot(t_track[0:l],totalE[0:l],'-',label = 'total energy')
    #totalE_graph_scaled, = ax_energy1.plot(t_track[0:l],totalE[0:l]*(max(totalKE[0:l])/max(totalE[0:l])),'g--',label = 'total energy enlarged')
    if injection_time>0:
        micro_start = (Time_max - injection_time)/2
        micro_end = micro_start+injection_time
        print('aaaaaaaaaaa')
        plt.axvline(x = micro_start, color = 'r')#, label = 'start of microwave heating')
        plt.axvline(x = micro_end, color = 'r')#, label = 'end of microwave heating')
    plt.xlabel('t')
    plt.ylabel('energy')
    ax_energy1.legend(handles=[totalKE_graph, totalU_graph])#, totalE_graph])
    plt.title('1st particle Energy',y=1.05)
    plt.text(0.50, -0.35, "KE initial and final = " + str(totalKE[:3]) +'\n' + str(totalKE[-3:]) 
             + '\n' + "trapU initial and final = " + str(trapU[:3]) +'\n' + str(trapU[-3:])
             + '\n' + "calcualtion time  = " + str(Time_max)
             + '\n Microwave applied for ' + str(injection_time) + ' s'
             , transform=plt.gcf().transFigure, fontsize=14, ha='center', color='blue')

    ax_energy2 = plt.subplot(122)
    plt.plot(t_track[0:l],totalE[0:l],'g-',label = 'total energy')
    #plt.text(1,1,str(lamda))
    plt.xlabel('t')
    plt.ylabel('energy')
    plt.title('1st total energy',y=1.05)

    print('fluctuation in total E: totalE/totalKE = ', max(totalE[0:l])/max(totalKE[0:l]))
    ##ASK about this phenomena of total energy change
    ##it is small compared to the KE and U (order of 10^-5) so it should not be much problem. Is this a computation error?

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    #print('current time = ', current_time)
    #print('time took = ', time_took)
    print('initial and final KE = ', [totalKE[:3],totalKE[-3:]])
    print('initial and final trap U = ', [trapU[:3],trapU[-3:]])
    #print('initial and final intU = ', [intU[:3],intU[-3:]])


# In[20]:


#plotting potentials
def plot_potential(t_track, Time_max, trapU, intU, injection_time= inj): #readme needs modifying
    start_op_time = time.time()
    
    if np.size(trapU) == np.size(t_track):
        #plt.subplot(121)
        plt.title('V_trap',y=1.05)
        plt.plot(t_track,np.array(trapU))
    if injection_time>0:
        micro_start = (Time_max - injection_time)/2
        micro_end = micro_start+injection_time
        plt.axvline(x = micro_start, color = 'r')#, label = 'start of microwave heating')
        plt.axvline(x = micro_end, color = 'r')#, label = 'end of microwave heating')
        
        plt.xlabel('t')
        plt.ylabel('V')
    
    '''
    if np.size(intU) == np.size(t_track):
        plt.subplot(122)
        plt.title('V_int',y=1.05)
        plt.xlabel('time (s)')
        plt.ylabel('V (J)')
        plt.plot(t_track,np.array(intU)*1e5)
    '''
      
    if injection_time>0:
        micro_start = (Time_max - injection_time)/2
        micro_end = micro_start+injection_time
        plt.axvline(x = micro_start, color = 'r')#, label = 'start of microwave heating')
        plt.axvline(x = micro_end, color = 'r')#, label = 'end of microwave heating')
        plt.xlabel('time (s)')

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)

def phi(w,t_track,v_track):
    vx, vy = v_track[:,:,0],v_track[:,:,1],v_track[:,:,2]
    phi = w*t_track - math.arccos(vx/max(vx))
    phi2 = w*t_track - math.arcsin(vy/max(vy))
    return phi, phi2  #readme: compare these two values to verify they are the same


#plotting guiding center
def guidingR(w, v_track, r_track, t_track):
    start_op_time = time.time()
    fig3, (ax1,ax2,ax3) = plt.subplots(1,3)
    fig3.tight_layout()

    x1, y1, z1 = r_track[:,:,0],r_track[:,:,1],r_track[:,:,2]
    vx, vy = v_track[:,:,0],v_track[:,:,1],v_track[:,:,2]
    
    if np.linalg.norm(w_c) == 0:
        w_c = 1e-10
    Rx1 = x1 - vy/np.linalg.norm(w) #readme: I think this is x1 + vy/np.linalg.norm(w_c)
    Ry1 = y1 - vx/np.linalg.norm(w)  
    Rz1 = z1

    ax1 = plt.subplot(131)
    #ax1.margins(-0.1,-0.1)
    ax1.plot(Rx1,Ry1, label = 'Guiding Center Positions')
    ax1.scatter(x1,y1, label = 'Parcle Positions')
    ax1.title.set_text('Guiding Center Position')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.legend()
    
    ax2 = plt.subplot(132)
    ax2.plot(t_track,Rx1)
    ax2.title.set_text('Change in Guiding Center x-Position')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('x (m)')
    ax2.legend()
    
    ax3 = plt.subplot(133)
    ax3.plot(t_track,Rx1)
    ax3.title.set_text('Change in Guiding Center y-Position')
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('y (m)')
    ax3.legend()

    #plt.gca().set_aspect('equal')


    plt.show()

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)
    
def v_perp(t_track, v_track):
    v_perp = np.stack((v_track[:,:,0],v_track[:,:,1]),axis=1)
    v_perp_mag = np.linalg.norm(v_perp,axis=1)
    plt.plot(t_track,v_perp_mag)
    plt.xlabel('time (s)')
    plt.ylabel('velocity (m/s)')
    plt.title('V Perpendicular')
    return v_perp

def v_parallel(t_track, v_track):
    v_par = v_track[:,:,2]
    plt.plot(t_track,v_par)
    plt.xlabel('time (s)')
    plt.ylabel('velocity (m/s)')
    plt.title('V Parallel')
    return v_par