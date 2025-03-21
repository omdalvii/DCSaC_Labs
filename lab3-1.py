# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 05:42:26 2025

@author: omdal
"""

import numpy as np
import matplotlib.pyplot as plt

# given
f_in = 1e9 # 1 Ghz
f_s = 10e9 # 10 Ghz
tau = 10e-12 # R*C = 10ps

# create time itself
T_end = 2e-9 # simulation length
dt = 1e-12 # time step = 1ps
t = np.arange(0, T_end, dt) # time array

# input signal @ 2V amplitude
v_in = 2 * np.cos(2 * np.pi * f_in * t)

# sampling signal (square wave)
v_sw = (np.sin(2 * np.pi * f_s * t) > 0).astype(int)

# capacitor charging equation:
#       v_c = (v_in-v_current) * (1 - e^(-t/tau)) + v_current
#           = (v_in-v_current) - (v_in-v_current)e^(-t/tau) + v_current
#           = v_in - (v_in - v_current)e^(-t/tau)
# for each dt:
#       v_c[t] = v_in[t-1] * (v_in[t-1] - v_c[t-1])e^(-dt/tau))  

# setup output array and sim
v_c = np.zeros_like(t)
v_c[0] = v_in[0] # initial capacitor charge
for i in range(1,len(v_c)):
    if v_sw[i]:
        v_c[i] = v_in[i-1] - (v_in[i-1] - v_c[i-1])*(1/np.exp(dt/tau))
    else:
        v_c[i] = v_c[i-1]
        
# Plot results
plt.figure(figsize=(8, 6))
plt.plot(t * 1e9, v_in, label="Input Signal")
plt.plot(t * 1e9, v_c, label="Sampled Output")
plt.xlabel("Time (ns)")
plt.ylabel("Voltage (V)")
plt.title("Sampling Circuit Output (Track and Hold)")
plt.legend()
plt.grid()
plt.show()

# if "true" ZOH sampler then there capacitor can instantly sample + hold value
v_zoh = np.zeros_like(t)
for i in range(len(v_c)):
    if i % (1/(f_s*dt)) == 0:
        v_zoh[i] = v_in[i]
    else:
        v_zoh[i] = v_zoh[i-1]
        
# Plot results
plt.figure(figsize=(8, 6))
plt.plot(t * 1e9, v_in, label="Input Signal")
plt.plot(t * 1e9, v_zoh, label="Sampled Output")
plt.xlabel("Time (ns)")
plt.ylabel("Voltage (V)")
plt.title("Sampling Circuit Output (True ZOH)")
plt.legend()
plt.grid()
plt.show()