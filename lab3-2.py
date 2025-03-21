# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 17:33:47 2025

@author: omdal
"""

import numpy as np

# part a
f_data = 10e9
t_charge = 0.5 * 1/f_data
v_lsb = 1/2**7
v_change = 0.5

tau = - t_charge / np.log(v_lsb/v_change)
print(f"a.\ntau = {tau*10**12:.04f}ps")

# part b

# calculate largest change in input signal
freqs = [0.2e9, 0.58e9, 1e9, 1.7e9, 2.4e9]
sum = 0
for freq in freqs:
    sum += 0.4 * np.sin(2 * np.pi * freq * 25e-12) 
print(f"b.\nV_change,max = {sum*10**3:0.1f}mV")

# calculate tau
tau = -50e-12 / np.log(7.8125e-3/sum)
print(f"tau = {tau*10**12:.04f}ps")