# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 19:43:12 2025

@author: omdal
"""

import numpy as np
import matplotlib.pyplot as plt

# arrays for plot
codes = np.array(["000", "001", "010", "011", "100", "101", "110", "111"])
x_nums = np.array([0, 1, 2, 3, 4, 5, 6, 7])

# given
dnl = np.array([0, -0.5, 0, 0.5, -1, 0.5, 0.5, 0])
ofs_err = 0.5
fs_err = 0.5

# find inl
inl = np.zeros_like(dnl)
for i in range(1, len(inl)):
    inl[i] = inl[i-1] + dnl[i]
    print(f"INL[{i}] = {inl[i]:.03f} ({i-1:03b} --> {i:03b})")

transfer = np.zeros(shape=2**3)
transfer[1] = 0.5 + ofs_err
for i in range(2, len(transfer)):
    transfer[i] = transfer[i-1] + (1 + dnl[i])
    
# plot
plt.figure(figsize=(8, 6))
plt.step(transfer, x_nums)
plt.yticks(x_nums, codes)
plt.xticks(x_nums)
plt.xlabel("Input Value (LSB)")
plt.ylabel("Code")
plt.title("ADC Transfer Function")
plt.text(4.1, 2.4, f"Max |INL| = {max(np.abs(inl)):.02f}")
plt.grid()
plt.show()