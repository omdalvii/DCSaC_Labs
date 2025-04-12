# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 18:28:30 2025

@author: omdal
"""

import numpy as np
import matplotlib.pyplot as plt

# arrays for plot
x_nums = np.arange(2**4)
codes = np.empty(shape=(2**4), dtype='U4')
for i in range(len(x_nums)):
    codes[i] = f"{x_nums[i]:04b}"
    
# given
ramp_hist = np.array([43, 115, 85, 101, 122, 170, 75, 146, 125, 60, 95, 95, 115, 40, 120, 242])
ramp_cumul = ramp_hist
for i in range(1, len(ramp_cumul)):
    ramp_cumul[i] += ramp_cumul[i-1]

# find ideal situation
ideal_line = x_nums+0.5
ideal_length = (ramp_cumul[15] - ramp_cumul[0])/15
normalized_ramp = ramp_cumul/ideal_length

# plot  
plt.figure(figsize=(8, 6))
plt.step(normalized_ramp, x_nums, label="Actual ADC function")
plt.step(ideal_line, x_nums, label="Ideal ADC function", ls=":", lw=2)
plt.yticks(x_nums, codes)
plt.xticks(x_nums)
plt.xlabel("Input Value (LSB)")
plt.ylabel("Code")
plt.xlim(0,15)
plt.title("Ideal vs. Actual ADC output")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# find dnl and inl
dnl = np.zeros(2**4)
for i in range(1, len(dnl)):
    dnl[i] = (normalized_ramp[i]-normalized_ramp[i-1]) - (ideal_line[i]-ideal_line[i-1])
    print(f"DNL[{i}] = {dnl[i]:.03f} ({i-1:03b} --> {i:03b})")
print("\n")
inl = np.zeros(2**4)
for i in range(1, len(inl)):
    inl[i] = dnl[i] + inl[i-1]
    print(f"INL[{i}] = {inl[i]:.03f} ({i-1:03b} --> {i:03b})")
print(f"\nMax |DNL| = {max(np.abs(dnl)):.03f}",
      f"\nMax |INL| = {max(np.abs(inl)):.03f}")

# check monotonicity
if np.any(dnl < -1):
    monotonic = False
else:
    monotonic = True
print(f"\nIs the ADC monotonic? {'Yes' if monotonic else 'No'}")

# plot dnl and inl
plt.figure(figsize=(8, 6))
plt.plot(x_nums, dnl, label="DNL")
plt.plot(x_nums, inl, label="INL")
plt.xticks(x_nums)
plt.xlabel("Code")
plt.ylabel("DNL/INL (LSB)")
plt.title("DNL vs. INL")
plt.legend(loc="lower left")
plt.text(11, 1.1, f"Max |DNL| = {max(np.abs(dnl)):.02f}")
plt.text(9.5, -1.2, f"Max |INL| = {max(np.abs(inl)):.02f}")
plt.grid()
plt.show()