# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 22:00:42 2025

@author: omdal
"""

import numpy as np
import matplotlib.pyplot as plt

on_times = np.array([100, 85.75, 72.25, 58.5, 45.25, 42.433, 28.75, 15.75])
vals = np.arange(8)
codes = np.array(["000", "001", "010", "011", "100", "101", "110", "111"])

# make ramp histogram
hist = np.empty(8)
hist[-1] = on_times[-1]
for i in range(1, len(hist)):
    hist[i-1] = on_times[i-1] - on_times[i]
hist_cumul = np.cumsum(hist)
print(hist)

# find ideal situation
ideal_line = vals+0.5
ideal_length = 100/7
normalized_ramp = hist_cumul/ideal_length
print(normalized_ramp)
print(ideal_line)

# find offset/full scale error
off_err = normalized_ramp[0] - ideal_line[0]
fsc_err = normalized_ramp[-1] - ideal_line[-1]

# gain error
gain_real = 7/(normalized_ramp[-1] - normalized_ramp[0])
gain_ideal = 7/(ideal_line[-1] - ideal_line[0])
print(gain_real, gain_ideal)
gain_err = gain_real - gain_ideal

# plot  
plt.figure(figsize=(8, 6))
plt.step(normalized_ramp, vals, label="Actual ADC function")
plt.step(ideal_line, vals, label="Ideal ADC function", ls=":", lw=2)
plt.yticks(vals, codes)
plt.xticks(vals)
plt.xlabel("Input Value (LSB)")
plt.ylabel("Code")
plt.xlim(0,7)
plt.title("Ideal vs. Actual ADC output")
plt.legend(loc="lower right")
plt.text(1.1, 0.1, f"Offset Error = {off_err:.02f} LSB")
plt.text(3.5, 6.7, f"Full Scale Error = {fsc_err:.02f} LSB")
plt.text(1.5, 4.7, f"Gain Error = {gain_err:.02f} LSB/code")
plt.grid()
plt.show()

# find dnl and inl
dnl = np.zeros(8)
for i in range(1, len(dnl)):
    dnl[i] = (normalized_ramp[i]-normalized_ramp[i-1]) - (ideal_line[i]-ideal_line[i-1])
    print(f"DNL[{i}] = {dnl[i]:.03f} ({i-1:03b} --> {i:03b})")
print("\n")
inl = np.zeros(8)
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
plt.plot(vals, dnl, label="DNL")
plt.plot(vals, inl, label="INL")
plt.xticks(vals)
plt.xlabel("Code")
plt.ylabel("DNL/INL (LSB)")
plt.title("DNL vs. INL")
plt.legend(loc="lower left")
plt.text(5, 0.1, f"Max |DNL| = {max(np.abs(dnl)):.02f} LSB")
plt.text(5, -1, f"Max |INL| = {max(np.abs(inl)):.02f} LSB")
plt.grid()
plt.show()