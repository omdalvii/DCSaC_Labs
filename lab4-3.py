# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 12:45:35 2025

@author: omdal
"""

import numpy as np
import matplotlib.pyplot as plt


# arrays for plot
codes = np.array(["000", "001", "010", "011", "100", "101", "110", "111"])
x_nums = np.array([0, 1, 2, 3, 4, 5, 6, 7])

# given values
LSB = 0.1
dac_out_levels = np.array([-0.01, 0.105, 0.195, 0.28, 0.37, 0.48, 0.6, 0.75])
ideal_out_levels = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

###################

# part a
print("Part A:\n")

# get offset and full scale error in LSB units
offset_error = (dac_out_levels[0] - ideal_out_levels[0]) / LSB
fullsc_error = (dac_out_levels[7] - ideal_out_levels[7]) / LSB
print(f"Offset Error: {offset_error:.3f} LSB",
      f"\nFull Scale Error: {fullsc_error:.3f} LSB")

# plot actual vs ideal transfer function
plt.figure(figsize=(8, 6))
plt.xticks(x_nums, codes)
plt.plot(x_nums, ideal_out_levels, label="Ideal Output")
plt.plot(x_nums, dac_out_levels, label="DAC Output")
plt.xlabel("Code")
plt.ylabel("Voltage (V)")
plt.title("Ideal vs. Actual DAC output")
plt.legend(loc="lower right")
plt.grid()
plt.show()

###################

# part b
print("\n------------\nPart B:\n")

# get ideal and actual gain in LSB/code units
gain_ideal = (ideal_out_levels[7] - ideal_out_levels[0]) / (7*LSB)
gain_actual = (dac_out_levels[7] - dac_out_levels[0]) / (7*LSB)
print(f"Ideal Gain: {gain_ideal:.03f} LSB/code",
      f"\nReal Gain: {gain_actual:.03f} LSB/code")

# get gain error
gain_error = gain_actual - gain_ideal
print(f"Gain Error: {gain_error:.03f} LSB/code")

###################

# part c
print("\n------------\nPart C/D:\n")

# end point corrent
#dac_out_corr = (dac_out_levels - dac_out_levels[0]) * (ideal_out_levels[7] / dac_out_levels[7])
dac_out_corr = dac_out_levels
dac_out_corr[0] = 0
dac_out_corr[7] = 0.7
dnl = np.zeros(8)
for i in range(1, len(dac_out_corr)):
    dnl[i] = (dac_out_corr[i] - dac_out_corr[i-1] - LSB) / LSB
    print(f"DNL[{i}] - {dnl[i]:.03f} ({i-1:03b} --> {i:03b})")
print("\n")
inl = np.zeros(8)
for i in range(1, len(dac_out_corr)):
    inl[i] = inl[i-1] + dnl[i]
    print(f"INL[{i}] - {inl[i]:.03f} ({i-1:03b} --> {i:03b})")
np.set_printoptions(precision=3)
print(f"\nMax |DNL| = {max(np.abs(dnl)):.03f}",
      f"\nMax |INL| = {max(np.abs(inl)):.03f}")

# plot corrected dac output
plt.figure(figsize=(8, 6))
plt.xticks(x_nums, codes)
plt.plot(x_nums, ideal_out_levels, label="Ideal Output")
plt.plot(x_nums, dac_out_corr, label="Endpont Corrected Output")
plt.xlabel("Code")
plt.ylabel("Voltage (V)")
plt.title("Ideal vs. Endpoint Corrected DAC output")
plt.legend(loc="lower right")
plt.text(4.6, 0.23, f"Max DNL = {max(np.abs(dnl)):.02f}\nMax INL = {max(np.abs(inl)):.02f}")
plt.grid()
plt.show()

# plot dnl and inl
plt.figure(figsize=(8, 6))
plt.xticks(x_nums, codes)
plt.plot(x_nums, dnl, label="DNL")
plt.plot(x_nums, inl, label="INL")
plt.xlabel("Code")
plt.ylabel("DNL/INL (LSB)")
plt.title("DNL vs. INL")
plt.legend(loc="lower right")
plt.text(4.2, 0.17, f"Max |DNL| = {max(np.abs(dnl)):.02f}")
plt.text(4.4, -0.28, f"Max |INL| = {max(np.abs(inl)):.02f}")
plt.grid()
plt.show()

