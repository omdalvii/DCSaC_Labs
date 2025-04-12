# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 02:32:35 2025

@author: omdal
"""

import numpy as np 
import matplotlib.pyplot as plt

# part a
print("Part A:\n")

# create time array
f_s = 100e9 # none given so choosing arbitrary rate
T = 5e-9 # length of simulation
dt = 1/f_s # time between samples
t = np.arange(0, T, dt)

# get signal amplitude
rms = 0.2
A = rms * np.sqrt(2)

# set signal frequency so that k cycles fit in simulation
k = 4
f_in = k / T

# create input signal
v_in = A * np.sin(2 * np.pi * f_in * t)

# set up ADC/create ADC output
v_fs = 1.2 # full-scale voltage
LSB = v_fs / (2**12) # for 12-bit adc
v_out = LSB * np.round(v_in/LSB)

# get quantization error
quant_err = v_out - v_in

# get power values and SNR
sig_pwr = np.mean(v_in**2)
noise_pwr = np.mean(quant_err**2)
SNR = 10 * np.log10(sig_pwr/noise_pwr)
print(f"Calculated SNR - {SNR:.2f} dB")
print("\n--Simulation values--",
      f"\nf_s = {f_s*1e-9} GHz",
      f"\nf_in = {f_in*1e-6} MHz",
      f"\nLSB = {LSB*1e6} uV")

# perform DFT on v_out
N = 2**12
v_out_fft = np.fft.fft(v_out, N)
freqs = np.fft.fftfreq(N, 1/(f_s))

# get PSD
v_out_psd = (np.abs(v_out_fft)**2) / (f_s/(2*N))
v_out_psd_norm = v_out_psd/max(v_out_psd)
v_out_psd_norm_db = 10 * np.log10(v_out_psd_norm)

# plot transient graph of input/output
plt.figure(figsize=(8, 6))
plt.plot(t * 1e9, v_in, label="Input Signal", c="blue")
plt.plot(t * 1e9, v_out, label="ADC Output", c="orange", lw=1)
plt.xlabel("Time (ns)")
plt.ylabel("Voltage (V)")
plt.title("Signal vs. ADC Output")
plt.legend(loc="lower right")
plt.show()

# plot the DFT
plt.figure(figsize=(8, 5))
plt.stem(freqs * 1e-9, np.abs(v_out_fft))
plt.xlim(0,f_s/20e9)
plt.xlabel("Frequency (GHz)")
plt.ylabel("FFT Magnitude")
plt.title("DFT of $V_{out}$")
plt.show()

# plot the PSD
plt.figure(figsize=(8, 5))
plt.stem(freqs * 1e-9, v_out_psd_norm)
plt.xlim(0,f_s/20e9)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Normalized Magnitude")
plt.title("Normalized PSD of $V_{out}$")
plt.text(2, 0.2, f"SNR = {SNR:.2f} dB")
plt.show()

# plot the PSD
plt.figure(figsize=(8, 5))
plt.stem(freqs * 1e-9, v_out_psd_norm_db)
plt.xlim(0,f_s/20e9)
plt.ylim(-100, 10)
plt.yscale("symlog")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Normalized Magnitude (dB)")
plt.title("Normalized PSD of $V_{out}$ in Decibels")
plt.text(2, 1, f"SNR = {SNR:.2f} dB")
plt.show()

#####################################################################

# part b
print("\n-----------------\nPart B:\n")

# create full-scale input signal
v_in = 0.6 * np.sin(2 * np.pi * f_in * t) # A = v_fs/2 = 0.6V

# Set seed for random noise so code is repeatable
np.random.seed(42) 

# add gaussian noise w/ std=0.5V to signal
noise = np.random.normal(loc=0.0, scale=0.5, size=v_in.shape)
v_in_noisy = v_in + noise

# get new v_out from noisy signal
v_out_noisy = LSB * np.round(v_in_noisy/LSB)
v_out_noisy = np.clip(v_out_noisy, -0.6, 0.6) # clip adc outpt to stay within [v_fs/2, -v_fs2]

# get quantization error
quant_err_noisy = v_out_noisy - v_in_noisy

# get SNR of input signal
noisy_sig_in_pwr = np.mean(v_in**2)
noisy_noise_in_pwr = np.mean(noise**2)
noisy_in_SNR = 10 * np.log10(noisy_sig_in_pwr/noisy_noise_in_pwr)

# get SNR of output signal
noisy_sig_out_pwr = np.mean(v_in_noisy**2)
noisy_noise_out_pwr = np.mean(quant_err_noisy**2)
noisy_out_SNR = 10 * np.log10(noisy_sig_out_pwr/noisy_noise_out_pwr)

print(f"Calculated SNR of input signal: {noisy_in_SNR} dB")
print(f"Calculated SNR of output signal: {noisy_out_SNR} dB")
print("\n--Simulation values--",
      f"\nV_pkpk of noise = {max(noise)-min(noise):.2f} V")

# plot transient graph of input with/without noise
plt.figure(figsize=(8, 6))
plt.plot(t * 1e9, v_in_noisy, label="Noisy Input Signal", c="orange")
plt.plot(t * 1e9, v_in, label="Pure Input Signal", c="blue", lw=3)
plt.xlabel("Time (ns)")
plt.ylabel("Voltage (V)")
plt.title("Pure vs. Noisy input signal")
plt.legend(loc="lower right")
plt.text(3.5, 1.6, f"SNR = {noisy_in_SNR:.2f} dB")
plt.show()

# plot transient graph of input/output
plt.figure(figsize=(8, 6))
plt.plot(t * 1e9, v_in_noisy, label="Noisy Input Signal", c="blue")
plt.plot(t * 1e9, v_out_noisy, label="ADC Output", c="orange", lw=1)
plt.xlabel("Time (ns)")
plt.ylabel("Voltage (V)")
plt.title("Signal vs. ADC Output, with gaussian noise added")
plt.legend(loc="lower right")
plt.text(3.5, 1.6, f"SNR = {noisy_out_SNR:.2f} dB")
plt.show()

#####################################################################

# part c
print("\n-----------------\nPart C:\n")

# create full-scale input signal
v_in = 0.6 * np.sin(2 * np.pi * f_in * t) # A = v_fs/2 = 0.6V

# add gaussian noise w/ std=0.5V to signal
noise = np.random.uniform(-0.5, 0.5, size=v_in.shape)
v_in_noisy = v_in + noise

# get new v_out from noisy signal
v_out_noisy = LSB * np.round(v_in_noisy/LSB)
v_out_noisy = np.clip(v_out_noisy, -0.6, 0.6) # clip adc outpt to stay within [v_fs/2, -v_fs2]

# get quantization error
quant_err_noisy = v_out_noisy - v_in_noisy

# get SNR of input signal
noisy_sig_in_pwr = np.mean(v_in**2)
noisy_noise_in_pwr = np.mean(noise**2)
noisy_in_SNR = 10 * np.log10(noisy_sig_in_pwr/noisy_noise_in_pwr)

# get SNR of output signal
noisy_sig_out_pwr = np.mean(v_in_noisy**2)
noisy_noise_out_pwr = np.mean(quant_err_noisy**2)
noisy_out_SNR = 10 * np.log10(noisy_sig_out_pwr/noisy_noise_out_pwr)

print(f"Calculated SNR of input signal: {noisy_in_SNR} dB")
print(f"Calculated SNR of output signal: {noisy_out_SNR} dB")
print("\n--Simulation values--",
      f"\nV_pkpk of noise = {max(noise)-min(noise):.2f} V")

# plot transient graph of input with/without noise
plt.figure(figsize=(8, 6))
plt.plot(t * 1e9, v_in_noisy, label="Noisy Input Signal", c="orange")
plt.plot(t * 1e9, v_in, label="Pure Input Signal", c="blue", lw=3)
plt.xlabel("Time (ns)")
plt.ylabel("Voltage (V)")
plt.ylim(-1.3,1.3)
plt.title("Pure vs. Noisy input signal")
plt.legend(loc="lower right")
plt.text(3.5, 1.15, f"SNR = {noisy_in_SNR:.2f} dB")
plt.show()

# plot transient graph of input/output
plt.figure(figsize=(8, 6))
plt.plot(t * 1e9, v_in_noisy, label="Noisy Input Signal", c="blue")
plt.plot(t * 1e9, v_out_noisy, label="ADC Output", c="orange", lw=1)
plt.xlabel("Time (ns)")
plt.ylabel("Voltage (V)")
plt.ylim(-1.3,1.3)
plt.title("Signal vs. ADC Output, with uniform noise added")
plt.legend(loc="lower right")
plt.text(3.5, 1.15, f"SNR = {noisy_out_SNR:.2f} dB")
plt.show()
