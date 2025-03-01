#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 19:36:24 2025

@author: omdalvii
"""

import numpy as np
import matplotlib.pyplot as plt

# Method for finding SNR & plotting PSD 
def getPSD(tone, noise, f_t, f_s, N, C, label):
    # Get noisy tone
    tone_noisy = tone + noise
    
    # Calculate SNR and convert to dB
    P_s = np.sum((np.abs(np.fft.fft(tone_sampled, N))**2) / (f_s/(2*N))) # Use DFT of tone by itself for P_s
    P_n = np.sum((np.abs(np.fft.fft(noise, N))**2) / (f_s/(2*N))) # Use DFT of noise by itself for P_n
    SNR = P_s/P_n
    SNR_dB = 10 * np.log10(SNR)
    
    # Perform DFT on noisy tone
    tone_noisy_fft = np.fft.fft(tone_noisy, N)
    freqs = np.fft.fftfreq(N, 1/f_s)
    
    # Calculate PSD from DFT, normalize & convert to dB
    tone_psd = (np.abs(tone_noisy_fft)**2) / (f_s/(2*N))
    tone_psd_norm = (tone_psd) / (max(tone_psd))
    tone_psd_norm_db = 10 * np.log10(tone_psd_norm)
    
    # Plot the PSD
    plt.figure(figsize=(8, 5))
    plt.plot(freqs * 1e-6, (tone_psd_norm_db))
    plt.xlim(0,f_s/2e6)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Normalized Magnitude (dB)")
    plt.title(f"PSD of {f_t/1e6} MHz tone, sampled at {f_s/1e6} MHz with C = {C} and N = {N}\n"+
              "$SNR_{dB}$ " + f"= {SNR_dB}\n"
              f"({label})")
    plt.grid()
    plt.show()

# Quantization function
def quantize(value, Vfs, bitCount):
    deltaV = Vfs / 2**bitCount
    return ((value//deltaV) + np.round(value%deltaV)) * deltaV


# A
# Set fullscale voltage, tone freq, and sample freq
Vfs = 1
f_t = 200e6
f_s = 400e6
# Set cycle count and bin count and calculate total sample length
N = 2**12
C = 30
T = C * 1/f_t
# Generate times of samples and use them to get sampled tone
t = np.arange(0, T, 1/f_s)
tone_sampled = (Vfs/2) * np.sin(2 * np.pi * f_t * t)
tone_quantized = quantize(tone_sampled, Vfs, 6)
quantization_noise = tone_quantized - tone_sampled
# Get PSD
getPSD(tone_sampled, quantization_noise, f_t, f_s, N, C, "quantized, 6-bit ADC")


# B
# Repeat as above using larger sample rate
f_t = 200e6
f_s = 500e6
t = np.arange(0, T, 1/f_s)
tone_sampled = (Vfs/2) * np.sin(2 * np.pi * f_t * t)
tone_quantized = quantize(tone_sampled, Vfs, 6)
quantization_noise = tone_quantized - tone_sampled
getPSD(tone_sampled, quantization_noise, f_t, f_s, N, C, "quantized, 6-bit ADC")


# C
# Repeat as above using larger bit count
tone_quantized = quantize(tone_sampled, Vfs, 12)
quantization_noise = tone_quantized - tone_sampled
getPSD(tone_sampled, quantization_noise, f_t, f_s, N, C, "quantized, 12-bit ADC")


# D
# Repeat as above using hanning window
hn_tone = tone_sampled * np.hanning(tone_sampled.size)
hn_quant = quantize(hn_tone, Vfs, 12)
hn_quant_noise = hn_quant-hn_tone
getPSD(hn_tone, hn_quant_noise, f_t, f_s, N, C, "quantized, 12-bit ADC, hanning window")


# E
# Add noise to signal (choosing gaussian)
# Set seed for random noise so code is repeatable
np.random.seed(42) 
g_noise_var = 0.00000515
g_noise_std = np.sqrt(g_noise_var)  
# Generate noise
g_noise = np.random.normal(loc=0.0, scale=g_noise_std, size=tone_sampled.shape)
noisy_tone = tone_sampled + g_noise
# Repeat C & D
tone_quantized = quantize(noisy_tone, Vfs, 12)
noise = tone_quantized - tone_sampled
getPSD(noisy_tone, noise, f_t, f_s, N, C, "quantized, gaussian noise, 12-bit ADC")

hn_tone = noisy_tone * np.hanning(noisy_tone.size)
hn_quant = quantize(hn_tone, Vfs, 12)
hn_noise = hn_quant-(tone_sampled * np.hanning(tone_sampled.size))
getPSD(hn_tone, hn_quant_noise, f_t, f_s, N, C, "quantized, gaussian noise, 12-bit ADC, hanning window")