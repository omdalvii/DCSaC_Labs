#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:55:54 2025

@author: omdalvii
"""


import numpy as np
import matplotlib.pyplot as plt

# Method for finding SNR & plotting PSD 
def getPSD(tone, noise, noise_var, f_t, f_s, N, C, label):
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
    plt.xlim(0,2.5)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Normalized Magnitude (dB)")
    plt.title(f"PSD of {f_t/1e6} MHz tone, sampled at {f_s/1e6} MHz with C = {C} and N = {N}\n"+
              "$\sigma^2$ " +f"= {noise_var}, " + " $SNR_{dB}$ " + f"= {SNR_dB}\n"
              f"({label})")
    plt.grid()
    plt.show()
    

# Set seed for random noise so code is repeatable
np.random.seed(42) 
# Tone frequency and sample frequency
f_t = 2e6
f_s = 5e6
# Set cycle count and bin count and calculate total sample length
N = 2**12
C = 1007
T = C * 1/f_t
# Generate times of samples and use them to get sampled tone
t = np.arange(0, T, 1/f_s)
tone_sampled = np.sin(2 * np.pi * f_t * t)



# Gaussian Noise
# Set variance of noise to be generated
g_noise_var = 0.00000515
g_noise_std = np.sqrt(g_noise_var)  
# Generate noise
g_noise = np.random.normal(loc=0.0, scale=g_noise_std, size=tone_sampled.shape)
# Get PSD
getPSD(tone_sampled, g_noise, g_noise_var, f_t, f_s, N, C, "Gaussian")


# Normal noise
# Set scale variance of noise to be generated
n_noise_var = 0.000001275
high = np.sqrt(n_noise_var * 12)
# Generate noise
n_noise = np.random.uniform(high=high, size=tone_sampled.shape)
# Get PSD
getPSD(tone_sampled, n_noise, n_noise_var, f_t, f_s, N, C, "Normal")


# Hanning Window
hn_tone = tone_sampled * np.hanning(tone_sampled.size)
hn_gaussian = g_noise * np.hanning(g_noise.size)
hn_normal = n_noise * np.hanning(n_noise.size)
getPSD(hn_tone, hn_gaussian, g_noise_var, f_t, f_s, N, C, "Gaussian, Hanning Window")
getPSD(hn_tone, hn_normal, n_noise_var, f_t, f_s, N, C, "Normal, Hanning Window")


# Hamming Window
hn_tone = tone_sampled * np.hamming(tone_sampled.size)
hn_gaussian = g_noise * np.hamming(g_noise.size)
hn_normal = n_noise * np.hamming(n_noise.size)
getPSD(hn_tone, hn_gaussian, g_noise_var, f_t, f_s, N, C, "Gaussian, Hamming Window")
getPSD(hn_tone, hn_normal, n_noise_var, f_t, f_s, N, C, "Normal, Blackman Window")


# Blackman Window
bl_tone = tone_sampled * np.blackman(tone_sampled.size)
bl_gaussian = g_noise * np.blackman(g_noise.size)
bl_normal = n_noise * np.blackman(n_noise.size)
getPSD(bl_tone, bl_gaussian, g_noise_var, f_t, f_s, N, C, "Gaussian, Blackman Window")
getPSD(bl_tone, bl_normal, n_noise_var, f_t, f_s, N, C, "Normal, Blackman Window")