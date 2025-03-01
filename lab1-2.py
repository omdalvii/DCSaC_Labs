# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 02:56:55 2025

@author: omdal
"""

import numpy as np
import matplotlib.pyplot as plt
import math


Fs = 500e6  # Sampling frequency (500 MHz)
F1 = 300e6  # First signal frequency (300 MHz)
F2 = 800e6  # Second signal frequency (800 MHz)

###############################################
# A

# Lets see what the alias frequency would be for these two signals
R1, R2 = round(F1/Fs), round(F2/Fs)
Fa1 = abs(F1 - R1*Fs)
Fa2 = abs(F2 - R2*Fs)
print(f"The alias frequencies are Fa1 = {Fa1/10**6}MHz, and Fa2 = {Fa2/10**6}MHz")

# As we can see, the alias frequencies are the same in both cases. 
# This is due to the input freqencies being far higher than the nyquist frequency for Fs=300MHz
# We can plot the original signals and show where they are sampled, then overlay the alias
# frequency to visualize how this is happening

# Make time array for sample points and use it to sample input signals
T = 20e-9   # Duration of signal (20ns)
t = np.arange(0, T, 1/Fs)  # Sample times
x1 = np.cos(2 * np.pi * F1 * t)
x2 = np.cos(2 * np.pi * F2 * t)

# Make "continuous" versions of input/alias signals for graphing
t_cont = np.arange(0, T, 1e-12)
x1_cont = np.cos(2 * np.pi * F1 * t_cont)
x2_cont = np.cos(2 * np.pi * F2 * t_cont)
alias = np.cos(2* np.pi * Fa1 * t_cont)

# Plot the signals
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_cont * 1e9, x1_cont, "-b", label=f"Orignial x1(t), F1 = {F1/1e6}MHz")
plt.plot(t_cont * 1e9, alias, "-r", label=f"Aliased signal of x1(t), Fa1 = {Fa1/1e6}MHz")
plt.scatter(t * 1e9, x1)
plt.title(f"Aliased Signal x1(n) @ Fs={Fs/1e6}MHz")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.legend(loc=4)

plt.subplot(1, 2, 2)
plt.plot(t_cont * 1e9, x2_cont, "-b", label=f"Orignial x2(t), F2 = {F2/1e6}MHz")
plt.plot(t_cont * 1e9, alias, "-r", label=f"Aliased signal of x2(t), Fa2 = {Fa2/1e6}MHz")
plt.scatter(t * 1e9, x2)
plt.title(f"Aliased Signal x2(n) @ Fs={Fs/1e6}MHz")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.legend(loc=4)

plt.show()

#################################################
# B

# Since the sample frequency is only 300MHz, the nyquist frequency is 150MHz. This means that any
# signal that has a frequency above 150MHz that is sampled will be aliased, and become unrecoverable

# To fix this, we can increase the sampling rate to be at least 1.6GHz, which would give us a new
# nyquist frequency of 800MHz. This should allow us to sample both x1(t) and x2(t) and recover the
# original signals

#################################################
#C

# plt.figure(figsize=(6,5))
# plt.step(t*1e9, x1, where="post")
# plt.plot(t_cont*1e9, x1_cont)
# plt.scatter(t*1e9, x1)
# plt.show()

#################################################
#D
np.seterr(divide='ignore', invalid='ignore')

def reconstruct(xs, Ts, t):
    sum=0
    for n in range(1,len(xs)):
        sum += xs[n] * (np.sin(np.pi * (t - n*Ts) / Ts)) / (np.pi * (t - n*Ts) / Ts)
    return sum

def mse(xr, x, size):
    sum = 0
    for r, o in zip(xr, x):
        if np.isnan(r): sum += 0
        else: sum += (r-o)**2
    return sum/size

def partD(Fs):
    F1 = 300e6 # x1 frequency
    T = 10/F1 # Duration of sample = 10 cycles
    Ts = 1/Fs
    
    # Sample x1(t)
    t = np.arange(0, T, Ts) # Sample times
    t_cont = np.arange(0, T, 1e-12)
    x1 = np.cos(2 * np.pi * F1 * t)
    x1_cont = np.cos(2 * np.pi * F1 * t_cont)
    
    # Reconstruct x1(t)
    xr = reconstruct(x1, Ts, t_cont)
    
    # Display MSE
    MSE = mse(xr, x1_cont, len(xr))
    print(f"\nMSE for reconstruction of x1(t) [{Fs/1e6}MHz, 0:Ts:T-Ts] = {MSE}")
    
    # Plot reconstructed vs original signal
    plt.figure(figsize=(12,5))
    
    plt.subplot(1, 2, 1)
    plt.plot(t_cont*1e9, xr, "-b", label="Reconstructed x1(t)")
    plt.plot(t_cont*1e9, x1_cont, ":r", label = "Original x(t)")
    plt.title(f"Original vs. Reconstructed signal x1(t) \nsampling @ Fs={Fs/1e6}MHz 0:Ts:T-Ts")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.legend(loc=3)
    
    # Sample x1(t) with samples shifted
    t = t+(Ts/2)
    x1 = np.cos(2 * np.pi * F1 * t)
    
    # Reconstruct x1(t)
    xr = reconstruct(x1, Ts, t_cont)
    
    # Display MSE
    MSE = mse(xr, x1_cont, len(xr))
    print(f"MSE for reconstruction of x1(t) [{Fs/1e6}MHz, Ts2:Ts:T-Ts/2] = {MSE}")
    
    # Plot reconstructed vs original signal
    plt.subplot(1, 2, 2)
    plt.plot(t_cont*1e9, xr, "-b", label="Reconstructed x1(t)")
    plt.plot(t_cont*1e9, x1_cont, ":r", label = "Original x(t)")
    plt.title(f"Original vs. Reconstructed signal x1(t) \nsampling @ Fs={Fs/1e6}MHz Ts2:Ts:T-Ts/2")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.legend(loc=3)
    
    plt.show()

partD(800e6)
partD(1000e6)
partD(500e6)

# We can see the MSE went down when Fs was increased to 1000MHz, which makes sense as more samples
# per second means better data. We can also see the MSE drastially increased when we dropped to 500MHz,
# the severity of the increase can be attributed to the frequency of the original signal (300MHz) being
# above the nyqist frequency (250MHz)
