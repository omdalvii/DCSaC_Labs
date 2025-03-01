# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 07:55:22 2025

@author: omdal
"""

import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

######################################
#A

# Given
F = 2e6 # Signal frequency (2MHz)
Fs = 5e6 #Sample frequency (5MHz)
N = 50 # DFT point count

# Sample x(t)
Ts = 1/Fs
T = Ts*N
t = np.arange(0, T, Ts)
xn = np.cos(2 * np.pi * F * t) 

# Do FFT
x_k = fft(xn, N)  # Compute FFT of the signal
frequencies = np.fft.fftfreq(N, Ts)  # Frequency axis

# Plot the DFT
plt.figure(figsize=(8, 5))
plt.stem(frequencies * 1e-6, np.abs(x_k))
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.title(f"DFT of x(t) with F = {F/1e6}MHz\nFs = {Fs/1e6}MHz")
plt.grid()
plt.show()

######################################
#B

# Given
F1 = 200e6 # Signal frequency 1 (200MHz)
F2 = 400e6 # Signal frequency 2 (400MHz)
Fs = 1e9 #Sample frequency (1GHz)
N = 50 # DFT point count

# Sample x(t)
Ts = 1/Fs
T = Ts*N
t = np.arange(0, T, Ts)
yn = np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F * t) 

# Do FFT
y_k = fft(yn, N)  # Compute FFT of the signal
frequencies = np.fft.fftfreq(N, Ts)  # Frequency axis

# Plot the DFT
plt.figure(figsize=(8, 5))
plt.stem(frequencies * 1e-6, np.abs(y_k))
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.title(f"DFT of y(t) with F1 = {F1/1e6}MHz and F2 = {F2/1e6}MHz\nFs = {Fs/1e6}MHz")
plt.grid()
plt.show()

# The 200MHz frequency is easily observable, however the 400MHz cannot be seen. This is odd, as 400MHz
# is under the nyquist frequency and thus should be observable. Instead, there is a large spike at
# around 0MHz

######################################
#C

# Given
F1 = 200e6 # Signal frequency 1 (200MHz)
F2 = 400e6 # Signal frequency 2 (400MHz)
Fs = 500e6 #Sample frequency (500MHz)
N = 50 # DFT point count

# Sample x(t)
Ts = 1/Fs
T = Ts*N
t = np.arange(0, T, Ts)
yn = np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F * t) 

# Do FFT
y_k = fft(yn, N)  # Compute FFT of the signal
frequencies = np.fft.fftfreq(N, Ts)  # Frequency axis

# Plot the DFT
plt.figure(figsize=(8, 5))
plt.stem(frequencies * 1e-6, np.abs(y_k))
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.title(f"DFT of y(t) with F1 = {F1/1e6}MHz and F2 = {F2/1e6}MHz\nFs = {Fs/1e6}MHz")
plt.grid()
plt.show()

# The 200MHz and 0MHz is still apparent, and the 400 MHZ is still not seen.

######################################
#D

# Given
F = 2e6 # x(t) Signal frequency (2MHz)
Fsx = 4e6 # x Sample frequency (4MHz)

F1 = 200e6 # y(t) Signal frequency 1 (200MHz)
F2 = 400e6 # y(t) Signal frequency 2 (400MHz)
Fsy = 500e6 #Sample frequency (400MHz)
N = 50 # DFT point count

# Sample x(t)
Tsx = 1/Fsx
Tx = Tsx*51
tx = np.arange(0, Tx, Tsx)
xn = np.cos(2 * np.pi * F * tx) 

# Sample y(t)
Tsy = 1/Fsy
Ty = Tsy*N
ty = np.arange(0, Ty, Tsy)
yn = np.cos(2 * np.pi * F1 * ty) + np.cos(2 * np.pi * F * ty)

# Apply blackman window
xn_bm = xn * np.blackman(51)
yn_bm = yn * np.blackman(N)

# Do FFT
x_k = fft(xn_bm, 51)  # Compute FFT of the signal
frequenciesx = np.fft.fftfreq(51, Tsx)  # Frequency axis
y_k = fft(yn_bm, N)  # Compute FFT of the signal
frequenciesy = np.fft.fftfreq(N, Tsy)  # Frequency axis

# Plot the DFTs
plt.figure(figsize=(8, 5))
plt.stem(frequenciesx * 1e-6, np.abs(x_k))
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.title(f"DFT of x(t) with F = {F/1e6}MHz\nFs = {Fs/1e6}MHz, blackman window")
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.stem(frequenciesy * 1e-6, np.abs(y_k))
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.title(f"DFT of y(t) with F1 = {F1/1e6}MHz and F2 = {F2/1e6}MHz\nFs = {Fs/1e6}MHz, blackman window")
plt.grid()
plt.show()

# After adding the blackman window, all the peaks are much more spread out than before