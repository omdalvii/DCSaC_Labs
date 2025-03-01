# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


###############################
#A

# FIR filters have no poles, and as such do not have denominators in their transfer functions.
# They come in the form:
#
#       H(x) = b_0 + b_1(z^-1) + b_2(z^-2) + ... + b_n(z^-n)
#
# They can be represented as a single array of coefficiencts (b)
#
# For example, b = [1, 1/2, 1/3] gives the following transfer function
#
#       H(x) = 1 + (1/2)(z^-1) + (1/3)(z^-2)

# IIR filters have poles, so their transfer function has to have a denominator. They are of the
# form:
#
#              (b_0 + b_1(z^-1) + b_2(z^-2) + ... + b_n(z^-n))
#       H(x) = ------------------------------------------------
#              (a_0 + a_1(z^-1) + a_2(z^-2) + ... + a_m(z^-m))
#
# They can be represented as two arrays of coefficients (a & b)
#
# For example, a = [1, 1/2], b = [1] gives the following transfer function
#
#                      1
#       H(x) = ------------------
#               1 + (1/2)(z^-1)

fir_a, fir_b = [1], [1, 1/2, 1/3]
fir_w, fir_h = signal.freqz(fir_b, fir_a)

fir_z, fir_p, _ = signal.tf2zpk(fir_b, fir_a)
print(f"Zeroes: {np.angle(fir_z)}")
print(f"Poles: {np.angle(fir_p)}")

plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(fir_w, 20 * np.log10(abs(fir_h)))
plt.plot(fir_w, 0*fir_w)
plt.grid()
plt.title(f"Frequency Response of FIR filter with \na = [1], b = [1, 1/2, 1/3]")
plt.ylabel("Amplitude in dB")
plt.xlabel("Frequency in rad/sample")

plt.subplot(1, 2, 2)
plt.plot(fir_w, np.angle(fir_h, deg=True))
plt.grid()
plt.title(f"Phase Response of FIR filter with \na = [1, 1/2], b = [1]")
plt.ylabel("Phase in degrees")
plt.xlabel("Frequency in rad/sample")

plt.show()


iir_a, iir_b = [1, 1/2], [1]
iir_w, iir_h = signal.freqz(iir_b, iir_a)

iir_z, iir_p, _ = signal.tf2zpk(iir_b, iir_a)
print(f"\nZeroes: {np.angle(iir_z)}")
print(f"Poles: {np.angle(iir_p)}")

plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(iir_w, 20 * np.log10(abs(iir_h)))
plt.plot(iir_w, 0*iir_w)
plt.grid()
plt.title(f"Frequency Response of IIR filter with \na = [1, 1/2], b = [1]")
plt.ylabel("Amplitude in dB")
plt.xlabel("Frequency in rad/sample")

plt.subplot(1, 2, 2)
plt.plot(iir_w, np.angle(iir_h, deg=True))
plt.grid()
plt.title(f"Phase Response of IIR filter with \na = [1, 1/2], b = [1]")
plt.ylabel("Phase in degrees")
plt.xlabel("Frequency in rad/sample")

plt.show()


###############################
#B


b = [1, 1, 1, 1, 1]
w, h = signal.freqz(b)

z, p, _ = signal.tf2zpk(iir_b, iir_a)
print(f"\nZeroes: {np.angle(z)}")
print(f"Poles: {np.angle(p)}")

plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(w, 20 * np.log10(abs(h)))
plt.plot(w, 0*w)
plt.grid()
plt.title(f"Frequency Response of $H(z) = 1 + z^-1 + z^-2 + z^-3 + z^-4$")
plt.ylabel("Amplitude in dB")
plt.xlabel("Frequency in rad/sample")

plt.subplot(1, 2, 2)
plt.plot(w, np.angle(h, deg=True))
plt.grid()
plt.title(f"Phase Response of $H(z) = 1 + z^-1 + z^-2 + z^-3 + z^-4$")
plt.ylabel("Phase in degrees")
plt.xlabel("Frequency in rad/sample")

plt.show()