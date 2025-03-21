#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 09:41:19 2025

@author: omdalvii
"""

import numpy as np

def rms(x1, x2):
    sum = 0
    for a, b in zip(x1, x2):
        sum += (a-b)**2

    return np.sqrt(sum/len(x1))


# 10khz tone, 100khz sample rate
vs1 = [0.746321,
      1.70447,
      2.00936,
      1.53165,
      0.461726,
      -0.799985,
      -1.751806,
      -2.052573,
      -1.566092,
      -0.51155]

vi1 = [0.79648,
      1.74241,
      2.01207,
      1.50904,
      0.420942,
      -0.830244,
      -1.76224,
      -2.039765,
      -1.544447,
      -0.456951]

rms10k = rms(vs1, vi1)
print(f"For Fi = 10kHz and Fs = 100kHz, RMS = {rms10k:.4f}")

# 20khz tone, 200khz sample rate
vs2 = [1.00418,
      1.8546,
      1.97978,
      1.34706,
      0.231288,
      -0.994358,
      -1.856872,
      -1.98015,
      -1.339568,
      -0.217673]

vi2 = [1.004,
      1.70736,
      1.84675,
      1.20081,
      -0.04943,
      -1.068237,
      -1.900026,
      -1.980239,
      -1.348763,
      -0.172453]

rms20k = rms(vs2, vi2)
print(f"For Fi = 20kHz and Fs = 200kHz, RMS = {rms20k:.4f}")

inc = rms20k/rms10k
print(f"The RMS error increased by {inc*100:.2f}%")