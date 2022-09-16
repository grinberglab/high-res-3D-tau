import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
import rawpy
from scipy.optimize import curve_fit
import skimage.transform as xform
from sklearn.decomposition import FastICA



def compute_a0(volume):  # compute a0
    N = volume.shape[2]
    I = np.sum(volume, axis=2)
    I = I / N
    return I

def compute_a2(volume):
    N = volume.shape[2]  # angles are always 0 to 170 degrees
    ang = np.arange(0, N*10, 10)
    rad = np.radians(ang)
    I = np.zeros([volume.shape[0], volume.shape[1]])
    for i in range(0, N):
        v = volume[..., i]
        v = v * np.cos(2 * rad[i])
        I = I + v

    I = 2 * (I / N)
    return I

def compute_b2(volume):
    N = volume.shape[2]  # angles are always 0 to 170 degrees
    ang = np.arange(0, N*10, 10)
    rad = np.radians(ang)
    I = np.zeros([volume.shape[0], volume.shape[1]])
    for i in range(0, N):
        v = volume[..., i]
        v = v * np.sin(2 * rad[i])
        I = I + v

    I = 2 * (I / N)
    return I

def compute_I0(a0):
    return 2*a0

def compute_direction_eq(a2,b2): # direction Theta
    a = (np.arctan2(b2,(-a2)))/2
    return a

def compute_sind_eq(a0,a2,b2):
    a2s = a2**2
    b2s = b2**2
    sind = np.sqrt(a2s+b2s)/a0
    return sind


# Computes retardation delta from map |sin D|
def compute_delta(sind):
    D = np.arcsin(sind)
    return D

# Computes inclination angle alpha
# if D is a matrix, anpla is computed element wise
#l: lambda (light wavelength)
#d: slice thickness
#RI: refraction index (n1-n0)
#D: retardation computed from |sin D|
def compute_alpha(l,d,RI,D):
    f = np.sqrt((l*D)/(2*np.pi*d*RI))
    a = np.arccos(f)
    return a

# create the function we want to fit
def sin_curve(x, freq, amplitude, phase, offset):
    return np.sin(x * freq + phase) * amplitude + offset

def fit_curve(data,t):
    guess_freq = 1
    guess_amplitude = 3 * np.std(data) / (2 ** 0.5)
    guess_phase = 0
    guess_offset = np.mean(data)
    p0 = [guess_freq, guess_amplitude,
          guess_phase, guess_offset]
    # now do the fit
    fit = curve_fit(sin_curve, t, data, p0=p0, maxfev=10000)
    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = sin_curve(t, *p0)
    # recreate the fitted curve using the optimized parameters
    #data_fit = my_sin(t, *fit[0])
    return fit,data_first_guess















