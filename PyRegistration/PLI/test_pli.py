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

# root_dir = '/Volumes/SUSHI_HD2/SUSHI/Posdoc/PLI/2017/P2694/block-4/TIFF2'
#root_dir = '/home/maryana/storage/Posdoc/PLI/2017/P2694/block-4/TIFF2'
#img_name = 'slice_0002_{:04d}.tif'
#root_dir = '/home/maryana/storage/Posdoc/PLI/2017/test_pli_R6'
root_dir='/Volumes/SUSHI_HD/SUSHI/Posdoc/PLI/2017/test_pli_R6'
mask_file = os.path.join(root_dir,'mask.tif')
img_name = 'Image {:06d}.tif'
nFiles = 18
rrate = 0.20
tmp = io.imread(os.path.join(root_dir,img_name.format(6)))
#tmp = xform.rescale(tmp,rrate)
mask = io.imread(mask_file)
if mask.ndim > 2:
    mask = mask[...,0]

vol = np.zeros([tmp.shape[0], tmp.shape[1], nFiles])
for f in range(nFiles):
    name = os.path.join(root_dir,img_name.format(f+6))
    img = io.imread(name)
    #img = xform.rescale(img,rrate)
    G = img[...,1]
    vol[...,f] = G
#
# vol2 = vol[305:325,652:555,:]

# mask_img = os.path.join(root_dir,'mask_slice_0002_small.tif')
# mask = io.imread(mask_img)

#vol = np.load('mini_data.npy')
# nPix = vol.shape[0]*vol.shape[1]
# ica = FastICA(n_components=2)
# for f in range(0,vol.shape[2]):
#     x = vol[...,f]
#     x = x.reshape([nPix])
#     S = ica.fit_transform(x)  # Reconstruct signals
#     A = ica.mixing_  # Get estimated mixing matrix
#     x2 = S.reshape([vol.shape[0],vol.shape[1]])
#     plt.imshow(x2)

a0 = compute_a0(vol)
I0 = compute_I0(a0)
a2 = compute_a2(vol)
b2 = compute_b2(vol)
dir_map2 = compute_direction_eq(a2,b2)
sind_map2 = compute_sind_eq(a0,a2,b2)

sind_map = np.zeros(I0.shape)
dir_map = np.zeros(I0.shape)

ang = np.arange(0, 180, 10)
t = np.radians(ang)
angles = np.arange(0,180)
t2 = np.radians(angles)
#idx_rows,idx_cols = np.nonzero(mask > 0)
idx_rows = np.arange(0,vol.shape[0])
idx_cols = np.arange(0,vol.shape[1])

for r in idx_rows:
    for c in idx_cols:
        if mask[r,c] == 0:
            continue
        data = vol[r,c,:]
        try:
            fit,data_guess = fit_curve(data,t)
            data_fit = sin_curve(t2, *fit[0])
            i0 = I0[r,c]
            sd = (data_fit.max() - data_fit.min()) / i0
            #|sin delta|
            sind_map[r,c] = sd
            #direction map theta
            idx_min = np.argmin(data_fit, axis=0)
            a = angles[idx_min]
            dir_map[r,c] = a
        except RuntimeError:
            print('Error')
            plt.plot(data)

plt.imshow(dir_map, cmap='viridis')


































