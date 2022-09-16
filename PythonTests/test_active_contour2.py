import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import skimage.io as io
from mahotas import bwperim
from skimage import img_as_int
from skimage.measure import find_contours
import scipy.ndimage.morphology as morph


file_name = '/home/maryana/storage/Posdoc/Brainstem/P2540/blockface/seg/P2540_001.png'
mask_name = '/home/maryana/storage/Posdoc/Brainstem/P2540/blockface/seg/mask/mask_P2540_001.png'
img = io.imread(file_name)
img = rgb2gray(img)
mask = io.imread(mask_name)
contour = find_contours(mask,1) #aranges the coordinates in CCW order
init = np.array([contour[0][:,1],contour[0][:,0]]).T


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
plt.gray()
ax.imshow(img)
#ax.plot(init[:, 0], init[:, 1], '.r', lw=3)

img2 = gaussian(img, 3)
snake = active_contour(img, init, alpha=0.010, beta=8, gamma=0.001)
perim = np.round(snake).astype(int)
mask2 = np.zeros(mask.shape,dtype=bool)
mask2[perim[:,1],perim[:,0]] = True
se = morph.generate_binary_structure(2,4)
mask2 = morph.binary_closing(mask2)
mask2 = morph.binary_fill_holes(mask2)


ax.plot(snake[:, 0], snake[:, 1], '-b')
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
plt.show()
