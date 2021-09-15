import nibabel as nib
import numpy as np
import skimage.io as io
import cv2
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

nii = nib.load('/home/maryana/storage2/Posdoc/AVID/AV13/blockface/nii/mri_brain.nii')
vol = nii.get_data()
vol2 = np.zeros(vol.shape)
nSlides = vol.shape[2]

# img = vol[:,:,160]
# img = img_as_ubyte(img)
# ret2,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# se = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
# th2 = cv2.morphologyEx(th, cv2.MORPH_OPEN, se)
#
# fig = plt.figure(figsize=(10, 5))
# ax1 = fig.add_subplot(1, 3, 1)
# ax1.imshow(img)
# ax2 = fig.add_subplot(1, 3, 2)
# ax2.imshow(th)
# ax3 = fig.add_subplot(1, 3, 3)
# ax3.imshow(th2)


for s in range(nSlides):
    img = vol[:, :, s]
    img = img_as_ubyte(img)
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(th, cv2.MORPH_OPEN, se)
    img[mask == 0] = 0
    vol2[:,:,s] = img

nii2 = nib.Nifti1Image(vol2,nii.affine)
nib.save(nii2,'/home/maryana/storage2/Posdoc/AVID/AV13/blockface/nii/mri_brain_thres.nii')






