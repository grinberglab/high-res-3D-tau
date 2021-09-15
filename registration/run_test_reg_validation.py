import nibabel as nib
import numpy as np
from misc.TiffTileLoader import ind2sub,sub2ind
import matplotlib.pyplot as plt
import skimage.morphology as morph
import skimage.measure as meas
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
from skimage import img_as_ubyte

def get_window(img,wsize,coords):
    row,col = coords[:]
    if wsize % 2 == 0:
        print('Error: wsize should be an odd value')
        return -1

    w_half = np.ceil(wsize/2.)
    row_c = int(row-w_half)
    col_c = int(col-w_half)
    w = img[row_c:row_c+wsize-1,col_c:col_c+wsize-1]
    return w

def compute_patch_coords(img,grid_rows,grid_cols):

        grid_rows = float(grid_rows)
        grid_cols = float(grid_cols)

        size = img.shape

        #compute row coords
        tile_coords = np.zeros([int(grid_rows*grid_cols),4]) #[row_upper_left, col_upper_left, row_lower_right, col_lower_right]

        row_off = np.floor(size[0]/grid_rows) #initial block size
        row_off = int(row_off)
        row_rem = size[0] % int(grid_rows)
        row_add = np.zeros([int(grid_rows),1]) #correction factor
        if row_rem > 0: # we have to compensate for uneven block sizes (reminder > 0). Make the last block in the row bigger since it's more likely to be background.
            row_add[-1] = row_rem

        col_off = np.floor(size[1]/grid_cols) #initial block size
        col_off = int(col_off)
        col_rem = size[1] % int(grid_cols)
        col_add = np.zeros([int(grid_cols),1]) #correction factor
        if col_rem > 0:
            col_add[-1] = col_rem

        tile_ind = 0
        up_row = 0
        up_col = 0
        for row_count in range(int(grid_rows)):
            for col_count in range(int(grid_cols)):
                #left upper corner
                tile_coords[tile_ind,0] = up_row
                tile_coords[tile_ind,1] = up_col

                low_row = up_row + row_off + row_add[row_count]
                low_col = up_col + col_off + col_add[col_count]
                tile_coords[tile_ind,2] = low_row
                tile_coords[tile_ind,3] = low_col

                #up_col = ((col_count+1) * col_off) + np.sum(col_add[0:col_count + 1])
                up_col = low_col
                tile_ind += 1
            up_col = 0
            #up_row = ((row_count+1) * row_off) + np.sum(row_add[0:row_count + 1])
            up_row = low_row

        return tile_coords

def mutual_information(hgram):
    #Mutual information for joint histogram
    #from: https://matthew-brett.github.io/teaching/mutual_information.html

    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


block_file = '1181_001-Whole-Brain_0478.png.nii'
histo_file = '478bspline.nii'

block_nii = nib.load(block_file)
histo_nii = nib.load(histo_file)

block_img = block_nii.get_data()
histo_img = histo_nii.get_data()

block_img = img_as_ubyte(block_img)
histo_img = img_as_ubyte(histo_img)


if block_img.shape[0] != histo_img.shape[0] or block_img.shape[1] != histo_img.shape[1]:
    print('Error: images differ in size! {}x{}/{}x{}'.format(block_img.shape[0],block_img.shape[1],histo_img.shape[0],histo_img.shape[1]))
    exit()


img_to_show = np.zeros((block_img.shape[0],block_img.shape[1],3))
img_to_show[:,:,0] = block_img
img_to_show[:,:,1] = histo_img

wsize = 81

#test mutual information with whole images
hist_2d, x_edges, y_edges = np.histogram2d(block_img.ravel(),histo_img.ravel(),bins=20)
# Plot as image, arranging axes as for scatterplot
# We transpose to put the T1 bins on the horizontal axis
# and use 'lower' to put 0, 0 at the bottom of the plot
plt.imshow(hist_2d.T, origin='lower', cmap='gray')
plt.xlabel('Block signal bin')
plt.ylabel('Histo signal bin')

# The histogram is easier to see if we show the log values to reduce the effect
# of the bins with a very large number of values:

# Show log histogram, avoiding divide by 0
hist_2d_log = np.zeros(hist_2d.shape)
non_zeros = hist_2d != 0
hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
plt.imshow(hist_2d_log.T, origin='lower', cmap='gray')
plt.xlabel('Block signal bin')
plt.ylabel('Histo signal bin')

print(mutual_information(hist_2d))

#compute Muttual Informatin map

#preprocess and get bounding boxes
block_mask = block_img > 0
histo_mask = histo_img > 0
mask = np.logical_or(block_mask, histo_mask)
mask = mask.astype('uint8')*255
label_mask = morph.label(mask)
props = meas.regionprops(label_mask)
for prop in props:
    # skip small images
    if prop.area < 300:
        continue
    minr, minc, maxr, maxc = prop.bbox

block_img_crop = block_img[minr:maxr,minc:maxc]
histo_img_crop = histo_img[minr:maxr,minc:maxc]

grid_rows = block_img_crop.shape[0]/wsize
grid_cols = block_img_crop.shape[1]/wsize

MI_map = np.zeros(histo_img_crop.shape)

patch_coords = compute_patch_coords(block_img_crop,grid_rows,grid_cols)
for coord in patch_coords:
    row_ul, col_ul, row_lr, col_lr = coord[:]

    row_ul = int(row_ul)
    col_ul = int(col_ul)
    row_lr = int(row_lr)
    col_lr = int(col_lr)

    #print('{},{},{},{}'.format(row_ul, col_ul, row_lr, col_lr))

    wblock = block_img_crop[row_ul:row_lr,col_ul:col_lr]
    whisto = histo_img_crop[row_ul:row_lr, col_ul:col_lr]
    hist_2d, x_edges, y_edges = np.histogram2d(wblock.ravel(), whisto.ravel(), bins=20)
    MI = mutual_information(hist_2d)
    MI_map[row_ul:row_lr,col_ul:col_lr] = MI


#plt.imshow(MI_map, cmap= 'viridis')

#checkerboard
checker = block_img_crop.copy()
counter = 0
for coord in patch_coords:
    row_ul, col_ul, row_lr, col_lr = coord[:]
    row_ul = int(row_ul)
    col_ul = int(col_ul)
    row_lr = int(row_lr)
    col_lr = int(col_lr)

    if counter % 2 != 0:
        checker[row_ul:row_lr,col_ul:col_lr] = histo_img_crop[row_ul:row_lr,col_ul:col_lr]

    counter += 1

plt.imshow(checker, cmap='gray')
plt.show()


norm = mpl.colors.Normalize(vmin=MI_map.min(), vmax=MI_map.max())
cmap = cm.viridis
MI_rgb = cmap(MI_map) #map "colors"
MI_rgb = img_as_ubyte(MI_rgb)
MI_rgb = MI_rgb[:,:,0:3]

overlay = MI_rgb.copy()
output = np.concatenate((img_as_ubyte(histo_img_crop[:,:,np.newaxis]),img_as_ubyte(histo_img_crop[:,:,np.newaxis]),img_as_ubyte(histo_img_crop[:,:,np.newaxis])),axis=2)
alpha = 0.5
output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
#plt.imshow(output)

fig = plt.figure()
ax1 = plt.subplot(121)
ax1.imshow(output)
ax2 = plt.subplot(122)
ax2.imshow(checker, cmap= 'gray')
pass







